"""Flask backend for the Question Paper Generator (replaces the Streamlit UI).
Run:  python server.py        ->  http://127.0.0.1:5000
Storage: SQLite (qpg.db). Bloom labels: LLM (cached in the DB) with rule-based fallback.
"""
import io, json, os, random, re
from collections import Counter

import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

from qpg import LEVELS, DIFFICULTIES, classify, add_difficulty, generate_by_constraints
from export import make_pdf, make_docx
from db import init_db, load_raw, save_raw
from llm_classifier import classify_all, DEFAULT_MODEL

CSV = "questions.csv"
BASE_COLS = ["id", "question", "topic", "marks", "true_bloom_level",
             "bloom_level", "bloom_confidence", "bloom_source"]
SECTIONS = ["Section A - Short Answer", "Section B - Medium Answer", "Section C - Long Answer"]
LEVEL_MARKS = {"Remember": 2, "Understand": 3, "Apply": 5, "Analyze": 5, "Evaluate": 6, "Create": 8}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TEMPLATES = {
    "Remember":   ["Define {topic}.", "List the key features of {topic}.", "State any two applications of {topic}."],
    "Understand": ["Explain the concept of {topic} with an example.", "Describe how {topic} works.",
                   "Summarize the main characteristics of {topic}."],
    "Apply":      ["Implement {topic} for a simple example.", "Solve a problem using {topic} and show the steps.",
                   "Demonstrate the working of {topic} with a small example."],
    "Analyze":    ["Compare {topic} with an alternative approach.", "Analyze the time and space complexity of {topic}.",
                   "Examine the advantages and limitations of {topic}."],
    "Evaluate":   ["Justify the use of {topic} in a real-world scenario.",
                   "Assess the suitability of {topic} for large-scale applications.",
                   "Critique the common implementations of {topic}."],
    "Create":     ["Design a solution to a practical problem using {topic}.",
                   "Develop an algorithm that makes use of {topic}.",
                   "Propose an improved variant of {topic} and explain your design."],
}
TEMPLATE_CSV = "question,topic,marks\nDefine a stack.,Stack,2\nExplain recursion with an example.,Recursion,5\n"
ALIASES = {"questions": "question", "question text": "question", "q": "question",
           "unit": "topic", "chapter": "topic", "module": "topic", "mark": "marks", "score": "marks"}

app = Flask(__name__, static_folder="static", static_url_path="/static")

# single-user demo state
S = {"raw": pd.DataFrame(columns=BASE_COLS), "source": "sample", "paper": [], "pool": [], "target": 0}


# ---------- helpers ----------
def section_of(marks):
    return SECTIONS[0] if marks <= 3 else SECTIONS[1] if marks <= 5 else SECTIONS[2]


def normalize(raw):
    raw = raw.copy()
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    for old, new in ALIASES.items():
        if old in raw.columns and new not in raw.columns:
            raw = raw.rename(columns={old: new})
    if "question" not in raw.columns:
        raise ValueError("the file needs a 'question' column")
    raw = raw.dropna(subset=["question"]).copy()
    raw["question"] = raw["question"].astype(str).str.strip()
    raw = raw[raw["question"] != ""].reset_index(drop=True)
    if "topic" not in raw.columns:
        raw["topic"] = "General"
    raw["topic"] = raw["topic"].fillna("General").astype(str).str.strip().replace("", "General")
    if "marks" not in raw.columns:
        raw["marks"] = 5
    raw["marks"] = pd.to_numeric(raw["marks"], errors="coerce").fillna(5).astype(int).clip(1, 50)
    if "true_bloom_level" not in raw.columns:
        raw["true_bloom_level"] = None
    for c in ("bloom_level", "bloom_confidence", "bloom_source"):
        if c not in raw.columns:
            raw[c] = None
    raw["id"] = [f"Q{i:03d}" for i in range(1, len(raw) + 1)]
    return raw[BASE_COLS]


def persist():
    save_raw(S["raw"], S["source"])


def prepare(raw):
    df = raw.copy()
    if not len(df):
        return df.assign(predicted_level=[], difficulty=[])
    rule = df.question.apply(classify)
    # use the cached LLM label when there is one, else the rule-based label
    df["predicted_level"] = df["bloom_level"].where(df["bloom_level"].isin(LEVELS), rule)
    return add_difficulty(df)


def clean(rec):
    conf, src = rec.get("bloom_confidence"), rec.get("bloom_source")
    return {"id": rec["id"], "question": rec["question"], "topic": rec["topic"], "marks": int(rec["marks"]),
            "level": rec["predicted_level"], "difficulty": rec["difficulty"],
            "confidence": None if conf is None or pd.isna(conf) else round(float(conf), 2),
            "source": None if src is None or pd.isna(src) else src}


def bank_payload():
    df = prepare(S["raw"])
    return [clean(r) for r in df.to_dict("records")]


def add_questions(items):
    raw = S["raw"]
    have = {str(q).strip().lower() for q in raw.question}
    fresh = []
    for it in items:
        key = it["question"].strip().lower()
        if key not in have:
            have.add(key)
            fresh.append(it)
    if fresh:
        nums = [int(i[1:]) for i in raw.id if str(i)[1:].isdigit()]
        start = (max(nums) + 1) if nums else 1
        rows = pd.DataFrame([{"id": f"Q{start + k:03d}", "bloom_level": None, "bloom_confidence": None,
                              "bloom_source": None, **it} for k, it in enumerate(fresh)])[BASE_COLS]
        S["raw"] = rows if raw.empty else pd.concat([raw, rows], ignore_index=True)
        S["raw"]["marks"] = S["raw"]["marks"].astype(int)
        persist()
    return len(fresh), len(items) - len(fresh)


def template_questions(topic, k):
    return [{"level": lvl, "question": tpl.replace("{topic}", topic)}
            for lvl in LEVELS for tpl in TEMPLATES[lvl][:k]]


def ai_questions(topic, k, api_key, model, course):
    import requests
    prompt = (f"Write university exam questions on the topic '{topic}'"
              + (f" for the course '{course}'" if course else "") + ". "
              f"For EACH of these Bloom's taxonomy levels: {', '.join(LEVELS)}, write {k} question(s) that clearly "
              "need that level of thinking and start with a suitable action verb "
              "(Define, Explain, Implement, Compare, Justify, Design, ...). "
              'Reply with JSON only: {"questions": [{"level": "Remember", "question": "..."}]}')
    r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=60, json={
        "model": model, "temperature": 0.7, "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": "You are an expert exam-paper setter. Reply with JSON only."},
                     {"role": "user", "content": prompt}]})
    if not r.ok:
        raise RuntimeError(f"Groq {r.status_code}: {r.text[:300]}")
    data = json.loads(r.json()["choices"][0]["message"]["content"])
    out = []
    for item in data.get("questions", []):
        lvl, q = str(item.get("level", "")).strip().title(), str(item.get("question", "")).strip()
        if lvl in LEVELS and q:
            out.append({"level": lvl, "question": q})
    if not out:
        raise ValueError("the AI returned no usable questions")
    return out


def llm_label_missing(api_key, force=False, model=DEFAULT_MODEL):
    """Label questions with the LLM and cache the result in the DB. Returns (done, failed)."""
    raw = S["raw"]
    todo = raw if force else raw[raw.bloom_level.isna()]
    items = [{"id": r.id, "text": r.question} for r in todo.itertuples()]
    if not items:
        return 0, 0
    res = classify_all(items, rule_fallback=classify, api_key=api_key, model=model)
    done = 0
    for qid, r in res.items():
        if r["source"] == "llm":          # never cache rule-based fallbacks
            m = raw.id == qid
            raw.loc[m, "bloom_level"] = r["level"]
            raw.loc[m, "bloom_confidence"] = r["confidence"]
            raw.loc[m, "bloom_source"] = "llm"
            done += 1
    persist()
    return done, len(items) - done


def paper_payload(sections_on=True):
    paper = S["paper"]
    order = (sorted(range(len(paper)), key=lambda i: (SECTIONS.index(section_of(paper[i]["marks"])), i))
             if sections_on else list(range(len(paper))))
    groups, last = [], "__start__"
    qno = 0
    for i in order:
        q = paper[i]
        sec = section_of(q["marks"]) if sections_on else None
        if sec != last:
            groups.append({"heading": sec, "items": []})
            last = sec
        qno += 1
        groups[-1]["items"].append({**clean(q), "no": qno})
    got = sum(q["marks"] for q in paper)
    return {"groups": groups, "total": got, "target": S["target"], "exact": got == S["target"],
            "count": len(paper), "manual": S.get("manual", False)}


def groups_for_export(sections_on):
    p = paper_payload(sections_on)
    return [(g["heading"], [(it["no"], it["question"], it["marks"]) for it in g["items"]]) for g in p["groups"]], p["total"]


# ---------- routes ----------
@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/bank")
def get_bank():
    return jsonify(bank=bank_payload(), source=S["source"], levels=LEVELS, difficulties=DIFFICULTIES)


@app.post("/api/source")
def set_source():
    src = request.json.get("source", "sample")
    S.update(source=src, paper=[], pool=[], target=0)
    err = None
    try:
        S["raw"] = load_raw(src)
    except Exception as e:
        S["raw"], err = pd.DataFrame(columns=BASE_COLS), str(e)
    return jsonify(bank=bank_payload(), source=src, error=err)


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file received"), 400
    data = f.read()
    try:
        try:
            df = pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
        S.update(raw=normalize(df), source="upload", paper=[], pool=[], target=0)
        persist()
    except Exception as e:
        return jsonify(error=f"Could not read the file: {e}"), 400
    return jsonify(bank=bank_payload(), source="upload")


@app.get("/api/template.csv")
def template():
    return send_file(io.BytesIO(TEMPLATE_CSV.encode()), mimetype="text/csv",
                     as_attachment=True, download_name="question_bank_template.csv")


@app.get("/api/bank.csv")
def bank_csv():
    return send_file(io.BytesIO(S["raw"][BASE_COLS].to_csv(index=False).encode()), mimetype="text/csv",
                     as_attachment=True, download_name="question_bank.csv")


@app.post("/api/bank/add")
def bank_add():
    j = request.json
    lines = [l.strip() for l in str(j.get("text", "")).splitlines() if l.strip()]
    if not lines:
        return jsonify(error="Type at least one question."), 400
    topic = (j.get("topic") or "General").strip() or "General"
    marks = max(1, min(50, int(j.get("marks", 5))))
    added, skipped = add_questions([{"question": l, "topic": topic, "marks": marks, "true_bloom_level": None}
                                    for l in lines])
    detected = dict(Counter(classify(l) for l in lines))
    return jsonify(bank=bank_payload(), added=added, skipped=skipped, detected=detected, topic=topic)


@app.post("/api/bank/save")
def bank_save():
    rows = request.json.get("rows", [])
    old = S["raw"].set_index("id") if not S["raw"].empty else None
    kept = []
    for r in rows:
        q = str(r.get("question", "")).strip()
        if not q:
            continue
        keep = {"true_bloom_level": None, "bloom_level": None, "bloom_confidence": None, "bloom_source": None}
        # unchanged questions keep their labels; edited ones are re-labelled
        if old is not None and r.get("id") in old.index and old.loc[r["id"], "question"] == q:
            for k in keep:
                v = old.loc[r["id"], k]
                keep[k] = None if pd.isna(v) else v
        kept.append({"id": r.get("id", ""), "question": q,
                     "topic": str(r.get("topic") or "General").strip() or "General",
                     "marks": max(1, min(50, int(r.get("marks") or 5))), **keep})
    S["raw"] = pd.DataFrame(kept, columns=BASE_COLS) if kept else pd.DataFrame(columns=BASE_COLS)
    if kept:
        S["raw"]["id"] = [f"Q{i:03d}" for i in range(1, len(S["raw"]) + 1)]
    persist()
    return jsonify(bank=bank_payload())


@app.post("/api/bank/classify")
def bank_classify():
    """Label questions with the LLM (only unlabelled ones unless force=true)."""
    j = request.get_json(silent=True) or {}
    key = j.get("api_key") or os.environ.get("GROQ_API_KEY", "")
    if S["raw"].empty:
        return jsonify(error="The question bank is empty."), 400
    if not key:
        return jsonify(error="Set GROQ_API_KEY or send api_key."), 400
    done, failed = llm_label_missing(key, bool(j.get("force", False)), j.get("model") or DEFAULT_MODEL)
    return jsonify(bank=bank_payload(), classified=done, failed=failed)


@app.post("/api/topics/generate")
def topics_generate():
    j = request.json
    names = list(dict.fromkeys(t.strip() for t in re.split(r"[,\n]", j.get("topics", "")) if t.strip()))
    method, k = j.get("method", "templates"), max(1, min(3, int(j.get("per_level", 1))))
    key = j.get("api_key") or os.environ.get("GROQ_API_KEY", "")
    if not names:
        return jsonify(error="Type at least one topic."), 400
    if method == "ai" and not key:
        return jsonify(error="Paste your Groq API key first."), 400
    try:
        items = []
        for name in names:
            got = (ai_questions(name, k, key, j.get("model") or DEFAULT_MODEL, (j.get("course") or "").strip())
                   if method == "ai" else template_questions(name, k))
            items += [{"question": g["question"], "topic": name, "marks": LEVEL_MARKS[g["level"]],
                       "true_bloom_level": g["level"]} for g in got]
        added, skipped = add_questions(items)
    except Exception as e:
        return jsonify(error=f"Could not generate questions: {e}"), 500
    return jsonify(bank=bank_payload(), added=added, skipped=skipped, topics=len(names))


@app.post("/api/paper/generate")
def paper_generate():
    j = request.json
    df = prepare(S["raw"])
    levels = j.get("levels") or LEVELS
    pool_df = df[df.predicted_level.isin(levels)] if len(df) else df
    if pool_df.empty:
        return jsonify(error="No questions available for the chosen levels."), 400
    paper, _ = generate_by_constraints(pool_df, int(j.get("n", 10)), int(j.get("marks", 40)),
                                       j.get("difficulty", "Mixed"), None, int(j.get("seed", 1)))
    S.update(paper=paper, pool=pool_df.to_dict("records"), target=int(j.get("marks", 40)), manual=False)
    return jsonify(paper_payload(j.get("sections", True)))


@app.post("/api/paper/manual")
def paper_manual():
    """Build a paper from hand-picked question ids (any number, no marks target)."""
    j = request.json
    ids = list(dict.fromkeys(j.get("ids") or []))
    df = prepare(S["raw"])
    if df.empty or not ids:
        return jsonify(error="Pick at least one question."), 400
    by_id = {r["id"]: r for r in df.to_dict("records")}
    paper = [by_id[i] for i in ids if i in by_id]
    if not paper:
        return jsonify(error="None of the selected questions exist any more."), 400
    S.update(paper=paper, pool=list(by_id.values()), target=sum(q["marks"] for q in paper), manual=True)
    return jsonify(paper_payload(j.get("sections", True)))


@app.get("/api/analytics")
def analytics():
    """Evaluate the Bloom classifier on questions that have a ground-truth label."""
    df = prepare(S["raw"])
    out = {"total": int(len(df)), "labeled": 0}
    if df.empty:
        return jsonify(out)
    lab = df.dropna(subset=["true_bloom_level"])
    lab = lab[lab.true_bloom_level.isin(LEVELS)]
    out["labeled"] = int(len(lab))
    if lab.empty:
        return jsonify(out)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    y, p = lab.true_bloom_level, lab.predicted_level
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=LEVELS, zero_division=0)
    out.update(
        accuracy=float(accuracy_score(y, p)),
        per_class=[{"level": l, "precision": float(a), "recall": float(b), "f1": float(c), "support": int(d)}
                   for l, a, b, c, d in zip(LEVELS, pr, rc, f1, sup)],
        confusion=confusion_matrix(y, p, labels=LEVELS).tolist(),
        levels=LEVELS)
    out["rule_accuracy"] = float(accuracy_score(y, lab.question.apply(classify)))
    try:   # ML baseline: TF-IDF + logistic regression with cross-validation
        from sklearn.pipeline import make_pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        smallest = int(y.value_counts().min())
        if y.nunique() >= 2 and smallest >= 2:
            k = min(5, smallest)
            pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
            s = cross_val_score(pipe, lab.question, y, cv=StratifiedKFold(k, shuffle=True, random_state=42))
            out["baseline"] = {"accuracy": float(s.mean()), "folds": k}
    except Exception as e:
        out["baseline_error"] = str(e)
    return jsonify(out)


@app.post("/api/paper/layout")
def paper_layout():
    return jsonify(paper_payload(request.json.get("sections", True)))


@app.post("/api/paper/swap")
def paper_swap():
    j = request.json
    paper = S["paper"]
    idx = next((i for i, q in enumerate(paper) if q["id"] == j.get("id")), None)
    if idx is None:
        return jsonify(error="Question not in paper"), 404
    old = paper[idx]
    used = {q["id"] for q in paper}
    free = [q for q in S["pool"] if q["id"] not in used and q["marks"] == old["marks"]]
    options = [q for q in free if q["predicted_level"] == old["predicted_level"]] or free
    msg = ""
    if options:
        paper[idx] = random.choice(options)
    else:
        msg = "No replacement with the same marks is available."
    return jsonify({**paper_payload(j.get("sections", True)), "msg": msg})


@app.post("/api/export/<fmt>")
def export(fmt):
    j = request.json
    groups, total = groups_for_export(j.get("sections", True))
    if not groups:
        return jsonify(error="Generate a paper first."), 400
    title, subject, duration = j.get("title", "Exam"), j.get("subject", ""), j.get("duration", "")
    if fmt == "pdf":
        data, mime, name = make_pdf(groups, title, subject, duration, total), "application/pdf", "question_paper.pdf"
    elif fmt == "docx":
        data = make_docx(groups, title, subject, duration, total)
        mime, name = "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "question_paper.docx"
    else:
        txt = "\n".join((f"\n{h}" if h else "") + "".join(f"\nQ{n}. {t} [{m} marks]" for n, t, m in items)
                        for h, items in groups)
        data, mime, name = txt.encode(), "text/plain", "question_paper.txt"
    return send_file(io.BytesIO(data), mimetype=mime, as_attachment=True, download_name=name)


def boot():
    init_db()
    if load_raw("sample").empty and os.path.exists(CSV):   # first run: seed the DB from the CSV
        save_raw(normalize(pd.read_csv(CSV)), "sample")
    S["raw"] = load_raw("sample")


if __name__ == "__main__":
    boot()
    app.run(debug=True)

    