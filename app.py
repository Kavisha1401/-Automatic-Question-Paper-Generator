import json
import os
import random
import re
from collections import Counter
import pandas as pd
import streamlit as st
from qpg import LEVELS, DIFFICULTIES, classify, add_difficulty, generate_by_constraints
from export import make_pdf, make_docx

CSV = "questions.csv"
BASE_COLS = ["id", "question", "topic", "marks", "true_bloom_level"]
SECTIONS = ["Section A - Short Answer", "Section B - Medium Answer", "Section C - Long Answer"]
SAMPLE, UPLOAD, TOPICS = "Sample bank", "Upload my own question bank", "Type topics and auto-generate questions"
LEVEL_MARKS = {"Remember": 2, "Understand": 3, "Apply": 5, "Analyze": 5, "Evaluate": 6, "Create": 8}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
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
TEMPLATE = "question,topic,marks\nDefine a stack.,Stack,2\nExplain recursion with an example.,Recursion,5\n"
ALIASES = {"questions": "question", "question text": "question", "q": "question",
           "unit": "topic", "chapter": "topic", "module": "topic",
           "mark": "marks", "score": "marks"}

st.set_page_config(page_title="Question Paper Generator", layout="wide")
ss = st.session_state


# ---------- helpers ----------
def section_of(marks):
    if marks <= 3:
        return SECTIONS[0]
    if marks <= 5:
        return SECTIONS[1]
    return SECTIONS[2]


def normalize(raw):
    """Accept almost any question-bank CSV: only a 'question' column is required."""
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
    raw["id"] = [f"Q{i:03d}" for i in range(1, len(raw) + 1)]
    return raw[BASE_COLS]


def read_any_csv(f):
    try:
        return pd.read_csv(f)
    except UnicodeDecodeError:
        f.seek(0)
        return pd.read_csv(f, encoding="latin-1")


def load_bank(source, uploaded):
    if source == SAMPLE:
        return normalize(pd.read_csv(CSV))
    if source == UPLOAD and uploaded is not None:
        return normalize(read_any_csv(uploaded))
    return pd.DataFrame(columns=BASE_COLS)


def persist():
    """Only the sample bank is saved back to questions.csv; other banks live in the session."""
    if ss.sig and ss.sig[0] == SAMPLE:
        ss.raw[BASE_COLS].to_csv(CSV, index=False)


def prepare(raw):
    df = raw.copy()
    df["predicted_level"] = df.question.apply(classify)
    return add_difficulty(df)


def swap(i):
    """Replace question i with an unused one of the same marks (keeps total marks)."""
    paper, old = ss.paper, ss.paper[i]
    used = {q["id"] for q in paper}
    free = [q for q in ss.pool if q["id"] not in used and q["marks"] == old["marks"]]
    same_level = [q for q in free if q["predicted_level"] == old["predicted_level"]]
    options = same_level or free
    if options:
        paper[i] = random.choice(options)
        ss.msg = ""
    else:
        ss.msg = "No replacement with the same marks is available."


def add_questions(items):
    """items = dicts with question, topic, marks, true_bloom_level. Skips duplicates."""
    have = {str(q).strip().lower() for q in ss.raw.question}
    fresh = []
    for it in items:
        key = it["question"].strip().lower()
        if key not in have:
            have.add(key)
            fresh.append(it)
    if fresh:
        nums = [int(i[1:]) for i in ss.raw.id if str(i)[1:].isdigit()]
        start = (max(nums) + 1) if nums else 1
        rows = pd.DataFrame([{"id": f"Q{start + k:03d}", **it} for k, it in enumerate(fresh)])[BASE_COLS]
        ss.raw = rows if ss.raw.empty else pd.concat([ss.raw, rows], ignore_index=True)
        ss.raw["marks"] = ss.raw["marks"].astype(int)
        persist()
        ss.ver += 1
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
    r.raise_for_status()
    data = json.loads(r.json()["choices"][0]["message"]["content"])
    out = []
    for item in data.get("questions", []):
        lvl, q = str(item.get("level", "")).strip().title(), str(item.get("question", "")).strip()
        if lvl in LEVELS and q:
            out.append({"level": lvl, "question": q})
    if not out:
        raise ValueError("the AI returned no usable questions")
    return out


# ---------- state ----------
for k, v in dict(paper=[], pool=[], target=0, msg="", ver=0, sig=None, load_err="").items():
    ss.setdefault(k, v)

st.title("Automatic Question Paper Generator")
st.caption("Bloom's Taxonomy based - intermediate demo")

# ---------- sidebar: choose the question bank ----------
sb = st.sidebar
sb.header("Question bank")
source = sb.radio("Source", [SAMPLE, UPLOAD, TOPICS], label_visibility="collapsed")
uploaded = None
if source == UPLOAD:
    uploaded = sb.file_uploader("Upload a CSV (needs a 'question' column; 'topic' and 'marks' are optional)",
                                type="csv")
    sb.download_button("Download CSV template", TEMPLATE, "question_bank_template.csv")

sig = (source, uploaded.name, uploaded.size) if uploaded else (source,)
if ss.sig != sig:
    try:
        ss.raw, ss.load_err = load_bank(source, uploaded), ""
    except Exception as e:
        ss.raw, ss.load_err = pd.DataFrame(columns=BASE_COLS), f"Could not read the file: {e}"
    ss.sig = sig
    ss.paper, ss.pool, ss.target, ss.msg = [], [], 0, ""
    ss.ver += 1
if ss.load_err:
    sb.error(ss.load_err)
sb.caption(f"{len(ss.raw)} questions in the bank")

if source == TOPICS:
    with sb.expander("Generate questions from topics", expanded=True):
        topics_text = st.text_area("Topics (separate with commas or new lines)",
                                   placeholder="Stack, Queue, Binary Trees")
        course = st.text_input("Subject / course (optional, helps the AI)")
        method = st.radio("Method", ["Offline templates", "AI (Groq)"],
                          help="Templates work offline with no key. AI writes better, topic-specific questions.")
        per_level = st.slider("Questions per Bloom's level", 1, 3, 1)
        api_key, model = "", DEFAULT_MODEL
        if method == "AI (Groq)":
            api_key = st.text_input("Groq API key", type="password",
                                    help="Free key at console.groq.com. Or set the GROQ_API_KEY environment variable.")
            model = st.text_input("Model", DEFAULT_MODEL)
        if st.button("Generate questions", type="primary"):
            names = list(dict.fromkeys(t.strip() for t in re.split(r"[,\n]", topics_text) if t.strip()))
            key = api_key or os.environ.get("GROQ_API_KEY", "")
            if not names:
                st.error("Type at least one topic.")
            elif method == "AI (Groq)" and not key:
                st.error("Paste your Groq API key first.")
            else:
                try:
                    items = []
                    with st.spinner("Generating questions..."):
                        for name in names:
                            got_q = (ai_questions(name, per_level, key, model, course.strip())
                                     if method == "AI (Groq)" else template_questions(name, per_level))
                            items += [{"question": g["question"], "topic": name, "marks": LEVEL_MARKS[g["level"]],
                                       "true_bloom_level": g["level"]} for g in got_q]
                    added, skipped = add_questions(items)
                    ss.flash = (f"Generated {added} question(s) for {len(names)} topic(s)"
                                + (f" ({skipped} duplicates skipped)" if skipped else "") + ".")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not generate questions: {e}")

df = prepare(ss.raw)

# ---------- sidebar: paper settings ----------
sb.header("Paper settings")
all_topics = sorted(df.topic.unique())
levels = sb.multiselect("Bloom's levels", LEVELS, default=LEVELS)
pool_df = df[df.predicted_level.isin(levels)]
n_max = max(1, len(pool_df))
n_questions = sb.number_input("Number of questions", 1, n_max, min(10, n_max))
total_marks = sb.number_input("Total marks", 1, 500, 40)
difficulty = sb.selectbox("Difficulty level", ["Mixed", "Easy", "Medium", "Hard"])
sections_on = sb.checkbox("Section-wise layout", value=True)
seed = sb.number_input("Seed (change for a new paper)", 0, 9999, 1)
with sb.expander("Paper details (for PDF / Word)"):
    exam_title = st.text_input("Exam title", "Internal Examination")
    subject = st.text_input("Subject", "Data Structures")
    duration = st.text_input("Duration", "2 Hours")

if ss.get("flash"):
    st.success(ss.flash)
    ss.flash = ""

tab1, tab2 = st.tabs(["Generate paper", "Question bank"])

# ---------- tab 1: generate / swap / export ----------
with tab1:
    if pool_df.empty:
        st.warning("No questions available. Type topics in the sidebar and click 'Generate questions', "
                   "upload a question bank, or add questions in the 'Question bank' tab.")
    elif st.button("Generate paper", type="primary"):
        paper, _ = generate_by_constraints(pool_df, int(n_questions), int(total_marks),
                                           difficulty, None, int(seed))
        ss.paper, ss.pool = paper, pool_df.to_dict("records")
        ss.target, ss.msg = int(total_marks), ""

    paper = ss.paper
    if not paper:
        if not pool_df.empty:
            st.info("Choose your settings in the sidebar and click **Generate paper**.")
    else:
        got = sum(q["marks"] for q in paper)
        if got == ss.target:
            st.success(f"Total marks: {got} (matches your target)")
        else:
            st.warning(f"Total marks: {got} - exact {ss.target} not possible with this bank; closest match shown.")
        if ss.msg:
            st.info(ss.msg)

        if sections_on:
            order = sorted(range(len(paper)), key=lambda i: (SECTIONS.index(section_of(paper[i]["marks"])), i))
        else:
            order = list(range(len(paper)))

        groups, current, last = [], None, "__start__"
        for qno, i in enumerate(order, 1):
            q = paper[i]
            sec = section_of(q["marks"]) if sections_on else None
            if sec != last:
                if sec:
                    st.markdown(f"#### {sec}")
                current = (sec, [])
                groups.append(current)
                last = sec
            current[1].append((qno, q["question"], q["marks"]))
            c1, c2, c3 = st.columns([0.07, 0.75, 0.18])
            c1.write(f"**Q{qno}.**")
            c2.write(q["question"])
            c2.caption(f"{q['marks']} marks | {q['difficulty']} | {q['predicted_level']} | {q['topic']}")
            c3.button("Swap", key=f"swap_{q['id']}", on_click=swap, args=(i,))

        out = pd.DataFrame(paper)
        st.divider()
        d1, d2, d3 = st.columns(3)
        d1.write("Bloom's levels")
        d1.bar_chart(out.predicted_level.value_counts().reindex(LEVELS).fillna(0))
        d2.write("Difficulty")
        d2.bar_chart(out.difficulty.value_counts().reindex(DIFFICULTIES).fillna(0))
        d3.write("Topics")
        d3.bar_chart(out.topic.value_counts())

        st.subheader("Download")
        txt = "\n".join((f"\n{h}" if h else "") + "".join(f"\nQ{n}. {t} [{m} marks]" for n, t, m in items)
                        for h, items in groups)
        e1, e2, e3 = st.columns(3)
        e1.download_button("PDF", make_pdf(groups, exam_title, subject, duration, got),
                           "question_paper.pdf", "application/pdf")
        e2.download_button("Word (.docx)", make_docx(groups, exam_title, subject, duration, got),
                           "question_paper.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        e3.download_button("Text (.txt)", txt, "question_paper.txt")

# ---------- tab 2: question bank (add / edit / delete) ----------
with tab2:
    st.subheader("Add questions")
    with st.form("add_q", clear_on_submit=True):
        q_text = st.text_area("Question(s) - one per line",
                              help="Paste many questions at once. Each line becomes a question "
                                   "with the topic and marks chosen below.")
        a, b = st.columns(2)
        topic_pick = a.selectbox("Topic", all_topics)
        topic_new = a.text_input("...or type a new topic")
        marks = b.number_input("Marks (for every question added)", 1, 20, 5)
        ok = st.form_submit_button("Add to question bank")
    if ok:
        lines = [l.strip() for l in q_text.splitlines() if l.strip()]
        if not lines:
            st.warning("Type at least one question.")
        else:
            topic = topic_new.strip() or topic_pick or "General"
            added, skipped = add_questions([{"question": l, "topic": topic, "marks": int(marks),
                                             "true_bloom_level": None} for l in lines])
            ss.flash = (f"Added {added} question(s) under '{topic}'. "
                        f"Detected Bloom's levels: {dict(Counter(classify(l) for l in lines))}"
                        + (f" ({skipped} duplicates skipped)" if skipped else ""))
            st.rerun()   # refresh topics, filters and the Generate button

    st.subheader("Edit or delete questions")
    if ss.raw.empty:
        st.info("The question bank is empty. Generate questions from topics in the sidebar, add them above, or upload a file.")
    else:
        view = ss.raw[["id", "question", "topic", "marks"]].copy()
        view["delete"] = False
        edited = st.data_editor(view, hide_index=True, key=f"editor_{ss.ver}", disabled=["id"],
                                column_config={"marks": st.column_config.NumberColumn(min_value=1, max_value=20, step=1)})
        s1, s2 = st.columns([1, 3])
        if s1.button("Save changes"):
            kept = edited[~edited["delete"]].drop(columns="delete").reset_index(drop=True)
            old = ss.raw.set_index("id")
            kept["true_bloom_level"] = [
                old.true_bloom_level.get(r.id) if old.question.get(r.id) == r.question else None
                for r in kept.itertuples()]
            kept["marks"] = kept["marks"].astype(int)
            ss.raw = kept[BASE_COLS]
            persist()
            ss.ver += 1
            st.rerun()
        s2.download_button("Download this question bank (CSV)", ss.raw[BASE_COLS].to_csv(index=False),
                           "question_bank.csv", "text/csv")

        st.subheader("Predicted Bloom's level for every question")
        st.dataframe(df[["id", "question", "topic", "marks", "predicted_level", "difficulty"]], hide_index=True)
        st.write("Questions per level:", df.predicted_level.value_counts().to_dict())