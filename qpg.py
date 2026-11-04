"""Automatic Question Paper Generator with Bloom's Taxonomy - intermediate demo.
Run:  python qpg.py            (add --seed 7 for a different paper)
"""
import argparse, random, re
from collections import Counter
import pandas as pd

LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

# ---------- Step 1: rule-based Bloom classifier ----------
VERBS = {
    "Remember":   ["define", "list", "state", "name", "identify", "recall", "write"],
    "Understand": ["explain", "describe", "summarize", "summarise", "illustrate", "classify", "discuss"],
    "Apply":      ["solve", "calculate", "implement", "use", "apply", "construct", "demonstrate", "convert", "compute"],
    "Analyze":    ["compare", "differentiate", "distinguish", "examine", "analyze", "analyse", "investigate", "break"],
    "Evaluate":   ["justify", "assess", "critique", "evaluate", "judge", "recommend", "argue", "defend"],
    "Create":     ["design", "develop", "propose", "create", "formulate", "invent", "compose"],
}
VERB2LEVEL = {v: lvl for lvl, vs in VERBS.items() for v in vs}
PATTERNS = [(r"^what is", "Remember"), (r"^(how|why)\b", "Understand")]

def classify(question: str) -> str:
    q = question.lower().strip()
    first = re.findall(r"[a-z]+", q)[0]
    if first in VERB2LEVEL:                      # verb at the start is most reliable
        return VERB2LEVEL[first]
    for pat, lvl in PATTERNS:
        if re.match(pat, q):
            return lvl
    for w in re.findall(r"[a-z]+", q):           # fallback: first known verb anywhere
        if w in VERB2LEVEL:
            return VERB2LEVEL[w]
    return "Understand"                          # default

# ---------- Step 2: paper generator ----------
def generate_paper(df, blueprint, seed=None):
    rng = random.Random(seed)
    chosen, topic_count = [], Counter()
    for lvl in LEVELS:
        need = blueprint.get(lvl, 0)
        pool = df[df.predicted_level == lvl].to_dict("records")
        rng.shuffle(pool)
        for _ in range(min(need, len(pool))):
            pool.sort(key=lambda r: topic_count[r["topic"]])   # prefer least-used topic
            pick = pool.pop(0)
            topic_count[pick["topic"]] += 1
            chosen.append(pick)
        if len(pool) + need > 0 and need > len(df[df.predicted_level == lvl]):
            print(f"[warning] only {len(df[df.predicted_level == lvl])} '{lvl}' questions available")
    return chosen

def print_paper(paper, title="Data Structures - Internal Exam"):
    total = sum(q["marks"] for q in paper)
    print("=" * 64, f"\n{title}\nTotal Marks: {total}\n" + "=" * 64)
    for n, q in enumerate(paper, 1):
        print(f"Q{n}. {q['question']}  [{q['marks']} marks | {q['predicted_level']} | {q['topic']}]")
    print("-" * 64)
    print("Bloom's coverage:", dict(Counter(q["predicted_level"] for q in paper)))
    print("Topic coverage  :", dict(Counter(q["topic"] for q in paper)))

# ---------- Step 3: evaluation of the classifier ----------
def evaluate(df):
    from sklearn.metrics import accuracy_score, classification_report
    df = df.dropna(subset=["true_bloom_level"])   # skip questions added via the app
    acc = accuracy_score(df.true_bloom_level, df.predicted_level)
    print(f"\nRule-based classifier accuracy: {acc:.2%}  ({len(df)} questions)")
    print(classification_report(df.true_bloom_level, df.predicted_level, labels=LEVELS, zero_division=0))
    # ML baseline for comparison (TF-IDF + Logistic Regression, 5-fold CV)
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    s = cross_val_score(pipe, df.question, df.true_bloom_level, cv=cv)
    print(f"ML baseline (TF-IDF + LogReg, 5-fold CV) accuracy: {s.mean():.2%}")


# ---------- Step 4: constraint-based generator (questions, marks, difficulty) ----------
DIFFICULTIES = ["Easy", "Medium", "Hard"]
DIFF_FROM_BLOOM = {"Remember": "Easy", "Understand": "Easy", "Apply": "Medium",
                   "Analyze": "Medium", "Evaluate": "Hard", "Create": "Hard"}
DIFF_MIX = {"Easy": (0.6, 0.3, 0.1), "Medium": (0.3, 0.5, 0.2),
            "Hard": (0.1, 0.3, 0.6), "Mixed": (0.34, 0.33, 0.33)}

def add_difficulty(df):
    """Use the CSV's own 'difficulty' column if present, else derive it from Bloom's level."""
    if "difficulty" not in df.columns:
        df["difficulty"] = df.predicted_level.map(DIFF_FROM_BLOOM)
    return df

def generate_by_constraints(df, n_questions, total_marks, difficulty="Mixed",
                            levels=None, seed=None, tries=5000):
    """Pick n_questions whose marks add up to total_marks (or as close as possible)
    with the requested overall difficulty. Returns (paper, exact_match_flag)."""
    rng = random.Random(seed)
    pool = df if not levels else df[df.predicted_level.isin(levels)]
    pool = pool.to_dict("records")
    n_questions = min(n_questions, len(pool))
    target = [round(p * n_questions) for p in DIFF_MIX[difficulty]]
    best, best_score = None, float("inf")
    for _ in range(tries):
        pick = rng.sample(pool, n_questions)
        marks_err = abs(sum(q["marks"] for q in pick) - total_marks)
        counts = Counter(q["difficulty"] for q in pick)
        mix_err = sum(abs(counts[d] - t) for d, t in zip(DIFFICULTIES, target))
        topic_pen = n_questions - len({q["topic"] for q in pick})   # prefer topic spread
        score = marks_err * 100 + mix_err * 5 + topic_pen
        if score < best_score:
            best, best_score = pick, score
            if marks_err == 0 and mix_err == 0 and topic_pen == 0:
                break
    best.sort(key=lambda q: (LEVELS.index(q["predicted_level"]), q["id"]))
    exact = sum(q["marks"] for q in best) == total_marks
    return best, exact

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="questions.csv")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["predicted_level"] = df.question.apply(classify)
    print(df[["id", "question", "predicted_level"]].head(8).to_string(index=False))
    evaluate(df)

    blueprint = {"Remember": 2, "Understand": 2, "Apply": 2, "Analyze": 2, "Evaluate": 1, "Create": 1}
    print("\nBlueprint:", blueprint, "\n")
    print_paper(generate_paper(df, blueprint, args.seed))