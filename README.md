# Automatic Question Paper Generator (Bloom's Taxonomy)

A web app that builds exam question papers from a question bank. Each question is automatically classified into a level of **Bloom's Taxonomy** (Remember, Understand, Apply, Analyze, Evaluate, Create), and papers are generated to match your chosen number of questions, total marks, and difficulty. Papers can be exported as **PDF, Word (.docx) or plain text**.

## Features

- **Rule-based Bloom's classifier** – predicts the cognitive level of a question from its leading action verb (e.g. *Define* → Remember, *Compare* → Analyze, *Design* → Create).
- **Constraint-based paper generation** – picks questions that hit your target total marks, difficulty mix (Easy / Medium / Hard / Mixed) and spread across topics.
- **Auto-select or manual pick** – let the app choose, or hand-pick questions from the bank.
- **Swap a question** – replace any question with another one of the same marks (and same Bloom's level where possible).
- **Section-wise layout** – Section A (≤3 marks), Section B (4–5 marks), Section C (6+ marks).
- **Three ways to build a question bank**
  - Use the bundled **sample bank** (`questions.csv`, Data Structures)
  - **Upload your own CSV**
  - **Generate from topics** – offline templates, or AI-written questions via the **Groq API**
- **Question bank manager** – add, edit, delete and download the bank as CSV.
- **Classifier analytics** – accuracy, precision/recall/F1, confusion matrix, and a TF-IDF + Logistic Regression baseline (for questions that have a ground-truth label).
- **Export** – PDF, Word, Text, or print directly from the browser.

## Project Structure

```
ILLM/
├── server.py          # Flask backend + REST API (main entry point)
├── qpg.py             # Bloom's classifier + paper generation + evaluation (also runs as a CLI)
├── export.py          # PDF (reportlab) and Word (python-docx) export
├── app.py             # Older Streamlit version of the UI (optional)
├── questions.csv      # Sample question bank (52 questions)
└── static/
    └── index.html     # Frontend (single-page UI)
```

## Prerequisites

- **Python 3.9+** (tested with 3.11)
- `pip`
- (Optional) A free **Groq API key** from <https://console.groq.com> for AI-generated questions

## Setup & Run

### 1. Go to the project folder

```bash
cd ILLM
```

### 2. (Recommended) Create a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install flask pandas reportlab python-docx requests scikit-learn
```

### 4. Start the server

```bash
python server.py
```

### 5. Open the app

Go to **http://127.0.0.1:5000** in your browser.

## How to Use

1. **Choose a question bank** (*Question bank* page)
   - **Sample bank** – works out of the box.
   - **Upload CSV** – needs a `question` column; `topic` and `marks` are optional. Use *Download CSV template* for the format.
   - **Generate from topics** – type topics (comma or newline separated), choose *Offline templates* or *AI (Groq)*, and click **Generate questions**.
2. **Generate a paper** (*Generate* page)
   - Select Bloom's levels, number of questions, total marks, difficulty, and a seed (change the seed for a different paper).
   - Fill in exam title, subject and duration, then click **Generate paper**.
   - Use **Swap** on any question, or **Regenerate** for a fresh set.
   - Or switch to **Pick manually** to choose specific questions.
3. **Export** – download as **PDF**, **Word** or **Text**, or **Print**.

## Question Bank CSV Format

```csv
question,topic,marks
Define a stack.,Stack,2
Explain recursion with an example.,Recursion,5
```

| Column | Required | Notes |
|---|---|---|
| `question` | Yes | Question text |
| `topic` | No | Defaults to `General` |
| `marks` | No | Defaults to `5` (clamped to 1–50) |
| `true_bloom_level` | No | Ground-truth label, used only for the analytics page |

Common alternative headers (`questions`, `unit`, `chapter`, `module`, `mark`, `score`) are recognised automatically.

## Using AI Question Generation (Groq)

Either paste the key in the UI, or set it as an environment variable before starting the server:

```bash
# macOS / Linux
export GROQ_API_KEY="your_key_here"

# Windows (PowerShell)
$env:GROQ_API_KEY="your_key_here"

python server.py
```

Default model: `llama-3.3-70b-versatile` (changeable in the UI).

## Other Ways to Run

**Command-line demo** (classifier evaluation + sample paper):

```bash
python qpg.py --seed 7
```

**Streamlit UI** (older interface; also needs `pip install streamlit`):

```bash
streamlit run app.py
```

## How It Works

1. **Classification** – The first word of a question is matched against Bloom's action-verb lists. If there is no match, fallback patterns (`What is…` → Remember, `How/Why…` → Understand) and then any verb found in the sentence are used. The default is *Understand*.
2. **Difficulty** – Derived from the Bloom's level: Remember/Understand → Easy, Apply/Analyze → Medium, Evaluate/Create → Hard.
3. **Paper generation** – Randomly samples up to 5000 candidate papers and scores each on how close its total marks are to the target (highest weight), how well its difficulty mix matches, and how widely it covers topics. The best-scoring paper is returned. If the exact total isn't achievable, the closest match is shown.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Re-run the `pip install ...` command in step 3 (with your virtual environment active). |
| Port 5000 already in use | Change `app.run(debug=True)` in `server.py` to `app.run(debug=True, port=5001)`. On macOS, disable AirPlay Receiver or use another port. |
| `questions.csv` not found | Run `python server.py` from inside the `ILLM` folder. |
| Groq errors | Check the API key and your internet connection. |
| "Total marks not exact" warning | The bank can't reach that exact total with the chosen number of questions. Add more questions with varied marks or adjust the target. |

## Notes

- This is a **single-user demo**: state is held in server memory and resets on restart.
- Edits to the **sample bank** are saved back to `questions.csv`; uploaded or topic-generated banks live only in the session (download them as CSV to keep them).
- `debug=True` is enabled for development – turn it off before any real deployment.

## Tech Stack

Python · Flask · pandas · scikit-learn · ReportLab · python-docx · HTML/CSS/JavaScript · Groq API (optional)