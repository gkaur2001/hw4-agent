# hw4-starter-agent — Policy/Procedure Helpdesk (LangGraph)

> **Report:** See [`report.md`](report.md) for the written submission covering design choices, observations, and example runs.

A LangGraph-based agentic system for the JHU HW4 assignment, extended with
vector retrieval, plan-then-execute reasoning, and LLM-as-judge evaluation.

---

## What this does

The agent answers "helpdesk" questions (late work policy, extensions, IT help, etc.)
from a small local knowledge base of Markdown files.

Each run produces:
- A **cited answer** (sources = KB filenames)
- A **reasoning artifact** (plan, assumptions, per-step tool calls, decision)
- An **evaluation report** (heuristic metrics + LLM-as-judge scores)
- A **JSON output file** in `outputs/`

---

## Setup

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running locally

### 2. Pull a model

```bash
ollama pull llama3.1:8b
# llama3.1:8b is recommended — reliable JSON and instruction following
# set MODEL_NAME in .env to match
```

### 3. Install dependencies

```bash
cd hw4-starter-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure

```bash
cp .env.example .env
# Edit MODEL_NAME in .env to match the model you pulled
```

### 5. Ingest the knowledge base

Run once before first use (and re-run after editing any KB files):

```bash
python scripts/ingest.py
```

This chunks the 5 Markdown files in `data/kb/` and stores embeddings in
`data/chroma_db/` using `all-MiniLM-L6-v2`.

---

## Running

### Single question

```bash
python run.py --question "How do late submissions work?"
python run.py --question "How do I request an extension?"
python run.py --question "What should I do if I can't access the VPN?"
```

### Offline evaluation (runs all 15 golden questions)

```bash
python scripts/eval_offline.py
# or equivalently:
python run.py --eval
```

---

## Graph flow

```
retrieve → reason → execute_plan → answer → evaluate → finalize
                 ↘ (needs_clarification) ↗
                         finalize
```

| Node | Responsibility |
|------|---------------|
| `retrieve` | Semantic vector search over ChromaDB; falls back to keyword if DB is empty |
| `reason` | LLM produces a 3–5 step plan, assumptions, and decision as a structured artifact |
| `execute_plan` | Executes each plan step: runs a targeted vector query per step, merges new chunks |
| `answer` | Generates a cited answer from the enriched context + plan |
| `evaluate` | Heuristic metrics + LLM-as-judge rubric |
| `finalize` | Validates output against Pydantic schema, writes JSON to `outputs/` |

---

## Output format

Each run writes a JSON file to `outputs/run_<timestamp>.json`:

```json
{
  "question": "...",
  "final_answer": "...",
  "citations": ["policy_late_work.md"],
  "reasoning_trace": {
    "plan": ["step 1", "step 2", "step 3"],
    "assumptions": ["..."],
    "tool_calls": [
      {"tool": "vector_retriever", "args": {"query": "step 1"}, "result_summary": "..."},
      {"tool": "vector_retriever", "args": {"query": "step 2"}, "result_summary": "..."}
    ],
    "decision": "..."
  },
  "eval_report": {
    "groundedness_score": 0.97,
    "citation_coverage": 1.0,
    "tool_use_score": 1.0,
    "notes": "ok",
    "llm_judge_factuality": 0.8,
    "llm_judge_relevance": 0.8,
    "llm_judge_citation": 0.6,
    "llm_judge_overall": 0.73,
    "llm_judge_reasoning": "..."
  },
  "run_meta": {
    "model": "llama3.1:8b",
    "latency_ms": 8500,
    "tokens_estimate": null
  }
}
```

---

## What was implemented

### Part A — Retrieval: Vector DB (ChromaDB + sentence-transformers)

- `scripts/ingest.py` — chunks KB Markdown files into paragraphs and stores
  embeddings in ChromaDB using `all-MiniLM-L6-v2`
- `app/tools/vector_retriever.py` — semantic similarity search over the
  persisted collection; returns `list[{"id", "text", "score"}]`
- `app/nodes/retrieve.py` — calls vector retriever, falls back to keyword
  retriever if the collection is empty

### Part B — Reasoning: Plan-then-execute

- `app/nodes/reason.py` — LLM produces a structured JSON artifact: 3–5 step
  plan, assumptions, and a one-sentence decision
- `app/nodes/execute_plan.py` — loops through each plan step, runs a targeted
  vector query per step, deduplicates chunks, and merges new evidence into
  context before the answer node runs; each step is logged as a tool_call
  in `reasoning_trace`

### Part C — Evaluation: LLM-as-judge

- `app/eval/metrics.py` — `llm_judge()` prompts the local Ollama model with a
  1–5 rubric scoring factuality, relevance, and citation quality; scores are
  normalised to [0, 1]; budget-guarded (skipped if answer or context is empty)
- `app/eval/online.py` — calls `llm_judge()` and merges scores into the
  heuristic eval report
- `app/models.py` / `app/nodes/finalize.py` — `EvalReport` extended with five
  judge fields persisted to the JSON output

---

## Repo structure

```
hw4-starter-agent/
├── run.py                          # CLI (single question + offline eval)
├── requirements.txt
├── .env.example
├── app/
│   ├── config.py                   # Settings (from .env)
│   ├── state.py                    # GraphState TypedDict
│   ├── models.py                   # AgentOutput Pydantic schema
│   ├── graph.py                    # LangGraph assembly
│   ├── nodes/
│   │   ├── retrieve.py             # Vector retrieval node (Part A)
│   │   ├── reason.py               # Plan generation node (Part B)
│   │   ├── execute_plan.py         # Plan execution node (Part B)
│   │   ├── answer.py               # Answer generation
│   │   ├── evaluate.py             # Evaluation node (Part C)
│   │   └── finalize.py             # Schema validation + file write
│   ├── tools/
│   │   ├── local_retriever.py      # Keyword overlap fallback retriever
│   │   ├── vector_retriever.py     # ChromaDB semantic retriever (Part A)
│   │   └── web_search_stub.py      # Web search stub (not implemented)
│   └── eval/
│       ├── metrics.py              # Heuristic metrics + LLM-as-judge (Part C)
│       └── online.py               # Online eval orchestration
├── scripts/
│   ├── ingest.py                   # KB chunking + ChromaDB indexing
│   └── eval_offline.py             # Batch evaluation on golden set
├── data/
│   ├── kb/                         # Markdown policy/IT docs (5 files)
│   ├── chroma_db/                  # ChromaDB vector store (generated)
│   └── golden/                     # golden_qa.jsonl (15 Q&A pairs)
└── outputs/                        # Run outputs and eval summaries
```

---

## Grading Rubric

| Part | Criterion | Points |
|------|-----------|--------|
| **A — Retrieval** | Implemented one valid retrieval extension beyond keyword overlap | 20 |
| | Citations are valid KB filenames | 5 |
| | Brief explanation of design choice in report | 5 |
| **B — Reasoning** | Implemented one valid reasoning/control extension | 20 |
| | Reasoning artifact (plan/decision) is visible and coherent | 5 |
| | Brief explanation of design choice in report | 5 |
| **C — Evaluation** | Implemented one valid evaluation extension | 20 |
| | Numeric scores reported; interpretation included in report | 5 |
| | Brief explanation of design choice in report | 5 |
| **Code quality** | Modular, readable, runnable; no crashes on edge cases | 5 |
| **Report** | 1–2 pages; covers what/why/what-you-observed; includes example runs | 5 |
| **Total** | | **100** |
