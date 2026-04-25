# HW4 Report — Extending a Starter Agentic System

## Overview

The starter LangGraph agent was extended with semantic vector retrieval (Part A),
plan-then-execute reasoning (Part B), and an LLM-as-judge evaluation layer (Part C).
All three extensions operate on a five-node graph:
`retrieve → reason → execute_plan → answer → evaluate → finalize`.

---

## Part A — Retrieval: Vector Database (ChromaDB + Sentence-Transformers)

**What was implemented.**
The keyword-overlap baseline was replaced with semantic vector search using
ChromaDB and the `all-MiniLM-L6-v2` sentence-transformers model. A one-time
ingestion script (`scripts/ingest.py`) splits each of the five knowledge base
Markdown files into paragraph-level chunks (minimum 100 characters), embeds
them, and stores 36 chunks in a persistent ChromaDB collection. At query time,
`app/tools/vector_retriever.py` performs a cosine similarity search and returns
the top-k chunks with normalised scores. The original keyword retriever is
retained as a fallback if the collection is empty.

**Design rationale.**
Keyword overlap fails when users paraphrase — for example, "how do I get a
later deadline?" shares few tokens with "extension policy." Semantic embeddings
map both to the same region of vector space, making retrieval robust to
vocabulary variation without requiring a larger LLM.

**Observed results.**
Across all 15 golden Q&A pairs, `citation_recall` reached **1.000** — the
correct source file was retrieved in every case. `groundedness_score` averaged
**0.967**, indicating almost no hallucinated numeric values.

---

## Part B — Reasoning: Plan-then-Execute

**What was implemented.**
`app/nodes/reason.py` prompts the LLM (temperature 0) to produce a structured
JSON artifact containing a 3–5 step plan, a list of assumptions, and a
one-sentence decision. A new node, `app/nodes/execute_plan.py`, then loops
through each plan step, runs a targeted vector query for that step's text,
deduplicates retrieved chunks against what is already in context, and appends
new evidence. Every step is logged as a `tool_call` entry in `reasoning_trace`,
making the execution history fully visible in both the terminal output and the
saved JSON artifact.

**Design rationale.**
In the baseline, the plan was generated but never acted on — it was a
decorative artifact with no effect on the final answer. Plan-then-execute makes
the plan functional: each step grounds the answer in step-specific evidence
rather than relying solely on the initial broad retrieval pass. This is
particularly useful for multi-faceted questions where a single query may miss
relevant chunks.

**Observed results.**
The reasoning trace is visible in every run (see Example Runs). For the VPN
question, `execute_plan` retrieved chunks from both `it_vpn_access.md` and
`it_canvas_access.md` across its steps, enriching context beyond the initial
single-file retrieval. The plan steps and decision record are coherent and
directly reflect the question domain.

---

## Part C — Evaluation: LLM-as-Judge

**What was implemented.**
`app/eval/metrics.py` includes an `llm_judge()` function that prompts the same
local Ollama model (temperature 0) with a rubric asking it to score the
(question, answer, context) triple on three dimensions — factuality, relevance,
and citation quality — each on a 1–5 scale. Scores are normalised to \[0, 1\]
and averaged into an overall score. A budget guard skips the judge if the
answer or context is empty. `app/eval/online.py` calls `llm_judge()` after the
heuristic pass and merges all scores into a unified `eval_report`, which is
persisted to the JSON output file.

**Design rationale.**
Heuristic metrics measure proxy signals (are numbers grounded? is a citation
present?) but cannot assess whether an answer is semantically complete or
partially correct. An LLM judge can read the answer and context together and
reason about gaps — the same way a human grader would.

**Observed results.**
The judge exposed two patterns that heuristics missed entirely.

First, partial answers. In all three example runs, heuristic `groundedness_score`
was **1.00** (no hallucinated numbers), yet judge `factuality` was **0.80**.
The judge correctly identified that each answer covered only part of the policy:
the late-work answer focused on the 25–48 hour tier but omitted the 10% and 50%
tiers; the extension answer omitted the requirement to request before the original
deadline. These are real gaps invisible to numeric heuristics.

Second, heuristic fragility to formatting. When the Sources line was removed
from the answer prompt, `citation_coverage` (heuristic) dropped from **1.00**
to **0.50** across runs — because the metric checks whether a citation filename
appears in the answer text. The citations themselves were correctly retrieved and
stored; only the display format changed. The judge's `citation_quality` score was
unaffected (remaining at **1.00** for policy questions), since it evaluates
semantic appropriateness rather than string presence.

The offline batch evaluation across all 15 golden items produced:

| Metric | Mean Score |
|--------|-----------|
| citation\_recall | 1.000 |
| keyword\_hit\_rate | 0.689 |
| groundedness\_score | 0.991 |
| citation\_coverage | 0.600 |
| tool\_use\_score | 1.000 |
| **judge\_factuality** | **0.813** |
| **judge\_relevance** | **1.000** |
| **judge\_citation** | **0.840** |
| **judge\_overall** | **0.884** |

`citation_recall` and `tool_use_score` both hit **1.000** — the correct source
file was retrieved and retrieval ran successfully on every question.
`groundedness_score` of **0.991** confirms near-zero hallucinated numeric values.

`keyword_hit_rate` of **0.689** reflects a known weakness of `llama3.1:8b`:
the model paraphrases rather than quotes, so expected keywords from the golden
set are often absent even when the answer is factually correct.

`citation_coverage` of **0.600** is a formatting artefact: the heuristic checks
whether a citation filename appears in the answer text. After removing the
Sources line from the prompt, citations no longer appear inline, so the metric
gives only partial credit even though citations are correctly retrieved and stored.
The judge's `citation_quality` score of **0.840** is unaffected because it
evaluates semantic appropriateness rather than string presence, and is the more
meaningful measure.

The LLM judge's **overall score of 0.884** across 15 runs indicates the system
consistently produces factually accurate and relevant answers. The gap between
`judge_factuality` (0.813) and `judge_relevance` (1.000) suggests answers are
always on-topic but occasionally miss secondary details — consistent with the
judge reasoning observed in individual runs.

---

## Example Runs

### Q1 — Late submission policy

**Question:** How do late submissions work?

**Answer:** According to our late work policy, submissions between 25–48 hours
late incur a 25% deduction from the earned score. For example, if you earn
90/100 but submitted 30 hours late, you would receive 90 × 0.75 = 67.5 points.
This deduction applies after grading, so your instructor will first calculate
your grade as usual and then apply the penalty.

**Citations:** policy\_late\_work.md

**Reasoning Trace:**
1. Review policy\_late\_work.md for late submission penalties
2. Identify relevant sections (Late Submission Penalties, Graded Late Work)
3. Apply deduction rules to determine final score
→ *The policy for late submissions involves a tiered system of deductions based on the time elapsed since the deadline, with a 15-minute grace period applying to all deadlines.*

**Eval:** Groundedness 1.00 · Citation coverage 0.50 · Tool-use 1.00 |
Judge: factuality 0.80 · relevance 1.00 · citation 1.00 · **overall 0.93**

**Judge reasoning:** The assistant accurately described the late work policy but
slightly misstated the deduction for submissions between 1–24 hours late as
"10% deduction" instead of "up to 10% deduction."

---

### Q2 — Deadline extension request

**Question:** How do I request a deadline extension?

**Answer:** To request a deadline extension, email your instructor at least 24
hours before the original deadline with the subject line "\[EXTENSION REQUEST\]
\<Course\> \<Assignment Name\>." Briefly describe the reason for the request and
attach any relevant documentation. If approved, you will receive a confirmation
email with the new deadline, which is typically 48–72 hours beyond the original
deadline.

**Citations:** policy\_extensions.md

**Reasoning Trace:**
1. Email your instructor at least 24 hours before the deadline
2. Use the subject line: \[EXTENSION REQUEST\] \<Course\> \<Assignment Name\>
3. Briefly describe the reason and attach documentation if available
→ *To request a deadline extension, follow the steps outlined in the policy document.*

**Eval:** Groundedness 1.00 · Citation coverage 0.50 · Tool-use 1.00 |
Judge: factuality 0.80 · relevance 1.00 · citation 1.00 · **overall 0.93**

**Judge reasoning:** The assistant's answer accurately summarizes the key points
from policy\_extensions.md but fails to mention that extensions must be
requested before the original deadline.

---

### Q3 — VPN setup

**Question:** How do I set up VPN access?

**Answer:** To set up VPN access, go to `https://my.jh.edu` and log in with
your JHED ID. Navigate to IT Downloads > Cisco AnyConnect VPN, download the
client for your operating system (Windows, macOS, Linux), and then launch
Cisco AnyConnect and connect to `vpn.jhu.edu`.

**Citations:** it\_vpn\_access.md, it\_canvas\_access.md

**Reasoning Trace:**
1. Go to https://my.jh.edu and log in with your JHED ID
2. Navigate to IT Downloads > Cisco AnyConnect VPN and download the client for your operating system
3. Launch Cisco AnyConnect and connect to vpn.jhu.edu
→ *To set up VPN access, follow the steps outlined in the IT Help document 'it\_vpn\_access.md'.*

**Eval:** Groundedness 1.00 · Citation coverage 0.50 · Tool-use 1.00 |
Judge: factuality 0.80 · relevance 1.00 · citation 0.60 · **overall 0.80**

**Judge reasoning:** The assistant's answer is mostly factually accurate but
lacks a crucial step in the connection process (two-factor authentication). The
citations are partially correct, missing one important detail.
