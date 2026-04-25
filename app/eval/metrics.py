"""
Offline + online evaluation metrics.

Baseline metrics:
  - citation_coverage:   does the answer contain at least one valid citation?
  - citation_validity:   do all cited filenames exist in the KB?
  - groundedness:        penalize numbers/dates in the answer not found in retrieved context.
  - tool_use_score:      did retrieval run and return non-empty context?

Part C extension — LLM-as-judge:
  llm_judge() prompts the local Ollama model with a rubric and scores
  (question, answer, context) on three dimensions (1–5 each):
    - factuality:       are the claims accurate and supported by context?
    - relevance:        does the answer address the question?
    - citation_quality: are citations present and appropriate?
  Scores are normalised to [0, 1] and an overall average is computed.
  Budget guard: skipped if answer or context is empty.
"""
import json
import re
from pathlib import Path

from app.config import KB_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_kb_ids(kb_dir: Path = None) -> set[str]:
    kb_dir = kb_dir or KB_DIR
    if not kb_dir.exists():
        return set()
    return {p.name for p in kb_dir.glob("*.md")}


def _extract_numbers_and_dates(text: str) -> set[str]:
    """Return all number/date tokens in text (very lightweight)."""
    # Match things like 10%, 25%, 48, 7, 2024, Jan, etc.
    tokens = re.findall(r"\b\d[\d,.%/]*\b", text)
    return set(tokens)


# ---------------------------------------------------------------------------
# Individual metrics (each returns a float in [0, 1])
# ---------------------------------------------------------------------------

def citation_coverage(answer: str, citations: list[str]) -> float:
    """
    1.0 if the answer has at least one citation that appears in the answer text
    or if the citations list is non-empty; 0.0 otherwise.

    Simpler than per-sentence coverage — good enough for a baseline.
    """
    if not citations:
        return 0.0
    # Check that at least one citation filename appears in the answer
    for cit in citations:
        if cit.replace(".md", "") in answer or cit in answer:
            return 1.0
    # Citations were produced even if not inline — partial credit
    return 0.5


def citation_validity(citations: list[str], kb_dir: Path = None) -> float:
    """
    Fraction of citations that correspond to real KB files.
    Returns 1.0 if citations list is empty (no false citations).
    """
    valid = _valid_kb_ids(kb_dir)
    if not citations:
        return 1.0  # No citations made — not penalized here (coverage handles it)
    valid_count = sum(1 for c in citations if c in valid)
    return valid_count / len(citations)


def groundedness_score(answer: str, context: str) -> float:
    """
    Heuristic groundedness check.
    Penalizes specific numbers/dates in the answer that are absent from the context.
    Returns a score in [0, 1]; 1.0 = fully grounded, lower = suspect hallucination.
    """
    if not answer:
        return 0.0
    answer_nums = _extract_numbers_and_dates(answer)
    if not answer_nums:
        return 1.0  # No numbers to check; assume grounded
    context_nums = _extract_numbers_and_dates(context)
    grounded = answer_nums & context_nums
    score = len(grounded) / len(answer_nums)
    return round(score, 4)


def tool_use_score(retrieved_docs: list[dict], context: str) -> float:
    """
    Simple check: did retrieval run and return useful content?
      1.0 — retrieval ran and context is non-empty
      0.5 — retrieval ran but returned no docs
      0.0 — retrieval did not run (retrieved_docs is None)
    """
    if retrieved_docs is None:
        return 0.0
    if not retrieved_docs or not context:
        return 0.5
    return 1.0


# ---------------------------------------------------------------------------
# LLM-as-judge (Part C)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an impartial evaluator. Score the assistant's answer using the rubric below.
Respond ONLY with a JSON object — no extra text.

Question: {question}

Retrieved context (truncated):
{context}

Assistant answer:
{answer}

Rubric (score each 1–5):
  factuality:       1=fabricated claims, 5=every claim fully supported by context
  relevance:        1=off-topic, 5=directly and completely answers the question
  citation_quality: 1=no citations or all wrong, 5=correct citations for every claim

JSON schema:
{{
  "factuality": <int 1-5>,
  "relevance": <int 1-5>,
  "citation_quality": <int 1-5>,
  "reasoning": "<one sentence explaining scores>"
}}
"""


def llm_judge(
    question: str,
    answer: str,
    context: str,
    citations: list[str],
) -> dict:
    """
    Call the local Ollama model as a judge and return normalised scores.

    Returns a dict with keys:
        llm_judge_factuality, llm_judge_relevance, llm_judge_citation,
        llm_judge_overall  (all floats in [0, 1])
        llm_judge_reasoning (str)

    Returns all zeros with an error note if the call fails or is skipped.
    """
    _SKIP = {
        "llm_judge_factuality": 0.0,
        "llm_judge_relevance": 0.0,
        "llm_judge_citation": 0.0,
        "llm_judge_overall": 0.0,
        "llm_judge_reasoning": "skipped",
    }

    if not answer or not context:
        return {**_SKIP, "llm_judge_reasoning": "skipped (empty answer or context)"}

    try:
        from langchain_ollama import ChatOllama
        from app.config import MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE

        llm = ChatOllama(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
            format="json",
        )

        # Truncate context to keep the judge prompt cheap
        ctx_snippet = context[:2000]
        cit_str = ", ".join(citations) if citations else "(none)"
        answer_with_cits = f"{answer}\n\nCitations provided: {cit_str}"

        prompt = _JUDGE_PROMPT.format(
            question=question,
            context=ctx_snippet,
            answer=answer_with_cits,
        )
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Parse JSON robustly using balanced-brace extraction
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        depth, start, candidate = 0, None, None
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = raw[start : i + 1]
                    break
        data = json.loads(candidate) if candidate else json.loads(raw)

        def _clamp(val, lo=1, hi=5):
            try:
                return max(lo, min(hi, int(val)))
            except (TypeError, ValueError):
                return lo

        f = _clamp(data.get("factuality", 1))
        r = _clamp(data.get("relevance", 1))
        c = _clamp(data.get("citation_quality", 1))
        f_norm = round(f / 5, 4)
        r_norm = round(r / 5, 4)
        c_norm = round(c / 5, 4)
        overall = round((f_norm + r_norm + c_norm) / 3, 4)

        return {
            "llm_judge_factuality": f_norm,
            "llm_judge_relevance": r_norm,
            "llm_judge_citation": c_norm,
            "llm_judge_overall": overall,
            "llm_judge_reasoning": str(data.get("reasoning", "")).strip(),
        }

    except Exception as exc:
        return {**_SKIP, "llm_judge_reasoning": f"judge error: {exc}"}


# ---------------------------------------------------------------------------
# Composite eval_report builder
# ---------------------------------------------------------------------------

def compute_eval_report(
    answer: str,
    citations: list[str],
    context: str,
    retrieved_docs: list[dict],
    kb_dir: Path = None,
) -> dict:
    """
    Compute all baseline metrics and return an eval_report dict.
    """
    cov = citation_coverage(answer, citations)
    val = citation_validity(citations, kb_dir)
    gnd = groundedness_score(answer, context)
    tus = tool_use_score(retrieved_docs, context)

    notes_parts = []
    if cov < 0.5:
        notes_parts.append("answer missing inline citations")
    if val < 1.0:
        notes_parts.append("some citations do not match KB files")
    if gnd < 0.7:
        notes_parts.append("answer contains numbers/dates not found in context")
    if tus < 1.0:
        notes_parts.append("retrieval returned no results")

    return {
        "groundedness_score": gnd,
        "citation_coverage": cov,
        "tool_use_score": tus,
        "notes": "; ".join(notes_parts) if notes_parts else "ok",
    }
