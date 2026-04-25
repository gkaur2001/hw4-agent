"""
Online (post-hoc, per-run) evaluation helpers.

Runs heuristic metrics from app/eval/metrics.py followed by the LLM-as-judge
rubric (Part C). All scores are merged into a single eval_report dict returned
to the evaluate node.
"""
from app.eval.metrics import compute_eval_report, llm_judge


def run_online_eval(state: dict) -> dict:
    """
    Run all online metrics for a completed graph state.

    Baseline heuristics (groundedness, citation coverage, tool-use) are always
    computed.  LLM-as-judge scores are appended on top.

    Args:
        state: the current GraphState dict (after answer node).

    Returns:
        An eval_report dict matching the EvalReport schema.
    """
    answer = state.get("final_answer") or state.get("draft_answer", "")
    citations = state.get("citations", [])
    context = state.get("context", "")
    retrieved_docs = state.get("retrieved_docs")
    question = state.get("question", "")

    report = compute_eval_report(
        answer=answer,
        citations=citations,
        context=context,
        retrieved_docs=retrieved_docs,
    )

    # Part C — LLM-as-judge
    judge_scores = llm_judge(
        question=question,
        answer=answer,
        context=context,
        citations=citations,
    )
    report.update(judge_scores)

    return report
