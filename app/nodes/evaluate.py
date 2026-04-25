"""
Evaluate node — computes online (post-hoc) evaluation metrics for the current run.

Runs heuristic metrics (groundedness, citation coverage, tool-use) followed by
the LLM-as-judge rubric (factuality, relevance, citation quality) from
app/eval/online.py. All scores are stored in eval_report and persisted to JSON.
"""
from app.eval.online import run_online_eval
from app.state import GraphState


def evaluate_node(state: GraphState) -> dict:
    """
    Compute eval_report metrics from the current state.

    Reads:  state["final_answer"], state["citations"], state["context"],
            state["retrieved_docs"]
    Writes: state["eval_report"]
    """
    eval_report = run_online_eval(state)
    return {**state, "eval_report": eval_report}
