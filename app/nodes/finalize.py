"""
Finalize node — validates the output schema (Pydantic) and writes JSON to outputs/.
"""
import json
import time
from datetime import datetime
from pathlib import Path

from app.config import MODEL_NAME, OUTPUTS_DIR
from app.eval.metrics import compute_eval_report
from app.models import (
    AgentOutput,
    EvalReport,
    ReasoningTrace,
    RunMeta,
    ToolCallRecord,
)
from app.state import GraphState


def finalize_node(state: GraphState) -> dict:
    """
    Validate graph output against AgentOutput schema and persist to outputs/.

    Reads:  all of state
    Writes: state["end_time"]; also writes a JSON file to OUTPUTS_DIR
    """
    end_time = time.time()
    start_time = state.get("start_time", end_time)
    latency_ms = int((end_time - start_time) * 1000)

    errors = list(state.get("errors", []))

    # Build Pydantic model (gracefully handle missing fields)
    raw_trace = state.get("reasoning_trace") or {}

    def _to_str_list(val) -> list[str]:
        """Coerce a list to list[str], filtering None/non-string entries."""
        if not val:
            return []
        return [str(item) for item in val if item is not None and item is not False]

    tool_calls = [
        ToolCallRecord(**tc) if isinstance(tc, dict) else tc
        for tc in (raw_trace.get("tool_calls") or [])
    ]
    reasoning_trace = ReasoningTrace(
        plan=_to_str_list(raw_trace.get("plan")),
        assumptions=_to_str_list(raw_trace.get("assumptions")),
        tool_calls=tool_calls,
        decision=str(raw_trace.get("decision") or ""),
    )

    raw_eval = state.get("eval_report")
    if not raw_eval:
        # Evaluate node was skipped (e.g. clarification path) — compute inline
        answer_text = state.get("final_answer") or state.get("draft_answer", "")
        raw_eval = compute_eval_report(
            answer=answer_text,
            citations=state.get("citations") or [],
            context=state.get("context", ""),
            retrieved_docs=state.get("retrieved_docs"),
        )
    eval_report = EvalReport(
        groundedness_score=raw_eval.get("groundedness_score", 0.0),
        citation_coverage=raw_eval.get("citation_coverage", 0.0),
        tool_use_score=raw_eval.get("tool_use_score", 0.0),
        notes=raw_eval.get("notes", ""),
        llm_judge_factuality=raw_eval.get("llm_judge_factuality", 0.0),
        llm_judge_relevance=raw_eval.get("llm_judge_relevance", 0.0),
        llm_judge_citation=raw_eval.get("llm_judge_citation", 0.0),
        llm_judge_overall=raw_eval.get("llm_judge_overall", 0.0),
        llm_judge_reasoning=raw_eval.get("llm_judge_reasoning", ""),
    )

    output = AgentOutput(
        question=state.get("question", ""),
        final_answer=state.get("final_answer") or state.get("draft_answer", ""),
        citations=state.get("citations") or [],
        reasoning_trace=reasoning_trace,
        eval_report=eval_report,
        run_meta=RunMeta(
            model=MODEL_NAME,
            latency_ms=latency_ms,
            tokens_estimate=None,
        ),
    )

    # Write to outputs/<timestamp>.json
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = OUTPUTS_DIR / f"run_{ts}.json"
    out_path.write_text(
        json.dumps(output.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        **state,
        "end_time": end_time,
        "output_path": str(out_path),
    }
