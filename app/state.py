"""
Internal graph state (TypedDict).
All nodes read from and return updates to GraphState.
"""
from typing import Any, Optional
from typing_extensions import TypedDict


class ToolCallRecord(TypedDict):
    tool: str
    args: dict
    result_summary: str


class ReasoningTrace(TypedDict):
    plan: list[str]
    assumptions: list[str]
    tool_calls: list[ToolCallRecord]
    decision: str


class EvalReport(TypedDict):
    groundedness_score: float
    citation_coverage: float
    tool_use_score: float
    notes: str
    llm_judge_factuality: float
    llm_judge_relevance: float
    llm_judge_citation: float
    llm_judge_overall: float
    llm_judge_reasoning: str


class RetrievedDoc(TypedDict):
    id: str
    text: str
    score: float


class GraphState(TypedDict, total=False):
    # Required input
    question: str

    # Retrieval outputs
    retrieved_docs: list[RetrievedDoc]
    context: str
    citations: list[str]

    # Reasoning outputs
    reasoning_trace: ReasoningTrace
    needs_clarification: bool
    draft_answer: str

    # Answer
    final_answer: str

    # Evaluation
    eval_report: EvalReport

    # Bookkeeping
    errors: list[str]
    start_time: float
    end_time: float
    output_path: str                  # set by finalize_node; used by run.py
    pending_tool_call: dict           # internal handoff from retrieve → reason
