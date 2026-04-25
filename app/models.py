"""
Output contract — Pydantic model for the final JSON artifact written to outputs/.
"""
from typing import Optional, Any
from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    result_summary: str


class ReasoningTrace(BaseModel):
    plan: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    decision: str = ""


class EvalReport(BaseModel):
    groundedness_score: float = 0.0
    citation_coverage: float = 0.0
    tool_use_score: float = 0.0
    notes: str = ""
    # LLM-as-judge scores (Part C) — normalised to [0, 1]
    llm_judge_factuality: float = 0.0
    llm_judge_relevance: float = 0.0
    llm_judge_citation: float = 0.0
    llm_judge_overall: float = 0.0
    llm_judge_reasoning: str = ""


class RunMeta(BaseModel):
    model: str
    latency_ms: int
    tokens_estimate: Optional[int] = None


class AgentOutput(BaseModel):
    """Final validated output written to outputs/<timestamp>.json."""
    question: str
    final_answer: str
    citations: list[str] = Field(default_factory=list)
    reasoning_trace: ReasoningTrace = Field(default_factory=ReasoningTrace)
    eval_report: EvalReport = Field(default_factory=EvalReport)
    run_meta: RunMeta
