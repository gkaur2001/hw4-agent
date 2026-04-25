"""
Reason node — produces a structured reasoning artifact (plan, assumptions, decision).

The output is an EXPLICIT artifact stored in reasoning_trace, not raw chain-of-thought.
The plan produced here is executed step-by-step by execute_plan_node before answer
generation, implementing the plan-then-execute reasoning strategy (Part B).
"""
import json
import re
from typing import Optional
from app.config import MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE
from app.state import GraphState

# Lazy import so missing optional deps don't break the import chain
def _get_llm():
    from langchain_ollama import ChatOllama
    # format="json" instructs Ollama to constrain output to valid JSON
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        format="json",
    )


_REASON_PROMPT = """\
You are a policy assistant. Respond ONLY with a JSON object using this exact schema:
{{
  "plan": ["step 1", "step 2", "step 3"],
  "assumptions": ["assumption"],
  "needs_clarification": false,
  "clarifying_question": "",
  "decision": "one sentence"
}}

Question: {question}

Context:
{context}

Rules:
- plan = 3-5 short steps describing how to answer from the context above.
- needs_clarification = false whenever context is available. Only set true if the question is completely unanswerable (e.g. refers to a specific person or course number not mentioned anywhere).
- clarifying_question = "" unless needs_clarification is true.
- decision = one sentence on how you will answer using the context.
"""


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} substring using brace counting, or None."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def _parse_json_from_llm(text: str) -> dict:
    """Extract and parse the first JSON object from an LLM response.

    Robust against markdown fences and extra surrounding text that small
    models sometimes emit alongside the JSON.
    """
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    candidate = _extract_first_json_object(text)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No JSON object found in LLM response: {text[:200]}")


_FALLBACK_TRACE = {
    "plan": [
        "Review the retrieved KB documents",
        "Identify the most relevant policy or procedure",
        "Formulate a concise, cited answer",
    ],
    "assumptions": ["The retrieved context contains the necessary information"],
    "tool_calls": [],
    "decision": "Answer directly from retrieved context",
}


def reason_node(state: GraphState) -> dict:
    """
    Produce a reasoning artifact and decide whether clarification is needed.

    Reads:  state["question"], state["context"], state.get("pending_tool_call")
    Writes: state["reasoning_trace"], state["needs_clarification"],
            state["draft_answer"] (set if needs_clarification)
    """
    question = state["question"]
    context = state.get("context", "")
    errors = list(state.get("errors", []))
    pending_tool_call = state.get("pending_tool_call")

    needs_clarification = False
    draft_answer = state.get("draft_answer", "")

    try:
        llm = _get_llm()
        prompt = _REASON_PROMPT.format(
            question=question,
            context=context[:3000] if context else "(none)",
        )
        response = llm.invoke(prompt)
        data = _parse_json_from_llm(response.content)

        plan = data.get("plan") or _FALLBACK_TRACE["plan"]
        assumptions = data.get("assumptions") or _FALLBACK_TRACE["assumptions"]
        decision = data.get("decision") or _FALLBACK_TRACE["decision"]
        needs_clarification = bool(data.get("needs_clarification", False))
        # If we have retrieved context, we can attempt an answer — don't stall on clarification
        if needs_clarification and context:
            needs_clarification = False
        if needs_clarification:
            cq = data.get("clarifying_question", "").strip()
            draft_answer = cq if cq else "Could you please provide more details about what you're looking for?"

    except Exception as exc:
        errors.append(f"reason_node error: {exc}")
        plan = _FALLBACK_TRACE["plan"]
        assumptions = _FALLBACK_TRACE["assumptions"]
        decision = _FALLBACK_TRACE["decision"]

    tool_calls = []
    if pending_tool_call:
        tool_calls.append(pending_tool_call)

    reasoning_trace = {
        "plan": plan,
        "assumptions": assumptions,
        "tool_calls": tool_calls,
        "decision": decision,
    }

    updated = {**state}
    updated.pop("pending_tool_call", None)
    updated.update({
        "reasoning_trace": reasoning_trace,
        "needs_clarification": needs_clarification,
        "draft_answer": draft_answer,
        "errors": errors,
    })
    return updated
