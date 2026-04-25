"""
Answer node — generates a concise, cited answer from context + reasoning trace.

The node calls the LLM once with a tightly constrained prompt.
It does NOT expose raw chain-of-thought; the reasoning artifact is already in state.

TODO: This node is usually fine as-is, but you may modify the prompt or
add a re-ranking step (e.g., check if the draft answer is consistent with
the reasoning plan).
"""
import re
from app.config import MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE
from app.state import GraphState


def _get_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )


_ANSWER_PROMPT = """\
You are a helpful policy assistant at Johns Hopkins University.
Answer ONLY from the context below.

Question: {question}

Context:
{context}

Plan: {plan}

Rules:
1. Use ONLY the context. Do not invent facts.
2. Be concise: 2-5 sentences.
3. Do not include a Sources or Citations line — citations are handled separately.
"""


def answer_node(state: GraphState) -> dict:
    """
    Generate the final answer.

    Reads:  state["question"], state["context"], state["reasoning_trace"]
    Writes: state["final_answer"], state["citations"] (refined from inline citations)
    """
    question = state["question"]
    context = state.get("context", "")
    reasoning_trace = state.get("reasoning_trace", {})
    plan_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(reasoning_trace.get("plan", [])))
    errors = list(state.get("errors", []))

    if not context:
        final_answer = (
            "I couldn't find relevant information in the knowledge base to answer your question. "
            "Please check with your instructor or the JHU IT Help Desk."
        )
        return {**state, "final_answer": final_answer, "errors": errors}

    try:
        llm = _get_llm()
        prompt = _ANSWER_PROMPT.format(
            question=question,
            context=context[:4000],
            plan=plan_text or "(no plan)",
        )
        response = llm.invoke(prompt)
        # Strip any Sources/Citations line the model writes despite the instruction
        final_answer = re.sub(r"\n*Sources:.*$", "", response.content.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()

    except Exception as exc:
        errors.append(f"answer_node error: {exc}")
        final_answer = (
            "I encountered an error while generating an answer. "
            "Please try again or contact support."
        )

    return {**state, "final_answer": final_answer, "errors": errors}
