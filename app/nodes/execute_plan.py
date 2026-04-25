"""
Execute-plan node — runs a targeted retrieval query for each step in the
reasoning plan produced by reason_node.

This implements the Part B "plan-then-execute" extension:
  1. reason_node produces a 3-5 step plan
  2. THIS node executes each step by querying the vector KB with that step's text
  3. New (non-duplicate) chunks are merged into the context
  4. Each step's retrieval is logged as a tool_call in reasoning_trace

The result is a richer, step-grounded context that answer_node uses to
produce a more accurate, better-cited final answer.
"""
from app.state import GraphState


def execute_plan_node(state: GraphState) -> dict:
    """
    For each plan step, run a retrieval query and merge new docs into context.

    Reads:  state["reasoning_trace"]["plan"], state["retrieved_docs"],
            state["context"], state["citations"]
    Writes: state["retrieved_docs"], state["context"], state["citations"],
            state["reasoning_trace"]["tool_calls"]
    """
    reasoning_trace = state.get("reasoning_trace", {})
    plan = reasoning_trace.get("plan", [])
    errors = list(state.get("errors", []))

    existing_docs = list(state.get("retrieved_docs", []))
    existing_citations = list(state.get("citations", []))
    tool_calls = list(reasoning_trace.get("tool_calls", []))

    # Dedup key: source file + first 60 chars of text
    seen = {d["id"] + d["text"][:60] for d in existing_docs}

    for step in plan:
        step_query = step.strip()
        if not step_query:
            continue

        # Try vector retrieval first, fall back to keyword
        try:
            from app.tools.vector_retriever import retrieve
            step_docs = retrieve(step_query, top_k=2)
            tool_used = "vector_retriever"
        except Exception as exc:
            errors.append(f"execute_plan vector error on step '{step_query[:40]}': {exc}")
            try:
                from app.tools.local_retriever import retrieve as kw_retrieve
                step_docs = kw_retrieve(step_query, top_k=2)
                tool_used = "local_retriever"
            except Exception as exc2:
                errors.append(f"execute_plan keyword error on step '{step_query[:40]}': {exc2}")
                step_docs = []
                tool_used = "none"

        # Only keep chunks not already in context
        new_docs = []
        for doc in step_docs:
            key = doc["id"] + doc["text"][:60]
            if key not in seen:
                seen.add(key)
                new_docs.append(doc)
                existing_docs.append(doc)
                if doc["id"] not in existing_citations:
                    existing_citations.append(doc["id"])

        tool_calls.append({
            "tool": tool_used,
            "args": {"query": step_query},
            "result_summary": (
                f"step '{step_query[:50]}': "
                f"{len(new_docs)} new chunk(s) from "
                f"{[d['id'] for d in new_docs]}"
            ),
        })

    # Rebuild context string from all docs (original + newly retrieved)
    context_parts = []
    for doc in existing_docs:
        header = f"--- {doc['id']} (score={doc['score']:.4f}) ---"
        context_parts.append(f"{header}\n{doc['text'][:1500]}")
    context = "\n\n".join(context_parts)

    updated_trace = {**reasoning_trace, "tool_calls": tool_calls}

    return {
        **state,
        "retrieved_docs": existing_docs,
        "context": context,
        "citations": existing_citations,
        "reasoning_trace": updated_trace,
        "errors": errors,
    }
