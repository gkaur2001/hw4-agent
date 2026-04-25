"""
Retrieve node — fetches relevant KB documents for the user question.

Uses vector_retriever (ChromaDB + sentence-transformers) for semantic similarity.
Falls back to local_retriever (keyword overlap) if the vector DB hasn't been
ingested yet — run `python scripts/ingest.py` once to enable vector retrieval.
"""
from app.state import GraphState


def retrieve_node(state: GraphState) -> dict:
    """
    Retrieve relevant documents and build context string.

    Reads:  state["question"]
    Writes: state["retrieved_docs"], state["context"], state["citations"]
    """
    question = state["question"]
    errors = list(state.get("errors", []))

    tool_name = "vector_retriever"
    try:
        from app.tools.vector_retriever import retrieve
        docs = retrieve(question)
    except Exception as exc:
        errors.append(f"vector_retriever unavailable, falling back to keyword: {exc}")
        tool_name = "local_retriever"
        try:
            from app.tools.local_retriever import retrieve as keyword_retrieve
            docs = keyword_retrieve(question)
        except Exception as exc2:
            errors.append(f"local_retriever error: {exc2}")
            docs = []

    tool_call = {
        "tool": tool_name,
        "args": {"query": question},
        "result_summary": f"retrieved {len(docs)} doc(s): {[d['id'] for d in docs]}",
    }

    context_parts = []
    for doc in docs:
        header = f"--- {doc['id']} (score={doc['score']}) ---"
        body = doc["text"][:1500]
        context_parts.append(f"{header}\n{body}")
    context = "\n\n".join(context_parts)

    citations = list(dict.fromkeys(doc["id"] for doc in docs))

    return {
        **state,
        "retrieved_docs": docs,
        "context": context,
        "citations": citations,
        "errors": errors,
        "pending_tool_call": tool_call,
    }
