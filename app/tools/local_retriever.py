"""
Baseline KB retriever — simple keyword overlap (TF-style) over data/kb/*.md.

TODO (Part A — student extension point):
  Replace or extend this with one of:
  - Vector DB retrieval: embed docs with sentence-transformers, store in ChromaDB/FAISS,
    query with cosine similarity.
  - Query decomposition: generate 2–5 subqueries, retrieve per subquery, merge + deduplicate.
  - Web search: call Wikipedia API or DuckDuckGo; add returned pages as retrieved docs.

  Keep the return type identical: list[{"id": str, "text": str, "score": float}]
  so that retrieve.py and the rest of the graph don't need to change.
"""
import re
from pathlib import Path

from app.config import KB_DIR, MAX_RETRIEVED_DOCS


def _keyword_score(doc_text: str, query: str) -> float:
    """Fraction of unique query tokens present in doc_text (case-insensitive)."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return 0.0
    doc_tokens = set(re.findall(r"\w+", doc_text.lower()))
    overlap = query_tokens & doc_tokens
    return len(overlap) / len(query_tokens)


def retrieve(query: str, kb_dir: Path = None, top_k: int = None) -> list[dict]:
    """
    Return the top-k KB documents most relevant to `query`.

    Args:
        query:  The user question (or any search string).
        kb_dir: Override the default KB directory (useful for testing).
        top_k:  Override the default number of results to return.

    Returns:
        List of dicts: [{"id": filename, "text": content, "score": float}, ...]
        Sorted descending by score. May return fewer than top_k if the KB is small.
        Returns an empty list if the KB directory is missing or empty.
    """
    kb_dir = kb_dir or KB_DIR
    top_k = top_k if top_k is not None else MAX_RETRIEVED_DOCS

    if not kb_dir.exists():
        return []

    scored_docs = []
    for md_file in sorted(kb_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        score = _keyword_score(text, query)
        scored_docs.append({"id": md_file.name, "text": text, "score": round(score, 4)})

    scored_docs.sort(key=lambda d: d["score"], reverse=True)
    return scored_docs[:top_k]
