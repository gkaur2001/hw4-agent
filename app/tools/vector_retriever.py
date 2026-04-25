"""
Vector DB retriever — semantic search over the KB using ChromaDB + sentence-transformers.

The collection is populated by running `python scripts/ingest.py` before first use.
After ingestion the collection is persistent on disk (data/chroma_db/) and this
module loads it lazily on the first call to retrieve().

Return type is identical to local_retriever.retrieve() so retrieve_node needs no changes:
    list[{"id": str, "text": str, "score": float}]

Scores are cosine similarities in [0, 1] (higher = more relevant).
"""
from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import CHROMA_DB_DIR, EMBEDDING_MODEL, MAX_RETRIEVED_DOCS

_collection = None


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    _collection = client.get_or_create_collection(
        name="kb_docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """
    Return the top-k KB chunks most semantically similar to `query`.

    Raises RuntimeError if the collection is empty (ingest.py hasn't been run).
    """
    top_k = top_k if top_k is not None else MAX_RETRIEVED_DOCS
    collection = _get_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "ChromaDB collection is empty. Run `python scripts/ingest.py` first."
        )

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance is in [0, 2]; with unit vectors it's in [0, 1]
        # where 0 = identical. Convert to similarity: score = 1 - distance.
        score = round(max(0.0, 1.0 - distance), 4)
        docs.append({"id": metadata["source"], "text": text, "score": score})

    return docs
