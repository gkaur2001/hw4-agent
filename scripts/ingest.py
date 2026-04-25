"""
Ingest script — embeds KB markdown files and stores them in ChromaDB.

Each document is split into paragraphs (double-newline boundaries) so that
retrieval returns focused chunks rather than entire files. Chunks smaller
than MIN_CHUNK_CHARS are merged with the preceding chunk to avoid noise.

Usage:
    python scripts/ingest.py

Re-running is idempotent: the collection is cleared and rebuilt each time,
so you can safely re-run after editing KB files.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console
from rich.table import Table

from app.config import CHROMA_DB_DIR, EMBEDDING_MODEL, KB_DIR

console = Console()

MIN_CHUNK_CHARS = 100  # merge chunks shorter than this with the previous one


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; merge tiny fragments into the preceding chunk."""
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for para in raw:
        if chunks and len(para) < MIN_CHUNK_CHARS:
            chunks[-1] = chunks[-1] + " " + para
        else:
            chunks.append(para)
    return chunks


def ingest(kb_dir: Path = None, chroma_dir: Path = None):
    kb_dir = kb_dir or KB_DIR
    chroma_dir = chroma_dir or CHROMA_DB_DIR

    if not kb_dir.exists():
        console.print(f"[red]KB directory not found: {kb_dir}[/red]")
        return

    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        console.print("[yellow]No .md files found in KB directory.[/yellow]")
        return

    # Set up ChromaDB
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Wipe and recreate for idempotency
    try:
        client.delete_collection("kb_docs")
    except Exception:
        pass
    collection = client.create_collection(
        name="kb_docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Chunk, embed, and add documents
    table = Table(title="KB Ingestion", show_header=True, header_style="bold blue")
    table.add_column("File", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Chars", justify="right")

    all_ids, all_docs, all_meta = [], [], []

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = _split_paragraphs(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{md_file.name}::{i}"
            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_meta.append({"source": md_file.name, "chunk_index": i})

        table.add_row(md_file.name, str(len(chunks)), str(len(text)))

    collection.add(ids=all_ids, documents=all_docs, metadatas=all_meta)

    console.print(table)
    console.print(
        f"\n[green]✓ Indexed {len(all_ids)} chunks from {len(md_files)} files "
        f"into ChromaDB at {chroma_dir}[/green]"
    )
    console.print(f"[dim]Embedding model: {EMBEDDING_MODEL}[/dim]")


if __name__ == "__main__":
    ingest()
