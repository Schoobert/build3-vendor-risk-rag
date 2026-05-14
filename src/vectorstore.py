import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")

_PLACEHOLDER_MARKERS = ("placeholder", "your-", "xxxxx", "<your", "<>")


def _is_real_credential(value: str | None) -> bool:
    if not value:
        return False
    v = value.lower()
    return not any(m in v for m in _PLACEHOLDER_MARKERS)


def get_backend() -> str:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if _is_real_credential(url) and _is_real_credential(key):
        return "supabase"
    return "chromadb"


# ── Shared helpers ────────────────────────────────────────────────────────────

def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)


def _embed_query(text: str) -> list[float]:
    response = _openai_client().embeddings.create(
        model="text-embedding-ada-002",
        input=[text],
    )
    return response.data[0].embedding


def _supabase_client():
    from supabase import create_client
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


# ── Public API ────────────────────────────────────────────────────────────────

def init_collection(collection_name: str = "vendor_risk") -> chromadb.Collection | None:
    """Returns a ChromaDB collection, or None when using the Supabase backend."""
    if get_backend() == "supabase":
        return None
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(
    embedded_chunks: list[dict],
    collection_name: str = "vendor_risk",
    source_filename: str = "",
) -> None:
    if get_backend() == "supabase":
        _store_supabase(embedded_chunks, source_filename)
    else:
        _store_chroma(embedded_chunks, collection_name)


def query_collection(
    query_text: str,
    n_results: int = 5,
    collection_name: str = "vendor_risk",
) -> list[dict]:
    if get_backend() == "supabase":
        return _query_supabase(query_text, n_results)
    return _query_chroma(query_text, n_results, collection_name)


# ── ChromaDB backend ──────────────────────────────────────────────────────────

def _store_chroma(embedded_chunks: list[dict], collection_name: str) -> None:
    collection = init_collection(collection_name)
    collection.add(
        ids=[str(c["chunk_index"]) for c in embedded_chunks],
        embeddings=[c["embedding"] for c in embedded_chunks],
        documents=[c["text"] for c in embedded_chunks],
        metadatas=[{"chunk_index": c["chunk_index"]} for c in embedded_chunks],
    )


def _query_chroma(query_text: str, n_results: int, collection_name: str) -> list[dict]:
    collection = init_collection(collection_name)
    query_embedding = _embed_query(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )
    return [
        {"text": doc, "distance": dist}
        for doc, dist in zip(results["documents"][0], results["distances"][0])
    ]


# ── Supabase / pgvector backend ───────────────────────────────────────────────

def _store_supabase(embedded_chunks: list[dict], source_filename: str) -> None:
    client = _supabase_client()
    # Clear existing chunks for this file to avoid duplicates on re-upload
    if source_filename:
        client.table("document_chunks").delete().eq(
            "source_filename", source_filename
        ).execute()
    rows = [
        {
            "text": c["text"],
            "embedding": c["embedding"],
            "chunk_index": c["chunk_index"],
            "source_filename": source_filename,
        }
        for c in embedded_chunks
    ]
    client.table("document_chunks").insert(rows).execute()


def _query_supabase(query_text: str, n_results: int) -> list[dict]:
    client = _supabase_client()
    query_embedding = _embed_query(query_text)
    result = client.rpc(
        "match_chunks",
        {"query_embedding": query_embedding, "match_count": n_results},
    ).execute()
    return [
        {"text": row["text"], "distance": 1 - row["similarity"]}
        for row in result.data
    ]


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest import ingest_pdf
    from embed import embed_chunks

    backend = get_backend()
    print(f"Backend detected: {backend}\n")

    pdf_path = "sample_data/NovaSoft_SOC2_TypeII_2024.pdf"
    filename = Path(pdf_path).name

    print("Step 1: Ingesting PDF...")
    chunks = ingest_pdf(pdf_path)
    print(f"  {len(chunks)} chunks extracted.\n")

    print("Step 2: Embedding all chunks...")
    embedded = embed_chunks(chunks)
    print(f"  {len(embedded)} chunks embedded.\n")

    print("Step 3: Storing chunks...")
    store_chunks(embedded, source_filename=filename)
    print("  Done.\n")

    query = "What exceptions were identified?"
    print(f'Step 4: Querying — "{query}"')
    print("-" * 60)
    results = query_collection(query, n_results=3)
    for i, r in enumerate(results, 1):
        preview = r["text"][:300].replace("\n", " ")
        print(f"  [{i}] distance={r['distance']:.4f}")
        print(f"       {preview}")
        print()
