import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")


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


def init_collection(collection_name: str = "vendor_risk") -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(
    embedded_chunks: list[dict],
    collection_name: str = "vendor_risk",
) -> None:
    collection = init_collection(collection_name)
    collection.add(
        ids=[str(c["chunk_index"]) for c in embedded_chunks],
        embeddings=[c["embedding"] for c in embedded_chunks],
        documents=[c["text"] for c in embedded_chunks],
        metadatas=[{"chunk_index": c["chunk_index"]} for c in embedded_chunks],
    )


def query_collection(
    query_text: str,
    n_results: int = 5,
    collection_name: str = "vendor_risk",
) -> list[dict]:
    collection = init_collection(collection_name)
    query_embedding = _embed_query(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )
    return [
        {
            "text": doc,
            "distance": dist,
        }
        for doc, dist in zip(results["documents"][0], results["distances"][0])
    ]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest import ingest_pdf
    from embed import embed_chunks

    pdf_path = "sample_data/NovaSoft_SOC2_TypeII_2024.pdf"

    print("Step 1: Ingesting PDF...")
    chunks = ingest_pdf(pdf_path)
    print(f"  {len(chunks)} chunks extracted.\n")

    print("Step 2: Embedding all chunks...")
    embedded = embed_chunks(chunks)
    print(f"  {len(embedded)} chunks embedded.\n")

    print("Step 3: Storing in ChromaDB...")
    store_chunks(embedded)
    print("  Done.\n")

    queries = [
        "What exceptions or findings were identified?",
        "What are the encryption and data security controls?",
    ]

    for query in queries:
        print(f"Query: \"{query}\"")
        print("-" * 60)
        results = query_collection(query, n_results=3)
        for i, r in enumerate(results, 1):
            preview = r["text"][:200].replace("\n", " ")
            print(f"  [{i}] distance={r['distance']:.4f}")
            print(f"       {preview}")
            print()
