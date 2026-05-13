import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key)


def embed_chunks(chunks: list[str]) -> list[dict]:
    client = get_client()
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=chunks,
    )
    return [
        {
            "chunk_index": i,
            "text": chunks[i],
            "embedding": data.embedding,
        }
        for i, data in enumerate(response.data)
    ]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest import ingest_pdf

    pdf_path = "sample_data/NovaSoft_SOC2_TypeII_2024.pdf"
    all_chunks = ingest_pdf(pdf_path)
    sample = all_chunks[:5]

    print(f"Embedding {len(sample)} chunks...\n")
    results = embed_chunks(sample)

    for r in results:
        preview = r["text"][:100].replace("\n", " ")
        print(f"[{r['chunk_index']}] {preview}")
        print(f"     Embedding length: {len(r['embedding'])}\n")
