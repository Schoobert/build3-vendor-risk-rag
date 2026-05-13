import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

SYSTEM_PROMPT = """You are a vendor risk analyst reviewing SOC 2 audit documentation.

Rules you must follow without exception:
1. Only use information present in the retrieved document chunks provided to you.
2. Always cite the specific control ID or document section (e.g. CC7.1, A1.3, Section I) when referencing a finding.
3. Never fabricate or infer findings not explicitly present in the context.
4. If the retrieved context does not contain enough information to answer the query, respond with:
   RISK RATING: Unknown
   SUMMARY: Insufficient context retrieved to answer this query. The document chunks provided do not contain relevant information.
   CITATIONS: N/A
   RECOMMENDED ACTION: Re-query with different search terms or review the full document.

Always respond in exactly this format:
RISK RATING: <Low | Medium | High | Critical | Unknown>
SUMMARY: <2-3 sentence summary of the finding>
CITATIONS: <Direct quotes from the source document with control IDs or section references>
RECOMMENDED ACTION: <Specific action the vendor or auditor should take>"""


def _get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    return anthropic.Anthropic(api_key=api_key)


def analyze_vendor_risk(query: str, n_results: int = 5) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from vectorstore import query_collection

    results = query_collection(query, n_results=n_results)

    context_blocks = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{r['text']}" for i, r in enumerate(results)
    )

    user_message = f"""Query: {query}

Retrieved document chunks:
{context_blocks}

Based solely on the chunks above, provide your structured risk analysis."""

    client = _get_client()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text
    parsed = {"query": query, "raw_response": raw, "chunks_used": len(results)}

    for line in raw.splitlines():
        if line.startswith("RISK RATING:"):
            parsed["risk_rating"] = line.split(":", 1)[1].strip()
        elif line.startswith("SUMMARY:"):
            parsed["summary"] = line.split(":", 1)[1].strip()
        elif line.startswith("CITATIONS:"):
            parsed["citations"] = line.split(":", 1)[1].strip()
        elif line.startswith("RECOMMENDED ACTION:"):
            parsed["recommended_action"] = line.split(":", 1)[1].strip()

    return parsed


def _print_result(result: dict) -> None:
    print(f'Query: "{result["query"]}"')
    print("=" * 70)
    print(result["raw_response"])
    print(f"\n[Chunks used: {result['chunks_used']}]")
    print()


if __name__ == "__main__":
    queries = [
        "What security exceptions or audit findings were identified?",
        "How is customer data encrypted and protected?",
        "What is the vendor's backup and recovery posture?",
    ]

    for query in queries:
        result = analyze_vendor_risk(query)
        _print_result(result)
