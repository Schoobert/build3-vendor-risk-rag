import sys
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def ingest_pdf(pdf_path: str) -> list[str]:
    text = extract_text(pdf_path)

    if len(text) < 100:
        print(
            f"WARNING: Only {len(text)} characters extracted from '{pdf_path}'. "
            "This may be a scanned document — OCR may be required.",
            file=sys.stderr,
        )

    return chunk_text(text)


if __name__ == "__main__":
    pdf_path = "sample_data/NovaSoft_SOC2_TypeII_2024.pdf"
    chunks = ingest_pdf(pdf_path)
    text = extract_text(pdf_path)

    print(f"Total characters extracted: {len(text)}")
    print(f"Number of chunks created:   {len(chunks)}")
    print()
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"--- Chunk {i} ---")
        print(chunk)
        print()
