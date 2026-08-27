from pathlib import Path
import pdfplumber

PDF_DIR = Path("pdfs")
CHUNK_SIZE = 800  # target chars per chunk
CHUNK_OVERLAP = 150  # overlap between consecutive chunks


def extract_text_from_pdf(pdf_path: Path) -> str:
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

    return "\n".join(full_text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Recursive-style splitting: try to split on paragraph breaks first.
    Falls back to raw character slicing if a paragraph is too long.
    """

    paragraphs = text.split("\n")  # rough paragraph split
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph would exceed chunk_size, save current chunk and start new one
        if current_chunk and len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # start new chunk with overlap from the end of the previous chunk
            current_chunk = current_chunk[-overlap:] + "\n" + para
        elif len(para) > chunk_size:
            # para itself is bigger than chunk_size: force-split it
            if current_chunk:
                chunks.append(current_chunk.strip())
            chunks.append(para[:chunk_size])
            current_chunk = para[chunk_size - overlap:]
        else:
            current_chunk += "\n" + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


if __name__ == "__main__":
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    all_chunks = []  # will hold (source_file, chunk_text) tuples

    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"{pdf_path.name}: {len(chunks)} chunks")

        for chunk in chunks:
            all_chunks.append((pdf_path.name, chunk))

    print(f"\nTotal chunks across all PDFs: {len(all_chunks)}")

    # inspect a couple of chunks concretely
    print("\n--- Example chunk #0 ---")
    print(all_chunks[0][0])
    print(all_chunks[0][1])

    print("\n--- Example chunk #1 ---")
    print(all_chunks[1][0])
    print(all_chunks[1][1])
