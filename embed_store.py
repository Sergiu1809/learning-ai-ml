from pathlib import Path
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

PDF_DIR = Path("pdfs")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DB_DIR = "chroma_db"  # Chroma will persist data here on disk
COLLECTION_NAME = "oop_labs"  # like a "table" name for this corpus


def extract_text_from_pdf(pdf_path: Path) -> str:
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

    return ("\n").join(full_text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current_chunk and len(current_chunk) + len(para) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + "\n" + para
        elif len(para) > chunk_size:
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
    # 1. Load the embedding model (dowwnloads once, then cached locally)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. Extract + chunk all PDFs, same as before
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    all_chunks = []  # list of chunk texts
    # list of dicts, one per chunk (source file, chunk index)
    all_metadatas = []
    all_ids = []  # unique string id per chunk, required by Chroma

    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": pdf_path.name, "chunk_index": i})
            # e.g. "Laborator1_Clase_Mosteniri (1)_0"
            all_ids.append(f"{pdf_path.stem}_{i}")

    print(f"Total chunks to embed: {len(all_chunks)}")

    # 3. Generate embeddings for all chunks at once (batched, much faster than one-by-one)
    print("Generating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)

    # 4. Store everything in Chroma
    print("Storing in Chroma...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=all_ids,
        embeddings=embeddings.tolist(),  # Chroma wants plain lists, not numpy arrays
        documents=all_chunks,  # store the raw text too,  so we can retrieve it directly
        metadatas=all_metadatas,
    )

    print(
        f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")
