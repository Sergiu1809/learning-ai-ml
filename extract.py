import pdfplumber
from pathlib import Path

PDF_DIR = Path("pdfs")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF, page by page"""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                full_text.append(text)

    return "\n".join(full_text)


if __name__ == "__main__":
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.\n")

    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        print(f"--- {pdf_path.name} ---")
        print(f"Extracted {len(text)} characters")
        print(text[:300])  # preview first 300 chars
        print()
