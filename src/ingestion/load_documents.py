from pathlib import Path
import fitz  # PyMuPDF


DOCUMENTS_DIR = Path("data/documents")


def load_pdf_documents():
    documents = []

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_path in pdf_files:
        pdf = fitz.open(pdf_path)

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append({
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number,
                    "file_path": str(pdf_path)
                }
            })

        pdf.close()

    return documents


if __name__ == "__main__":
    documents = load_pdf_documents()

    print(f"Total pages extracted: {len(documents)}")

    if documents:
        print("\nFirst document:")
        print("----------------")
        print(documents[0]["text"][:1000])

        print("\nMetadata:")
        print(documents[0]["metadata"])