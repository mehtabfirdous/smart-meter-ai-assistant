from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.load_documents import load_pdf_documents
from src.ingestion.clean_text import clean_text


def create_chunks(documents):
    """
    Split documents into smaller chunks while preserving metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for document in documents:

        cleaned_text = clean_text(document["text"])

        split_texts = splitter.split_text(cleaned_text)

        for chunk_index, chunk_text in enumerate(split_texts):

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **document["metadata"],
                    "chunk_id": chunk_index
                }
            })

    return chunks


if __name__ == "__main__":

    # Step 1: Load PDF documents
    documents = load_pdf_documents()

    print(f"Documents/pages loaded: {len(documents)}")

    # Step 2: Create chunks
    chunks = create_chunks(documents)

    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Display first few chunks
    print("\nFirst 3 chunks")
    print("=" * 60)

    for i, chunk in enumerate(chunks[:3]):

        print(f"\nChunk {i + 1}")
        print("-" * 60)

        print(chunk["text"])

        print("\nMetadata:")
        print(chunk["metadata"])