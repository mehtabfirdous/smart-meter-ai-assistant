from sentence_transformers import SentenceTransformer

from src.chunking.chunk_documents import create_chunks
from src.ingestion.load_documents import load_pdf_documents


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def create_embeddings(chunks):
    """
    Convert text chunks into embedding vectors.
    """

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":

    # Load PDF pages
    documents = load_pdf_documents()

    # Create chunks
    chunks = create_chunks(documents)

    print(f"Total chunks: {len(chunks)}")

    # Create embeddings
    embeddings = create_embeddings(chunks)

    print(f"Embedding shape: {embeddings.shape}")

    print("\nFirst embedding:")
    print(embeddings[0])