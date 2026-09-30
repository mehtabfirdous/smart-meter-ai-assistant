import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking.chunk_documents import create_chunks
from src.ingestion.load_documents import load_pdf_documents


MODEL_NAME = "BAAI/bge-small-en-v1.5"

INDEX_PATH = "data/vectorstore/smart_meter.index"
CHUNKS_PATH = "data/vectorstore/chunks.pkl"


def create_vectorstore():

    # 1. Load documents
    documents = load_pdf_documents()

    # 2. Create chunks
    chunks = create_chunks(documents)

    print(f"Total chunks: {len(chunks)}")

    # 3. Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # 4. Extract chunk text
    texts = [chunk["text"] for chunk in chunks]

    # 5. Generate embeddings
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # 6. Convert embeddings to NumPy float32
    embeddings = np.asarray(embeddings, dtype="float32")

    print(f"Embedding shape: {embeddings.shape}")

    # 7. Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # 8. Add embeddings to FAISS
    index.add(embeddings)

    print(f"Vectors stored in FAISS: {index.ntotal}")

    # 9. Create output directory
    os.makedirs("data/vectorstore", exist_ok=True)

    # 10. Save FAISS index
    faiss.write_index(index, INDEX_PATH)

    # 11. Save chunks and metadata separately
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("\nVector store created successfully!")

    print(f"FAISS index: {INDEX_PATH}")
    print(f"Chunk metadata: {CHUNKS_PATH}")


if __name__ == "__main__":
    create_vectorstore()