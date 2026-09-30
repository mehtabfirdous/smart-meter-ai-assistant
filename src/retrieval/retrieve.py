import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

INDEX_PATH = "data/vectorstore/smart_meter.index"
CHUNKS_PATH = "data/vectorstore/chunks.pkl"


print("Loading vector store...")

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)


print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("Embedding model loaded successfully.")


def retrieve(query, top_k=3):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position == -1:
            continue

        chunk = chunks[index_position]

        results.append({
            "score": float(score),
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        })

    return results


if __name__ == "__main__":

    question = (
        "Why is my smart meter not communicating with HES?"
    )

    results = retrieve(
        question,
        top_k=3
    )

    print("\nUSER QUESTION")
    print("=" * 60)
    print(question)

    print("\nRETRIEVED CHUNKS")
    print("=" * 60)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(f"\nResult {i}")
        print("-" * 60)

        print(
            f"Similarity score: "
            f"{result['score']:.4f}"
        )

        print("\nSource:")
        print(
            result["metadata"]["source"]
        )

        print("\nPage:")
        print(
            result["metadata"]["page"]
        )

        print("\nText:")
        print(result["text"])