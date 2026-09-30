import requests

from src.retrieval.retrieve import retrieve


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

# Minimum similarity required before asking the LLM
MIN_RETRIEVAL_SCORE = 0.50


REFUSAL_MESSAGE = (
    "I could not find enough information in the available knowledge base."
)


def build_prompt(question, retrieved_chunks):

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        source = chunk["metadata"]["source"]
        page = chunk["metadata"]["page"]

        context_parts.append(
            f"""
[CONTEXT {i}]
Source: {source}
Page: {page}

Content:
{chunk["text"]}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a Smart Meter AI Assistant for utility operations.

Answer the user's question using ONLY the information provided
in the context.

Rules:

1. Use only information contained in the context.
2. Do not use outside knowledge.
3. Do not invent technical procedures, values, causes, or facts.
4. Do not mention or expose CONTEXT numbers.
5. Do not generate document citations.
6. Do not create a References section.
7. Keep the answer concise, factual, and professional.
8. If the context does not contain enough information to answer
   the question, say exactly:

"I could not find enough information in the available knowledge base."

9. If the question is only partially answerable, clearly state
   what information is available and what information is missing.

At the END of your answer, provide the numbers of the context
sections that directly support your answer.

Use this exact format:

SUPPORTING_CONTEXTS: 1, 3

Do not include anything after the SUPPORTING_CONTEXTS line.

Return ONLY the answer and the SUPPORTING_CONTEXTS line.

Context:
====================
{context}
====================

User Question:
{question}

Answer:
"""

    return prompt


def parse_supporting_contexts(response_text, retrieved_chunks):

    lines = response_text.strip().splitlines()

    supporting_contexts = []

    answer_lines = []

    for line in lines:

        if line.strip().upper().startswith(
            "SUPPORTING_CONTEXTS:"
        ):

            value = line.split(
                ":",
                1
            )[1].strip()

            for item in value.split(","):

                item = item.strip()

                try:

                    context_number = int(item)

                    if (
                        1 <= context_number
                        <= len(retrieved_chunks)
                    ):

                        supporting_contexts.append(
                            context_number
                        )

                except ValueError:

                    continue

        else:

            answer_lines.append(line)

    answer = "\n".join(answer_lines).strip()

    return answer, supporting_contexts


def add_source_citations(
    answer,
    retrieved_chunks,
    supporting_contexts
):

    if not answer:
        return answer

    if answer == REFUSAL_MESSAGE:
        return answer

    # If the model did not identify supporting contexts,
    # fall back to the highest-ranked retrieved source.
    if not supporting_contexts:

        supporting_contexts = [1]

    seen = set()
    citations = []

    for context_number in supporting_contexts:

        chunk = retrieved_chunks[
            context_number - 1
        ]

        source = chunk["metadata"]["source"]
        page = chunk["metadata"]["page"]

        citation = (
            f"[Source: {source}, Page: {page}]"
        )

        if citation not in seen:

            citations.append(citation)
            seen.add(citation)

    if citations:

        answer = (
            answer.rstrip()
            + "\n\n"
            + "\n".join(citations)
        )

    return answer


def generate_answer(question, top_k=3):

    retrieved_chunks = retrieve(
        question,
        top_k=top_k
    )

    # ---------------------------------------------------------
    # ANSWERABILITY GUARD
    # ---------------------------------------------------------

    if not retrieved_chunks:

        return {
            "answer": REFUSAL_MESSAGE,
            "sources": []
        }

    top_score = retrieved_chunks[0]["score"]

    if top_score < MIN_RETRIEVAL_SCORE:

        return {
            "answer": REFUSAL_MESSAGE,
            "sources": retrieved_chunks
        }

    # ---------------------------------------------------------
    # GENERATION
    # ---------------------------------------------------------

    prompt = build_prompt(
        question,
        retrieved_chunks
    )

    print(
        f"\nTop retrieval score: {top_score:.4f}"
    )

    print(
        f"Prompt length: {len(prompt)} characters"
    )

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

    except requests.RequestException as e:

        print("\nOllama connection error:")
        print(e)

        return {
            "answer": "Ollama generation failed.",
            "sources": retrieved_chunks
        }

    if response.status_code != 200:

        print("\nOllama returned an error:")
        print(response.text)

        return {
            "answer": "Ollama generation failed.",
            "sources": retrieved_chunks
        }

    result = response.json()

    raw_answer = result.get(
        "response",
        REFUSAL_MESSAGE
    ).strip()

    if not raw_answer:

        raw_answer = REFUSAL_MESSAGE

    # ---------------------------------------------------------
    # PARSE SUPPORTING CONTEXTS
    # ---------------------------------------------------------

    if raw_answer == REFUSAL_MESSAGE:

        answer = REFUSAL_MESSAGE
        supporting_contexts = []

    else:

        answer, supporting_contexts = (
            parse_supporting_contexts(
                raw_answer,
                retrieved_chunks
            )
        )

        if not answer:

            answer = REFUSAL_MESSAGE

    # ---------------------------------------------------------
    # PROGRAMMATIC CITATIONS
    # ---------------------------------------------------------

    answer = add_source_citations(
        answer,
        retrieved_chunks,
        supporting_contexts
    )

    return {
        "answer": answer,
        "sources": retrieved_chunks
    }


if __name__ == "__main__":

    question = (
        "Why is my smart meter not communicating with HES?"
    )

    result = generate_answer(
        question,
        top_k=3
    )

    print("\nUSER QUESTION")
    print("=" * 60)
    print(question)

    print("\nAI ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\nRETRIEVED SOURCES")
    print("=" * 60)

    for source in result["sources"]:

        print(
            f"- {source['metadata']['source']} "
            f"(Page {source['metadata']['page']}, "
            f"Score {source['score']:.4f})"
        )