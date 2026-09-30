import json
import traceback

from src.generation.rag_pipeline import generate_answer


QUESTIONS_PATH = "data/evaluation/questions.json"
OUTPUT_PATH = "data/evaluation/results.json"


def load_questions():

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def evaluate():

    questions = load_questions()

    print("=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    print(f"Total questions: {len(questions)}")

    results = []

    for i, item in enumerate(questions, start=1):

        question_id = item["id"]
        question = item["question"]

        print("\n" + "=" * 70)
        print(f"Question {i}/{len(questions)}")
        print("=" * 70)

        print(f"ID       : {question_id}")
        print(f"Question : {question}")

        try:

            result = generate_answer(
                question,
                top_k=3
            )

            answer = result["answer"]

            print("\nAI ANSWER")
            print("-" * 70)
            print(answer)

            sources = []

            print("\nRETRIEVED SOURCES")
            print("-" * 70)

            for source in result["sources"]:

                metadata = source["metadata"]

                source_info = {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "score": source["score"]
                }

                sources.append(source_info)

                print(
                    f"- {metadata['source']} | "
                    f"Page {metadata['page']} | "
                    f"Score {source['score']:.4f}"
                )

            results.append({
                "id": question_id,
                "question": question,
                "answer": answer,
                "sources": sources,
                "status": "success"
            })

        except Exception as e:

            print("\nERROR")
            print("-" * 70)
            print(str(e))

            traceback.print_exc()

            results.append({
                "id": question_id,
                "question": question,
                "answer": "",
                "sources": [],
                "status": "failed",
                "error": str(e)
            })

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    successful = sum(
        1 for item in results
        if item["status"] == "success"
    )

    failed = len(results) - successful

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print(f"Total     : {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed    : {failed}")

    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate()