import json


RESULTS_PATH = "data/evaluation/results.json"


def load_results():

    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def generate_report():

    results = load_results()

    print()
    print("=" * 80)
    print("RAG QUESTION-WISE EVALUATION REPORT")
    print("=" * 80)

    for item in results:

        question_id = item["id"]
        question = item["question"]
        answer = item["answer"]

        scores = [
            source["score"]
            for source in item["sources"]
        ]

        top_score = max(scores) if scores else 0

        # Retrieval status
        if top_score >= 0.70:
            retrieval_status = "STRONG"
        else:
            retrieval_status = "WEAK"

        # Generation status
        if "generation failed" in answer.lower():
            generation_status = "FAILED"
        else:
            generation_status = "SUCCESS"

        # Answerability
        if "could not find enough information" in answer.lower():
            answerability = "UNANSWERABLE"
        else:
            answerability = "ANSWERABLE"

        # Grounding
        if "[source:" in answer.lower():
            grounding = "GROUNDED"
        else:
            grounding = "NOT GROUNDED"

        print()
        print("-" * 80)

        print(f"ID            : {question_id}")
        print(f"Question      : {question}")
        print(f"Top Score     : {top_score:.4f}")
        print(f"Retrieval     : {retrieval_status}")
        print(f"Generation    : {generation_status}")
        print(f"Answerability : {answerability}")
        print(f"Grounding     : {grounding}")

        print()
        print("Retrieved Sources:")

        for source in item["sources"]:

            print(
                f"  - {source['source']} "
                f"| Page {source['page']} "
                f"| Score {source['score']:.4f}"
            )

    print()
    print("=" * 80)
    print("REPORT COMPLETED")
    print("=" * 80)


if __name__ == "__main__":

    generate_report()