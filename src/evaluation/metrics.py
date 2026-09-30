import json
import statistics


RESULTS_PATH = "data/evaluation/results.json"
QUESTIONS_PATH = "data/evaluation/questions.json"
GROUND_TRUTH_PATH = "data/evaluation/ground_truth.json"


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def load_results():
    return load_json(RESULTS_PATH)


def load_questions():
    return load_json(QUESTIONS_PATH)


def load_ground_truth():
    return load_json(GROUND_TRUTH_PATH)


def calculate_metrics():

    results = load_results()
    questions = load_questions()
    ground_truth = load_ground_truth()

    total = len(results)

    # --------------------------------------------------
    # Create lookup dictionaries
    # --------------------------------------------------

    question_lookup = {
        item["id"]: item
        for item in questions
    }

    ground_truth_lookup = {
        item["id"]: item
        for item in ground_truth
    }

    # --------------------------------------------------
    # Counters
    # --------------------------------------------------

    successful = 0
    failed = 0

    strong_retrieval = 0
    weak_retrieval = 0

    all_scores = []

    correct_source_retrieval = 0

    answerability_correct = 0
    correct_refusals = 0

    grounded_answers = 0
    cited_answers = 0

    # --------------------------------------------------
    # Category statistics
    # --------------------------------------------------

    categories = {}

    # --------------------------------------------------
    # Process every evaluation result
    # --------------------------------------------------

    for item in results:

        question_id = item["id"]
        answer = item["answer"]

        answer_lower = answer.lower()

        # --------------------------------------------------
        # Question metadata
        # --------------------------------------------------

        question_info = question_lookup.get(
            question_id,
            {}
        )

        category = question_info.get(
            "category",
            "unknown"
        )

        # --------------------------------------------------
        # Ground truth
        # --------------------------------------------------

        truth = ground_truth_lookup.get(
            question_id,
            {}
        )

        expected_sources = truth.get(
            "expected_sources",
            []
        )

        expected_answerable = truth.get(
            "answerable",
            True
        )

        # --------------------------------------------------
        # Initialize category
        # --------------------------------------------------

        if category not in categories:

            categories[category] = {
                "total": 0,
                "successful": 0,
                "strong_retrieval": 0,
                "expected_source": 0,
                "answerability": 0,
                "grounded": 0,
                "citation": 0
            }

        categories[category]["total"] += 1

        # --------------------------------------------------
        # 1. Generation success
        # --------------------------------------------------

        generation_failed = (
            "generation failed"
            in answer_lower
        )

        if generation_failed:

            failed += 1

        else:

            successful += 1
            categories[category]["successful"] += 1

        # --------------------------------------------------
        # 2. Retrieval scores
        # --------------------------------------------------

        scores = [
            source["score"]
            for source in item.get("sources", [])
        ]

        if scores:

            max_score = max(scores)

            all_scores.append(max_score)

            if max_score >= 0.70:

                strong_retrieval += 1

                categories[category][
                    "strong_retrieval"
                ] += 1

            else:

                weak_retrieval += 1

        # --------------------------------------------------
        # 3. Expected source retrieval
        # --------------------------------------------------

        retrieved_sources = [
            source["source"]
            for source in item.get("sources", [])
        ]

        if expected_answerable:

            source_match = any(
                expected_source in retrieved_sources
                for expected_source
                in expected_sources
            )

            if source_match:

                correct_source_retrieval += 1

                categories[category][
                    "expected_source"
                ] += 1

        else:

            # For out-of-knowledge questions,
            # no expected source is correct.

            if not expected_sources:

                correct_source_retrieval += 1

                categories[category][
                    "expected_source"
                ] += 1

        # --------------------------------------------------
        # 4. Answerability
        # --------------------------------------------------

        refusal = (
            "i could not find enough information"
            in answer_lower
        )

        if expected_answerable:

            if not refusal:

                answerability_correct += 1

                categories[category][
                    "answerability"
                ] += 1

        else:

            if refusal:

                answerability_correct += 1
                correct_refusals += 1

                categories[category][
                    "answerability"
                ] += 1

        # --------------------------------------------------
        # 5. Grounding
        # --------------------------------------------------

        grounded = (
            "[source:" in answer_lower
        )

        if grounded:

            grounded_answers += 1

            categories[category][
                "grounded"
            ] += 1

        # --------------------------------------------------
        # 6. Citation coverage
        # --------------------------------------------------

        cited = False

        for source_name in retrieved_sources:

            if source_name.lower() in answer_lower:

                cited = True
                break

        # Out-of-knowledge answer has no expected
        # source but can still be considered correctly
        # cited if it explicitly states knowledge-base
        # limitation.

        if not cited and not expected_answerable:

            if refusal:

                cited = True

        if cited:

            cited_answers += 1

            categories[category][
                "citation"
            ] += 1

    # ==================================================
    # Retrieval statistics
    # ==================================================

    if all_scores:

        average_score = statistics.mean(
            all_scores
        )

        minimum_score = min(
            all_scores
        )

        maximum_score = max(
            all_scores
        )

    else:

        average_score = 0
        minimum_score = 0
        maximum_score = 0

    # ==================================================
    # Overall percentages
    # ==================================================

    if total > 0:

        generation_rate = (
            successful / total
        ) * 100

        retrieval_rate = (
            strong_retrieval / total
        ) * 100

        expected_source_rate = (
            correct_source_retrieval / total
        ) * 100

        answerability_rate = (
            answerability_correct / total
        ) * 100

        grounding_rate = (
            grounded_answers / total
        ) * 100

        citation_rate = (
            cited_answers / total
        ) * 100

    else:

        generation_rate = 0
        retrieval_rate = 0
        expected_source_rate = 0
        answerability_rate = 0
        grounding_rate = 0
        citation_rate = 0

    # ==================================================
    # Overall RAG Quality Score
    # ==================================================

    overall_rag_score = (
        generation_rate
        + expected_source_rate
        + answerability_rate
        + grounding_rate
        + citation_rate
    ) / 5

    # ==================================================
    # Display
    # ==================================================

    print()

    print("=" * 80)
    print("RAG EVALUATION METRICS")
    print("=" * 80)

    print()

    print(
        f"Total questions              : "
        f"{total}"
    )

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------

    print()

    print("GENERATION")
    print("-" * 80)

    print(
        f"Successful generations       : "
        f"{successful}"
    )

    print(
        f"Failed generations           : "
        f"{failed}"
    )

    print(
        f"Generation success rate      : "
        f"{generation_rate:.2f}%"
    )

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    print()

    print("RETRIEVAL")
    print("-" * 80)

    print(
        f"Average top similarity       : "
        f"{average_score:.4f}"
    )

    print(
        f"Minimum top similarity       : "
        f"{minimum_score:.4f}"
    )

    print(
        f"Maximum top similarity       : "
        f"{maximum_score:.4f}"
    )

    print(
        f"Strong retrieval (>= 0.70)   : "
        f"{strong_retrieval}"
    )

    print(
        f"Weak retrieval (< 0.70)      : "
        f"{weak_retrieval}"
    )

    print(
        f"Similarity-based retrieval   : "
        f"{retrieval_rate:.2f}%"
    )

    # --------------------------------------------------
    # Expected source
    # --------------------------------------------------

    print()

    print("EXPECTED SOURCE RETRIEVAL")
    print("-" * 80)

    print(
        f"Correct source retrieval     : "
        f"{correct_source_retrieval}"
    )

    print(
        f"Expected-source accuracy     : "
        f"{expected_source_rate:.2f}%"
    )

    # --------------------------------------------------
    # Answerability
    # --------------------------------------------------

    print()

    print("ANSWERABILITY")
    print("-" * 80)

    print(
        f"Answerability accuracy       : "
        f"{answerability_rate:.2f}%"
    )

    print(
        f"Correct refusals             : "
        f"{correct_refusals}"
    )

    # --------------------------------------------------
    # Grounding
    # --------------------------------------------------

    print()

    print("GROUNDING")
    print("-" * 80)

    print(
        f"Grounded answers             : "
        f"{grounded_answers}"
    )

    print(
        f"Grounded answer rate         : "
        f"{grounding_rate:.2f}%"
    )

    # --------------------------------------------------
    # Citation
    # --------------------------------------------------

    print()

    print("CITATION")
    print("-" * 80)

    print(
        f"Answers with source citation : "
        f"{cited_answers}"
    )

    print(
        f"Citation coverage rate       : "
        f"{citation_rate:.2f}%"
    )

    # --------------------------------------------------
    # Overall score
    # --------------------------------------------------

    print()

    print("OVERALL RAG QUALITY")
    print("-" * 80)

    print(
        f"Overall RAG Quality Score    : "
        f"{overall_rag_score:.2f}%"
    )

    # --------------------------------------------------
    # Category-wise evaluation
    # --------------------------------------------------

    print()

    print("=" * 80)
    print("CATEGORY-WISE EVALUATION")
    print("=" * 80)

    for category, stats in categories.items():

        category_total = stats["total"]

        if category_total > 0:

            category_generation = (
                stats["successful"]
                / category_total
            ) * 100

            category_retrieval = (
                stats["strong_retrieval"]
                / category_total
            ) * 100

            category_expected = (
                stats["expected_source"]
                / category_total
            ) * 100

            category_answerability = (
                stats["answerability"]
                / category_total
            ) * 100

            category_grounding = (
                stats["grounded"]
                / category_total
            ) * 100

            category_citation = (
                stats["citation"]
                / category_total
            ) * 100

        else:

            category_generation = 0
            category_retrieval = 0
            category_expected = 0
            category_answerability = 0
            category_grounding = 0
            category_citation = 0

        print()

        print(
            f"Category: {category}"
        )

        print("-" * 80)

        print(
            f"Questions                    : "
            f"{category_total}"
        )

        print(
            f"Generation success           : "
            f"{category_generation:.2f}%"
        )

        print(
            f"Strong retrieval             : "
            f"{category_retrieval:.2f}%"
        )

        print(
            f"Expected source retrieval    : "
            f"{category_expected:.2f}%"
        )

        print(
            f"Answerability accuracy       : "
            f"{category_answerability:.2f}%"
        )

        print(
            f"Grounded answers             : "
            f"{category_grounding:.2f}%"
        )

        print(
            f"Citation coverage            : "
            f"{category_citation:.2f}%"
        )

    # ==================================================
    # Completion
    # ==================================================

    print()

    print("=" * 80)
    print("METRICS CALCULATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":

    calculate_metrics()