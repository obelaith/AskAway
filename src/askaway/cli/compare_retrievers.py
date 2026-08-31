import argparse
import json
from pathlib import Path

import numpy as np

from askaway.evaluation.retrieval import recall_at_k, reciprocal_rank
from askaway.retrieval.bm25 import BM25Retriever, load_chunks
from askaway.retrieval.dense import DenseRetriever


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate(
    retriever,
    questions: list[dict],
) -> dict:
    recall_1 = []
    recall_3 = []
    recall_5 = []
    reciprocal_ranks = []

    for item in questions:
        results = retriever.search(
            item["question"],
            k=5,
        )

        relevant = item["relevant"]

        recall_1.append(
            recall_at_k(results, relevant, 1)
        )
        recall_3.append(
            recall_at_k(results, relevant, 3)
        )
        recall_5.append(
            recall_at_k(results, relevant, 5)
        )
        reciprocal_ranks.append(
            reciprocal_rank(results, relevant)
        )

    return {
        "questions": len(questions),
        "recall@1": np.mean(recall_1),
        "recall@3": np.mean(recall_3),
        "recall@5": np.mean(recall_5),
        "mrr": np.mean(reciprocal_ranks),
    }


def print_results(
    name: str,
    results: dict,
) -> None:
    print(
        f"{name:<8} "
        f"{results['questions']:>3}  "
        f"{results['recall@1']:.3f}  "
        f"{results['recall@3']:.3f}  "
        f"{results['recall@5']:.3f}  "
        f"{results['mrr']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BM25 and dense retrieval by language."
    )

    parser.add_argument("corpus", type=Path)
    parser.add_argument("questions", type=Path)

    args = parser.parse_args()

    chunks = load_chunks(args.corpus)
    questions = load_questions(args.questions)

    english = [
        item
        for item in questions
        if item["language"] == "en"
    ]

    arabic = [
        item
        for item in questions
        if item["language"] == "ar"
    ]

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)

    print()
    print("Retriever comparison")
    print("--------------------")
    print("Method     N   R@1    R@3    R@5    MRR")

    for language, subset in [
        ("ALL", questions),
        ("EN", english),
        ("AR", arabic),
    ]:
        print()
        print(language)

        print_results(
            "BM25",
            evaluate(bm25, subset),
        )

        print_results(
            "Dense",
            evaluate(dense, subset),
        )


if __name__ == "__main__":
    main()