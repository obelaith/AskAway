import argparse
import json
from pathlib import Path

import numpy as np

from askaway.evaluation.retrieval import recall_at_k, reciprocal_rank
from askaway.retrieval.bm25 import BM25Retriever, load_chunks


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 on the retrieval benchmark."
    )

    parser.add_argument("corpus", type=Path)
    parser.add_argument("questions", type=Path)

    args = parser.parse_args()

    chunks = load_chunks(args.corpus)
    questions = load_questions(args.questions)

    retriever = BM25Retriever(chunks)

    recall_1 = []
    recall_3 = []
    recall_5 = []
    reciprocal_ranks = []
    misses = []

    for item in questions:
        if not item.get("answerable", True):
            continue

        results = retriever.search(
            item["question"],
            k=5,
        )

        relevant = item["relevant"]

        r1 = recall_at_k(results, relevant, 1)
        r3 = recall_at_k(results, relevant, 3)
        r5 = recall_at_k(results, relevant, 5)

        recall_1.append(r1)
        recall_3.append(r3)
        recall_5.append(r5)

        reciprocal_ranks.append(
            reciprocal_rank(results, relevant)
        )

        if r5 == 0:
            misses.append(
                (item["id"], item["question"])
            )

    print()
    print("BM25 retrieval results")
    print("----------------------")
    print(f"Questions: {len(recall_1)}")
    print(f"Recall@1: {np.mean(recall_1):.3f}")
    print(f"Recall@3: {np.mean(recall_3):.3f}")
    print(f"Recall@5: {np.mean(recall_5):.3f}")
    print(f"MRR:      {np.mean(reciprocal_ranks):.3f}")

    if misses:
        print()
        print("Missed at k=5")
        print("-------------")

        for question_id, question in misses:
            print(f"{question_id}: {question}")


if __name__ == "__main__":
    main()