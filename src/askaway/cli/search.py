import argparse
from pathlib import Path

from askaway.retrieval.bm25 import (
    BM25Retriever,
    load_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the document collection with BM25."
    )

    parser.add_argument(
        "corpus",
        type=Path,
        help="Path to the processed corpus.",
    )

    parser.add_argument(
        "query",
        type=str,
        help="The question or search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to show.",
    )

    args = parser.parse_args()

    chunks = load_chunks(args.corpus)
    retriever = BM25Retriever(chunks)

    results = retriever.search(
        args.query,
        k=args.top_k,
    )

    for result in results:
        print()
        print(
            f"[{result['rank']}] "
            f"{result['filename']} — "
            f"page {result['page_number']}"
        )

        print(f"Score: {result['score']:.4f}")
        print(result["text"][:700])


if __name__ == "__main__":
    main()