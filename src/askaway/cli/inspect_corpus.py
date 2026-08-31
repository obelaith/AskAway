import argparse
from collections import Counter
from pathlib import Path

from askaway.retrieval.bm25 import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show a quick summary of the processed document collection."
    )

    parser.add_argument(
        "corpus",
        type=Path,
        help="Path to the processed JSONL corpus.",
    )

    args = parser.parse_args()

    chunks = load_chunks(args.corpus)

    counts = Counter(
        chunk.filename
        for chunk in chunks
    )

    print()
    print("Corpus summary")
    print("--------------")

    for filename, count in sorted(counts.items()):
        print(f"{filename}: {count} chunks")

    print()
    print(f"Total chunks: {len(chunks)}")

    print()
    print("Sample text")
    print("-----------")

    seen = set()

    for chunk in chunks:
        if chunk.filename in seen:
            continue

        seen.add(chunk.filename)

        preview = chunk.text[:300].replace("\n", " ")

        print()
        print(f"{chunk.filename} — page {chunk.page_number}")
        print(preview)


if __name__ == "__main__":
    main()