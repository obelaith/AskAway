import argparse
import json
from collections import defaultdict
from pathlib import Path

from askaway.retrieval.bm25 import load_chunks


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the stored text for Arabic benchmark pages."
    )

    parser.add_argument("corpus", type=Path)
    parser.add_argument("questions", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("arabic_pages.txt"),
    )

    args = parser.parse_args()

    chunks = load_chunks(args.corpus)
    questions = load_questions(args.questions)

    targets = defaultdict(list)

    for question in questions:
        if question["language"] != "ar":
            continue

        for relevant in question["relevant"]:
            key = (
                relevant["document_id"],
                relevant["page"],
            )

            targets[key].append(
                (question["id"], question["question"])
            )

    lines = []

    for (document_id, page), page_questions in targets.items():
        lines.append("=" * 70)
        lines.append(f"{document_id} — page {page}")
        lines.append("=" * 70)
        lines.append("")
        lines.append("Questions:")

        for question_id, question in page_questions:
            lines.append(f"- {question_id}: {question}")

        lines.append("")
        lines.append("Extracted text:")

        matching_chunks = [
            chunk
            for chunk in chunks
            if chunk.document_id == document_id
            and chunk.page_number == page
        ]

        if not matching_chunks:
            lines.append("[No chunks found for this page]")
        else:
            for chunk in matching_chunks:
                lines.append("")
                lines.append(f"[{chunk.chunk_id}]")
                lines.append(chunk.text)

        lines.append("")

    report = "\n".join(lines)

    args.output.write_text(
        report,
        encoding="utf-8",
    )

    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()