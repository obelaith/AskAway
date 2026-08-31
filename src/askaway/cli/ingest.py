import argparse
import json
from pathlib import Path

from askaway.ingestion.chunker import chunk_document
from askaway.ingestion.parser import extract_pdf


def make_document_id(path: Path, root: Path) -> str:
    relative_path = path.relative_to(root).with_suffix("")
    return "__".join(relative_path.parts)


def ingest_directory(input_dir: Path, output_file: Path) -> None:
    pdf_files = sorted(input_dir.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_chunks = 0
    pages_without_text = 0

    with output_file.open("w", encoding="utf-8") as output:
        for pdf_path in pdf_files:
            document_id = make_document_id(pdf_path, input_dir)
            pages = extract_pdf(pdf_path, document_id)

            total_pages += len(pages)
            pages_without_text += sum(
                len(page.text.strip()) < 20
                for page in pages
            )

            chunks = chunk_document(pages)
            total_chunks += len(chunks)

            for chunk in chunks:
                output.write(
                    json.dumps(
                        chunk.to_dict(),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            print(
                f"{pdf_path.name}: "
                f"{len(pages)} pages, "
                f"{len(chunks)} chunks"
            )

    print()
    print("Corpus created")
    print(f"Documents: {len(pdf_files)}")
    print(f"Pages: {total_pages}")
    print(f"Chunks: {total_chunks}")
    print(f"Pages with little/no extracted text: {pages_without_text}")
    print(f"Saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and chunk PDFs for Askaway."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the source PDFs.",
    )

    parser.add_argument(
        "output_file",
        type=Path,
        help="Path for the generated JSONL corpus.",
    )

    args = parser.parse_args()

    ingest_directory(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()