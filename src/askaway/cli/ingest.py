import argparse
import json
from pathlib import Path

from askaway.ingestion.chunker import chunk_document
from askaway.ingestion.ocr import extract_pdf_with_ocr
from askaway.ingestion.parser import extract_pdf


def make_document_id(
    path: Path,
    root: Path,
) -> str:
    relative_path = path.relative_to(root).with_suffix("")
    return "__".join(relative_path.parts)


def detect_language_folder(
    pdf_path: Path,
    input_dir: Path,
) -> str:
    relative_parts = pdf_path.relative_to(
        input_dir
    ).parts

    if relative_parts:
        return relative_parts[0]

    return ""


def ingest_directory(
    input_dir: Path,
    output_file: Path,
) -> None:

    pdf_files = sorted(
        input_dir.rglob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {input_dir}"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_pages = 0
    total_chunks = 0
    ocr_documents = 0

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as output:

        for pdf_path in pdf_files:

            document_id = make_document_id(
                pdf_path,
                input_dir,
            )

            language = detect_language_folder(
                pdf_path,
                input_dir,
            )

            print()
            print(
                f"Processing {pdf_path.name}"
            )

            if language == "arabic":
                print(
                    "Extraction: OCR"
                )

                pages = extract_pdf_with_ocr(
                    pdf_path,
                    document_id,
                )

                ocr_documents += 1

            else:
                print(
                    "Extraction: native PDF text"
                )

                pages = extract_pdf(
                    pdf_path,
                    document_id,
                )

            total_pages += len(pages)

            chunks = chunk_document(
                pages
            )

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
                f"Finished {pdf_path.name}: "
                f"{len(pages)} pages, "
                f"{len(chunks)} chunks"
            )

    print()
    print(
        "AskAway ingestion complete"
    )
    print(
        "------------------------"
    )
    print(
        f"Documents: {len(pdf_files)}"
    )
    print(
        f"Pages:     {total_pages}"
    )
    print(
        f"Chunks:    {total_chunks}"
    )
    print(
        f"OCR docs:  {ocr_documents}"
    )
    print(
        f"Saved to:  {output_file}"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build AskAway corpus using "
            "OCR for Arabic documents and "
            "native extraction otherwise."
        )
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing PDFs.",
    )

    parser.add_argument(
        "output_file",
        type=Path,
        help="Output JSONL corpus.",
    )

    args = parser.parse_args()

    ingest_directory(
        args.input_dir,
        args.output_file,
    )


if __name__ == "__main__":
    main()