import argparse
from pathlib import Path

import pymupdf

from askaway.ingestion.ocr import extract_page_with_ocr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR on one PDF page."
    )

    parser.add_argument(
        "pdf",
        type=Path,
    )

    parser.add_argument(
        "page",
        type=int,
        help="Physical PDF page number, starting from 1.",
    )

    args = parser.parse_args()

    with pymupdf.open(args.pdf) as document:
        if args.page < 1 or args.page > len(document):
            raise ValueError(
                f"Page must be between 1 and {len(document)}"
            )

        page = document[args.page - 1]

        text = extract_page_with_ocr(page)

    output = Path("ocr_test.txt")

    output.write_text(
        text,
        encoding="utf-8",
    )

    print(f"OCR text saved to {output}")


if __name__ == "__main__":
    main()