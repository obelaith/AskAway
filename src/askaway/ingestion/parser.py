from pathlib import Path

import pymupdf

from askaway.models import Page


def extract_pdf(path: Path, document_id: str) -> list[Page]:
    pages = []

    with pymupdf.open(path) as document:
        for page_index, pdf_page in enumerate(document):
            text = pdf_page.get_text("text", sort=True)

            pages.append(
                Page(
                    document_id=document_id,
                    filename=path.name,
                    page_number=page_index + 1,
                    text=text,
                )
            )

    return pages