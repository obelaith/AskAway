from askaway.ingestion.text import clean_text, detect_language
from askaway.models import Chunk, Page


def chunk_page(
    page: Page,
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[Chunk]:
    text = clean_text(page.text)

    if not text:
        return []

    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])

        chunk_id = (
            f"{page.document_id}:"
            f"p{page.page_number}:"
            f"c{chunk_index}"
        )

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                filename=page.filename,
                page_number=page.page_number,
                chunk_index=chunk_index,
                language=detect_language(chunk_text),
                text=chunk_text,
            )
        )

        if end == len(words):
            break

        start = end - overlap
        chunk_index += 1

    return chunks


def chunk_document(
    pages: list[Page],
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[Chunk]:
    chunks = []

    for page in pages:
        chunks.extend(
            chunk_page(
                page,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return chunks