from askaway.ingestion.chunker import chunk_page
from askaway.models import Page


def test_chunk_page_preserves_metadata():
    text = " ".join(f"word{i}" for i in range(300))

    page = Page(
        document_id="sample_doc",
        filename="sample.pdf",
        page_number=3,
        text=text,
    )

    chunks = chunk_page(
        page,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.document_id == "sample_doc"
        assert chunk.filename == "sample.pdf"
        assert chunk.page_number == 3


def test_empty_page_returns_no_chunks():
    page = Page(
        document_id="empty_doc",
        filename="empty.pdf",
        page_number=1,
        text="",
    )

    assert chunk_page(page) == []