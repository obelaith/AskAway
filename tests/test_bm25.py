from askaway.models import Chunk
from askaway.retrieval.bm25 import BM25Retriever, tokenize


def make_chunk(
    chunk_id: str,
    text: str,
    language: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        filename=f"{chunk_id}.pdf",
        page_number=1,
        chunk_index=0,
        language=language,
        text=text,
    )


def test_bm25_finds_english_passage():
    chunks = [
        make_chunk(
            "employment",
            "Employees may qualify for overtime pay after working more than forty hours.",
            "en",
        ),
        make_chunk(
            "transport",
            "Drivers must renew their vehicle registration before it expires.",
            "en",
        ),
        make_chunk(
            "health",
            "Patients should speak with a doctor about treatment options.",
            "en",
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "overtime employees forty hours",
        k=1,
    )

    assert results[0]["chunk_id"] == "employment"


def test_bm25_finds_arabic_passage():
    chunks = [
        make_chunk(
            "finance",
            "يهدف الدليل إلى حماية المستهلك المالي وتوضيح حقوق العملاء.",
            "ar",
        ),
        make_chunk(
            "health",
            "تقدم الوزارة خدمات صحية للمواطنين والمقيمين.",
            "ar",
        ),
        make_chunk(
            "transport",
            "تشمل الخدمات ترخيص المركبات وتجديد رخص القيادة.",
            "ar",
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "حماية المستهلك المالي",
        k=1,
    )

    assert results[0]["chunk_id"] == "finance"


def test_arabic_normalization():
    tokens = tokenize("الإجازة السنوية")

    assert "الاجازة" in tokens
    assert "السنوية" in tokens