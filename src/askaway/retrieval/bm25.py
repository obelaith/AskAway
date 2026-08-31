import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from askaway.models import Chunk

ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+|[\u0600-\u06FF]+"
)


def normalize_for_search(text: str) -> str:
    text = text.lower()
    text = ARABIC_DIACRITICS.sub("", text)

    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
    )

    return text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(
        normalize_for_search(text)
    )


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            chunks.append(Chunk(**data))

    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError(
                "Cannot build an index from an empty corpus"
            )

        self.chunks = chunks

        tokenized_corpus = [
            tokenize(chunk.text)
            for chunk in chunks
        ]

        self.index = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict]:
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:k]

        results = []

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):
            chunk = self.chunks[int(index)]

            results.append(
                {
                    "rank": rank,
                    "score": float(scores[index]),
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "language": chunk.language,
                    "text": chunk.text,
                }
            )

        return results