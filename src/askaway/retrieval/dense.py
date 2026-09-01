import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from askaway.config import load_config
from askaway.models import Chunk

DEFAULT_MODEL = None

def corpus_fingerprint(chunks: list[Chunk]) -> str:
    hasher = hashlib.sha256()

    for chunk in chunks:
        hasher.update(chunk.chunk_id.encode("utf-8"))
        hasher.update(chunk.text.encode("utf-8"))

    return hasher.hexdigest()


class DenseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str | None = None,
        cache_dir: Path = Path("data/processed/dense"),
    ):
        if not chunks:
            raise ValueError(
                "Cannot build a dense index from an empty corpus"
            )

        self.chunks = chunks

        if model_name is None:
            config = load_config()
            model_name = config["models"]["embedding"]

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        cache_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_path = cache_dir / "embeddings.npy"
        self.metadata_path = cache_dir / "metadata.json"

        self.embeddings = self._load_or_create_embeddings()

    def _load_or_create_embeddings(self) -> np.ndarray:
        fingerprint = corpus_fingerprint(self.chunks)

        if self.embeddings_path.exists() and self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)

            cache_matches = (
                metadata.get("model_name") == self.model_name
                and metadata.get("corpus_fingerprint") == fingerprint
                and metadata.get("chunk_count") == len(self.chunks)
            )

            if cache_matches:
                print("Loading cached document embeddings...")
                return np.load(self.embeddings_path)

        passages = [
            f"passage: {chunk.text}"
            for chunk in self.chunks
        ]

        print(f"Embedding {len(passages)} document chunks...")

        embeddings = self.model.encode(
            passages,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        np.save(
            self.embeddings_path,
            embeddings,
        )

        metadata = {
            "model_name": self.model_name,
            "chunk_count": len(self.chunks),
            "corpus_fingerprint": fingerprint,
        }

        with self.metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )

        print("Document embeddings saved.")

        return embeddings

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict]:
        query_embedding = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
        )[0]

        scores = self.embeddings @ query_embedding
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