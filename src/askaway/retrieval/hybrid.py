from askaway.retrieval.bm25 import BM25Retriever
from askaway.retrieval.dense import DenseRetriever


def min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [1.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        bm25_weight: float = 0.5,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.bm25_weight = bm25_weight
        self.dense_weight = 1 - bm25_weight

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_k: int = 10,
    ) -> list[dict]:

        bm25_results = self.bm25.search(
            query,
            k=candidate_k,
        )

        dense_results = self.dense.search(
            query,
            k=candidate_k,
        )

        combined = {}

        bm25_scores = min_max_normalize(
            [
                result["score"]
                for result in bm25_results
            ]
        )

        dense_scores = min_max_normalize(
            [
                result["score"]
                for result in dense_results
            ]
        )

        for result, score in zip(
            bm25_results,
            bm25_scores,
        ):
            combined[result["chunk_id"]] = {
                **result,
                "hybrid_score": (
                    self.bm25_weight * score
                ),
            }

        for result, score in zip(
            dense_results,
            dense_scores,
        ):
            chunk_id = result["chunk_id"]

            if chunk_id in combined:
                combined[chunk_id]["hybrid_score"] += (
                    self.dense_weight * score
                )
            else:
                combined[chunk_id] = {
                    **result,
                    "hybrid_score": (
                        self.dense_weight * score
                    ),
                }

        ranked = sorted(
            combined.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        for rank, result in enumerate(
            ranked[:k],
            start=1,
        ):
            result["rank"] = rank

        return ranked[:k]