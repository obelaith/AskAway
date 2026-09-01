from sentence_transformers import CrossEncoder

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        k: int = 5,
    ) -> list[dict]:

        pairs = [
            (
                query,
                document["text"],
            )
            for document in documents
        ]

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for document, score in zip(
            documents,
            scores,
        ):
            result = document.copy()
            result["rerank_score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        for rank, result in enumerate(
            reranked[:k],
            start=1,
        ):
            result["rank"] = rank

        return reranked[:k]