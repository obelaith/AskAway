from sentence_transformers import CrossEncoder

from askaway.config import load_config

DEFAULT_MODEL = None

class Reranker:
    def __init__(
        self,
        model_name: str | None = None,
    ):

        if model_name is None:
            config = load_config()
            model_name = config["models"]["reranker"]

        self.model_name = model_name
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