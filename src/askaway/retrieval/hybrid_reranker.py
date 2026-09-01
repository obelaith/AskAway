from askaway.config import load_config
from askaway.retrieval.hybrid import HybridRetriever
from askaway.retrieval.reranker import Reranker


class HybridRerankerRetriever:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        candidate_k: int | None = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

        if candidate_k is None:
            config = load_config()
            candidate_k = config["retrieval"]["candidate_k"]

        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        k: int | None = None,
    ) -> list[dict]:

        if k is None:
            config = load_config()
            k = config["retrieval"]["top_k"]

        candidates = self.hybrid_retriever.search(
            query,
            k=self.candidate_k,
        )

        results = self.reranker.rerank(
            query,
            candidates,
            k=k,
        )

        return results