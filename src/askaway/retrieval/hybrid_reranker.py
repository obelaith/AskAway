from askaway.retrieval.hybrid import HybridRetriever
from askaway.retrieval.reranker import Reranker


class HybridRerankerRetriever:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        candidate_k: int = 10,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict]:

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