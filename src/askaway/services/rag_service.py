from pathlib import Path

from askaway.config import load_config
from askaway.generation.openai_llm import OpenAILLM
from askaway.rag import RAGPipeline
from askaway.retrieval.bm25 import (
    BM25Retriever,
    load_chunks,
)
from askaway.retrieval.dense import DenseRetriever
from askaway.retrieval.hybrid import HybridRetriever
from askaway.retrieval.hybrid_reranker import (
    HybridRerankerRetriever,
)
from askaway.retrieval.reranker import Reranker


class RAGService:

    def __init__(self, corpus_path):

        config = load_config()

        corpus_path = Path(corpus_path)

        chunks = load_chunks(corpus_path)


        bm25 = BM25Retriever(chunks)


        dense = DenseRetriever(chunks)


        hybrid = HybridRetriever(
            bm25,
            dense,
            bm25_weight=config["retrieval"]["bm25_weight"],
        )


        reranker = Reranker()


        retriever = HybridRerankerRetriever(
            hybrid,
            reranker,
            candidate_k=config["retrieval"]["candidate_k"],
        )


        llm = OpenAILLM()


        self.rag = RAGPipeline(
            retriever,
            llm,
        )


    def ask(self, question: str):

        return self.rag.answer(question)