import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer questions using AskAway RAG."
    )

    parser.add_argument(
        "corpus",
        type=Path,
        help="Path to processed chunks.",
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question to answer.",
    )

    args = parser.parse_args()

    config = load_config()

    chunks = load_chunks(args.corpus)

    print("Loading retrieval pipeline...")

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

    rag = RAGPipeline(
        retriever,
        llm,
    )

    result = rag.answer(
        args.question,
    )

    print()
    print("Answer")
    print("------")
    print(result["answer"])

    print()
    print("Sources")
    print("-------")

    for source in result["sources"]:
        print(
            f"- {source['filename']} "
            f"(page {source['page_number']})"
        )


if __name__ == "__main__":
    main()