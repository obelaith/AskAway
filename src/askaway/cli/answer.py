import argparse
from pathlib import Path

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

    chunks = load_chunks(args.corpus)

    print("Loading retrieval pipeline...")

    bm25 = BM25Retriever(chunks)

    dense = DenseRetriever(chunks)

    hybrid = HybridRetriever(
        bm25,
        dense,
        bm25_weight=0.7,
    )

    reranker = Reranker()

    retriever = HybridRerankerRetriever(
        hybrid,
        reranker,
        candidate_k=10,
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