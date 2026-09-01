from askaway.generation.llm import BaseLLM
from askaway.retrieval.hybrid_reranker import (
    HybridRerankerRetriever,
)


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRerankerRetriever,
        llm: BaseLLM,
    ):
        self.retriever = retriever
        self.llm = llm

    def build_prompt(
        self,
        question: str,
        context: list[dict],
    ) -> str:

        context_text = "\n\n".join(
            [
                (
                    f"Source: {item['filename']} "
                    f"(page {item['page_number']})\n"
                    f"{item['text']}"
                )
                for item in context
            ]
        )

        prompt = f"""
You are a helpful assistant answering questions from provided documents.

Use only the provided context.
If the answer is not in the context, say that you do not know.

Question:
{question}

Context:
{context_text}

Answer:
"""

        return prompt

    def answer(
        self,
        question: str,
    ) -> dict:

        chunks = self.retriever.search(
            question,
            k=5,
        )

        prompt = self.build_prompt(
            question,
            chunks,
        )

        answer = self.llm.generate(
            prompt
        )

        sources = [
            {
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "score": chunk.get(
                    "rerank_score",
                    None,
                ),
            }
            for chunk in chunks
            if chunk.get(
                "rerank_score",
                0,
            ) >= 0.6
        ]

        return {
            "answer": answer,
            "sources": sources,
        }