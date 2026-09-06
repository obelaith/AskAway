# AskAway

AskAway is a multilingual Retrieval-Augmented Generation (RAG) system for answering questions over PDF documents.

The project focuses on building a complete document question-answering pipeline, including document ingestion, OCR processing, retrieval, reranking, and LLM-based answer generation.

The system supports both English and Arabic documents, including scanned PDFs that require OCR before they can be searched.

---

## 

Traditional keyword search can struggle when users ask questions using different wording than the original document.

AskAway combines multiple retrieval approaches:

- BM25 lexical retrieval for exact term matching
- Dense retrieval for semantic similarity
- Hybrid retrieval combining both methods
- Cross-encoder reranking to improve final result ordering
- LLM generation using retrieved document context

The goal is to retrieve relevant document passages first, then generate answers grounded only on the retrieved information.


## Features

- Multilingual document question answering
- Arabic OCR support for scanned PDFs
- Native PDF text extraction
- Document chunking
- BM25 retrieval
- Dense semantic retrieval
- Hybrid retrieval
- Cross-encoder reranking
- Source-aware answers
- Configurable retrieval and model settings

---

## Retrieval Models

### Sparse Retrieval

BM25 is used to capture exact keyword matches and terminology overlap.

### Dense Retrieval

Dense embeddings are generated using:


intfloat/multilingual-e5-base


This allows semantic matching across English and Arabic queries.

### Hybrid Retrieval

Hybrid retrieval combines BM25 and dense retrieval scores.

Current configuration:

''yaml
bm25_weight: 0.7
dense_weight: 0.3
Reranking

Retrieved candidates are reranked using:

BAAI/bge-reranker-v2-m3

The reranker improves the ordering of retrieved passages before sending context to the LLM.

Evaluation
Dataset

Current benchmark:

10 PDF documents
20 manually created questions
English and Arabic queries

Metrics:

Recall@1
Recall@3
Recall@5
Mean Reciprocal Rank (MRR)
Retrieval Results
Without OCR

Initial evaluation using only extracted PDF text:

Method	Recall@1	Recall@3	Recall@5	MRR
BM25	0.500	0.700	0.750	0.602
Dense Retrieval	0.400	0.600	0.650	0.504
OCR-Enhanced Retrieval

Arabic scanned documents were processed using OCR before indexing.

Method	Recall@1	Recall@3	Recall@5	MRR
BM25	0.650	0.850	0.900	0.752
Dense Retrieval	0.550	0.950	1.000	0.738

OCR improved retrieval performance, especially for Arabic documents without an embedded text layer.

Hybrid Retrieval

Different BM25/dense weighting configurations were tested.

BM25 Weight	Dense Weight	Recall@1	Recall@3	Recall@5	MRR
0.5	0.5	0.600	0.950	1.000	0.785
0.3	0.7	0.600	1.000	1.000	0.792
0.7	0.3	0.700	0.950	0.950	0.808

The best hybrid configuration on this benchmark used a higher BM25 contribution.

Hybrid Retrieval + Reranking

A cross-encoder reranker was added after retrieving the top candidates.

Pipeline:

Hybrid Retrieval
(top 10 candidates)

        ↓

Cross Encoder Reranking

        ↓

Final top 5 results

Results:

Method	Recall@1	Recall@3	Recall@5	MRR
Hybrid	0.700	0.950	0.950	0.808
Hybrid + Reranker	0.850	1.000	1.000	0.925

The reranker improved the ranking of relevant passages, especially when multiple documents contained similar terminology.

Project Structure
AskAway/
│
├── configs/
│   └── default.yaml
│
├── src/
│   └── askaway/
│       │
│       ├── ingestion/
│       │   ├── parser.py
│       │   ├── ocr.py
│       │   └── chunker.py
│       │
│       ├── retrieval/
│       │   ├── bm25.py
│       │   ├── dense.py
│       │   ├── hybrid.py
│       │   ├── reranker.py
│       │   └── hybrid_reranker.py
│       │
│       ├── generation/
│       │   ├── llm.py
│       │   └── openai_llm.py
│       │
│       └── rag.py
│
├── tests/
│
├── results/
│   └── retrieval_results.md
│
└── pyproject.toml
Configuration

Model and retrieval parameters are stored separately from the code.

Example:

retrieval:
  bm25_weight: 0.7
  top_k: 5
  candidate_k: 10

models:
  embedding: intfloat/multilingual-e5-base
  reranker: BAAI/bge-reranker-v2-m3

This makes it easier to experiment with different retrieval settings.

Running the Project

Install dependencies:

pip install -e .

Run ingestion:

python -m askaway.cli.ingest <input_folder> <output_file>

Search documents:

python -m askaway.cli.search <corpus> "<query>"

Generate an answer:

python -m askaway.cli.answer <corpus> "<question>"


Future Improvements:
Expand evaluation benchmark to a larger multilingual document collection
Add more OCR-heavy Arabic documents
Add latency measurements
Add a web interface for document upload and question answering
Support additional LLM backends
