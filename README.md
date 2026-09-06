# AskAway

AskAway is a multilingual Retrieval-Augmented Generation (RAG) system for answering questions over PDF documents.

The project focuses on building a complete document question-answering pipeline, including document ingestion, OCR processing, retrieval, reranking, LLM-based answer generation, and API deployment.

The system supports both English and Arabic documents, including scanned PDFs that require OCR before they can be searched.

---

# Overview

Traditional keyword search can struggle when users ask questions using different wording than the original document.

AskAway combines multiple retrieval approaches:

- BM25 lexical retrieval for exact term matching
- Dense retrieval for semantic similarity
- Hybrid retrieval combining both methods
- Cross-encoder reranking to improve final result ordering
- LLM generation using retrieved document context

The goal is to retrieve relevant document passages first, then generate answers grounded only on the retrieved information.

---

# System Pipeline

```text
PDF Documents
      |
      v
Document Processing
(OCR + Text Extraction)
      |
      v
Text Chunking
      |
      v
Retrieval
(BM25 + Dense + Hybrid)
      |
      v
Cross Encoder Reranking
      |
      v
LLM Generation
      |
      v
Answer + Sources
```

---

# Features

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
- REST API interface
- Docker-based deployment

---

# Retrieval Models

## Sparse Retrieval

BM25 is used to capture exact keyword matches and terminology overlap.

## Dense Retrieval

Dense embeddings are generated using:

```text
intfloat/multilingual-e5-base
```

This allows semantic matching across English and Arabic queries.

## Hybrid Retrieval

Hybrid retrieval combines BM25 and dense retrieval scores.

The best performing configuration on the benchmark used:

```yaml
retrieval:
  bm25_weight: 0.7
  top_k: 5
  candidate_k: 10
```

## Reranking

Retrieved candidates are reranked using:

```text
BAAI/bge-reranker-v2-m3
```

The reranker improves the ordering of retrieved passages before sending context to the LLM.

---

# API Service

AskAway includes a FastAPI service that exposes the RAG pipeline through an HTTP interface.

The API separates the retrieval and generation logic from the user interface layer, allowing the system to be accessed by external applications.

Example workflow:

```text
Client Application
        |
        v
FastAPI Endpoint
        |
        v
RAG Pipeline
        |
        |
+-------+-------+
|               |
Retrieval    LLM Generation
|
v
Answer + Sources
```

Example request:

```json
{
  "question": "What documents are required for this service?"
}
```

Example response:

```json
{
  "answer": "The required documents are...",
  "sources": [
    {
      "filename": "document.pdf",
      "page_number": 12
    }
  ]
}
```

---

# Docker Deployment

AskAway can be containerized using Docker to provide a reproducible environment.

The Docker image includes:

- Python environment
- Project dependencies
- Retrieval components
- API service configuration

Build the image:

```bash
docker build -t askaway .
```

Run the container:

```bash
docker run -p 8000:8000 askaway
```

The API becomes available at:

```text
http://localhost:8000
```

---

# Evaluation

## Dataset

Current benchmark:

- 10 PDF documents
- 20 manually created questions
- English and Arabic queries

Metrics:

- Recall@1
- Recall@3
- Recall@5
- Mean Reciprocal Rank (MRR)

---

# Retrieval Results

## Initial Retrieval

Evaluation using extracted PDF text:

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.500 | 0.700 | 0.750 | 0.602 |
| Dense Retrieval | 0.400 | 0.600 | 0.650 | 0.504 |

---

## OCR-Enhanced Retrieval

Arabic scanned documents were processed using OCR before indexing.

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.650 | 0.850 | 0.900 | 0.752 |
| Dense Retrieval | 0.550 | 0.950 | 1.000 | 0.738 |

OCR improved retrieval performance on documents where normal PDF extraction was insufficient.

---

## Hybrid Retrieval

Different BM25/dense weighting configurations were tested.

| BM25 Weight | Dense Weight | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---:|---:|---:|---:|
| 0.5 | 0.5 | 0.600 | 0.950 | 1.000 | 0.785 |
| 0.3 | 0.7 | 0.600 | 1.000 | 1.000 | 0.792 |
| 0.7 | 0.3 | 0.700 | 0.950 | 0.950 | 0.808 |

---

## Hybrid Retrieval + Reranking

Pipeline:

```text
Hybrid Retrieval
(top 10 candidates)

        |
        v

Cross Encoder Reranking

        |
        v

Final top 5 results
```

Results:

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Hybrid | 0.700 | 0.950 | 0.950 | 0.808 |
| Hybrid + Reranker | 0.850 | 1.000 | 1.000 | 0.925 |

The reranker improved the ranking of relevant passages, especially when multiple documents contained similar terminology.

---

# Project Structure

```text
AskAway/
│
├── configs/
│   └── default.yaml
│
├── src/
│   └── askaway/
│       │
│       ├── api/
│       │   └── app.py
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
├── results/
├── Dockerfile
└── pyproject.toml
```

---

# Configuration

Model and retrieval parameters are stored separately from the code.

Example:

```yaml
retrieval:
  bm25_weight: 0.7
  top_k: 5
  candidate_k: 10

models:
  embedding: intfloat/multilingual-e5-base
  reranker: BAAI/bge-reranker-v2-m3
```

This makes it easier to experiment with different retrieval settings.

---

# Running the Project

Install dependencies:

```bash
pip install -e .
```

Run ingestion:

```bash
python -m askaway.cli.ingest <input_folder> <output_file>
```

Search documents:

```bash
python -m askaway.cli.search <corpus> "<query>"
```

Generate an answer:

```bash
python -m askaway.cli.answer <corpus> "<question>"
```

Run API:

```bash
uvicorn askaway.api.app:app --host 0.0.0.0 --port 8000
```

---

# Future Improvements

- Expand evaluation benchmark to a larger multilingual document collection
- Add more OCR-heavy Arabic documents
- Add latency measurements
- Add document upload endpoint
- Add web interface
- Support additional LLM backends


This project is for educational and research purposes.
