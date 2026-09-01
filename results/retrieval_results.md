# Retrieval Evaluation

## Dataset

The retrieval benchmark consists of:

- 10 PDF documents
- 20 manually created benchmark questions
- English and Arabic queries

Evaluation metrics:

- Recall@1
- Recall@3
- Recall@5
- Mean Reciprocal Rank (MRR)

---

# Baseline Retrieval

## Without OCR

Initial evaluation was performed using only machine-readable PDF text extraction.

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.500 | 0.700 | 0.750 | 0.602 |
| Dense Retrieval | 0.400 | 0.600 | 0.650 | 0.504 |

---

# OCR-Enhanced Retrieval

Several Arabic documents were scanned PDFs without an embedded text layer. An OCR pipeline was added to extract searchable text from these documents.

The OCR-enhanced corpus improved retrieval performance:

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.650 | 0.850 | 0.900 | 0.752 |
| Dense Retrieval | 0.550 | 0.950 | 1.000 | 0.738 |

The improvement was mainly observed on Arabic documents where the original PDFs contained little or no extractable text.

---

# Hybrid Retrieval

Hybrid retrieval combines:

- BM25 lexical retrieval
- Multilingual dense retrieval

Different weighting configurations were tested.

| BM25 Weight | Dense Weight | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---:|---:|---:|---:|
| 0.5 | 0.5 | 0.600 | 0.950 | 1.000 | 0.785 |
| 0.3 | 0.7 | 0.600 | 1.000 | 1.000 | 0.792 |
| 0.7 | 0.3 | 0.700 | 0.950 | 0.950 | 0.808 |

The highest MRR on this benchmark was achieved using a 70% BM25 / 30% dense retrieval weighting.

---

# Error Analysis

One example failure case:

**Question**


لأي غرض تُصدر الشهادة الصحية للوافد؟


The correct document page was retrieved, but it was ranked lower than other pages containing similar administrative terminology.

The retrieved results contained several references to health certificates, laboratory procedures, and related services. This caused ranking confusion between general procedural pages and the page containing the actual purpose of issuing the certificate.

This suggests that a reranking stage could improve final result ordering by scoring retrieved passages against the specific query meaning.



---

# Hybrid Retrieval + Reranking

A cross-encoder reranker was added after hybrid retrieval.

Pipeline:

Hybrid retrieval
(top 10 candidates)

↓

Cross-encoder reranking

↓

Final top 5 results

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Hybrid | 0.700 | 0.950 | 0.950 | 0.808 |
| Hybrid + Reranker | 0.850 | 1.000 | 1.000 | 0.925 |

The reranker improved the ordering of retrieved passages, increasing the number of questions where the correct passage appeared as the first result.


# Hybrid Retrieval + Reranking

A cross-encoder reranker was added after hybrid retrieval.

Pipeline:

Hybrid retrieval
(top 10 candidates)

↓

Cross-encoder reranking

↓

Final top 5 results

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Hybrid | 0.700 | 0.950 | 0.950 | 0.808 |
| Hybrid + Reranker | 0.850 | 1.000 | 1.000 | 0.925 |