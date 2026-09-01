# Retrieval Evaluation

Dataset:
- 10 PDFs
- 20 benchmark questions
- English + Arabic

## Without OCR

| Method | R@1 | R@3 | R@5 | MRR |
|---     |---  |---  |---  |---  |
| BM25 | 0.500 | 0.700 | 0.750 | 0.602 |
| Dense | 0.400 | 0.600 | 0.650 | 0.504 |

## With OCR

| Method | R@1 | R@3 | R@5 | MRR |
|---     |---  |---  |---  |---  |
| BM25 | 0.650 | 0.850 | 0.900 | 0.752 |
| Dense | 0.550 | 0.950 | 1.000 | 0.738 |



## Hybrid retrieval results 50/50
-----------------------
Questions: 20
Recall@1: 0.600
Recall@3: 0.950
Recall@5: 1.000
MRR:      0.785


## Hybrid retrieval results 30(bm25)/70
-----------------------
Questions: 20
Recall@1: 0.600
Recall@3: 1.000
Recall@5: 1.000
MRR:      0.792


## Hybrid retrieval results 70(bm25)/30
-----------------------
Questions: 20
Recall@1: 0.700
Recall@3: 0.950
Recall@5: 0.950
MRR:      0.808

Missed at k=5
-------------
ar_health_001: لأي غرض تُصدر الشهادة الصحية للوافد؟