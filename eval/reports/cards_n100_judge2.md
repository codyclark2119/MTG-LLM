# Section 9 Evaluation Report (recalibrated judge)

100 questions, re-scored from `cards_n100.jsonl` with the v2 judge: correctness and citation scored separately, length/style explicitly excluded, candidates anonymized behind randomized A/B/C/D labels.

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 3.68 (n=95) | 3.72 | 1275 |
| base | 3.60 (n=95) | 3.62 | 1249 |
| base_rag_cards | 3.79 (n=95) | 3.69 | 1290 |
| finetuned_rag | 3.07 (n=95) | 2.73 | 884 |
| finetuned | 3.05 (n=95) | 2.60 | 789 |
| finetuned_rag_cards | 3.15 (n=95) | 2.89 | 897 |

Correlation(answer length, correctness): **r = +0.042** (v1 judge measured r = +0.21 against its single blended score).

