# Section 9 Evaluation Report (recalibrated judge)

45 questions, re-scored from `pilot_v3.jsonl` with the v2 judge: correctness and citation scored separately, length/style explicitly excluded, candidates anonymized behind randomized A/B/C/D labels.

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 3.90 (n=41) | 4.51 | 1207 |
| base | 3.90 (n=41) | 4.76 | 1213 |
| finetuned_rag | 2.55 (n=41) | 3.20 | 377 |
| finetuned | 2.71 (n=41) | 2.95 | 303 |

Correlation(answer length, correctness): **r = +0.439** (v1 judge measured r = +0.21 against its single blended score).

