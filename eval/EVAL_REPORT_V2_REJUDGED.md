# Section 9 Evaluation Report (recalibrated judge)

110 questions, re-scored from `eval_results_v2.jsonl` with the v2 judge: correctness and citation scored separately, length/style explicitly excluded, candidates anonymized behind randomized A/B/C/D labels.

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base | 3.04 (n=110) | 3.57 | 1224 |
| base_rag | 3.75 (n=110) | 4.37 | 1043 |
| finetuned | 2.37 (n=110) | 3.11 | 450 |
| finetuned_rag | 3.22 (n=110) | 3.95 | 503 |

Correlation(answer length, correctness): **r = -0.005** (v1 judge measured r = +0.21 against its single blended score).

