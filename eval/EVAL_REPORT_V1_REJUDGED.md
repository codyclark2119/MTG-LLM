# Section 9 Evaluation Report (recalibrated judge)

110 questions, re-scored from `eval_results.jsonl` with the v2 judge: correctness and citation scored separately, length/style explicitly excluded, candidates anonymized behind randomized A/B/C/D labels.

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base | 3.20 (n=110) | 3.78 | 1224 |
| base_rag | 3.87 (n=110) | 4.47 | 1043 |
| finetuned | 2.01 (n=110) | 3.04 | 270 |
| finetuned_rag | 2.53 (n=110) | 3.54 | 295 |

Correlation(answer length, correctness): **r = +0.270** (v1 judge measured r = +0.21 against its single blended score).

