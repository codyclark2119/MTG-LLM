# Section 9 Evaluation Report (recalibrated judge)

110 questions, re-scored from `rules_v2_rejudged.jsonl` with the v2 judge: correctness and citation scored separately, length/style explicitly excluded, candidates anonymized behind randomized A/B/C/D labels.

- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base | 2.94 (n=109) | 3.17 | 1224 |
| base_rag | 3.71 (n=109) | 3.72 | 1043 |
| finetuned | 2.06 (n=109) | 1.74 | 450 |
| finetuned_rag | 3.29 (n=109) | 3.06 | 503 |

Correlation(answer length, correctness): **r = +0.075** (v1 judge measured r = +0.21 against its single blended score).

