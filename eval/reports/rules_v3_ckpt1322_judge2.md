# Section 9 Evaluation Report (recalibrated judge)

110 questions, re-scored from `rules_v3_ckpt1322.jsonl` with the v2 judge: correctness and citation scored separately, length/style explicitly excluded, candidates anonymized behind randomized A/B/C/D labels.

- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 3.94 (n=109) | 3.87 | 1043 |
| base | 2.93 (n=109) | 3.01 | 1224 |
| finetuned_rag | 3.32 (n=109) | 3.27 | 512 |
| finetuned | 2.20 (n=109) | 2.15 | 446 |

Correlation(answer length, correctness): **r = +0.106** (v1 judge measured r = +0.21 against its single blended score).

