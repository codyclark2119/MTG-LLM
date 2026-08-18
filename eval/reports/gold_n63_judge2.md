# Section 9 Evaluation Report (recalibrated judge)

63 questions, re-scored from `gold_n63.jsonl` with the v2 judge: correctness and citation scored separately, length/style explicitly excluded, candidates anonymized behind randomized A/B/C/D labels.

- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 3.39 (n=60) | 4.63 | 1075 |
| base | 3.22 (n=60) | 4.60 | 1150 |
| finetuned_rag | 2.30 (n=59) | 3.69 | 492 |
| finetuned | 2.11 (n=60) | 3.12 | 505 |

Correlation(answer length, correctness): **r = +0.347** (v1 judge measured r = +0.21 against its single blended score).

