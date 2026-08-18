# Section 9 Evaluation Report (recalibrated judge)

24 questions, re-scored from `segtest.jsonl` with the v2 judge: correctness and citation scored separately, length/style explicitly excluded, candidates anonymized behind randomized A/B/C/D labels.

- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 2.99 (n=22) | 4.36 | 1153 |
| base | 3.78 (n=22) | 4.64 | 1177 |
| finetuned_rag | 2.09 (n=22) | 3.36 | 586 |
| finetuned | 1.95 (n=22) | 2.91 | 571 |

Correlation(answer length, correctness): **r = +0.483** (v1 judge measured r = +0.21 against its single blended score).

