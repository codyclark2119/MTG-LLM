# Section 9 Evaluation Report (recalibrated judge)

99 questions, re-scored from `gold_n99.jsonl` with the **V3 rubric judge**: the judge reports which enumerated key points and which common errors each answer made, and the score is computed in Python from those counts.

The judge PROMPT is identical to the first pass; only the judge MODEL differs. That is what Section 9.9 requires — vary the judge and nothing else.

- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 3.16 (n=92) | 4.39 | 1104 |
| base | 3.09 (n=92) | 4.43 | 1163 |
| finetuned_rag | 2.36 (n=91) | 3.56 | 521 |
| finetuned | 2.18 (n=92) | 3.23 | 500 |

Correlation(answer length, correctness): **r = +0.280** (v1 judge measured r = +0.21 against its single blended score).

