# Section 9 Evaluation Report (recalibrated judge)

99 questions, re-scored from `gold_n99.jsonl` with the **V3 rubric judge**: the judge reports which enumerated key points and which common errors each answer made, and the score is computed in Python from those counts.

The judge PROMPT is identical to the first pass; only the judge MODEL differs. That is what Section 9.9 requires — vary the judge and nothing else.

- judge: `mlx-community/Qwen2.5-32B-Instruct-4bit`

- **unverifiable claims discarded: 0** across 0/396 arm-answers. Each was a key point or common error the judge asserted and then could not quote from the candidate it was grading (V4, Section 21.7).

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 1.60 (n=99) | 2.25 | 1104 |
| base | 1.40 (n=99) | 2.01 | 1163 |
| finetuned_rag | 1.23 (n=99) | 2.25 | 521 |
| finetuned | 1.09 (n=99) | 2.03 | 500 |

Correlation(answer length, correctness): **r = +0.080** (v1 judge measured r = +0.21 against its single blended score).

