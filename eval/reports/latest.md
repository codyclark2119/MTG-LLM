# Section 9 Evaluation Report (recalibrated judge)

2 questions, re-scored from `tiny_run3.jsonl` with the **V3 rubric judge**: the judge reports which enumerated key points and which common errors each answer made, and the score is computed in Python from those counts.

The judge PROMPT is identical to the first pass; only the judge MODEL differs. That is what Section 9.9 requires — vary the judge and nothing else.

- judge: `mlx-community/Qwen2.5-7B-Instruct-4bit`

- scoring: `points_only` — correctness is `points_hit / n_points`; `errors_made` is reported but does not move the score (Section 21.28)

- **unverifiable claims discarded: 0** across 0/8 arm-answers. Each was a key point or common error the judge asserted and then could not quote from the candidate it was grading (V4, Section 21.7).

- **every listed error fired at once: 3/8 blunder calls (38%)**, of which 1 also credit the answer with half the key points or more — two claims that cannot both hold. `common_errors` are alternative wrong answers; committing all of them is usually not something an answer can do, and blunder rate is defined on this field (Section 21.26).

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 2.00 (n=2) | 3.00 | 753 |
| base | 3.00 (n=2) | 4.00 | 1192 |
| finetuned_rag | 4.00 (n=2) | 3.00 | 566 |
| finetuned | 2.00 (n=2) | 3.00 | 610 |

Correlation(answer length, correctness): **r = -0.113** (v1 judge measured r = +0.21 against its single blended score).

