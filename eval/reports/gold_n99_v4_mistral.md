# Section 9 Evaluation Report (recalibrated judge)

99 questions, re-scored from `gold_n99_v4_32b.jsonl` with the **V3 rubric judge**: the judge reports which enumerated key points and which common errors each answer made, and the score is computed in Python from those counts.

The judge PROMPT is identical to the first pass; only the judge MODEL differs. That is what Section 9.9 requires — vary the judge and nothing else.

- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

- scoring: `points_only` — correctness is `points_hit / n_points`; `errors_made` is reported but does not move the score (Section 21.28)

- **unverifiable claims discarded: 0** across 0/396 arm-answers. Each was a key point or common error the judge asserted and then could not quote from the candidate it was grading (V4, Section 21.7).

- **every listed error fired at once: 29/190 blunder calls (15%)**, of which 1 also credit the answer with half the key points or more — two claims that cannot both hold. `common_errors` are alternative wrong answers; committing all of them is usually not something an answer can do, and blunder rate is defined on this field (Section 21.26).

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 1.89 (n=99) | 2.39 | 1104 |
| base | 1.66 (n=99) | 1.85 | 1163 |
| finetuned_rag | 1.34 (n=99) | 2.01 | 326 |
| finetuned | 1.57 (n=99) | 2.29 | 252 |

Correlation(answer length, correctness): **r = +0.052** (v1 judge measured r = +0.21 against its single blended score).

