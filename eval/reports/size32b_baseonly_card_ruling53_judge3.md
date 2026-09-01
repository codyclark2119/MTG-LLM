# Section 9 Evaluation Report (recalibrated judge)

53 questions, re-scored from `size32b_baseonly_card_ruling53.jsonl` with the **V3 rubric judge**: the judge reports which enumerated key points and which common errors each answer made, and the score is computed in Python from those counts.

The judge PROMPT is identical to the first pass; only the judge MODEL differs. That is what Section 9.9 requires — vary the judge and nothing else.

- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

- scoring: `points_only` — correctness is `points_hit / n_points`; `errors_made` is reported but does not move the score (Section 21.28)

- **unverifiable claims discarded: 0** across 0/159 arm-answers. Each was a key point or common error the judge asserted and then could not quote from the candidate it was grading (V4, Section 21.7).

- **every listed error fired at once: 7/46 blunder calls (15%)**, of which 1 also credit the answer with half the key points or more — two claims that cannot both hold. `common_errors` are alternative wrong answers; committing all of them is usually not something an answer can do, and blunder rate is defined on this field (Section 21.26).

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 3.41 (n=53) | 2.81 | 1153 |
| base | 3.70 (n=53) | 4.13 | 1437 |
| base_rag_cards_rulings | 4.19 (n=53) | 3.15 | 1178 |

Correlation(answer length, correctness): **r = +0.115** (v1 judge measured r = +0.21 against its single blended score).

