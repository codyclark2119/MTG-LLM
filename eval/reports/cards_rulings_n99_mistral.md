# Section 9 Evaluation Report (recalibrated judge)

99 questions, re-scored from `cards_rulings_n99.jsonl` with the **V3 rubric judge**: the judge reports which enumerated key points and which common errors each answer made, and the score is computed in Python from those counts.

The judge PROMPT is identical to the first pass; only the judge MODEL differs. That is what Section 9.9 requires — vary the judge and nothing else.

- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

- scoring: `points_only` — correctness is `points_hit / n_points`; `errors_made` is reported but does not move the score (Section 21.28)

- **unverifiable claims discarded: 0** across 0/594 arm-answers. Each was a key point or common error the judge asserted and then could not quote from the candidate it was grading (V4, Section 21.7).

- **every listed error fired at once: 27/259 blunder calls (10%)**, of which 1 also credit the answer with half the key points or more — two claims that cannot both hold. `common_errors` are alternative wrong answers; committing all of them is usually not something an answer can do, and blunder rate is defined on this field (Section 21.26).

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 1.65 (n=99) | 2.35 | 1104 |
| base | 1.63 (n=99) | 1.87 | 1163 |
| base_rag_cards_rulings | 1.68 (n=99) | 2.39 | 1104 |
| finetuned_rag | 1.31 (n=99) | 1.85 | 521 |
| finetuned | 1.23 (n=99) | 1.75 | 500 |
| finetuned_rag_cards_rulings | 1.30 (n=99) | 1.91 | 521 |

Correlation(answer length, correctness): **r = +0.053** (v1 judge measured r = +0.21 against its single blended score).

