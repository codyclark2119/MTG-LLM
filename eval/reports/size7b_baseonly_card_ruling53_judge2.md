# Section 9 Evaluation Report (recalibrated judge)

53 questions, re-scored from `size7b_baseonly_card_ruling53.jsonl` with the **V3 rubric judge**: the judge reports which enumerated key points and which common errors each answer made, and the score is computed in Python from those counts.

The judge PROMPT is identical to the first pass; only the judge MODEL differs. That is what Section 9.9 requires — vary the judge and nothing else.

- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

- scoring: `points_only` — correctness is `points_hit / n_points`; `errors_made` is reported but does not move the score (Section 21.28)

- **unverifiable claims discarded: 0** across 0/159 arm-answers. Each was a key point or common error the judge asserted and then could not quote from the candidate it was grading (V4, Section 21.7).

> **9 of 159 arm-answers (6%) went unjudged** — the judge returned JSON that could not be parsed, usually by running out of output tokens. They are excluded rather than counted as wrong.

- **every listed error fired at once: 4/81 blunder calls (5%)**, of which 1 also credit the answer with half the key points or more — two claims that cannot both hold. `common_errors` are alternative wrong answers; committing all of them is usually not something an answer can do, and blunder rate is defined on this field (Section 21.26).

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 4.41 (n=50) | 4.20 | 1398 |
| base | 4.66 (n=50) | 4.76 | 1798 |
| base_rag_cards_rulings | 4.53 (n=50) | 4.72 | 1551 |

Correlation(answer length, correctness): **r = +0.287** (v1 judge measured r = +0.21 against its single blended score).

