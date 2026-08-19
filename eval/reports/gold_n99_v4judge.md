# Section 9 Evaluation Report (recalibrated judge)

99 questions, re-scored from `gold_n99_v4judge.jsonl` with the **V4 rubric judge**: the judge reports which enumerated key points and which common errors each answer made, and the score is computed in Python from those counts, and every claim must be backed by a verbatim quote from the candidate or it is discarded.

> **This rescore changed the judge PROMPT (V3 -> V4), not only the judge model.** Section 9.9 asks for one variable at a time, so this is not a two-judge agreement measurement and must not be read against a V3 run as though it were.

- judge: `mlx-community/Qwen2.5-7B-Instruct-4bit`

- **unverifiable claims discarded: 26** across 16/56 arm-answers. Each was a key point or common error the judge asserted and then could not quote from the candidate it was grading (V4, Section 21.7).

> **340 of 396 arm-answers (86%) went unjudged** — the judge returned JSON that could not be parsed, usually by running out of output tokens. They are excluded rather than counted as wrong.

> The failures are not spread evenly: longer rubrics need longer judge output, so what remains is a subset selected by rubric size rather than by anything about the answers. **Read nothing into the means below** until the run is repeated with a larger `--judge-max-tokens`.

| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |
| --- | --- | --- | --- |
| base_rag | 2.04 (n=14) | 3.14 | 1104 |
| base | 1.68 (n=14) | 3.43 | 1163 |
| finetuned_rag | 1.25 (n=14) | 2.14 | 521 |
| finetuned | 1.04 (n=14) | 2.29 | 500 |

Correlation(answer length, correctness): **r = +0.324** (v1 judge measured r = +0.21 against its single blended score).

