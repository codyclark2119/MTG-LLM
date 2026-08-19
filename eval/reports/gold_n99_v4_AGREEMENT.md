# Inter-judge agreement: `gold_n99_v4judge.jsonl` vs `gold_n99_v4judge2.jsonl`

99 questions x 4 arms, identical stored answers.
Segmented by who wrote the rubric — the variable Section 14.6 found dominates agreement.

| Rubric source | Questions | Pairs | Pearson r | Exact | Mean gap | Same points_hit |
| --- | --- | --- | --- | --- | --- | --- |
| **ALL** | 6 | 24 | +0.46 | 38% | 1.00 | 38% |
| Cody Clark | 6 | 24 | +0.46 | 38% | 1.00 | 38% |

Reference: Section 14.6 measured r **+0.30** machine-drafted vs **+0.62** hand-authored on 20 questions.

> Only **Cody Clark** rubrics are present, so there is no contrast to read here — the row is a baseline for when another source arrives.

## Questions the judges disagree on most

Rewrite these rubrics before adding more (Section 16.12: disagreement localizes, so a handful of items carries most of it).

| Gold id | Arm | Judge A | Judge B | Gap | Rubric |
| --- | --- | --- | --- | --- | --- |
| `qa-alex-casts-solitude-braelyn-passes-prior` | finetuned | 1.0 | 5.0 | 4.0 | Cody Clark |
| `qa-alden-controls-elesh-norn-mother-of-mach` | base_rag | 3.0 | 5.0 | 2.0 | Cody Clark |
| `qa-alden-controls-elesh-norn-mother-of-mach` | base | 1.0 | 3.0 | 2.0 | Cody Clark |
| `qa-anton-casts-xenagos-god-of-revels-can-na` | base | 1.0 | 3.0 | 2.0 | Cody Clark |
| `rg-19` | base_rag | 3.0 | 5.0 | 2.0 | Cody Clark |
| `rg-47` | base | 3.0 | 1.0 | 2.0 | Cody Clark |
| `rg-49` | base | 4.0 | 2.0 | 2.0 | Cody Clark |
| `qa-alden-controls-elesh-norn-mother-of-mach` | finetuned_rag | 2.0 | 1.0 | 1.0 | Cody Clark |
| `qa-alex-casts-solitude-braelyn-passes-prior` | base_rag | 2.0 | 3.0 | 1.0 | Cody Clark |
| `qa-alex-casts-solitude-braelyn-passes-prior` | base | 1.0 | 2.0 | 1.0 | Cody Clark |
| `qa-alex-casts-solitude-braelyn-passes-prior` | finetuned_rag | 1.0 | 2.0 | 1.0 | Cody Clark |
| `qa-anton-casts-xenagos-god-of-revels-can-na` | finetuned | 1.0 | 2.0 | 1.0 | Cody Clark |
