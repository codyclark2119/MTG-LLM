# Inter-judge agreement: `gold_n99_judge2.jsonl` vs `gold_n99_judge3.jsonl`

99 questions x 4 arms, identical stored answers.
Segmented by who wrote the rubric — the variable Section 14.6 found dominates agreement.

| Rubric source | Questions | Pairs | Pearson r | Exact | Mean gap | Same points_hit |
| --- | --- | --- | --- | --- | --- | --- |
| **ALL** | 92 | 367 | +0.22 | 20% | 1.53 | 19% |
| Cody Clark | 71 | 283 | +0.24 | 22% | 1.53 | 19% |
| Steve Steve | 21 | 84 | +0.08 | 15% | 1.52 | 17% |

Reference: Section 14.6 measured r **+0.30** machine-drafted vs **+0.62** hand-authored on 20 questions.

> At least one segment is under 25 questions. Section 14.6 needed 20 to separate hand from machine, an effect far larger than the difference between two careful authors — read a split here as a prompt to look at specific records, not as a measured difference.

## Questions the judges disagree on most

Worth reading to understand *how* the judges differ — but not a repair list. Section 21.20 measured disagreement as diffuse rather than localized (Gini 0.45–0.48 across rubric items; 91 of 99 records carry at least one dispute), so rewriting the rows below would not move kappa. The lever is the judge, not this table.

| Gold id | Arm | Judge A | Judge B | Gap | Rubric |
| --- | --- | --- | --- | --- | --- |
| `gloss-choose-a-background` | base_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `gloss-last-known-information` | base_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `gloss-summoning-sickness-rule` | finetuned | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-alden-controls-elesh-norn-mother-of-mach` | base_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-alex-casts-thoughtseize-targeting-neil-i` | base_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-anton-casts-xenagos-god-of-revels-can-na` | base_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-anton-casts-xenagos-god-of-revels-can-na` | base | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-armando-controls-leyline-of-the-guildpac` | base | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-ashley-controls-karn-liberated-and-exile` | base | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-avery-controls-magus-of-the-moon-enchant` | base_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-avery-controls-magus-of-the-moon-enchant` | base | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-avery-controls-magus-of-the-moon-enchant` | finetuned | 5.0 | 1.0 | 4.0 | Cody Clark |
