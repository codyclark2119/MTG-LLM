# Inter-judge agreement: `gold_n99.jsonl` vs `gold_n99_judge3.jsonl`

99 questions x 4 arms, identical stored answers.
Segmented by who wrote the rubric — the variable Section 14.6 found dominates agreement.

| Rubric source | Questions | Pairs | Pearson r | Exact | Mean gap | Same points_hit |
| --- | --- | --- | --- | --- | --- | --- |
| **ALL** | 99 | 396 | +0.20 | 31% | 1.07 | 32% |
| Cody Clark | 74 | 296 | +0.23 | 32% | 1.06 | 34% |
| Steve Steve | 24 | 96 | +0.11 | 24% | 1.12 | 25% |
| judge:CC | 1 | 4 | +nan | 50% | 1.00 | 50% |

Reference: Section 14.6 measured r **+0.30** machine-drafted vs **+0.62** hand-authored on 20 questions.

> At least one segment is under 25 questions. Section 14.6 needed 20 to separate hand from machine, an effect far larger than the difference between two careful authors — read a split here as a prompt to look at specific records, not as a measured difference.

## Questions the judges disagree on most

Worth reading to understand *how* the judges differ — but not a repair list. Section 21.20 measured disagreement as diffuse rather than localized (Gini 0.45–0.48 across rubric items; 91 of 99 records carry at least one dispute), so rewriting the rows below would not move kappa. The lever is the judge, not this table.

| Gold id | Arm | Judge A | Judge B | Gap | Rubric |
| --- | --- | --- | --- | --- | --- |
| `qa-alex-casts-thoughtseize-targeting-neil-i` | base | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-aubree-has-murderous-rider-in-their-grav` | base_rag | 1.0 | 5.0 | 4.0 | Cody Clark |
| `qa-avery-controls-magus-of-the-moon-enchant` | base_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-avery-controls-magus-of-the-moon-enchant` | base | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-avery-controls-magus-of-the-moon-enchant` | finetuned | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-in-a-game-of-two-headed-giant-both-adaly` | base | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-nico-controls-striped-riverwinder-annika` | base | 5.0 | 1.0 | 4.0 | Cody Clark |
| `rg-1012` | base_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `rg-1012` | finetuned_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `rg-1060` | base | 5.0 | 1.0 | 4.0 | Cody Clark |
| `rg-1124` | finetuned_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `rg-1124` | finetuned | 5.0 | 1.0 | 4.0 | Cody Clark |
