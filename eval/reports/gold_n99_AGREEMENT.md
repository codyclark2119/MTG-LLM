# Inter-judge agreement: `gold_n99.jsonl` vs `gold_n99_judge2.jsonl`

99 questions x 4 arms, identical stored answers.
Segmented by who wrote the rubric — the variable Section 14.6 found dominates agreement.

| Rubric source | Questions | Pairs | Pearson r | Exact | Mean gap | Same points_hit |
| --- | --- | --- | --- | --- | --- | --- |
| **ALL** | 92 | 367 | +0.49 | 30% | 0.99 | 32% |
| Cody Clark | 71 | 283 | +0.51 | 33% | 0.98 | 35% |
| Steve Steve | 21 | 84 | +0.40 | 23% | 1.01 | 20% |

Reference: Section 14.6 measured r **+0.30** machine-drafted vs **+0.62** hand-authored on 20 questions.

> At least one segment is under 25 questions. Section 14.6 needed 20 to separate hand from machine, an effect far larger than the difference between two careful authors — read a split here as a prompt to look at specific records, not as a measured difference.

## Questions the judges disagree on most

Rewrite these rubrics before adding more (Section 16.12: disagreement localizes, so a handful of items carries most of it).

| Gold id | Arm | Judge A | Judge B | Gap | Rubric |
| --- | --- | --- | --- | --- | --- |
| `qa-alex-casts-thoughtseize-targeting-neil-i` | base_rag | 1.0 | 5.0 | 4.0 | Cody Clark |
| `rg-308` | finetuned | 1.0 | 5.0 | 4.0 | Steve Steve |
| `rg-47` | base_rag | 1.0 | 5.0 | 4.0 | Cody Clark |
| `gloss-choose-a-background` | base_rag | 1.5 | 5.0 | 3.5 | Cody Clark |
| `rg-1012` | finetuned_rag | 5.0 | 1.5 | 3.5 | Cody Clark |
| `rg-109` | base | 1.5 | 5.0 | 3.5 | Cody Clark |
| `rg-109` | finetuned | 1.5 | 5.0 | 3.5 | Cody Clark |
| `rg-189` | base | 5.0 | 1.5 | 3.5 | Steve Steve |
| `rg-2599` | base | 1.5 | 5.0 | 3.5 | Cody Clark |
| `rg-2599` | finetuned_rag | 1.5 | 5.0 | 3.5 | Cody Clark |
| `rg-308` | base_rag | 1.5 | 5.0 | 3.5 | Steve Steve |
| `gloss-last-known-information` | base_rag | 1.7 | 5.0 | 3.3 | Cody Clark |
