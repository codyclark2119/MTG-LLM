# Inter-judge agreement: `segtest.jsonl` vs `segtest_judge2.jsonl`

24 questions x 4 arms, identical stored answers.
Segmented by who wrote the rubric — the variable Section 14.6 found dominates agreement.

| Rubric source | Questions | Pairs | Pearson r | Exact | Mean gap | Same points_hit |
| --- | --- | --- | --- | --- | --- | --- |
| **ALL** | 22 | 88 | +0.49 | 40% | 0.98 | 43% |
| Cody Clark | 10 | 40 | +0.57 | 42% | 0.97 | 48% |
| Steve Steve | 11 | 44 | +0.40 | 39% | 0.94 | 36% |
| judge:CC | 1 | 4 | +0.58 | 25% | 1.50 | 75% |

Reference: Section 14.6 measured r **+0.30** machine-drafted vs **+0.62** hand-authored on 20 questions.

> At least one segment is under 25 questions. Section 14.6 needed 20 to separate hand from machine, an effect far larger than the difference between two careful authors — read a split here as a prompt to look at specific records, not as a measured difference.

## Questions the judges disagree on most

Rewrite these rubrics before adding more (Section 16.12: disagreement localizes, so a handful of items carries most of it).

| Gold id | Arm | Judge A | Judge B | Gap | Rubric |
| --- | --- | --- | --- | --- | --- |
| `qa-adaline-casts-seasoned-pyromancer-and-ha` | base | 1.0 | 5.0 | 4.0 | Cody Clark |
| `rg-86` | base_rag | 1.0 | 5.0 | 4.0 | Cody Clark |
| `rg-104` | finetuned_rag | 1.5 | 5.0 | 3.5 | Steve Steve |
| `rg-204` | base_rag | 1.5 | 5.0 | 3.5 | Steve Steve |
| `qa-alex-casts-thoughtseize-targeting-neil-i` | base_rag | 2.0 | 5.0 | 3.0 | Cody Clark |
| `qa-alex-casts-thoughtseize-targeting-neil-i` | base | 2.0 | 5.0 | 3.0 | Cody Clark |
| `qa-anson-activates-karn-the-great-creator-t` | finetuned | 2.0 | 5.0 | 3.0 | Cody Clark |
| `rg-1555` | base | 5.0 | 2.0 | 3.0 | Steve Steve |
| `rg-86` | base | 2.0 | 5.0 | 3.0 | Cody Clark |
| `rg-1208` | base_rag | 2.5 | 5.0 | 2.5 | Steve Steve |
| `rg-1208` | base | 2.5 | 5.0 | 2.5 | Steve Steve |
| `qa-amy-casts-assassin-s-trophy-targeting-ni` | base | 3.0 | 5.0 | 2.0 | judge:CC |
