# Inter-judge agreement: `cards_rulings_n99.jsonl` vs `cards_rulings_n99_mistral.jsonl`

- cards_rulings_n99: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- cards_rulings_n99_mistral: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

99 questions x 6 arms, identical stored answers.
Segmented by who wrote the rubric — the variable Section 14.6 found dominates agreement.

| Rubric source | Questions | Pairs | Pearson r | Exact | Mean gap | Same points_hit |
| --- | --- | --- | --- | --- | --- | --- |
| **ALL** | 99 | 594 | +0.21 | 39% | 1.54 | 37% |
| Cody Clark | 74 | 444 | +0.24 | 43% | 1.43 | 41% |
| Steve Steve | 24 | 144 | +0.16 | 26% | 1.91 | 24% |
| judge:CC | 1 | 6 | +0.24 | 33% | 1.33 | 33% |

Reference: Section 14.6 measured r **+0.30** machine-drafted vs **+0.62** hand-authored on 20 questions.

> At least one segment is under 25 questions. Section 14.6 needed 20 to separate hand from machine, an effect far larger than the difference between two careful authors — read a split here as a prompt to look at specific records, not as a measured difference.

## Questions the judges disagree on most

Worth reading to understand *how* the judges differ — but not a repair list. Section 21.20 measured disagreement as diffuse rather than localized (Gini 0.45–0.48 across rubric items; 91 of 99 records carry at least one dispute), so rewriting the rows below would not move kappa. The lever is the judge, not this table.

| Gold id | Arm | Judge A | Judge B | Gap | Rubric |
| --- | --- | --- | --- | --- | --- |
| `qa-alice-controls-ral-monsoon-mage-and-cast` | base_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-alice-controls-ral-monsoon-mage-and-cast` | base | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-alice-controls-ral-monsoon-mage-and-cast` | base_rag_cards_rulings | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-alice-controls-ral-monsoon-mage-and-cast` | finetuned_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-alice-controls-ral-monsoon-mage-and-cast` | finetuned_rag_cards_rulings | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-anton-casts-xenagos-god-of-revels-can-na` | base_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-anton-casts-xenagos-god-of-revels-can-na` | base_rag_cards_rulings | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-armando-controls-leyline-of-the-guildpac` | base_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-armando-controls-leyline-of-the-guildpac` | base | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-armando-controls-leyline-of-the-guildpac` | base_rag_cards_rulings | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-armando-controls-leyline-of-the-guildpac` | finetuned_rag | 5.0 | 1.0 | 4.0 | Cody Clark |
| `qa-armando-controls-leyline-of-the-guildpac` | finetuned | 5.0 | 1.0 | 4.0 | Cody Clark |
