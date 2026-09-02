# Inter-judge agreement: `rulesonly7b_k3.jsonl` vs `rulesonly7b_k3_qwen32bjudge.jsonl`

- rulesonly7b_k3: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`
- rulesonly7b_k3_qwen32bjudge: `mlx-community/Qwen2.5-32B-Instruct-4bit`

20 questions x 2 arms, identical stored answers.
Segmented by who wrote the rubric — the variable Section 14.6 found dominates agreement.

| Rubric source | Questions | Pairs | Pearson r | Exact | Mean gap | Same points_hit |
| --- | --- | --- | --- | --- | --- | --- |
| **ALL** | 20 | 40 | +0.46 | 52% | 0.96 | 52% |
| machine-drafted | 20 | 40 | +0.46 | 52% | 0.96 | 52% |

Reference: Section 14.6 measured r **+0.30** machine-drafted vs **+0.62** hand-authored on 20 questions.

> Only **machine-drafted** rubrics are present, so there is no contrast to read here — the row is a baseline for when another source arrives.

## Questions the judges disagree on most

Worth reading to understand *how* the judges differ — but not a repair list. Section 21.20 measured disagreement as diffuse rather than localized (Gini 0.45–0.48 across rubric items; 91 of 99 records carry at least one dispute), so rewriting the rows below would not move kappa. The lever is the judge, not this table.

| Gold id | Arm | Judge A | Judge B | Gap | Rubric |
| --- | --- | --- | --- | --- | --- |
| `ro-0018` | base_rag | 5.0 | 1.0 | 4.0 | machine-drafted |
| `ro-0020` | base | 5.0 | 1.0 | 4.0 | machine-drafted |
| `ro-0001` | base | 4.0 | 1.0 | 3.0 | machine-drafted |
| `ro-0005` | base_rag | 5.0 | 2.0 | 3.0 | machine-drafted |
| `ro-0005` | base | 4.0 | 1.0 | 3.0 | machine-drafted |
| `ro-0018` | base | 4.0 | 1.0 | 3.0 | machine-drafted |
| `ro-0003` | base | 5.0 | 2.3 | 2.7 | machine-drafted |
| `ro-0010` | base_rag | 2.3 | 5.0 | 2.7 | machine-drafted |
| `ro-0004` | base_rag | 3.7 | 2.3 | 1.3 | machine-drafted |
| `ro-0004` | base | 3.7 | 2.3 | 1.3 | machine-drafted |
| `ro-0008` | base_rag | 3.7 | 2.3 | 1.3 | machine-drafted |
| `ro-0014` | base | 3.7 | 2.3 | 1.3 | machine-drafted |
