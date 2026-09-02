# Section 9 Evaluation Report

20 questions (20 machine-drafted (Claude) from the pinned CR; card-free by construction)

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: **none (`--base-only`)** — 3 arms, not 6. Comparable only to another `--base-only` run (Section 21.5: arm count changes scores)
- max tokens: `800`
- k (rules chunks retrieved): `3`
- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

**Answers stopped by the token ceiling** (not by finishing): `base` 2/20

20 scored against enumerated rubrics (V3 judge); 0 scored against a prose reference (V2 judge). Rubric correctness is computed from key points hit, not assigned holistically.

| Arm | Avg score (1-5) | N scored | Grounded | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- | --- | --- |
| base_rag | 3.69 | 20/20 | 19/20 | 0/20 | n/a |
| base | 3.04 | 20/20 | 16/20 | 4/20 | n/a |

Consistency (base_rag, 15 questions rerun): 15/15 identical on rerun.

## Lowest-scoring cases (for human review)

- [machine-drafted (Claude) from the pinned CR; card-free by construction] "What happens when I target a creature that has ward 2 and I do not pay?" — worst: base (score 1.0, Incorrect rule cited and misapplied)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "When exactly are state-based actions checked?" — worst: base (score 1.0, Incorrect rule cited and misinterpreted)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "After a spell resolves, who gets priority?" — worst: base (score 1.0, Incorrectly states that the player who didn't cast the spell gets priority and cites a fabricated rule.)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "If a creature with double strike has deathtouch, what happens when it is blocked by a large creature" — worst: base_rag (score 1.0, Incorrectly states that deathtouch does not apply in the first-strike step.)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "What does flash let me do?" — worst: base (score 1.0, Incorrect rules cited and incorrect information given)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "If a creature would be destroyed and it has a regeneration shield, what happens?" — worst: base (score 1.0, Incorrect rules cited and regeneration misdescribed)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "What does protection from red actually stop?" — worst: base_rag (score 1.8, Cites no rules)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "Can a creature with defender attack?" — worst: base_rag (score 2.33, fabricated rule cited)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "My creature has first strike and is blocked by a creature with deathtouch. Does my creature die?" — worst: base (score 3.0, Incorrectly cites rule 510.1 and 510.1a, which do not exist. Incorrectly states that damage is dealt simultaneously.)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "Can I regenerate or prevent a creature I am sacrificing?" — worst: base (score 3.0, Cites correct rules)
