# Section 9 Evaluation Report

20 questions (20 machine-drafted (Claude) from the pinned CR; card-free by construction)

- base model: `mlx-community/Qwen2.5-32B-Instruct-4bit`
- adapter: **none (`--base-only`)** — 3 arms, not 6. Comparable only to another `--base-only` run (Section 21.5: arm count changes scores)
- max tokens: `800`
- k (rules chunks retrieved): `3`
- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

20 scored against enumerated rubrics (V3 judge); 0 scored against a prose reference (V2 judge). Rubric correctness is computed from key points hit, not assigned holistically.

| Arm | Avg score (1-5) | N scored | Grounded | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- | --- | --- |
| base_rag | 3.74 | 20/20 | 20/20 | 0/20 | n/a |
| base | 3.34 | 20/20 | 18/20 | 2/20 | n/a |

Consistency (base_rag, 15 questions rerun): 15/15 identical on rerun.

## Lowest-scoring cases (for human review)

- [machine-drafted (Claude) from the pinned CR; card-free by construction] "If a 5/5 creature with deathtouch and trample is blocked by a 4/4, how much damage is dealt to the b" — worst: base (score 1.0, Incorrectly states that all 5 damage is assigned to the blocker and that trample is irrelevant.)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "If a creature with indestructible has its toughness reduced to 0, does it die?" — worst: base (score 1.0, Cites wrong rules and misinterprets indestructible)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "If a creature would be destroyed and it has a regeneration shield, what happens?" — worst: base (score 1.0, Incorrect rules cited and regeneration does not set toughness to 0)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "If my creature with lifelink deals damage to two different creatures at once, how much life do I gai" — worst: base_rag (score 1.0, Incorrect rule cited and misinterpreted lifelink as a triggered ability)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "If an attacking creature with trample is blocked by a creature with protection from that attacker's " — worst: base (score 2.0, Mentions trample damage going through but incorrectly states protection prevents all damage and incorrectly states protection prevents trample damage to the player.)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "After a spell resolves, who gets priority?" — worst: base_rag (score 2.33, misses 2, 3)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "Can a creature with defender attack?" — worst: base_rag (score 2.33, Incorrect rule cited)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "Why can a creature with haste attack the turn it comes under my control?" — worst: base (score 2.33, Incorrect rules cited)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "What does protection from red actually stop?" — worst: base_rag (score 3.4, Misses point 3 and 5, and cites a non-existent rule 702.16d)
- [machine-drafted (Claude) from the pinned CR; card-free by construction] "Can a creature with reach block a creature with flying?" — worst: base_rag (score 3.67, misses key point 3)
