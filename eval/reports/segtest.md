# Section 9 Evaluation Report

24 questions (24 gold)

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter under test: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen2.5-7B-Instruct-4bit`  (**same model as the `base` arm** — self-preference bias is not ruled out; re-judge with --rescore-from and an independent judge, Section 9.9)

24 scored against enumerated rubrics (V3 judge); 0 scored against a prose reference (V2 judge). Rubric correctness is computed from key points hit, not assigned holistically.

| Arm | Avg score (1-5) | N scored | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- | --- |
| base_rag | 2.19 | 24/24 | 1/24 | 0/24 |
| base | 2.71 | 24/24 | 11/24 | 1/24 |
| finetuned_rag | 1.76 | 24/24 | 1/24 | 0/24 |
| finetuned | 1.60 | 24/24 | 1/24 | 0/24 |

Consistency (finetuned_rag, 15 questions rerun): 15/15 identical on rerun.

**Section 9.4 verdict:** finetuned_rag avg 1.76 vs base_rag avg 2.19. Fine-tuning (with RAG) does NOT beat RAG-only on this exam — per Section 9.4, fix the SFT data, not the hyperparameters.

## Lowest-scoring cases (for human review)

- [gold:rulesguru:104] "Player A controls Painter's Servant naming "blue" and Sword of Body and Mind. Player B casts Turn ta" — worst: finetuned (score 1.0, Does not provide an answer based on given rules)
- [gold:rulesguru:202] "Player A controls Drought and Derelor. How many Swamps would they have to sacrifice in order to cast" — worst: finetuned (score 1.0, No key points or errors made, but no relevant information provided.)
- [gold:rulesguru:273] "Player A controls Leyline of Anticipation. In Player B's end step, Player A casts Fury of the Horde." — worst: finetuned_rag (score 1.0, Makes all common errors and no key points)
- [gold:rulesguru:115] "Player A controls Fresh Volunteers and Muraganda Petroglyphs. They cast and resolve Weapon Surge, ta" — worst: finetuned_rag (score 1.0, Incorrectly describes Fresh Volunteers' abilities and interactions.)
- [gold:rulesguru:1099] "Player A controls Night of Souls' Betrayal and casts Watcher for Tomorrow. What happens as Watcher f" — worst: base_rag (score 1.0, Incorrect sequence, denies leaves-the-battlefield trigger, and fixes order)
- [gold:rulesguru:204] "Player A controls Drought. How many Swamps would Player B have to sacrifice in order to cast Beckon " — worst: finetuned_rag (score 1.0, Incorrectly cites a non-existent rule and does not address the question.)
- [gold:rulesguru:280] "Player A controls a Soar that was cast during their declare blockers step. During Player A's end ste" — worst: base_rag (score 1.0, Incorrectly states that Player A does not lose life and cites an irrelevant rule.)
- [gold:rulesguru:1394] "Player A controls Reflecting Pool and River of Tears. They have not played a land this turn. What ma" — worst: finetuned_rag (score 1.0, Incorrect color, extra info, and errors)
- [gold:rulesguru:1804] "Player A activates Karn, the Great Creator to make Player B's Cryptic Coat an artifact creature, whi" — worst: base_rag (score 1.0, Does not hit any key points, makes multiple errors, and cites nothing relevant.)
- [gold:rules-qa-paste] "In a game of Two-Headed Giant, both Player B and Player A have no cards left in their library. Playe" — worst: finetuned (score 1.0, Incorrectly cites rule 103.4, awards win to Player B, and states loss happens at same time as win ending in a draw)
