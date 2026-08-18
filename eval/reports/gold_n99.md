# Section 9 Evaluation Report

99 questions (99 gold)

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter under test: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen2.5-7B-Instruct-4bit`  (**same model as the `base` arm** — self-preference bias is not ruled out; re-judge with --rescore-from and an independent judge, Section 9.9)

99 scored against enumerated rubrics (V3 judge); 0 scored against a prose reference (V2 judge). Rubric correctness is computed from key points hit, not assigned holistically.

| Arm | Avg score (1-5) | N scored | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- | --- |
| base_rag | 2.38 | 99/99 | 1/99 | 10/99 |
| base | 2.46 | 99/99 | 35/99 | 2/99 |
| finetuned_rag | 1.88 | 99/99 | 4/99 | 5/99 |
| finetuned | 1.69 | 99/99 | 9/99 | 0/99 |

Consistency (finetuned_rag, 15 questions rerun): 15/15 identical on rerun.

**Section 9.4 verdict:** finetuned_rag avg 1.88 vs base_rag avg 2.38. Fine-tuning (with RAG) does NOT beat RAG-only on this exam — per Section 9.4, fix the SFT data, not the hyperparameters.

## Lowest-scoring cases (for human review)

- [gold:rulesguru:1156] "Player A casts Assassin's Trophy targeting Nihil Spellbomb. Player B searches for a Snow-Covered Swa" — worst: base_rag (score 1.0, Incorrectly cites a rule that does not apply to this scenario)
- [gold:rulesguru:1182] "Player A casts Seasoned Pyromancer and has two other cards in hand, one of which is a land card. Pla" — worst: base (score 1.0, Incorrectly states the number of tokens, cites irrelevant rules)
- [gold:rulesguru:1215] "Player A has Murderous Rider in their graveyard. Can they flash back Murderous Rider's Adventure wit" — worst: base_rag (score 1.0, Incorrectly states an Adventure can have Flashback, confuses Adventure with Flashback, and incorrectly states an Adventure can be Flashbacked)
- [gold:rulesguru:1414] "Player A controls Urborg, Tomb of Yawgmoth. Player B controls no lands and plays Castle Locthwain. D" — worst: base_rag (score 1.0, Incorrectly cites a rule about artifact lands and makes an error by not considering the specific card text of Castle Locthwain.)
- [gold:rulesguru:1596] "Player A casts Xenagos, God of Revels. Can Player B counter it with Spell Pierce?" — worst: finetuned_rag (score 1.0, Incorrectly states Xenagos is not a creature while on the stack, and incorrectly cites a rule.)
- [gold:rulesguru:1784] "Can Player A cast Pact of Negation targeting Player B's Ox of Agonas that is escaping from their gra" — worst: finetuned (score 1.0, Incorrectly states the rules for casting spells and targeting, and does not correctly identify the stack interaction)
- [gold:rulesguru:1796] "Player A casts Thoughtseize targeting Player B. In response, Player B casts their last card, and whe" — worst: base_rag (score 1.0, Incorrectly cites a rule that does not apply and makes errors.)
- [gold:rulesguru:1804] "Player A activates Karn, the Great Creator to make Player B's Cryptic Coat an artifact creature, whi" — worst: base_rag (score 1.0, Incorrectly interprets Karn's ability and its effects.)
- [gold:rulesguru:1836] "Player A would like to choose colorless for Utopia Sprawl. Can they?" — worst: finetuned (score 1.0, Incorrect, but no citation)
- [gold:rulesguru:2818] "Player A casts Solitude. Player B passes priority. Player C passes priority. Player D taps a land, f" — worst: finetuned_rag (score 1.0, Incorrectly describes the resolution and the sequence of events, and cites a non-existent rule.)
