# Section 9 Evaluation Report

99 questions (90 rulesguru, 8 glossary, 1 rules-qa-paste)

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter under test: `models/mtg-rules-adapter-v5-gameplay`
- judge: `mlx-community/Qwen2.5-32B-Instruct-4bit`

- **`finetuned_rag`**: this adapter's training set contains its system prompt **zero** times. The prompt fingerprint matches — nothing was edited — but the weights never saw this shape, which is Section 8.7's mechanism reached without a prompt change. Read the row as a statement about the training *shape*, not about the training *data*.

99 scored against enumerated rubrics (V3 judge); 0 scored against a prose reference (V2 judge). Rubric correctness is computed from key points hit, not assigned holistically.

| Arm | Avg score (1-5) | N scored | Grounded | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- | --- | --- |
| base_rag | 1.68 | 99/99 | 86/99 | 1/99 | n/a |
| base | 1.43 | 99/99 | 45/99 | 35/99 | n/a |
| finetuned_rag | 1.16 | 99/99 | 82/99 | 12/99 | n/a |
| finetuned | 1.20 | 99/99 | 83/99 | 14/99 | n/a |

Consistency (finetuned_rag, 15 questions rerun): 15/15 identical on rerun.

**Section 9.4 verdict:** finetuned_rag avg 1.16 vs base_rag avg 1.68. Fine-tuning (with RAG) does NOT beat RAG-only on this exam — per Section 9.4, fix the SFT data, not the hyperparameters.

## Lowest-scoring cases (for human review)

- [rulesguru:1156] "Player A casts Assassin's Trophy targeting Nihil Spellbomb. Player B searches for a Snow-Covered Swa" — worst: base_rag (score 1.0, Incorrect and irrelevant information)
- [rulesguru:1182] "Player A casts Seasoned Pyromancer and has two other cards in hand, one of which is a land card. Pla" — worst: base (score 1.0, Incorrect rule application and conclusion)
- [rulesguru:1215] "Player A has Murderous Rider in their graveyard. Can they flash back Murderous Rider's Adventure wit" — worst: base (score 1.0, Incorrect and self-contradicting)
- [rulesguru:1414] "Player A controls Urborg, Tomb of Yawgmoth. Player B controls no lands and plays Castle Locthwain. D" — worst: base_rag (score 1.0, Incorrect and irrelevant rule citation.)
- [rulesguru:1469] "Player A controls Leyline of the Guildpact and casts Magus of the Moon. What are the characteristics" — worst: base_rag (score 1.0, Incorrect and misleading)
- [rulesguru:1596] "Player A casts Xenagos, God of Revels. Can Player B counter it with Spell Pierce?" — worst: base_rag (score 1.0, Incorrect mana value and irrelevant rule references)
- [rulesguru:1784] "Can Player A cast Pact of Negation targeting Player B's Ox of Agonas that is escaping from their gra" — worst: base_rag (score 1.0, Incorrectly states escape and targeting rules, cites irrelevant rules)
- [rulesguru:1796] "Player A casts Thoughtseize targeting Player B. In response, Player B casts their last card, and whe" — worst: base_rag (score 1.0, Incorrect rules and reasoning)
- [rulesguru:1804] "Player A activates Karn, the Great Creator to make Player B's Cryptic Coat an artifact creature, whi" — worst: base_rag (score 1.0, Incorrect interpretation of Karn's ability)
- [rulesguru:1836] "Player A would like to choose colorless for Utopia Sprawl. Can they?" — worst: base (score 1.0, Incorrect and irrelevant rules cited)
