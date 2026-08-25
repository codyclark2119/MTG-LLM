# Section 9 Evaluation Report

99 questions (90 rulesguru, 8 glossary, 1 rules-qa-paste)

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter under test: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen2.5-7B-Instruct-4bit`  (**same model as the `base` arm** — self-preference bias is not ruled out; re-judge with --rescore-from and an independent judge, Section 9.9)

99 scored against enumerated rubrics (V3 judge); 0 scored against a prose reference (V2 judge). Rubric correctness is computed from key points hit, not assigned holistically.

| Arm | Avg score (1-5) | N scored | Grounded | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- | --- | --- |
| base_rag | 3.09 | 99/99 | 86/99 | 1/99 | n/a |
| base | 3.15 | 99/99 | 45/99 | 35/99 | n/a |
| base_rag_cards_rulings | 3.06 | 99/99 | 86/99 | 1/99 | n/a |
| finetuned_rag | 2.62 | 99/99 | 68/99 | 4/99 | n/a |
| finetuned | 2.35 | 99/99 | 62/99 | 9/99 | n/a |
| finetuned_rag_cards_rulings | 2.65 | 99/99 | 68/99 | 4/99 | n/a |

> **The highest-scoring arm is not the best-grounded one.** `base` scores best while fabricating 35/99 citations, against `base_rag`'s 1/99. The rubric judge scores which enumerated claims an answer made; a fabricated rule id is not one of them, so it costs almost nothing here. Do not read the score column without this one (Section 19.1).

Consistency (finetuned_rag, 15 questions rerun): 15/15 identical on rerun.

**Section 9.4 verdict:** finetuned_rag avg 2.62 vs base_rag avg 3.09. Fine-tuning (with RAG) does NOT beat RAG-only on this exam — per Section 9.4, fix the SFT data, not the hyperparameters.

## Lowest-scoring cases (for human review)

- [rulesguru:1156] "Player A casts Assassin's Trophy targeting Nihil Spellbomb. Player B searches for a Snow-Covered Swa" — worst: base_rag_cards_rulings (score 1.0, Incorrectly states the triggered ability resolves immediately and cannot be responded to)
- [rulesguru:1182] "Player A casts Seasoned Pyromancer and has two other cards in hand, one of which is a land card. Pla" — worst: base (score 1.0, Incorrectly cites rules and does not hit key points.)
- [rulesguru:1215] "Player A has Murderous Rider in their graveyard. Can they flash back Murderous Rider's Adventure wit" — worst: base (score 1.0, Confuses Adventure and Flashback, and provides incorrect rules and a fabricated citation.)
- [rulesguru:1414] "Player A controls Urborg, Tomb of Yawgmoth. Player B controls no lands and plays Castle Locthwain. D" — worst: base_rag_cards_rulings (score 1.0, Mentions Urborg, Tomb of Yawgmoth, but does not address the question. Mentions incorrect and irrelevant information.)
- [rulesguru:1596] "Player A casts Xenagos, God of Revels. Can Player B counter it with Spell Pierce?" — worst: base (score 1.0, Incorrectly states that Xenagos can be countered by Spell Pierce, citing irrelevant rules.)
- [rulesguru:1796] "Player A casts Thoughtseize targeting Player B. In response, Player B casts their last card, and whe" — worst: finetuned_rag (score 1.0, Incorrectly cites a rule and makes errors)
- [rulesguru:1804] "Player A activates Karn, the Great Creator to make Player B's Cryptic Coat an artifact creature, whi" — worst: base_rag (score 1.0, Incorrectly interprets Karn's ability and its effects.)
- [rulesguru:1836] "Player A would like to choose colorless for Utopia Sprawl. Can they?" — worst: base (score 1.0, Incorrectly states colorless can be chosen, citing a non-existent rule.)
- [rules-qa-paste] "In a game of Two-Headed Giant, both Player B and Player A have no cards left in their library. Playe" — worst: finetuned_rag (score 1.0, Incorrectly states the win and loss timing and cites an invalid rule.)
- [rulesguru:2818] "Player A casts Solitude. Player B passes priority. Player C passes priority. Player D taps a land, f" — worst: finetuned_rag (score 1.0, Incorrectly states that Player A gets priority and that Solitude causes creatures to be sacrificed, which is not relevant to the scenario.)
