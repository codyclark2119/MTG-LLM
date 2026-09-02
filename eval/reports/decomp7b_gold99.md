# Section 9 Evaluation Report

99 questions (90 rulesguru, 8 glossary, 1 rules-qa-paste)

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: **none (`--base-only`)** — 3 arms (`base_rag`, `base`, `base_rag_cards_rulings`), not 6. Comparable only to a run with the SAME arms (Section 21.5: arm count changes scores)
- max tokens: `800`
- k (rules chunks retrieved): `3`
- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

**Answers stopped by the token ceiling** (not by finishing): `base_rag` 2/99, `base` 5/99, `base_rag_cards_rulings` 5/99

99 scored against enumerated rubrics (V3 judge); 0 scored against a prose reference (V2 judge). Rubric correctness is computed from key points hit, not assigned holistically.

| Arm | Avg score (1-5) | N scored | Cited a rule | Fabricated citation | Fabrication rate | Citation matches reference |
| --- | --- | --- | --- | --- | --- | --- |
| base_rag | 1.88 | 99/99 | 89/99 | 1/99 | 1% | n/a |
| base | 1.37 | 99/99 | 96/99 | 45/99 | 47% | n/a |
| base_rag_cards_rulings | 2.75 | 99/99 | 51/99 | 5/99 | 10% | n/a |

> **The highest-scoring arm is not the least-fabricating one.** `base_rag_cards_rulings` scores best while fabricating 5/99 citations, against `base_rag`'s 1/99. The rubric judge scores which enumerated claims an answer made; a fabricated rule id is not one of them, so it costs almost nothing here. Do not read the score column without this one (Section 19.1).

Consistency (base_rag, 15 questions rerun): 15/15 identical on rerun.

## Lowest-scoring cases (for human review)

- [rulesguru:1156] "Player A casts Assassin's Trophy targeting Nihil Spellbomb. Player B searches for a Snow-Covered Swa" — worst: base_rag (score 1.0, Incorrectly denies the trigger entirely because Nihil Spellbomb has already been destroyed.)
- [rulesguru:1182] "Player A casts Seasoned Pyromancer and has two other cards in hand, one of which is a land card. Pla" — worst: base (score 1.0, Incorrect rules cited)
- [rulesguru:1215] "Player A has Murderous Rider in their graveyard. Can they flash back Murderous Rider's Adventure wit" — worst: base (score 1.0, cites correct rules but misapplies them)
- [rulesguru:1414] "Player A controls Urborg, Tomb of Yawgmoth. Player B controls no lands and plays Castle Locthwain. D" — worst: base_rag (score 1.0, irrelevant to the question)
- [rulesguru:1469] "Player A controls Leyline of the Guildpact and casts Magus of the Moon. What are the characteristics" — worst: base_rag (score 1.0, Does not address the question.)
- [rulesguru:1596] "Player A casts Xenagos, God of Revels. Can Player B counter it with Spell Pierce?" — worst: base_rag (score 1.0, cites correct rules but does not address the key points)
- [rulesguru:1784] "Can Player A cast Pact of Negation targeting Player B's Ox of Agonas that is escaping from their gra" — worst: base_rag (score 1.0, cites non-existent rules)
- [rulesguru:1796] "Player A casts Thoughtseize targeting Player B. In response, Player B casts their last card, and whe" — worst: base_rag (score 1.0, Incorrectly states that no life is lost and miscites rules.)
- [rulesguru:1804] "Player A activates Karn, the Great Creator to make Player B's Cryptic Coat an artifact creature, whi" — worst: base_rag (score 1.0, cites non-existent rule 701.108a)
- [rulesguru:1836] "Player A would like to choose colorless for Utopia Sprawl. Can they?" — worst: base (score 1.0, Incorrectly cites a non-existent rule and misinterprets the card's ability)
