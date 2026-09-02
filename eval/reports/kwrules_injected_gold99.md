# Section 9 Evaluation Report

99 questions (90 rulesguru, 8 glossary, 1 rules-qa-paste)

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: **none (`--base-only`)** — 2 arms (`base`, `base_rag_cards_rulings`), not 6. Comparable only to a run with the SAME arms (Section 21.5: arm count changes scores)
- max tokens: `800`
- k (rules chunks retrieved): `3`
- **keyword rules injected** (`--keyword-rules`, Section 21.159): the CR text for rules a resolved card's own keywords name, deduped against the dense hits. Card arms only.
- **plain `_rag` arm dropped** (`--no-plain-rag`), so this run's arm count differs from a default run and the two are not comparable (Section 21.5)
- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

**Answers stopped by the token ceiling** (not by finishing): `base` 5/99, `base_rag_cards_rulings` 1/99

99 scored against enumerated rubrics (V3 judge); 0 scored against a prose reference (V2 judge). Rubric correctness is computed from key points hit, not assigned holistically.

> **2 of 198 arm-answers (1%) went unjudged** — the judge returned JSON that could not be parsed, usually by running out of output tokens. They are excluded rather than counted as wrong.

| Arm | Avg score (1-5) | N scored | Cited a rule | Fabricated citation | Fabrication rate | Citation matches reference |
| --- | --- | --- | --- | --- | --- | --- |
| base | 1.27 | 98/99 | 96/99 | 45/99 | 47% | n/a |
| base_rag_cards_rulings | 2.70 | 98/99 | 47/99 | 1/99 | 2% | n/a |

Consistency (base, 15 questions rerun): 15/15 identical on rerun.

## Lowest-scoring cases (for human review)

- [rulesguru:1156] "Player A casts Assassin's Trophy targeting Nihil Spellbomb. Player B searches for a Snow-Covered Swa" — worst: base (score 1.0, Incorrectly states the trigger is put on the stack before Assassin's Trophy resolves and incorrectly states the payment is made when the trigger is put on the stack.)
- [rulesguru:1182] "Player A casts Seasoned Pyromancer and has two other cards in hand, one of which is a land card. Pla" — worst: base (score 1.0, Incorrect rule cited)
- [rulesguru:1215] "Player A has Murderous Rider in their graveyard. Can they flash back Murderous Rider's Adventure wit" — worst: base (score 1.0, Cites correct rules but misapplies them)
- [rulesguru:1414] "Player A controls Urborg, Tomb of Yawgmoth. Player B controls no lands and plays Castle Locthwain. D" — worst: base_rag_cards_rulings (score 1.0, No key points hit, common error 1 made, no citation)
- [rulesguru:1469] "Player A controls Leyline of the Guildpact and casts Magus of the Moon. What are the characteristics" — worst: base (score 1.0, Incorrectly applies Magus of the Moon's effect to all lands, including basic lands)
- [rulesguru:1596] "Player A casts Xenagos, God of Revels. Can Player B counter it with Spell Pierce?" — worst: base (score 1.0, Incorrectly states that Spell Pierce counters spells with a converted mana cost of 1 or less, and incorrectly states that Xenagos' devotion is relevant)
- [rulesguru:1784] "Can Player A cast Pact of Negation targeting Player B's Ox of Agonas that is escaping from their gra" — worst: base (score 1.0, Incorrectly states that Ox of Agonas creates a token when it dies and that this token can be targeted)
- [rulesguru:1796] "Player A casts Thoughtseize targeting Player B. In response, Player B casts their last card, and whe" — worst: base (score 1.0, Misses all key points and makes multiple errors)
- [rulesguru:1836] "Player A would like to choose colorless for Utopia Sprawl. Can they?" — worst: base (score 1.0, Fabricated rule cited)
- [rules-qa-paste] "In a game of Two-Headed Giant, both Player B and Player A have no cards left in their library. Playe" — worst: base (score 1.0, Incorrectly cites non-existent rules and does not address Jace's effect)
