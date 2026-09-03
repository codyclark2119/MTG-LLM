# Section 9 Evaluation Report

99 questions (90 rulesguru, 8 glossary, 1 rules-qa-paste)

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: **none (`--base-only`)** — 3 arms (`base`, `base_rag_cards_rulings`, `base_rag_cards_rulings_verified`), not 6. Comparable only to a run with the SAME arms (Section 21.5: arm count changes scores)
- max tokens: `800`
- k (rules chunks retrieved): `3`
- **verification pass** (`--verify`, Section 21.164): second greedy pass over the card arm's own draft; prompt digest `6b6020c957d4`. **3 answers lacked the `FINAL ANSWER:` marker** and fell back to the whole second-pass text, so the judge graded a critique on those
- **plain `_rag` arm dropped** (`--no-plain-rag`), so this run's arm count differs from a default run and the two are not comparable (Section 21.5)
- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

**Answers stopped by the token ceiling** (not by finishing): `base` 5/99, `base_rag_cards_rulings` 5/99

99 scored against enumerated rubrics (V3 judge); 0 scored against a prose reference (V2 judge). Rubric correctness is computed from key points hit, not assigned holistically.

> **3 of 297 arm-answers (1%) went unjudged** — the judge returned JSON that could not be parsed, usually by running out of output tokens. They are excluded rather than counted as wrong.

| Arm | Avg score (1-5) | N scored | Cited a rule | Fabricated citation | Fabrication rate | Citation matches reference |
| --- | --- | --- | --- | --- | --- | --- |
| base | 1.48 | 98/99 | 96/99 | 45/99 | 47% | n/a |
| base_rag_cards_rulings | 2.53 | 98/99 | 51/99 | 5/99 | 10% | n/a |
| base_rag_cards_rulings_verified | 2.03 | 98/99 | 7/99 | 0/99 | 0% | n/a |

> **The highest-scoring arm is not the least-fabricating one.** `base_rag_cards_rulings` scores best while fabricating 5/99 citations, against `base_rag_cards_rulings_verified`'s 0/99. The rubric judge scores which enumerated claims an answer made; a fabricated rule id is not one of them, so it costs almost nothing here. Do not read the score column without this one (Section 19.1).

Consistency (base, 15 questions rerun): 15/15 identical on rerun.

## Lowest-scoring cases (for human review)

- [rulesguru:1156] "Player A casts Assassin's Trophy targeting Nihil Spellbomb. Player B searches for a Snow-Covered Swa" — worst: base (score 1.0, Incorrectly states the trigger is placed on the stack before Assassin's Trophy resolves and incorrectly states the payment is made when the trigger is placed on the stack.)
- [rulesguru:1182] "Player A casts Seasoned Pyromancer and has two other cards in hand, one of which is a land card. Pla" — worst: base (score 1.0, Incorrect rules cited and incorrect conclusion)
- [rulesguru:1215] "Player A has Murderous Rider in their graveyard. Can they flash back Murderous Rider's Adventure wit" — worst: base (score 1.0, Incorrect rules cited)
- [rulesguru:1414] "Player A controls Urborg, Tomb of Yawgmoth. Player B controls no lands and plays Castle Locthwain. D" — worst: base_rag_cards_rulings (score 1.0, No key points stated, common error 1 stated)
- [rulesguru:1469] "Player A controls Leyline of the Guildpact and casts Magus of the Moon. What are the characteristics" — worst: base (score 1.0, Incorrectly applies Magus of the Moon's effect before Leyline of the Guildpact's effect)
- [rulesguru:1596] "Player A casts Xenagos, God of Revels. Can Player B counter it with Spell Pierce?" — worst: base (score 1.0, No key points mentioned)
- [rulesguru:1784] "Can Player A cast Pact of Negation targeting Player B's Ox of Agonas that is escaping from their gra" — worst: base (score 1.0, Incorrectly discusses Ox of Agonas dying and creating a token, which is irrelevant to the escape mechanic.)
- [rulesguru:1796] "Player A casts Thoughtseize targeting Player B. In response, Player B casts their last card, and whe" — worst: base (score 1.0, fabricated rules)
- [rulesguru:1804] "Player A activates Karn, the Great Creator to make Player B's Cryptic Coat an artifact creature, whi" — worst: base_rag_cards_rulings (score 1.0, Incorrectly states that Cryptic Coat can be equipped to Memnite and does not mention unattaching)
- [rulesguru:1836] "Player A would like to choose colorless for Utopia Sprawl. Can they?" — worst: base (score 1.0, Fabricated rule)
