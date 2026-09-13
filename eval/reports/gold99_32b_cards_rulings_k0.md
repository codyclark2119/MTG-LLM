# Section 9 Evaluation Report

99 questions (90 rulesguru, 8 glossary, 1 rules-qa-paste)

- base model: `mlx-community/Qwen2.5-32B-Instruct-4bit`
- adapter: **none (`--base-only`)** — 2 arms (`base`, `base_rag_cards_rulings`), not 6. Comparable only to a run with the SAME arms (Section 21.5: arm count changes scores)
- max tokens: `800`
- k (rules chunks retrieved): `0`
- **plain `_rag` arm dropped** (`--no-plain-rag`), so this run's arm count differs from a default run and the two are not comparable (Section 21.5)
- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

**Answers stopped by the token ceiling** (not by finishing): `base` 4/99, `base_rag_cards_rulings` 5/99

99 scored against enumerated rubrics (V3 judge); 0 scored against a prose reference (V2 judge). Rubric correctness is computed from key points hit, not assigned holistically.

| Arm | Avg score (1-5) | N scored | Cited a rule | Fabricated citation | Fabrication rate | Citation matches reference |
| --- | --- | --- | --- | --- | --- | --- |
| base | 1.70 | 99/99 | 96/99 | 29/99 | 30% | n/a |
| base_rag_cards_rulings | 2.78 | 99/99 | 85/99 | 21/99 | 25% | n/a |

> **No CR text was retrieved for this run (`--k-rules 0`), and the prompt still asks for rule citations.** Measured on the 32B card benchmark, removing that section made the model cite the CR *more* -- 32 answers to 41 -- from memory, and fabricate on 3 against 7 (Section 21.158). Read the fabrication RATE against a `k=3` run before reading the score column; the score alone moved +0.25 in the opposite direction.

Consistency (base, 15 questions rerun): 15/15 identical on rerun.

## Lowest-scoring cases (for human review)

- [rulesguru:1156] "Player A casts Assassin's Trophy targeting Nihil Spellbomb. Player B searches for a Snow-Covered Swa" — worst: base_rag_cards_rulings (score 1.0, Incorrect rules cited and misinterpreted)
- [rulesguru:1182] "Player A casts Seasoned Pyromancer and has two other cards in hand, one of which is a land card. Pla" — worst: base (score 1.0, Incorrectly cites rules and misinterprets card abilities)
- [rulesguru:1414] "Player A controls Urborg, Tomb of Yawgmoth. Player B controls no lands and plays Castle Locthwain. D" — worst: base (score 1.0, Incorrectly cites a rule and misapplies it)
- [rulesguru:1469] "Player A controls Leyline of the Guildpact and casts Magus of the Moon. What are the characteristics" — worst: base (score 1.0, Incorrect rules cited)
- [rulesguru:1596] "Player A casts Xenagos, God of Revels. Can Player B counter it with Spell Pierce?" — worst: base (score 1.0, Cites correct rules but misses key points)
- [rulesguru:1784] "Can Player A cast Pact of Negation targeting Player B's Ox of Agonas that is escaping from their gra" — worst: base (score 1.0, Mistakenly treats escape as a triggered ability and claims it is uncounterable.)
- [rulesguru:1796] "Player A casts Thoughtseize targeting Player B. In response, Player B casts their last card, and whe" — worst: base (score 1.0, Incorrect rule cited)
- [rules-qa-paste] "In a game of Two-Headed Giant, both Player B and Player A have no cards left in their library. Playe" — worst: base (score 1.0, Incorrectly states that the team loses the game due to the mill condition, and does not mention Jace, Wielder of Mysteries.)
- [rulesguru:2818] "Player A casts Solitude. Player B passes priority. Player C passes priority. Player D taps a land, f" — worst: base_rag_cards_rulings (score 1.0, No key points are stated.)
- [rulesguru:2836] "Player A controls Elesh Norn, Grand Cenobite. Player B casts Murderous Cut targeting Elesh Norn, Gra" — worst: base_rag_cards_rulings (score 1.0, A does not state that the returned Elesh Norn is a new object, and incorrectly states that Murderous Cut resolves on the returned Elesh Norn.)
