# Section 9 Evaluation Report

45 questions (45 gold)

45 scored against enumerated rubrics (V3 judge); 0 scored against a prose reference (V2 judge). Rubric correctness is computed from key points hit, not assigned holistically.

| Arm | Avg score (1-5) | N scored | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- | --- |
| base_rag | 2.76 | 45/45 | 0/45 | 1/45 |
| base | 3.17 | 45/45 | 8/45 | 0/45 |
| finetuned_rag | 1.91 | 45/45 | 3/45 | 2/45 |
| finetuned | 2.32 | 45/45 | 5/45 | 0/45 |

Consistency (finetuned_rag, 5 questions rerun): 5/5 identical on rerun.

**Section 9.4 verdict:** finetuned_rag avg 1.91 vs base_rag avg 2.76. Fine-tuning (with RAG) does NOT beat RAG-only on this exam — per Section 9.4, fix the SFT data, not the hyperparameters.

## Lowest-scoring cases (for human review)

- [gold:rulesguru:2] "Natalee controls a Kruphix, God of Horizons and has 3 red mana and 2 colorless mana in their mana po" — worst: base (score 1.0, Incorrectly describes the sequence of events and the final mana pools.)
- [gold:rulesguru:19] "Avery controls Possibility Storm and casts a Whetwheel for its morph cost. Avery will exile cards fr" — worst: base_rag (score 1.0, Incorrect and contradicts the rules.)
- [gold:rulesguru:8] "Ally controls a 2/2 green Bear Creature token. They cast Cloudshift, targeting the token. What happe" — worst: finetuned_rag (score 1.0, Incorrectly states the token enters the battlefield with a +1/+1 counter, which is not the effect of Cloudshift.)
- [gold:rulesguru:4] "Axton attacks with Wild Beastmaster and Hyena Pack. Nico casts Jace's Scrutiny, targeting the Wild B" — worst: finetuned_rag (score 1.0, Incorrectly states Hyena Pack is put into the owner's graveyard. Incorrectly cites a rule that does not apply.)
- [gold:rulesguru:6] "Ari controls a Falkenrath Gorger and has 8 cards in their hand. During their cleanup step, they disc" — worst: finetuned_rag (score 1.0, Incorrectly states the answer and misinterprets the rules.)
- [gold:rulesguru:17] "Alfred has to choose a card name for Council of the Absolute. Can they choose Rune-Tail's Essence?" — worst: base_rag (score 1.0, Incorrectly states the player cannot choose a card name that appears in the glossary, and cites a rule that does not apply.)
- [gold:rulesguru:86] "Augustus attacks Nia with a Daggerback Basilisk, which is their only creature. Nia blocks with their" — worst: base_rag (score 1.0, Incorrect timing, incorrect rule citation)
- [gold:rulesguru:172] "Allen controls Leovold, Emissary of Trest and Notion Thief. Nataly casts Brainstorm. What happens?" — worst: base_rag (score 1.0, Incorrectly applies rules about leaving the game, and misinterprets Brainstorm's effect.)
- [gold:rulesguru:76] "Alex controls a Great Furnace that has been animated with Skilled Animator's ability. Nehemiah plays" — worst: finetuned_rag (score 1.0, Incorrectly states the Great Furnace is a creature and fails to mention it's still a land.)
- [gold:rulesguru:47] "Nick controls Dryad Militant. Audrina casts Hanabi Blast. Does it return to their hand?" — worst: base_rag (score 1.0, Incorrectly describes the scenario and the rules.)
