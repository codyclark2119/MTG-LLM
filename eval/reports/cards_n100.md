# Section 9 Evaluation Report

100 questions (0 synthetic + 100 reddit)

| Arm | Avg score (1-5) | N scored | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- | --- |
| base_rag | 3.36 | 100/100 | 1/100 | 73/100 |
| base | 3.40 | 100/100 | 22/100 | 3/100 |
| base_rag_cards | 3.35 | 100/100 | 1/100 | 53/100 |
| finetuned_rag | 3.08 | 100/100 | 3/100 | 37/100 |
| finetuned | 3.11 | 100/100 | 8/100 | 4/100 |
| finetuned_rag_cards | 3.17 | 100/100 | 1/100 | 40/100 |

Consistency (finetuned_rag, 5 questions rerun): 5/5 identical on rerun.

**Section 9.4 verdict:** finetuned_rag avg 3.08 vs base_rag avg 3.36. Fine-tuning (with RAG) does NOT beat RAG-only on this exam — per Section 9.4, fix the SFT data, not the hyperparameters.

## Lowest-scoring cases (for human review)

- [reddit] "Discard down to 7, but you discard a madness card
Had a situation last night where at the end of my " — worst: finetuned (score 1, Contradicts the reference, cites incorrect rule.)
- [reddit] "Cascade and additional costs
Yo! If I cascade into spells that require additional costs to be cast l" — worst: finetuned (score 1, Incorrect ruling, contradicts reference. No citation.)
- [reddit] "Stony Silence/Colossus Hammer/Sigarda's Aid
Question about the Modern deck Hammertime and an interac" — worst: finetuned_rag (score 1, Incorrect ruling, cites nothing relevant)
- [reddit] "Xyris -vs- Removal
If a player casts [[Expansion // Explosion]] targeting [[Xyris, the Writhing Stor" — worst: base_rag_cards (score 1, Incorrect ruling, no creature tokens are created. Cites nothing relevant.)
- [reddit] "How many lives do I gain?
I only have an [[Ajani's Welcome]] on the battlefield, and I play a [[Comm" — worst: base_rag (score 1, Incorrectly guesses the life gain and +1/+1 counters, which are not the same as the reference. Cites no rules.)
- [reddit] "Mind Flayer question.
Why don’t I regain control of my creature that was stolen by [[Mind Flayer]] i" — worst: base_rag (score 1, Incorrect, contradicts the reference.)
- [reddit] "enchantment creature tokens and opalescence
If I have a 2/1 enchantment creature token on the battle" — worst: finetuned_rag (score 1, Incorrect, contradicts the reference)
- [reddit] "If I cast Calamity's wake and there's non creature spells on the stack below it, would those spells " — worst: finetuned (score 1, Contradicts reference, incorrect rule cited)
- [reddit] "Elvish Warmaster
The ability of [[Elvish Warmaster]] says "Whenever one or more other Elves enter th" — worst: base_rag (score 1, Incorrect and cites no rule)
- [reddit] "Question about [[Skyclave relic]]
If I kick skyclave relic when it enters the battlefield but then I" — worst: base (score 1, Incorrect, contradicts the reference answer)
