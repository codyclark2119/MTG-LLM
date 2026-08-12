# Section 9 Evaluation Report

60 questions (0 synthetic + 60 reddit)

| Arm | Avg score (1-5) | N scored | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- | --- |
| base_rag | 3.47 | 60/60 | 2/60 | 41/60 |
| base | 3.42 | 60/60 | 13/60 | 4/60 |
| base_rag_cards | 3.65 | 60/60 | 1/60 | 41/60 |
| finetuned_rag | 3.25 | 60/60 | 3/60 | 20/60 |
| finetuned | 3.03 | 60/60 | 7/60 | 2/60 |
| finetuned_rag_cards | 3.18 | 60/60 | 0/60 | 21/60 |

Consistency (finetuned_rag, 15 questions rerun): 15/15 identical on rerun.

**Section 9.4 verdict:** finetuned_rag avg 3.25 vs base_rag avg 3.47. Fine-tuning (with RAG) does NOT beat RAG-only on this exam — per Section 9.4, fix the SFT data, not the hyperparameters.

## Lowest-scoring cases (for human review)

- [reddit] "Exile/Shuffle into library question: if you exile a card like Nexus of Fate and then cast it for fre" — worst: finetuned (score 1, Incorrect ruling, cites nothing relevant)
- [reddit] "Infinite loop?
My friends and I were playing. One of them played [[sanguine bond]] and then [[exquis" — worst: base_rag (score 1, Incorrect, the scenario does create an infinite loop. No relevant rule number cited.)
- [reddit] "Basic MTG combat question
Hey, I haven't played in like 7 years and am really rusty. A friend got me" — worst: base (score 1, Incorrect and contradicts the reference. Cites wrong rule.)
- [reddit] "Chaos warping a "marked" card
We were playing a game and my commander got warped the issue was that " — worst: finetuned (score 1, Incorrectly states the rules do not address the specific scenario, but provides no relevant context or rule numbers.)
- [reddit] "Can I choose in which order suspend cards are cast?
Let me give you the example I have in mind:
Usin" — worst: finetuned_rag (score 1, Contradicts the reference, cites the correct rule.)
- [reddit] "Help with these two. Does doom blade still work on najeela. I think it still works due to it being r" — worst: base (score 1, Incorrect, cites wrong card. Does not cite relevant rules.)
- [reddit] "With 0 counters on my tapped Kalamax, this is 94 counters, right?" — worst: finetuned (score 1, Incorrectly states the number of counters and cites an irrelevant rule.)
- [reddit] "Do Minsc & Boo create the dame token as Minsc, Beloved Ranger?
So I would like to know if [[Minsc & " — worst: base_rag (score 1, Incorrectly cites rules and provides a wrong answer.)
- [reddit] "Does Deathrite need to target a land?
Since the "exile target land from a graveyard" is after the co" — worst: base_rag (score 1, Incorrect ruling, contradicts reference)
- [reddit] "Korvold attacks along with Kozilek while I have 0 cards in library. Would sacrificing Kozilek trigge" — worst: finetuned_rag (score 1, Incorrect, contradicts the reference answer and the rules. Does not correctly state the ruling.)
