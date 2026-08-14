# Section 9 Evaluation Report

110 questions (70 synthetic + 40 reddit)

| Arm | Avg score (1-5) | N scored | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- | --- |
| base | 3.53 | 110/110 | 33/110 | 5/107 |
| base_rag | 3.25 | 110/110 | 1/110 | 65/107 |
| finetuned | 2.49 | 110/110 | 11/110 | 5/107 |
| finetuned_rag | 2.80 | 110/110 | 6/110 | 50/107 |

Consistency (finetuned_rag, 15 questions rerun): 15/15 identical on rerun.

**Section 9.4 verdict:** finetuned_rag avg 2.80 vs base_rag avg 3.25. Fine-tuning (with RAG) does NOT beat RAG-only on this exam — per Section 9.4, fix the SFT data, not the hyperparameters.

## Lowest-scoring cases (for human review)

- [synthetic] "What does "Reveal" mean in Magic: The Gathering?" — worst: finetuned (score 1, Contradicts the reference answer)
- [synthetic] "What does "Max Speed" mean in Magic: The Gathering?" — worst: base (score 1, Incorrect and contradicted)
- [synthetic] "What does "Archenemy Commander" mean in Magic: The Gathering?" — worst: finetuned_rag (score 1, Incorrectly focuses on the Archenemy commander card rather than the format)
- [synthetic] "What happens when a player chooses to put their commander into the command zone using the replacemen" — worst: finetuned_rag (score 1, Contradicted rule citation)
- [synthetic] "What is a dungeon card and how is it brought into the game?" — worst: finetuned (score 1, Contradicts the reference answer)
- [synthetic] "What is the role of an emperor in the Emperor variant of Magic: The Gathering?" — worst: finetuned (score 1, Contradicts the reference answer and is overly detailed)
- [synthetic] "What does "Firebending" mean in Magic: The Gathering?" — worst: base_rag (score 1, Contradicts the reference answer)
- [synthetic] "What does "Meld" mean in Magic: The Gathering?" — worst: finetuned (score 1, Wrong or fabricated)
- [synthetic] "What does "Undaunted" mean in Magic: The Gathering?" — worst: base_rag (score 1, Incorrect and fabricated)
- [synthetic] "What does "Proliferate" mean in Magic: The Gathering?" — worst: finetuned (score 1, Wrong or fabricated)
