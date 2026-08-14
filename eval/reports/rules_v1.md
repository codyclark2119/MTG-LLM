# Section 9 Evaluation Report

110 questions (70 synthetic + 40 reddit)

| Arm | Avg score (1-5) | N scored | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- | --- |
| base | 3.62 | 110/110 | 33/110 | 5/107 |
| base_rag | 3.19 | 110/110 | 1/110 | 65/107 |
| finetuned | 2.47 | 110/110 | 26/110 | 5/107 |
| finetuned_rag | 2.37 | 110/110 | 5/110 | 58/107 |

Consistency (finetuned_rag, 15 questions rerun): 15/15 identical on rerun.

**Section 9.4 verdict:** finetuned_rag avg 2.37 vs base_rag avg 3.19. Fine-tuning (with RAG) does NOT beat RAG-only on this exam — per Section 9.4, fix the SFT data, not the hyperparameters.

## Lowest-scoring cases (for human review)

- [synthetic] "What does "Max Speed" mean in Magic: The Gathering?" — worst: base (score 1, Incorrect and contradicted)
- [synthetic] "What are the characteristics of a face-down permanent on the battlefield?" — worst: finetuned (score 1, Contradicts the reference answer)
- [synthetic] "What does "Archenemy Commander" mean in Magic: The Gathering?" — worst: finetuned_rag (score 1, Contradicts the reference answer by incorrectly citing rule 903 for Archenemy)
- [synthetic] "What happens when a player chooses to put their commander into the command zone using the replacemen" — worst: finetuned_rag (score 1, Wrong or fabricated, contradicts the reference answer)
- [synthetic] "What is the role of an emperor in the Emperor variant of Magic: The Gathering?" — worst: finetuned_rag (score 1, Contradicts the reference answer and missing key details)
- [synthetic] "What does "Firebending" mean in Magic: The Gathering?" — worst: base_rag (score 1, Contradicts the reference)
- [synthetic] "What does "Meld" mean in Magic: The Gathering?" — worst: finetuned (score 1, Wrong, describes a different keyword)
- [synthetic] "What happens when a creature spell resolves?" — worst: finetuned_rag (score 1, Incorrect and contradicts the reference answer)
- [synthetic] "What does "Undaunted" mean in Magic: The Gathering?" — worst: base_rag (score 1, Incorrect and fabricated)
- [synthetic] "What does "Mutate" mean in Magic: The Gathering?" — worst: base_rag (score 1, Contradicts the reference answer, incorrect rule citation)
