# Section 9 Evaluation Report

110 questions (70 synthetic, 40 reddit)

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter under test: `models/mtg-rules-adapter-v3-ckpt1322`
- judge: `mlx-community/Qwen2.5-7B-Instruct-4bit`  (**same model as the `base` arm** — self-preference bias is not ruled out; re-judge with --rescore-from and an independent judge, Section 9.9)

| Arm | Avg score (1-5) | N scored | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- | --- |
| base_rag | 3.65 | 110/110 | 1/110 | 65/107 |
| base | 3.21 | 110/110 | 33/110 | 5/107 |
| finetuned_rag | 3.25 | 110/110 | 2/110 | 56/107 |
| finetuned | 2.40 | 110/110 | 8/110 | 10/107 |

Consistency (finetuned_rag, 15 questions rerun): 15/15 identical on rerun.

**Section 9.4 verdict:** finetuned_rag avg 3.25 vs base_rag avg 3.65. Fine-tuning (with RAG) does NOT beat RAG-only on this exam — per Section 9.4, fix the SFT data, not the hyperparameters.

## Lowest-scoring cases (for human review)

- [synthetic] "What are the subtypes for enchantments, lands, and planeswalkers?" — worst: finetuned (score 1, Incorrect subtypes for all card types, incorrect citation)
- [synthetic] "What are the characteristics of a face-down permanent on the battlefield?" — worst: finetuned (score 1, Incorrect, cites wrong rule.)
- [synthetic] "What is a dungeon card and how is it brought into the game?" — worst: base (score 1, Incorrect, does not address the question about dungeon cards.)
- [synthetic] "What is the role of an emperor in the Emperor variant of Magic: The Gathering?" — worst: finetuned (score 1, Incorrect and cites nothing relevant.)
- [synthetic] "What does "Undaunted" mean in Magic: The Gathering?" — worst: finetuned_rag (score 1, Incorrect, wrong definition)
- [synthetic] "What does the rule say about the restrictions and additional effects of mana produced by spells or a" — worst: base (score 1, Incorrect, does not address the question and cites irrelevant rules.)
- [synthetic] "What does "Proliferate" mean in Magic: The Gathering?" — worst: finetuned_rag (score 1, Incorrect, cites an incorrect rule number)
- [synthetic] "What does "Mutate" mean in Magic: The Gathering?" — worst: base_rag (score 1, Incorrect definition, wrong rule number)
- [synthetic] "What is a scheme card and how does it function in the game?" — worst: base (score 1, Incorrect definition, wrong rule number, no relevant rules cited)
- [synthetic] "What is a kindred card and how does it function in the game?" — worst: finetuned (score 1, Incorrect and contradicts the reference. Mentions a non-existent kindred number and supertype, and cites an incorrect rule number 702.116, which does not exist.)
