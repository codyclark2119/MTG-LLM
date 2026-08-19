# SFT audit — `data/datasets`

Checked against 568 eval questions from `gold_questions.jsonl`, `gold_questions_eval.jsonl`, `reddit_card_questions.jsonl`, `reddit_questions.jsonl`, `rules_questions.jsonl`.

Excluded as a **source pool, not an eval set**: `rulesguru_candidates.jsonl`. The training set is drawn from it by design. The consequence is real and worth stating: a model trained here can never be evaluated on that pool — only on the records promoted into the gold set.

## Contamination

**3 training questions are eval questions verbatim.**

| Split | Eval record | Question |
| --- | --- | --- |
| train | `gold_questions.jsonl:gloss-companion` | What does "Companion" mean in Magic: The Gathering?… |
| train | `gold_questions.jsonl:gloss-face-down` | What does "Face Down" mean in Magic: The Gathering?… |
| valid | `gold_questions.jsonl:gloss-mutating-creature-spell` | What does "Mutating Creature Spell" mean in Magic: The Gathering?… |

20 pairs above 75% token overlap — **for review, not an automatic failure.** RulesGuru re-randomizes card and player names, so the same ruling legitimately appears twice with different nouns.

| Overlap | Split | Nearest eval record | Question |
| --- | --- | --- | --- |
| 100% | valid | `gold_questions_eval.jsonl:gloss-mutating-creature-spell` | What does "Mutating Creature Spell" mean in Magic: The Gathering?… |
| 100% | train | `rules_questions.jsonl:?` | Player A casts a spell that reads 'All creatures get +1/+1 until end of turn' and Player B… |
| 100% | train | `rules_questions.jsonl:?` | Player A casts a spell that reads 'All creatures get +1/+1 until end of turn' and Player B… |
| 100% | train | `gold_questions_eval.jsonl:gloss-face-down` | What does "Face Down" mean in Magic: The Gathering?… |
| 100% | train | `gold_questions_eval.jsonl:gloss-companion` | What does "Companion" mean in Magic: The Gathering?… |
| 80% | valid | `rules_questions.jsonl:?` | What does "Speed" mean in Magic: The Gathering?… |
| 80% | valid | `rules_questions.jsonl:?` | What does "Permanent Spell" mean in Magic: The Gathering?… |
| 80% | valid | `gold_questions_eval.jsonl:gloss-companion` | What does "Doctor’s Companion" mean in Magic: The Gathering?… |
| 80% | train | `rules_questions.jsonl:?` | What does "Special Action" mean in Magic: The Gathering?… |
| 80% | train | `rules_questions.jsonl:?` | What does "Permanent Card" mean in Magic: The Gathering?… |
| 80% | train | `rules_questions.jsonl:?` | What does "Merged Permanent" mean in Magic: The Gathering?… |
| 80% | train | `rules_questions.jsonl:?` | What does "Meld Cards" mean in Magic: The Gathering?… |
| 80% | train | `rules_questions.jsonl:?` | What does "Double Strike" mean in Magic: The Gathering?… |
| 80% | train | `rules_questions.jsonl:?` | What does "Double Agenda" mean in Magic: The Gathering?… |
| 80% | train | `rules_questions.jsonl:?` | What does "Commander" mean in Magic: The Gathering?… |
| 80% | train | `gold_questions_eval.jsonl:gloss-choose-a-background` | What does "Background" mean in Magic: The Gathering?… |
| 75% | train | `rules_questions.jsonl:?` | When does the active player get priority during the end step?… |
| 75% | train | `rules_questions.jsonl:?` | When does the active player get priority during the end step?… |
| 75% | train | `rules_questions.jsonl:?` | What is a permanent in Magic: The Gathering?… |
| 75% | train | `rules_questions.jsonl:?` | What is a permanent in Magic: The Gathering?… |

## Targets

| Split | Lines | Refusal-shaped | Cites a rule inline | Empty |
| --- | --- | --- | --- | --- |
| train | 2644 | 465 (17.6%) | 2288 (87%) | 0 |
| valid | 327 | 54 (16.5%) | 287 (88%) | 0 |

> Section 21.6 measured **18%** refusals in the synthetic set and **0.1%** in the verified one. A refusal target teaches the model to decline.

## Duplicate targets (train)

- **1138 distinct targets repeat**, accounting for 1166 redundant lines (44.1% of the split).
- 1261 pairs above 75% token overlap.

> A repeated target is trained on once per copy, so the effective epoch count over that content is higher than the configured one — the same arithmetic error as a config comment claiming an epoch it never ran.

## Target length (train, characters)

- min 5 · p25 237 · median 504 · p75 1003 · max 1529
- mean 623

- 55 targets (2%) are under 80 characters. The fine-tuned arm produces the shortest answers in every run; a set of one-liners is one explanation for that.

