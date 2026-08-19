# SFT audit — `data/datasets/verified`

Checked against 568 eval questions from `gold_questions.jsonl`, `gold_questions_eval.jsonl`, `reddit_card_questions.jsonl`, `reddit_questions.jsonl`, `rules_questions.jsonl`.

Excluded as a **source pool, not an eval set**: `rulesguru_candidates.jsonl`. The training set is drawn from it by design. The consequence is real and worth stating: a model trained here can never be evaluated on that pool — only on the records promoted into the gold set.

## Contamination

**No exact matches** across 1112 training lines x 568 eval questions (brute force, no pruning).

No pair reaches 75% token overlap with an eval question.

## Targets

| Split | Lines | Refusal-shaped | Cites a rule inline | Empty |
| --- | --- | --- | --- | --- |
| train | 1001 | 1 (0.1%) | 1001 (100%) | 0 |
| valid | 111 | 0 (0.0%) | 111 (100%) | 0 |

> Section 21.6 measured **18%** refusals in the synthetic set and **0.1%** in the verified one. A refusal target teaches the model to decline.

## Duplicate targets (train)

- **4 distinct targets repeat**, accounting for 5 redundant lines (0.5% of the split).
- 22 pairs above 75% token overlap.

> A repeated target is trained on once per copy, so the effective epoch count over that content is higher than the configured one — the same arithmetic error as a config comment claiming an epoch it never ran.

## Target length (train, characters)

- min 33 · p25 167 · median 242 · p75 342 · max 2461
- mean 277

- 45 targets (4%) are under 80 characters. The fine-tuned arm produces the shortest answers in every run; a set of one-liners is one explanation for that.

