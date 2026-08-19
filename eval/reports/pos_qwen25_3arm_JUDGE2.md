# Position Evaluation Report

22 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

| Arm | Blunder rate | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- |
| base_open | 47% (n=19) | 3.26 | 100% | 2.4 | 68% | 0% |
| base_closed | 68% (n=19) | 2.95 | 100% | 2.5 | 73% | 0% |
| base_cards_open | 74% (n=19) | 2.44 | 100% | 1.7 | 64% | 0% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 100% (need ≥90%); worst closed-arm legality 73% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** Blunder rate spans 47%–74% (spread 26%). A spread near zero means the positions are not separating the arms, and more positions will not fix that.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_open` at 53% over 17 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| basic | 8 | 25% | 75% | 88% |
| intermediate | 9 | 86% | 71% | 57% |
| advanced | 5 | 25% | 50% | 75% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| blocking | 4 | 25% | 100% | 75% |
| combat math | 6 | 40% | 60% | 80% |
| land sequencing | 3 | 50% | 100% | 50% |
| mulligan | 2 | 100% | 100% | 50% |
| race vs stabilize | 2 | 100% | 50% | 50% |
| removal timing | 3 | 33% | 33% | 100% |
| trigger ordering | 2 | 0% | 0% | 100% |

9 arm-position pairs went unjudged (judge returned unparseable JSON) and are excluded rather than counted as clean.

