# Position Evaluation Report

22 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

| Arm | Blunder rate | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- |
| base_open | 40% (n=20) | 3.34 | 100% | 2.4 | 68% | 0% |
| base_closed | 60% (n=20) | 3.22 | 100% | 2.5 | 73% | 0% |
| base_cards_open | 50% (n=20) | 2.85 | 100% | 1.7 | 64% | 0% |
| ft_cards_open | 70% (n=20) | 3.01 | 77% | 10.7 | 36% | 0% |
| base_open_think | 90% (n=20) | 2.95 | 100% | 1.8 | 23% | 5% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 77% (need ≥90%); worst closed-arm legality 73% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** Blunder rate spans 40%–90% (spread 50%). A spread near zero means the positions are not separating the arms, and more positions will not fix that.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_open` at 27% over 17 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| basic | 8 | 29% | 57% | 14% | 57% | 100% |
| intermediate | 9 | 25% | 62% | 62% | 88% | 88% |
| advanced | 5 | 80% | 60% | 80% | 60% | 80% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| blocking | 4 | 50% | 100% | 25% | 75% | 100% |
| combat math | 6 | 33% | 33% | 50% | 67% | 83% |
| land sequencing | 3 | 100% | 0% | 100% | 100% | 100% |
| mulligan | 2 | 50% | 100% | 50% | 50% | 100% |
| race vs stabilize | 2 | 0% | 50% | 50% | 100% | 100% |
| removal timing | 3 | 33% | 67% | 67% | 67% | 67% |
| trigger ordering | 2 | 50% | 50% | 50% | 50% | 100% |

10 arm-position pairs went unjudged (judge returned unparseable JSON) and are excluded rather than counted as clean.

