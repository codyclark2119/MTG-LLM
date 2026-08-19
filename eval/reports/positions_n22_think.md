# Position Evaluation Report

22 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen2.5-7B-Instruct-4bit`  (**same model as the base arms** — self-preference bias not ruled out)

| Arm | Blunder rate | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- |
| base_open | 73% (n=22) | 1.94 | 100% | 2.4 | 68% | 0% |
| base_closed | 86% (n=22) | 2.33 | 100% | 2.5 | 73% | 0% |
| base_cards_open | 64% (n=22) | 2.07 | 100% | 1.7 | 64% | 0% |
| ft_cards_open | 73% (n=22) | 2.32 | 77% | 10.7 | 36% | 0% |
| base_open_think | 86% (n=22) | 2.19 | 100% | 1.8 | 23% | 5% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 77% (need ≥90%); worst closed-arm legality 73% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** Blunder rate spans 64%–86% (spread 23%). A spread near zero means the positions are not separating the arms, and more positions will not fix that.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_cards_open` at 65% over 17 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| basic | 8 | 75% | 88% | 50% | 62% | 62% |
| intermediate | 9 | 67% | 89% | 78% | 78% | 100% |
| advanced | 5 | 80% | 80% | 60% | 80% | 100% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| blocking | 4 | 50% | 75% | 25% | 25% | 75% |
| combat math | 6 | 83% | 67% | 67% | 83% | 100% |
| land sequencing | 3 | 100% | 100% | 100% | 100% | 100% |
| mulligan | 2 | 0% | 100% | 50% | 50% | 50% |
| race vs stabilize | 2 | 50% | 100% | 50% | 100% | 100% |
| removal timing | 3 | 100% | 100% | 67% | 67% | 67% |
| trigger ordering | 2 | 100% | 100% | 100% | 100% | 100% |

> **The judge is the same model as the base arms**, so self-preference bias is not ruled out (Section 9.9).

