# Position Evaluation Report

22 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen2.5-7B-Instruct-4bit`  (**same model as the base arms** — self-preference bias not ruled out)

| Arm | Blunder rate | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- |
| base_open | 50% (n=22) | 2.62 | 100% | 2.0 | 68% | 5% |
| base_closed | 77% (n=22) | 2.55 | 100% | 2.2 | 82% | 5% |
| base_cards_open | 64% (n=22) | 2.14 | 100% | 1.7 | 64% | 0% |
| ft_cards_open | 82% (n=22) | 1.95 | 77% | 4.7 | 36% | 5% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 77% (need ≥90%); worst closed-arm legality 82% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** Blunder rate spans 50%–82% (spread 32%). A spread near zero means the positions are not separating the arms, and more positions will not fix that.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_open` at 47% over 17 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open | ft_cards_open |
| --- | --- | --- | --- | --- | --- |
| basic | 8 | 50% | 75% | 62% | 75% |
| intermediate | 9 | 44% | 78% | 56% | 89% |
| advanced | 5 | 60% | 80% | 80% | 80% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open | ft_cards_open |
| --- | --- | --- | --- | --- | --- |
| blocking | 4 | 25% | 75% | 25% | 25% |
| combat math | 6 | 83% | 67% | 83% | 100% |
| land sequencing | 3 | 100% | 100% | 100% | 100% |
| mulligan | 2 | 0% | 100% | 50% | 50% |
| race vs stabilize | 2 | 0% | 50% | 0% | 100% |
| removal timing | 3 | 33% | 67% | 67% | 100% |
| trigger ordering | 2 | 50% | 100% | 100% | 100% |

> **The judge is the same model as the base arms**, so self-preference bias is not ruled out (Section 9.9).

