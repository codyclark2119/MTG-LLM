# Position Evaluation Report

22 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen2.5-7B-Instruct-4bit`  (**same model as the base arms** — self-preference bias not ruled out)

| Arm | Blunder rate | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- |
| base_open | 73% (n=22) | 1.79 | 100% | 2.4 | 68% | 0% |
| base_closed | 82% (n=22) | 2.55 | 100% | 2.5 | 73% | 0% |
| base_cards_open | 73% (n=22) | 1.96 | 100% | 1.7 | 64% | 0% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 100% (need ≥90%); worst closed-arm legality 73% (need ≥95%).

**Gate 2 — the eval discriminates: FAIL.** Blunder rate spans 73%–82% (spread 9%). A spread near zero means the positions are not separating the arms, and more positions will not fix that.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_cards_open` at 71% over 17 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| basic | 8 | 75% | 75% | 62% |
| intermediate | 9 | 78% | 78% | 78% |
| advanced | 5 | 60% | 100% | 80% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| blocking | 4 | 50% | 100% | 25% |
| combat math | 6 | 83% | 83% | 67% |
| land sequencing | 3 | 100% | 100% | 100% |
| mulligan | 2 | 100% | 100% | 100% |
| race vs stabilize | 2 | 50% | 50% | 50% |
| removal timing | 3 | 67% | 33% | 100% |
| trigger ordering | 2 | 50% | 100% | 100% |

> **The judge is the same model as the base arms**, so self-preference bias is not ruled out (Section 9.9).

