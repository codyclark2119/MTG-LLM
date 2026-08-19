# Position Evaluation Report

22 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen3-14B-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

| Arm | Blunder rate | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- |
| base_open | 41% (n=22) | 4.01 | 82% | 2.0 | 55% | 0% |
| base_closed | 36% (n=22) | 4.01 | 68% | 1.6 | 50% | 0% |
| base_cards_open | 50% (n=22) | 3.79 | 55% | 1.1 | 50% | 0% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 55% (need ≥90%); worst closed-arm legality 50% (need ≥95%).

**Gate 2 — the eval discriminates: FAIL.** Blunder rate spans 36%–50% (spread 14%). A spread near zero means the positions are not separating the arms, and more positions will not fix that.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_closed` at 41% over 17 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| basic | 8 | 62% | 62% | 25% |
| intermediate | 9 | 33% | 22% | 67% |
| advanced | 5 | 20% | 20% | 60% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| blocking | 4 | 75% | 50% | 25% |
| combat math | 6 | 17% | 50% | 33% |
| land sequencing | 3 | 0% | 33% | 100% |
| mulligan | 2 | 0% | 0% | 50% |
| race vs stabilize | 2 | 50% | 0% | 100% |
| removal timing | 3 | 100% | 67% | 33% |
| trigger ordering | 2 | 50% | 0% | 50% |
