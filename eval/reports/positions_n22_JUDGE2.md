# Position Evaluation Report

22 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

| Arm | Blunder rate | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- |
| base_open | 45% (n=22) | 2.96 | 100% | 2.0 | 68% | 5% |
| base_closed | 82% (n=22) | 2.61 | 100% | 2.2 | 82% | 5% |
| base_cards_open | 50% (n=22) | 3.06 | 100% | 1.7 | 64% | 0% |
| ft_cards_open | 64% (n=22) | 2.89 | 77% | 4.7 | 36% | 5% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 77% (need ≥90%); worst closed-arm legality 82% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** Blunder rate spans 45%–82% (spread 36%). A spread near zero means the positions are not separating the arms, and more positions will not fix that.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_open` at 47% over 17 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open | ft_cards_open |
| --- | --- | --- | --- | --- | --- |
| basic | 8 | 12% | 88% | 75% | 38% |
| intermediate | 9 | 78% | 78% | 44% | 78% |
| advanced | 5 | 40% | 80% | 20% | 80% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open | ft_cards_open |
| --- | --- | --- | --- | --- | --- |
| blocking | 4 | 0% | 100% | 50% | 25% |
| combat math | 6 | 33% | 50% | 17% | 67% |
| land sequencing | 3 | 67% | 67% | 100% | 67% |
| mulligan | 2 | 100% | 100% | 100% | 50% |
| race vs stabilize | 2 | 100% | 100% | 50% | 50% |
| removal timing | 3 | 33% | 100% | 33% | 100% |
| trigger ordering | 2 | 50% | 100% | 50% | 100% |
