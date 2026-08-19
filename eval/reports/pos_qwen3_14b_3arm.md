# Position Evaluation Report

22 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen3-14B-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen3-14B-4bit`  (**same model as the base arms** — self-preference bias not ruled out)

| Arm | Blunder rate | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- |
| base_open | 33% (n=9) | 4.00 | 82% | 2.0 | 55% | 0% |
| base_closed | 33% (n=9) | 4.17 | 68% | 1.6 | 50% | 0% |
| base_cards_open | 22% (n=9) | 4.44 | 55% | 1.1 | 50% | 0% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 55% (need ≥90%); worst closed-arm legality 50% (need ≥95%).

**Gate 2 — the eval discriminates: FAIL.** Blunder rate spans 22%–33% (spread 11%). A spread near zero means the positions are not separating the arms, and more positions will not fix that.

**Gate 3 — blunder rate ≤25% on basic+intermediate: PASS.** Best arm `base_closed` at 25% over 17 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| basic | 8 | 60% | 40% | 40% |
| intermediate | 9 | 0% | 0% | 0% |
| advanced | 5 | 0% | 100% | 0% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| blocking | 4 | 33% | 67% | 33% |
| combat math | 6 | 50% | 0% | 0% |
| land sequencing | 3 | 0% | 0% | 0% |
| mulligan | 2 | 0% | 0% | 0% |
| race vs stabilize | 2 | — | — | — |
| removal timing | 3 | 100% | 100% | 100% |
| trigger ordering | 2 | — | — | — |

39 arm-position pairs went unjudged (judge returned unparseable JSON) and are excluded rather than counted as clean.


> **The judge is the same model as the base arms**, so self-preference bias is not ruled out (Section 9.9).

