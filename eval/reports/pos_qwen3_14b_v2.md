# Position Evaluation Report

24 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen3-14B-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen3-14B-4bit`  (**same model as the base arms** — self-preference bias not ruled out)

| Arm | Blunder rate | Errors/answer | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_open | 29% (n=7) | 0.29 | 2.95 | 100% | 2.7 | 71% | 0% |
| base_closed | 14% (n=7) | 0.29 | 3.64 | 92% | 2.5 | 62% | 0% |
| base_cards_open | 29% (n=7) | 0.29 | 3.60 | 100% | 3.8 | 67% | 0% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 92% (need ≥90%); worst closed-arm legality 62% (need ≥95%).

**Gate 2 — the eval discriminates: FAIL.** Blunder rate spans 14%–29% (spread 14%). A spread near zero means the positions are not separating the arms, and more positions will not fix that.

**Gate 3 — blunder rate ≤25% on basic+intermediate: PASS.** Best arm `base_closed` at 14% over 19 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| basic | 8 | 25% | 0% | 25% |
| intermediate | 11 | 33% | 33% | 33% |
| advanced | 5 | — | — | — |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| blocking | 5 | 0% | 0% | 0% |
| combat math | 6 | 67% | 33% | 67% |
| land sequencing | 3 | — | — | — |
| mulligan | 2 | 0% | 0% | 0% |
| race vs stabilize | 2 | — | — | — |
| removal timing | 4 | 0% | 0% | 0% |
| trigger ordering | 2 | — | — | — |

51 arm-position pairs went unjudged (judge returned unparseable JSON) and are excluded rather than counted as clean.


> **The judge is the same model as the base arms**, so self-preference bias is not ruled out (Section 9.9).

