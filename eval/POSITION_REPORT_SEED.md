# Position Evaluation Report

8 positions from `data/gold/positions_seed.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen2.5-7B-Instruct-4bit`  (**same model as the base arms** — self-preference bias not ruled out)

> **8/8 positions are machine-drafted seed fixtures.** They verify the pipeline; they are not Gate 3 evidence. Section 14.6 measured hand-authored rubrics beating machine drafts (inter-judge r +0.30 → +0.62).


| Arm | Blunder rate | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- |
| base_open | 75% (n=8) | 2.21 | 100% | 2.0 | 62% | 0% |
| base_closed | 75% (n=8) | 2.33 | 100% | 1.9 | 88% | 0% |
| base_cards_open | 75% (n=8) | 2.08 | 100% | 1.6 | 62% | 0% |
| ft_cards_open | 75% (n=8) | 2.46 | 100% | 2.4 | 38% | 0% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 100% (need ≥90%); worst closed-arm legality 88% (need ≥95%).

**Gate 2 — the eval discriminates: FAIL.** Blunder rate spans 75%–75% (spread 0%). A spread near zero means the positions are not separating the arms, and more positions will not fix that.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_open` at 83% over 6 positions. Seed fixtures cannot settle this gate — it needs judge-authored positions.

## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open | ft_cards_open |
| --- | --- | --- | --- | --- | --- |
| blocking | 1 | 100% | 100% | 100% | 100% |
| combat math | 2 | 50% | 50% | 50% | 50% |
| land sequencing | 1 | 100% | 100% | 100% | 100% |
| mulligan | 1 | 100% | 100% | 100% | 100% |
| race vs stabilize | 1 | 100% | 100% | 100% | 100% |
| removal timing | 2 | 50% | 50% | 50% | 50% |

> **Single judge, and it is the same model as the base arms.** Blunder rate here is one judge's opinion of whether a listed error was committed, and spot-checking the first run found a false positive (an answer that targeted the right creature was marked as committing the 'aimed it at the face' error). Section 9.9's two-judge protocol is the check: re-run with `--judge-model` pointed at an independent model before treating any blunder rate as settled.

