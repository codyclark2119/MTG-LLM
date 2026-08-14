# Position Evaluation Report

8 positions from `data/gold/positions_seed.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

> **8/8 positions are machine-drafted seed fixtures.** They verify the pipeline; they are not Gate 3 evidence. Section 14.6 measured hand-authored rubrics beating machine drafts (inter-judge r +0.30 → +0.62).


| Arm | Blunder rate | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- |
| base_open | 50% (n=8) | 2.08 | 100% | 2.0 | 62% | 0% |
| base_closed | 50% (n=8) | 2.08 | 100% | 1.9 | 88% | 0% |
| base_cards_open | 25% (n=8) | 2.46 | 100% | 1.6 | 62% | 0% |
| ft_cards_open | 62% (n=8) | 2.33 | 100% | 2.4 | 38% | 0% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 100% (need ≥90%); worst closed-arm legality 88% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** Blunder rate spans 25%–62% (spread 38%). A spread near zero means the positions are not separating the arms, and more positions will not fix that.

**Gate 3 — blunder rate ≤25% on basic+intermediate: PASS.** Best arm `base_cards_open` at 17% over 6 positions. Seed fixtures cannot settle this gate — it needs judge-authored positions.

## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open | ft_cards_open |
| --- | --- | --- | --- | --- | --- |
| blocking | 1 | 0% | 0% | 0% | 100% |
| combat math | 2 | 0% | 0% | 0% | 0% |
| land sequencing | 1 | 100% | 100% | 100% | 100% |
| mulligan | 1 | 100% | 100% | 0% | 100% |
| race vs stabilize | 1 | 100% | 100% | 100% | 100% |
| removal timing | 2 | 50% | 50% | 0% | 50% |
