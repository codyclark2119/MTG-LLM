# Position Evaluation Report

24 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

| Arm | Blunder rate | Errors/answer | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_open | 38% (n=24) | 0.75 | 1.89 | 100% | 2.3 | 71% | 0% |
| base_closed | 38% (n=24) | 0.71 | 3.18 | 100% | 2.5 | 75% | 0% |
| base_cards_open | 33% (n=24) | 0.62 | 1.54 | 100% | 1.7 | 62% | 0% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 100% (need ≥90%); worst closed-arm legality 75% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** **16/24 positions (67%) separate the arms** by ≥0.5 on the 1–5 correctness scale (need ≥50%). Compare the rules gold set at 74% (Section 21.11).

> Blunder-rate spread across arms is 4% (33%–38%), reported as a diagnostic rather than as the gate. Section 21.17: the binary call separates the arms on roughly half as many positions as correctness does, and averaging it per arm before comparing cancels what survives.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_cards_open` at 26% over 19 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| basic | 8 | 25% | 38% | 12% |
| intermediate | 11 | 45% | 45% | 36% |
| advanced | 5 | 40% | 20% | 60% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| blocking | 5 | 20% | 60% | 0% |
| combat math | 6 | 67% | 33% | 50% |
| land sequencing | 3 | 100% | 67% | 100% |
| mulligan | 2 | 0% | 50% | 0% |
| race vs stabilize | 2 | 0% | 0% | 0% |
| removal timing | 4 | 25% | 25% | 25% |
| trigger ordering | 2 | 0% | 0% | 50% |

> **One judge only.** Section 16.12 measured Gates 2 and 3 BOTH reversing between judges on byte-identical answers, so a gate verdict from a single judge is a statement about the judge. Re-run with `--second-judge mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.

