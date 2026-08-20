# Position Evaluation Report

24 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen2.5-32B-Instruct-4bit`

| Arm | Blunder rate | Errors/answer | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_open | 58% (n=24) | 1.58 | 2.08 | 100% | 2.3 | 71% | 0% |
| base_closed | 46% (n=24) | 1.17 | 3.04 | 100% | 2.5 | 75% | 0% |
| base_cards_open | 71% (n=24) | 1.83 | 2.04 | 100% | 1.7 | 62% | 0% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 100% (need ≥90%); worst closed-arm legality 75% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** **16/24 positions (67%) separate the arms** by ≥0.5 on the 1–5 correctness scale (need ≥50%). Compare the rules gold set at 74% (Section 21.11).

> Blunder-rate spread across arms is 25% (46%–71%), reported as a diagnostic rather than as the gate. Section 21.17: the binary call separates the arms on roughly half as many positions as correctness does, and averaging it per arm before comparing cancels what survives.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_closed` at 42% over 19 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| basic | 8 | 62% | 12% | 62% |
| intermediate | 11 | 64% | 64% | 73% |
| advanced | 5 | 40% | 60% | 80% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| blocking | 5 | 80% | 40% | 80% |
| combat math | 6 | 83% | 50% | 67% |
| land sequencing | 3 | 100% | 67% | 100% |
| mulligan | 2 | 0% | 100% | 50% |
| race vs stabilize | 2 | 50% | 50% | 50% |
| removal timing | 4 | 25% | 25% | 50% |
| trigger ordering | 2 | 0% | 0% | 100% |

> **One judge only.** Section 16.12 measured Gates 2 and 3 BOTH reversing between judges on byte-identical answers, so a gate verdict from a single judge is a statement about the judge. Re-run with `--second-judge mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.

