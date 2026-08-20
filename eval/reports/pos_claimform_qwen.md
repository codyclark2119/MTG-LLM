# Position Evaluation Report

22 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen2.5-7B-Instruct-4bit`  (**same model as the base arms** — self-preference bias not ruled out)

| Arm | Blunder rate | Errors/answer | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_open | 59% (n=22) | 1.73 | 2.85 | 100% | 2.4 | 68% | 0% |
| base_closed | 77% (n=22) | 2.18 | 3.80 | 100% | 2.5 | 73% | 0% |
| base_cards_open | 59% (n=22) | 1.86 | 2.95 | 100% | 1.7 | 64% | 0% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 100% (need ≥90%); worst closed-arm legality 73% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** **14/22 positions (64%) separate the arms** by ≥0.5 on the 1–5 correctness scale (need ≥50%). Compare the rules gold set at 74% (Section 21.11).

> Blunder-rate spread across arms is 18% (59%–77%), reported as a diagnostic rather than as the gate. Section 21.17: the binary call separates the arms on roughly half as many positions as correctness does, and averaging it per arm before comparing cancels what survives.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_cards_open` at 53% over 17 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| basic | 8 | 62% | 75% | 38% |
| intermediate | 9 | 67% | 78% | 67% |
| advanced | 5 | 40% | 80% | 80% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| blocking | 4 | 25% | 75% | 25% |
| combat math | 6 | 83% | 83% | 67% |
| land sequencing | 3 | 100% | 67% | 100% |
| mulligan | 2 | 50% | 100% | 0% |
| race vs stabilize | 2 | 50% | 100% | 50% |
| removal timing | 3 | 67% | 67% | 67% |
| trigger ordering | 2 | 0% | 50% | 100% |

> **The judge is the same model as the base arms**, so self-preference bias is not ruled out (Section 9.9).


> **One judge only.** Section 16.12 measured Gates 2 and 3 BOTH reversing between judges on byte-identical answers, so a gate verdict from a single judge is a statement about the judge. Re-run with `--second-judge mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.

