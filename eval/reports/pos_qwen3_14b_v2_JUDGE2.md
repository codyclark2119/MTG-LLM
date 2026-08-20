# Position Evaluation Report

24 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen3-14B-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

> **6/72 arm-position pairs went unjudged.** The judge carried roughly 709 characters of candidate text per batched call (~177 tokens) across 3 arms. If that is small against `--judge-max-tokens 1200`, the failure is in what the judge PRODUCES, not what it reads — a reasoning judge spends its budget thinking before it emits JSON.


| Arm | Blunder rate | Errors/answer | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_open | 55% (n=22) | 1.09 | 3.21 | 100% | 2.7 | 71% | 0% |
| base_closed | 68% (n=22) | 1.23 | 3.11 | 92% | 2.5 | 62% | 0% |
| base_cards_open | 45% (n=22) | 0.82 | 3.52 | 100% | 3.8 | 67% | 0% |

## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 92% (need ≥90%); worst closed-arm legality 62% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** **15/22 positions (68%) separate the arms** by ≥0.5 on the 1–5 correctness scale (need ≥50%). Compare the rules gold set at 74% (Section 21.11).

> Blunder-rate spread across arms is 23% (45%–68%), reported as a diagnostic rather than as the gate. Section 21.17: the binary call separates the arms on roughly half as many positions as correctness does, and averaging it per arm before comparing cancels what survives.

> 2 position(s) had an unjudged arm and are excluded from the count above rather than scored as ties.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_cards_open` at 44% over 19 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| basic | 8 | 62% | 75% | 50% |
| intermediate | 11 | 70% | 70% | 40% |
| advanced | 5 | 0% | 50% | 50% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| blocking | 5 | 60% | 60% | 20% |
| combat math | 6 | 50% | 67% | 67% |
| land sequencing | 3 | 50% | 100% | 100% |
| mulligan | 2 | 50% | 50% | 50% |
| race vs stabilize | 2 | 50% | 50% | 0% |
| removal timing | 4 | 75% | 75% | 50% |
| trigger ordering | 2 | 0% | 100% | 0% |

6 arm-position pairs went unjudged (judge returned unparseable JSON) and are excluded rather than counted as clean.


> **One judge only.** Section 16.12 measured Gates 2 and 3 BOTH reversing between judges on byte-identical answers, so a gate verdict from a single judge is a statement about the judge. Re-run with `--second-judge mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.

