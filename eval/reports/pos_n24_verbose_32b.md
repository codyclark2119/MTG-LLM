# Position Evaluation Report

24 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen2.5-32B-Instruct-4bit`

| Arm | Blunder rate | Errors/answer | Correctness | Parsed ok | Actions/answer | All legal | Did nothing | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_open | 75% (n=24) | 2.00 | 1.46 | 88% | 14.0 | 46% | 21% | 25% |
| base_closed | 50% (n=24) | 1.33 | 3.00 | 100% | 5.2 | 67% | 8% | 17% |
| base_cards_open | 71% (n=24) | 2.04 | 1.83 | 100% | 13.6 | 33% | 12% | 33% |

> **`base_open` declines to play on 21% of positions.** Its only action is `PASS`, which always matches `legal_actions` (16.13), so it scores legal; and doing nothing commits none of the enumerated `common_errors`, which are strategies. **Both gates are blind to it.** Read the blunder rate of any arm with a high figure here as a statement about the rubric, not the play (Section 21.58).


- **72 phase declarations disagree with the board; 181 declared taps the board could not produce.** Both are checked against the position and the oracle text, never the judge. A wrong phase is the one error the enumerated-`legal_actions` check cannot see — a legal action taken under a wrong belief looks identical to a correct one (Section 21.61).


- **3 credited key points name an action the answer never took** — 3/77 of all credited points (4%), 3/12 of those that state an action (25%). The action list comes from the parser, which never sees the judge, so this is checkable rather than a second opinion. Correctness above is unchanged — this sits beside it (Section 21.56).


## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 88% (need ≥90%); worst closed-arm legality 67% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** **16/24 positions (67%) separate the arms** by ≥0.5 on the 1–5 correctness scale (need ≥50%). Compare the rules gold set at 74% (Section 21.11).

> Blunder-rate spread across arms is 25% (50%–75%), reported as a diagnostic rather than as the gate. Section 21.17: the binary call separates the arms on roughly half as many positions as correctness does, and averaging it per arm before comparing cancels what survives.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_closed` at 47% over 19 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| basic | 8 | 75% | 12% | 62% |
| intermediate | 11 | 82% | 73% | 73% |
| advanced | 5 | 60% | 60% | 80% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| blocking | 5 | 60% | 40% | 80% |
| combat math | 6 | 100% | 67% | 83% |
| land sequencing | 3 | 67% | 67% | 100% |
| mulligan | 2 | 100% | 50% | 50% |
| race vs stabilize | 2 | 50% | 50% | 100% |
| removal timing | 4 | 75% | 50% | 25% |
| trigger ordering | 2 | 50% | 0% | 50% |

> **One judge only.** Section 16.12 measured Gates 2 and 3 BOTH reversing between judges on byte-identical answers, so a gate verdict from a single judge is a statement about the judge. Re-run with `--second-judge mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.

