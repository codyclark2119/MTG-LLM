# Position Evaluation Report

26 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen2.5-32B-Instruct-4bit`

| Arm | Blunder rate | Errors/answer | Correctness | Parsed ok | Plays/answer | Valid turn | All legal | Legal plays | Did nothing | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_open | 96% (n=26) | 2.46 | 1.46 | 88% | 5.5 | 23% | 46% | 72% (103/144) | 23% | 8% |
| base_closed | 58% (n=26) | 1.54 | 3.42 | 100% | 2.7 | 58% | 69% | 90% (63/70) | 12% | 0% |
| base_cards_open | 92% (n=26) | 3.31 | 1.65 | 100% | 6.2 | 19% | 35% | 61% (99/161) | 12% | 15% |

> **`base_open` declines to play on 23% of positions.** Its only action is `PASS`, which always matches `legal_actions` (16.13), so it scores legal; and doing nothing commits none of the enumerated `common_errors`, which are strategies. **Both gates are blind to it.** Read the blunder rate of any arm with a high figure here as a statement about the rubric, not the play (Section 21.58).


- **20/78 answers declare a phase that disagrees with the board (26%); 21/78 declare a tap the board could not produce (27%).** Raw problem counts are 72 and 181, but one looping answer contributes many, so the per-answer rate is the one to read. Both are checked against the position and the oracle text, never the judge. A wrong phase is the one error the enumerated-`legal_actions` check cannot see — a legal action taken under a wrong belief looks identical to a correct one (Section 21.61).


- **25/78 answers declare taps that do not pay for what they cast (32%); 8/78 cast a permanent already on the battlefield (10%).** Neither is visible to `legal_actions`: an over-tapped payment names only legal taps, and a spell already in play is absent from the list for a reason the list cannot state. The payment check is silent on answers that declare no taps, so it reports nothing on runs made before the verbose grammar rather than crediting them (Section 21.66).


- **Protocol errors, checked by parser: precision 37% (47/126), recall 43% (47/110).** These six rubric entries are decidable from the board, so every charge the judge makes against them is confirmed or refuted mechanically — no human, no second judge. 62 checks could not be decided here and are excluded rather than counted as the judge being wrong (Section 21.70).


## Turn scenarios

| Scenario | Steps | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| `turn-payment-combat-0001` | 2 | 0/2 steps | 1/2 steps | 1/2 steps |

> A turn counts as valid only when **every** step of it is. Steps are scored independently and the board advances on the REFERENCE line, so a mistake at step 1 never makes step 2 unanswerable — what this does not test is recovery from one's own mistake, which needs a rules engine (Section 21.71).


- **3 credited key points name an action the answer never took** — 3/90 of all credited points (3%), 3/13 of those that state an action (23%). The action list comes from the parser, which never sees the judge, so this is checkable rather than a second opinion. Correctness above is unchanged — this sits beside it (Section 21.56).


## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 88% (need ≥90%); worst closed-arm legality 69% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** **17/26 positions (65%) separate the arms** by ≥0.5 on the 1–5 correctness scale (need ≥50%). Compare the rules gold set at 74% (Section 21.11).

> Blunder-rate spread across arms is 38% (58%–96%), reported as a diagnostic rather than as the gate. Section 21.17: the binary call separates the arms on roughly half as many positions as correctness does, and averaging it per arm before comparing cancels what survives.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_closed` at 57% over 21 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| basic | 8 | 100% | 50% | 88% |
| intermediate | 13 | 92% | 62% | 92% |
| advanced | 5 | 100% | 60% | 100% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open |
| --- | --- | --- | --- | --- |
| blocking | 5 | 100% | 60% | 100% |
| combat math | 6 | 100% | 67% | 100% |
| land sequencing | 3 | 67% | 100% | 100% |
| mulligan | 2 | 100% | 50% | 100% |
| race vs stabilize | 2 | 100% | 50% | 100% |
| removal timing | 4 | 100% | 25% | 50% |
| trigger ordering | 2 | 100% | 0% | 100% |
| turn: payment then combat | 2 | 100% | 100% | 100% |

> **One judge only.** Section 16.12 measured Gates 2 and 3 BOTH reversing between judges on byte-identical answers, so a gate verdict from a single judge is a statement about the judge. Re-run with `--second-judge mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.

