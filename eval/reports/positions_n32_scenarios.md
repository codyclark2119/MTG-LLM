# Position Evaluation Report

34 positions from `data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen2.5-32B-Instruct-4bit`

| Arm | Blunder rate | Errors/answer | Correctness | Parsed ok | Plays/answer | Valid turn | All legal | Legal plays | Did nothing | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_open | 85% (n=34) | 1.76 | 1.74 | 94% | 23.2 | 32% | 53% | 82% (647/789) | 21% | 6% |
| base_closed | 50% (n=34) | 1.41 | 3.22 | 100% | 2.8 | 71% | 74% | 85% (81/95) | 3% | 3% |
| base_cards_open | 85% (n=34) | 2.44 | 1.68 | 100% | 11.3 | 24% | 41% | 65% (251/384) | 15% | 12% |
| ft_cards_open | 97% (n=34) | 2.91 | 1.63 | 79% | 7.7 | 18% | 26% | 83% (219/263) | 6% | 15% |
| base_open_think | 91% (n=34) | 2.32 | 2.03 | 94% | 13.6 | 26% | 38% | 89% (412/463) | 12% | 3% |

> **`base_open` declines to play on 21% of positions.** Its only action is `PASS`, which always matches `legal_actions` (16.13), so it scores legal; and doing nothing commits none of the enumerated `common_errors`, which are strategies. **Both gates are blind to it.** Read the blunder rate of any arm with a high figure here as a statement about the rubric, not the play (Section 21.58).


- **32/170 answers declare a phase that disagrees with the board (19%); 40/170 declare a tap the board could not produce (24%).** Raw problem counts are 1034 and 6665, but one looping answer contributes many, so the per-answer rate is the one to read. Both are checked against the position and the oracle text, never the judge. A wrong phase is the one error the enumerated-`legal_actions` check cannot see — a legal action taken under a wrong belief looks identical to a correct one (Section 21.61).


- **49/170 answers declare taps that do not pay for what they cast (29%); 12/170 cast a permanent already on the battlefield (7%).** Neither is visible to `legal_actions`: an over-tapped payment names only legal taps, and a spell already in play is absent from the list for a reason the list cannot state. The payment check is silent on answers that declare no taps, so it reports nothing on runs made before the verbose grammar rather than crediting them (Section 21.66).


- **11/170 answers cast a CREATURE spell with a TARGET (6%).** A creature spell does not target on cast; the model is treating it like removal. Four reviewer notes name this independently, and it is checked against oracle text rather than a judge. It has **no rubric entry** — it is convicted only as entry 3, for the wrong reason, and the one reviewer who met it ticked `not_covered`. Reported as a diagnostic pending a decision on an eighth `PROTOCOL_ERRORS` entry, which would invalidate every collected v4 verdict (Section 21.80).


- **18/170 answers declare mana and then spend none of it. The mana empties at end of step, so nothing is held up by it. `PROTOCOL_ERRORS` entry 6 is over-tapping and does NOT cover this — its wording presumes spells were cast, and it fires on 0 of these (11%).** Parser-decided, named in reviewer notes, and with no rubric entry (Section 21.83).


- **38/170 answers make a play without ever declaring a PHASE. `phase_problems` checks a declaration against the board and stays silent when there is none, which was right when the grammar merely allowed the line and wrong since 21.60 made it required (22%).** Parser-decided, named in reviewer notes, and with no rubric entry (Section 21.83).


- **Protocol errors, checked by parser: precision 36% (80/221), recall 29% (80/274).** These six rubric entries are decidable from the board, so every charge the judge makes against them is confirmed or refuted mechanically — no human, no second judge. 156 checks could not be decided here and are excluded rather than counted as the judge being wrong (Section 21.70).


## Blunder rate, decomposed

| Arm | n | strategy (judge) | protocol (judge) | protocol (**parser**) | headline |
| --- | --- | --- | --- | --- | --- |
| base_open | 34 | 29% | 76% | **85%** | 85% |
| base_closed | 34 | 35% | 32% | **88%** | 50% |
| base_cards_open | 34 | 35% | 82% | **88%** | 85% |
| ft_cards_open | 34 | 29% | 88% | **97%** | 97% |
| base_open_think | 34 | 44% | 76% | **100%** | 91% |

> **The headline mixes these two rubrics and adding the second moved it silently.** `errors_made` is the whole judge verdict, and since Section 21.70 the judge is given `common_errors + PROTOCOL_ERRORS` — so blunder rate went from *committed a listed strategy error* to *...or the judge thinks it committed a protocol one*, with no rename. **A run made before 21.70 is not comparable to one made after on this number.**

> **And the two halves can rank the arms differently**, so the mixed number is not a tie-break between them. An answer that declines to play commits no listed *strategy* while committing protocol entry 1 — the blindness 21.58 documented and `PROTOCOL_ERRORS` was added to close. Read the **parser** column for the protocol half: 5 of its 7 classes are 100% decidable from the board, and the judge runs 37% precision on the same checks (21.74). Gate 3 is deliberately NOT redefined here — which number it should read is B3's call, the same as `only_pass`.


## Turn scenarios

| Scenario | Steps | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| `turn-payment-combat-0001` | 2 | 0/2 steps | 1/2 steps | 1/2 steps | 0/2 steps | 1/2 steps |

> A turn counts as valid only when **every** step of it is. Steps are scored independently and the board advances on the REFERENCE line, so a mistake at step 1 never makes step 2 unanswerable — what this does not test is recovery from one's own mistake, which needs a rules engine (Section 21.71).


- **1 credited key points name an action the answer never took** — 1/155 of all credited points (1%), 1/16 of those that state an action (6%). The action list comes from the parser, which never sees the judge, so this is checkable rather than a second opinion. Correctness above is unchanged — this sits beside it (Section 21.56).


## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 79% (need ≥90%); worst closed-arm legality 74% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** **27/34 positions (79%) separate the arms** by ≥0.5 on the 1–5 correctness scale (need ≥50%). Compare the rules gold set at 74% (Section 21.11).

> Blunder-rate spread across arms is 47% (50%–97%), reported as a diagnostic rather than as the gate. Section 21.17: the binary call separates the arms on roughly half as many positions as correctness does, and averaging it per arm before comparing cancels what survives.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_closed` at 54% over 28 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| basic | 10 | 100% | 50% | 70% | 100% | 90% |
| intermediate | 18 | 78% | 56% | 89% | 94% | 89% |
| advanced | 6 | 83% | 33% | 100% | 100% | 100% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| blocking | 5 | 100% | 40% | 100% | 100% | 100% |
| closing the turn | 2 | 0% | 0% | 50% | 100% | 100% |
| combat math | 6 | 100% | 50% | 83% | 100% | 100% |
| land sequencing | 3 | 67% | 100% | 100% | 100% | 100% |
| mulligan | 2 | 100% | 50% | 100% | 100% | 100% |
| payment | 6 | 83% | 67% | 83% | 100% | 50% |
| race vs stabilize | 2 | 50% | 50% | 100% | 100% | 100% |
| removal timing | 4 | 100% | 25% | 50% | 75% | 100% |
| trigger ordering | 2 | 100% | 0% | 100% | 100% | 100% |
| turn: payment then combat | 2 | 100% | 100% | 100% | 100% | 100% |
