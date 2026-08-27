# Position Evaluation Report

34 positions from `data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

> **5/170 arm-position pairs went unjudged.** The judge carried roughly 13,447 characters of candidate text per batched call (~3,362 tokens) across 5 arms. If that is small against `--judge-max-tokens 2000`, the failure is in what the judge PRODUCES, not what it reads — a reasoning judge spends its budget thinking before it emits JSON.


| Arm | Blunder rate | Errors/answer | Correctness | Parsed ok | Plays/answer | Valid turn | All legal | Legal plays | Did nothing | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_open | 94% (n=33) | 5.91 | 2.31 | 94% | 23.2 | 32% | 53% | 82% (647/789) | 21% | 6% |
| base_closed | 58% (n=33) | 3.39 | 3.81 | 100% | 2.8 | 71% | 74% | 85% (81/95) | 3% | 3% |
| base_cards_open | 88% (n=33) | 6.03 | 2.42 | 100% | 11.3 | 24% | 41% | 65% (251/384) | 15% | 12% |
| ft_cards_open | 97% (n=33) | 6.36 | 1.99 | 79% | 7.7 | 18% | 26% | 83% (219/263) | 6% | 15% |
| base_open_think | 88% (n=33) | 6.73 | 2.38 | 94% | 13.6 | 26% | 38% | 89% (412/463) | 12% | 3% |

> **`base_open` declines to play on 21% of positions.** Its only action is `PASS`, which always matches `legal_actions` (16.13), so it scores legal; and doing nothing commits none of the enumerated `common_errors`, which are strategies. **Both gates are blind to it.** Read the blunder rate of any arm with a high figure here as a statement about the rubric, not the play (Section 21.58).


- **32/170 answers declare a phase that disagrees with the board (19%); 40/170 declare a tap the board could not produce (24%).** Raw problem counts are 1034 and 6665, but one looping answer contributes many, so the per-answer rate is the one to read. Both are checked against the position and the oracle text, never the judge. A wrong phase is the one error the enumerated-`legal_actions` check cannot see — a legal action taken under a wrong belief looks identical to a correct one (Section 21.61).


- **49/170 answers declare taps that do not pay for what they cast (29%); 12/170 cast a permanent already on the battlefield (7%).** Neither is visible to `legal_actions`: an over-tapped payment names only legal taps, and a spell already in play is absent from the list for a reason the list cannot state. The payment check is silent on answers that declare no taps, so it reports nothing on runs made before the verbose grammar rather than crediting them (Section 21.66).


- **11/170 answers cast a CREATURE spell with a TARGET (6%).** A creature spell does not target on cast; the model is treating it like removal. Four reviewer notes name this independently, and it is checked against oracle text rather than a judge. It has **no rubric entry** — it is convicted only as entry 3, for the wrong reason, and the one reviewer who met it ticked `not_covered`. Reported as a diagnostic pending a decision on an eighth `PROTOCOL_ERRORS` entry, which would invalidate every collected v4 verdict (Section 21.80).


- **18/170 answers declare mana and then spend none of it. The mana empties at end of step, so nothing is held up by it. `PROTOCOL_ERRORS` entry 6 is over-tapping and does NOT cover this — its wording presumes spells were cast, and it fires on 0 of these (11%).** Parser-decided, named in reviewer notes, and with no rubric entry (Section 21.83).


- **38/170 answers make a play without ever declaring a PHASE. `phase_problems` checks a declaration against the board and stays silent when there is none, which was right when the grammar merely allowed the line and wrong since 21.60 made it required (22%).** Parser-decided, named in reviewer notes, and with no rubric entry (Section 21.83).


- **Protocol errors, checked by parser: precision 31% (80/255), recall 43% (80/185).** These six rubric entries are decidable from the board, so every charge the judge makes against them is confirmed or refuted mechanically — no human, no second judge. 114 checks could not be decided here and are excluded rather than counted as the judge being wrong (Section 21.70).


## Blunder rate, decomposed

| Arm | n | strategy (judge) | protocol (judge) | protocol (**parser**) | headline |
| --- | --- | --- | --- | --- | --- |
| base_open | 33 | 39% | 91% | **85%** | 94% |
| base_closed | 33 | 27% | 52% | **88%** | 58% |
| base_cards_open | 33 | 39% | 85% | **88%** | 88% |
| ft_cards_open | 33 | 42% | 94% | **97%** | 97% |
| base_open_think | 33 | 48% | 88% | **100%** | 88% |

> **The headline mixes these two rubrics and adding the second moved it silently.** `errors_made` is the whole judge verdict, and since Section 21.70 the judge is given `common_errors + PROTOCOL_ERRORS` — so blunder rate went from *committed a listed strategy error* to *...or the judge thinks it committed a protocol one*, with no rename. **A run made before 21.70 is not comparable to one made after on this number.**

> **And the two halves can rank the arms differently**, so the mixed number is not a tie-break between them. An answer that declines to play commits no listed *strategy* while committing protocol entry 1 — the blindness 21.58 documented and `PROTOCOL_ERRORS` was added to close. Read the **parser** column for the protocol half: 5 of its 7 classes are 100% decidable from the board, and the judge runs 37% precision on the same checks (21.74). Gate 3 is deliberately NOT redefined here — which number it should read is B3's call, the same as `only_pass`.


## Turn scenarios

| Scenario | Steps | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| `turn-payment-combat-0001` | 2 | 0/2 steps | 1/2 steps | 1/2 steps | 0/2 steps | 1/2 steps |

> A turn counts as valid only when **every** step of it is. Steps are scored independently and the board advances on the REFERENCE line, so a mistake at step 1 never makes step 2 unanswerable — what this does not test is recovery from one's own mistake, which needs a rules engine (Section 21.71).


- **10 credited key points name an action the answer never took** — 10/233 of all credited points (4%), 10/35 of those that state an action (29%). The action list comes from the parser, which never sees the judge, so this is checkable rather than a second opinion. Correctness above is unchanged — this sits beside it (Section 21.56).


## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 79% (need ≥90%); worst closed-arm legality 74% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** **31/33 positions (94%) separate the arms** by ≥0.5 on the 1–5 correctness scale (need ≥50%). Compare the rules gold set at 74% (Section 21.11).

> Blunder-rate spread across arms is 39% (58%–97%), reported as a diagnostic rather than as the gate. Section 21.17: the binary call separates the arms on roughly half as many positions as correctness does, and averaging it per arm before comparing cancels what survives.

> 1 position(s) had an unjudged arm and are excluded from the count above rather than scored as ties.

**Gate 3 — blunder rate ≤25% on basic+intermediate: FAIL.** Best arm `base_closed` at 56% over 28 positions.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| basic | 10 | 100% | 56% | 78% | 100% | 100% |
| intermediate | 18 | 94% | 56% | 89% | 94% | 78% |
| advanced | 6 | 83% | 67% | 100% | 100% | 100% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| blocking | 5 | 100% | 100% | 100% | 100% | 100% |
| closing the turn | 2 | 0% | 0% | 50% | 100% | 100% |
| combat math | 6 | 100% | 50% | 100% | 100% | 100% |
| land sequencing | 3 | 100% | 100% | 100% | 100% | 100% |
| mulligan | 2 | 100% | 50% | 100% | 100% | 100% |
| payment | 6 | 100% | 50% | 83% | 83% | 50% |
| race vs stabilize | 2 | 100% | 50% | 100% | 100% | 100% |
| removal timing | 4 | 100% | 50% | 50% | 100% | 75% |
| trigger ordering | 2 | 100% | 50% | 100% | 100% | 100% |
| turn: payment then combat | 2 | 100% | 50% | 100% | 100% | 100% |

5 arm-position pairs went unjudged (judge returned unparseable JSON) and are excluded rather than counted as clean.

