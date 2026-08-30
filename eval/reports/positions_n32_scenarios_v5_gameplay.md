# Position Evaluation Report

34 positions from `/Users/codyclark/Documents/personal_code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v5-gameplay`
- judge: `mlx-community/Qwen2.5-32B-Instruct-4bit`

| Arm | Blunder rate | Errors/answer | Correctness | Parsed ok | Plays/answer | Valid turn | All legal | Legal plays | Did nothing | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_open | 94% (n=34) | 4.06 | 1.98 | 97% | 8.8 | 9% | 15% | 63% (187/298) | 3% | 3% |
| base_closed | 65% (n=34) | 2.47 | 2.97 | 100% | 3.9 | 53% | 53% | 88% (116/132) | 3% | 3% |
| base_cards_open | 97% (n=34) | 4.56 | 1.30 | 97% | 14.2 | 3% | 9% | 82% (396/483) | 3% | 21% |
| ft_cards_open | 100% (n=34) | 2.94 | 1.20 | 88% | 3.0 | 0% | 32% | 44% (45/103) | 32% | 9% |
| base_open_think | 88% (n=34) | 3.65 | 1.99 | 85% | 11.3 | 9% | 32% | 78% (300/383) | 18% | 21% |

> **`ft_cards_open` declines to play on 32% of positions.** Its only action is `PASS`, which always matches `legal_actions` (16.13), so it scores legal; and doing nothing commits none of the enumerated `common_errors`, which are strategies. **Both gates are blind to it.** Read the blunder rate of any arm with a high figure here as a statement about the rubric, not the play (Section 21.58).


- **48/170 answers declare a phase that disagrees with the board (28%); 49/170 declare a tap the board could not produce (29%).** Raw problem counts are 233 and 441, but one looping answer contributes many, so the per-answer rate is the one to read. Both are checked against the position and the oracle text, never the judge. A wrong phase is the one error the enumerated-`legal_actions` check cannot see — a legal action taken under a wrong belief looks identical to a correct one (Section 21.61).


- **40/170 answers declare taps that do not pay for what they cast (24%); 20/170 cast a permanent already on the battlefield (12%).** Neither is visible to `legal_actions`: an over-tapped payment names only legal taps, and a spell already in play is absent from the list for a reason the list cannot state. The payment check is silent on answers that declare no taps, so it reports nothing on runs made before the verbose grammar rather than crediting them (Section 21.66).


- **4/170 answers cast a CREATURE spell with a TARGET (2%).** A creature spell does not target on cast; the model is treating it like removal. Four reviewer notes name this independently, and it is checked against oracle text rather than a judge. It has **no rubric entry** — it is convicted only as entry 3, for the wrong reason, and the one reviewer who met it ticked `not_covered`. Reported as a diagnostic pending a decision on an eighth `PROTOCOL_ERRORS` entry, which would invalidate every collected v4 verdict (Section 21.80).


> **2/34 boards in this run are flagged for editing** (split x1, shorten x1). They are generated for, judged and counted like any other — a flag is metadata, not a filter — so every number here includes boards their author has already marked as needing work (Section 21.94).


- **33/170 answers declare mana and then spend none of it. The mana empties at end of step, so nothing is held up by it. `PROTOCOL_ERRORS` entry 6 is over-tapping and does NOT cover this — its wording presumes spells were cast, and it fires on 0 of these (19%).** Parser-decided, named in reviewer notes, and with no rubric entry (Section 21.83).


- **7/170 answers make a play without ever declaring a PHASE. `phase_problems` checks a declaration against the board and stays silent when there is none, which was right when the grammar merely allowed the line and wrong since 21.60 made it required (4%).** Parser-decided, named in reviewer notes, and with no rubric entry (Section 21.83).


- **Protocol errors, checked by parser: precision 39% (134/341), recall 46% (134/293).** These six rubric entries are decidable from the board, so every charge the judge makes against them is confirmed or refuted mechanically — no human, no second judge. 142 checks could not be decided here and are excluded rather than counted as the judge being wrong (Section 21.70).


## Blunder rate, decomposed

| Arm | n | strategy (judge) | protocol (judge) | protocol (**parser**) | headline |
| --- | --- | --- | --- | --- | --- |
| base_open | 34 | 44% | 94% | **97%** | 94% |
| base_closed | 34 | 53% | 47% | **65%** | 65% |
| base_cards_open | 34 | 44% | 97% | **100%** | 97% |
| ft_cards_open | 34 | 26% | 97% | **100%** | 100% |
| base_open_think | 34 | 47% | 88% | **97%** | 88% |

> **The headline mixes these two rubrics and adding the second moved it silently.** `errors_made` is the whole judge verdict, and since Section 21.70 the judge is given `common_errors + PROTOCOL_ERRORS` — so blunder rate went from *committed a listed strategy error* to *...or the judge thinks it committed a protocol one*, with no rename. **A run made before 21.70 is not comparable to one made after on this number.**

> **And the two halves can rank the arms differently**, so the mixed number is not a tie-break between them. An answer that declines to play commits no listed *strategy* while committing protocol entry 1 — the blindness 21.58 documented and `PROTOCOL_ERRORS` was added to close. Read the **parser** column for the protocol half: 5 of its 7 classes are 100% decidable from the board, and the judge runs 37% precision on the same checks (21.74). Gate 3 is deliberately NOT redefined here — which number it should read is B3's call, the same as `only_pass`.


## Turn scenarios

| Scenario | Steps | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| `turn-payment-combat-0001` | 2 | 0/2 steps | 0/2 steps | 0/2 steps | 0/2 steps | 0/2 steps |

> A turn counts as valid only when **every** step of it is. Steps are scored independently and the board advances on the REFERENCE line, so a mistake at step 1 never makes step 2 unanswerable — what this does not test is recovery from one's own mistake, which needs a rules engine (Section 21.71).


- **4 credited key points name an action the answer never took** — 4/134 of all credited points (3%), 4/19 of those that state an action (21%). The action list comes from the parser, which never sees the judge, so this is checkable rather than a second opinion. Correctness above is unchanged — this sits beside it (Section 21.56).


## Gates

**Gate 1 — protocol works: FAIL.** Worst-arm parse rate 85% (need ≥90%); worst closed-arm legality 53% (need ≥95%).

**Gate 2 — the eval discriminates: PASS.** **24/34 positions (71%) separate the arms** by ≥0.5 on the 1–5 correctness scale (need ≥50%). Compare the rules gold set at 74% (Section 21.11).

> Blunder-rate spread across arms is 35% (65%–100%), reported as a diagnostic rather than as the gate. Section 21.17: the binary call separates the arms on roughly half as many positions as correctness does, and averaging it per arm before comparing cancels what survives.

**Blunder rate — MEASURED, not gated** (B3, Section 21.122). Lowest arm `base_closed` at 68% over 28 basic+intermediate positions; the retired bar was 25%.

> No PASS/FAIL, because nothing supports a threshold at any value. The lowest arm is
> the one HANDED its legal actions, and it wins on protocol compliance rather than
> play — read the strategy column beside this, not this number alone. Reweighting
> for the judge's unreliable clean verdicts (21.81) moves the best arm from 54% to ~70%.

> **15 of 28 boards cannot move this rate** — every arm agrees on them, so only **13** are doing work. Read the rate against n=13, not n=28. A board pinned at all-blundered says the set is too hard for these arms; one pinned at all-clean says it is too easy. Neither is fixed by adjudicating it again.

## Blunder rate by difficulty band

| Difficulty | n | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| basic | 10 | 100% | 50% | 90% | 100% | 100% |
| intermediate | 18 | 94% | 78% | 100% | 100% | 89% |
| advanced | 6 | 83% | 50% | 100% | 100% | 67% |

> Bands under 25 positions (basic, intermediate, advanced) are too small to resolve a blunder-rate difference on their own. Read them as composition, and take the gate from the combined figure above.


## Blunder rate by category

| Category | n | base_open | base_closed | base_cards_open | ft_cards_open | base_open_think |
| --- | --- | --- | --- | --- | --- | --- |
| blocking | 5 | 80% | 80% | 100% | 100% | 100% |
| closing the turn | 2 | 100% | 100% | 100% | 100% | 0% |
| combat math | 6 | 100% | 33% | 100% | 100% | 83% |
| land sequencing | 3 | 100% | 100% | 100% | 100% | 100% |
| mulligan | 2 | 100% | 50% | 100% | 100% | 100% |
| payment | 6 | 83% | 50% | 100% | 100% | 83% |
| race vs stabilize | 2 | 100% | 100% | 100% | 100% | 100% |
| removal timing | 4 | 100% | 50% | 75% | 100% | 100% |
| trigger ordering | 2 | 100% | 50% | 100% | 100% | 100% |
| turn: payment then combat | 2 | 100% | 100% | 100% | 100% | 100% |

> **One judge only.** Section 16.12 measured Gates 2 and 3 BOTH reversing between judges on byte-identical answers, so a gate verdict from a single judge is a statement about the judge. Re-run with `--second-judge mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.

