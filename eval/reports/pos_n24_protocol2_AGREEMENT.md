# Position Judge Agreement

`pos_n24_protocol2_32b.jsonl` vs `pos_n24_protocol2_mistral.jsonl` — identical stored answers.

- pos_n24_protocol2_32b: `mlx-community/Qwen2.5-32B-Instruct-4bit`
- pos_n24_protocol2_mistral: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

- 78 arm-position blunder calls over 26 positions x 3 arms
- **Cohen's kappa on the blunder call: +0.67** (raw agreement 91%, chance 73%)
- correctness correlation: r = +0.70
- blunder rate: pos_n24_protocol2_32b 82%, pos_n24_protocol2_mistral 86%

**Reading:** kappa at or above 0.40 is moderate agreement — comparable to the r = +0.62 hand-authored rubrics reached on the rules eval (Section 14.6).

| Arm | pos_n24_protocol2_32b blunder | pos_n24_protocol2_mistral blunder | agree |
| --- | --- | --- | --- |
| base_open | 96% | 96% | 92% |
| base_closed | 58% | 65% | 85% |
| base_cards_open | 92% | 96% | 96% |

## Gate verdicts under each judge

| Gate | pos_n24_protocol2_32b | pos_n24_protocol2_mistral | |
| --- | --- | --- | --- |
| 2 — discriminates | 65% PASS | 85% PASS | agree |
| 3 — blunder ≤25% | 57% (base_closed) FAIL | 67% (base_closed) FAIL | agree |

## The parser as arbiter

On 58 answers the two judges fired a different set of protocol errors. `protocol_truth` is decided from the answer and the board with no judge involved, so it can say which was closer — the one comparison here that does not need a person (Section 21.76).

- parser sides with `pos_n24_protocol2_32b`: **10**
- parser sides with `pos_n24_protocol2_mistral`: **19**
- neither closer: 29

> **This settles a disagreement, not a judge.** It covers the protocol entries only; the strategy entries have no mechanical check and are exactly where 81% of adjudicated answers came back `not_covered`. A judge that loses here is worse at the half a parser could have done anyway (21.74).

## 7 disputed calls

Each is a position where one judge saw a blunder and the other did not. These are the cases to read by hand — they show whether the rubric is ambiguous or a judge is simply wrong.

- `pos-blocking-0004` / base_closed
- `pos-land-sequencing-0001` / base_open
- `pos-land-sequencing-0001` / base_closed
- `pos-race-vs-stabilize-0001` / base_open
- `pos-race-vs-stabilize-0002` / base_closed
- `pos-removal-timing-0001` / base_cards_open
- `pos-removal-timing-0004` / base_closed
