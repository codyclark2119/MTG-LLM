# Position Judge Agreement

`positions_n32_scenarios.jsonl` vs `positions_n32_scenarios_judge2.jsonl` — identical stored answers.

- positions_n32_scenarios: `mlx-community/Qwen2.5-32B-Instruct-4bit`
- positions_n32_scenarios_judge2: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`

- 165 arm-position blunder calls over 34 positions x 5 arms
- **Cohen's kappa on the blunder call: +0.59** (raw agreement 88%, chance 72%)
- correctness correlation: r = +0.59
- blunder rate: positions_n32_scenarios 82%, positions_n32_scenarios_judge2 85%

**Reading:** kappa at or above 0.40 is moderate agreement — comparable to the r = +0.62 hand-authored rubrics reached on the rules eval (Section 14.6).

| Arm | positions_n32_scenarios blunder | positions_n32_scenarios_judge2 blunder | agree |
| --- | --- | --- | --- |
| base_open | 85% | 94% | 91% |
| base_closed | 52% | 58% | 70% |
| base_cards_open | 85% | 88% | 97% |
| ft_cards_open | 97% | 97% | 94% |
| base_open_think | 91% | 88% | 91% |

## Gate verdicts under each judge

| Gate | positions_n32_scenarios | positions_n32_scenarios_judge2 | |
| --- | --- | --- | --- |
| 2 — discriminates | 79% PASS | 94% PASS | agree |
| 3 — blunder ≤25% | 54% (base_closed) FAIL | 56% (base_closed) FAIL | agree |

## The parser as arbiter

On 134 answers the two judges fired a different set of protocol errors. `protocol_truth` is decided from the answer and the board with no judge involved, so it can say which was closer — the one comparison here that does not need a person (Section 21.76).

- parser sides with `positions_n32_scenarios`: **13**
- parser sides with `positions_n32_scenarios_judge2`: **74**
- neither closer: 47

> **This settles a disagreement, not a judge.** It covers the protocol entries only; the strategy entries have no mechanical check and are exactly where 81% of adjudicated answers came back `not_covered`. A judge that loses here is worse at the half a parser could have done anyway (21.74).

## 19 disputed calls

Each is a position where one judge saw a blunder and the other did not. These are the cases to read by hand — they show whether the rubric is ambiguous or a judge is simply wrong.

- `pos-blocking-0001` / base_closed
- `pos-blocking-0002` / base_closed
- `pos-combat-math-0003` / base_closed
- `pos-combat-math-0005` / base_cards_open
- `pos-combat-math-0006` / base_closed
- `pos-land-sequencing-0001` / base_open
- `pos-race-vs-stabilize-0002` / base_open
- `pos-removal-timing-0002` / base_open_think
- `pos-removal-timing-0004` / base_closed
- `pos-removal-timing-0004` / ft_cards_open
- `pos-trigger-ordering-0002` / base_closed
- `sample-stage3-payment-0001` / base_open_think
- `sample-stage3-payment-0002` / base_closed
- `sample-stage3-payment-0002` / ft_cards_open
- `sample-stage3-payment-0003` / base_open
