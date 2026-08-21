# Position Judge Agreement

`pos_n24_32b.jsonl` vs `pos_n24_mistral24b.jsonl` — identical stored answers, judge varied.

- 72 arm-position blunder calls over 24 positions x 3 arms
- **Cohen's kappa on the blunder call: +0.47** (raw agreement 72%, chance 48%)
- correctness correlation: r = +0.71
- blunder rate: pos_n24_32b 58%, pos_n24_mistral24b 36%

**Reading:** kappa at or above 0.40 is moderate agreement — comparable to the r = +0.62 hand-authored rubrics reached on the rules eval (Section 14.6).

| Arm | pos_n24_32b blunder | pos_n24_mistral24b blunder | agree |
| --- | --- | --- | --- |
| base_open | 58% | 38% | 79% |
| base_closed | 46% | 38% | 75% |
| base_cards_open | 71% | 33% | 62% |

## Gate verdicts under each judge

| Gate | pos_n24_32b | pos_n24_mistral24b | |
| --- | --- | --- | --- |
| 2 — discriminates | 67% PASS | 67% PASS | agree |
| 3 — blunder ≤25% | 42% (base_closed) FAIL | 26% (base_cards_open) FAIL | agree |

> **Gate 3 agrees but is not stable.** The two judges are 16% apart on the same answers, against a 25% threshold. It returns the same verdict only because both land far from the line — the agreement is a fact about how far the model is from passing, not about the metric. Expect it to reverse as soon as an arm gets close.

## 20 disputed calls

Each is a position where one judge saw a blunder and the other did not. These are the cases to read by hand — they show whether the rubric is ambiguous or a judge is simply wrong.

- `pos-blocking-0001` / base_closed
- `pos-blocking-0002` / base_open
- `pos-blocking-0002` / base_closed
- `pos-blocking-0002` / base_cards_open
- `pos-blocking-0003` / base_open
- `pos-blocking-0003` / base_closed
- `pos-blocking-0003` / base_cards_open
- `pos-blocking-0004` / base_open
- `pos-blocking-0004` / base_cards_open
- `pos-blocking-0005` / base_cards_open
- `pos-combat-math-0003` / base_open
- `pos-combat-math-0003` / base_cards_open
- `pos-combat-math-0004` / base_closed
- `pos-mulligan-0001` / base_closed
- `pos-mulligan-0001` / base_cards_open
