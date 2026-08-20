# Position Judge Agreement

`pos_qwen3_14b_v2.jsonl` vs `pos_qwen3_14b_v2_judge2.jsonl` — identical stored answers, judge varied.

- 21 arm-position blunder calls over 24 positions x 3 arms
- **Cohen's kappa on the blunder call: +0.32** (raw agreement 62%, chance 44%)
- correctness correlation: r = +0.69
- blunder rate: pos_qwen3_14b_v2 24%, pos_qwen3_14b_v2_judge2 62%

**Reading:** kappa between 0.20 and 0.40 is weak agreement; usable for large effects only.

| Arm | pos_qwen3_14b_v2 blunder | pos_qwen3_14b_v2_judge2 blunder | agree |
| --- | --- | --- | --- |
| base_open | 29% | 57% | 71% |
| base_closed | 14% | 71% | 43% |
| base_cards_open | 29% | 57% | 71% |

## 8 disputed calls

Each is a position where one judge saw a blunder and the other did not. These are the cases to read by hand — they show whether the rubric is ambiguous or a judge is simply wrong.

- `pos-blocking-0004` / base_open
- `pos-blocking-0004` / base_closed
- `pos-blocking-0004` / base_cards_open
- `pos-combat-math-0005` / base_closed
- `pos-combat-math-0006` / base_closed
- `pos-removal-timing-0001` / base_open
- `pos-removal-timing-0001` / base_closed
- `pos-removal-timing-0001` / base_cards_open
