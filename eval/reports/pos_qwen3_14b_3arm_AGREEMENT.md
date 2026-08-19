# Position Judge Agreement

`pos_qwen3_14b_3arm.jsonl` vs `pos_qwen3_14b_3arm_judge2.jsonl` — identical stored answers, judge varied.

- 27 arm-position blunder calls over 22 positions x 3 arms
- **Cohen's kappa on the blunder call: +0.28** (raw agreement 67%, chance 54%)
- correctness correlation: r = +0.15
- blunder rate: pos_qwen3_14b_3arm 30%, pos_qwen3_14b_3arm_judge2 41%

**Reading:** kappa between 0.20 and 0.40 is weak agreement; usable for large effects only.

| Arm | pos_qwen3_14b_3arm blunder | pos_qwen3_14b_3arm_judge2 blunder | agree |
| --- | --- | --- | --- |
| base_open | 33% | 44% | 89% |
| base_closed | 33% | 33% | 78% |
| base_cards_open | 22% | 44% | 33% |

## 9 disputed calls

Each is a position where one judge saw a blunder and the other did not. These are the cases to read by hand — they show whether the rubric is ambiguous or a judge is simply wrong.

- `pos-blocking-0001` / base_closed
- `pos-blocking-0001` / base_cards_open
- `pos-blocking-0002` / base_open
- `pos-blocking-0004` / base_cards_open
- `pos-combat-math-0003` / base_closed
- `pos-combat-math-0003` / base_cards_open
- `pos-land-sequencing-0002` / base_cards_open
- `pos-mulligan-0002` / base_cards_open
- `pos-removal-timing-0003` / base_cards_open
