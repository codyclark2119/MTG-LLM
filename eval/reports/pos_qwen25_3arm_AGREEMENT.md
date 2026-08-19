# Position Judge Agreement

`pos_qwen25_3arm.jsonl` vs `pos_qwen25_3arm_judge2.jsonl` — identical stored answers, judge varied.

- 57 arm-position blunder calls over 22 positions x 3 arms
- **Cohen's kappa on the blunder call: +0.34** (raw agreement 72%, chance 57%)
- correctness correlation: r = +0.32
- blunder rate: pos_qwen25_3arm 77%, pos_qwen25_3arm_judge2 63%

**Reading:** kappa between 0.20 and 0.40 is weak agreement; usable for large effects only.

| Arm | pos_qwen25_3arm blunder | pos_qwen25_3arm_judge2 blunder | agree |
| --- | --- | --- | --- |
| base_open | 79% | 47% | 58% |
| base_closed | 79% | 68% | 89% |
| base_cards_open | 74% | 74% | 68% |

## 16 disputed calls

Each is a position where one judge saw a blunder and the other did not. These are the cases to read by hand — they show whether the rubric is ambiguous or a judge is simply wrong.

- `pos-blocking-0002` / base_cards_open
- `pos-blocking-0004` / base_open
- `pos-blocking-0004` / base_cards_open
- `pos-combat-math-0002` / base_closed
- `pos-combat-math-0004` / base_open
- `pos-combat-math-0005` / base_open
- `pos-combat-math-0006` / base_open
- `pos-land-sequencing-0001` / base_cards_open
- `pos-land-sequencing-0003` / base_open
- `pos-mulligan-0001` / base_cards_open
- `pos-race-vs-stabilize-0001` / base_cards_open
- `pos-race-vs-stabilize-0002` / base_open
- `pos-race-vs-stabilize-0002` / base_cards_open
- `pos-removal-timing-0002` / base_open
- `pos-trigger-ordering-0001` / base_open
