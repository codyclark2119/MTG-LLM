# Position Judge Agreement

`positions_n22.jsonl` vs `positions_n22_judge2.jsonl` — identical stored answers, judge varied.

- 88 arm-position blunder calls over 22 positions x 4 arms
- **Cohen's kappa on the blunder call: +0.24** (raw agreement 65%, chance 54%)
- correctness correlation: r = +0.43
- blunder rate: positions_n22 68%, positions_n22_judge2 60%

**Reading:** kappa between 0.20 and 0.40 is weak agreement; usable for large effects only.

| Arm | positions_n22 blunder | positions_n22_judge2 blunder | agree |
| --- | --- | --- | --- |
| base_open | 50% | 45% | 59% |
| base_closed | 77% | 82% | 68% |
| base_cards_open | 64% | 50% | 59% |
| ft_cards_open | 82% | 64% | 73% |

## 31 disputed calls

Each is a position where one judge saw a blunder and the other did not. These are the cases to read by hand — they show whether the rubric is ambiguous or a judge is simply wrong.

- `pos-blocking-0001` / ft_cards_open
- `pos-blocking-0002` / base_cards_open
- `pos-blocking-0003` / base_open
- `pos-blocking-0003` / ft_cards_open
- `pos-blocking-0004` / base_closed
- `pos-combat-math-0001` / base_cards_open
- `pos-combat-math-0002` / base_open
- `pos-combat-math-0002` / base_closed
- `pos-combat-math-0002` / base_cards_open
- `pos-combat-math-0002` / ft_cards_open
- `pos-combat-math-0004` / base_closed
- `pos-combat-math-0004` / base_cards_open
- `pos-combat-math-0005` / base_open
- `pos-combat-math-0005` / base_closed
- `pos-combat-math-0005` / base_cards_open
