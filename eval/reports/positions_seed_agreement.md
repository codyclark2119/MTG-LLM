# Position Judge Agreement

`positions_seed.jsonl` vs `positions_seed_judge2.jsonl` — identical stored answers, judge varied.

- 32 arm-position blunder calls over 8 positions x 4 arms
- **Cohen's kappa on the blunder call: +0.48** (raw agreement 75%, chance 52%)
- correctness correlation: r = +0.69
- blunder rate: position_results_seed 62%, position_results_seed_judge2 56%

**Reading:** kappa at or above 0.40 is moderate agreement — comparable to the r = +0.62 hand-authored rubrics reached on the rules eval (Section 14.6).

| Arm | position_results_seed blunder | position_results_seed_judge2 blunder | agree |
| --- | --- | --- | --- |
| base_open | 62% | 62% | 75% |
| base_closed | 62% | 62% | 75% |
| base_cards_open | 62% | 25% | 62% |
| ft_cards_open | 62% | 75% | 88% |

## 8 disputed calls

Each is a position where one judge saw a blunder and the other did not. These are the cases to read by hand — they show whether the rubric is ambiguous or a judge is simply wrong.

- `pos-seed-0001` / base_open
- `pos-seed-0001` / base_closed
- `pos-seed-0001` / ft_cards_open
- `pos-seed-0002` / base_open
- `pos-seed-0002` / base_closed
- `pos-seed-0002` / base_cards_open
- `pos-seed-0003` / base_cards_open
- `pos-seed-0006` / base_cards_open
