# Position Judge Agreement

`positions_n22_think.jsonl` vs `positions_n22_think_judge2.jsonl` — identical stored answers, judge varied.

- 100 arm-position blunder calls over 22 positions x 5 arms
- **Cohen's kappa on the blunder call: +0.14** (raw agreement 62%, chance 56%)
- correctness correlation: r = +0.24
- blunder rate: positions_n22_think 74%, positions_n22_think_judge2 62%

**Reading:** kappa below 0.20 means the two judges are barely agreeing beyond chance, so blunder rate is not yet a usable gate metric — fix the rubrics or the judge prompt before authoring more positions.

| Arm | positions_n22_think blunder | positions_n22_think_judge2 blunder | agree |
| --- | --- | --- | --- |
| base_open | 70% | 40% | 50% |
| base_closed | 85% | 60% | 65% |
| base_cards_open | 60% | 50% | 50% |
| ft_cards_open | 70% | 70% | 70% |
| base_open_think | 85% | 90% | 75% |

## 38 disputed calls

Each is a position where one judge saw a blunder and the other did not. These are the cases to read by hand — they show whether the rubric is ambiguous or a judge is simply wrong.

- `pos-blocking-0001` / base_open
- `pos-blocking-0001` / base_cards_open
- `pos-blocking-0001` / ft_cards_open
- `pos-blocking-0003` / base_cards_open
- `pos-blocking-0004` / base_open
- `pos-blocking-0004` / base_closed
- `pos-blocking-0004` / ft_cards_open
- `pos-blocking-0004` / base_open_think
- `pos-combat-math-0001` / base_cards_open
- `pos-combat-math-0001` / ft_cards_open
- `pos-combat-math-0002` / base_open_think
- `pos-combat-math-0003` / base_open
- `pos-combat-math-0003` / base_cards_open
- `pos-combat-math-0004` / base_cards_open
- `pos-combat-math-0004` / ft_cards_open
