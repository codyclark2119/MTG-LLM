# Position Evaluation Report

24 positions from `/Users/codyclark/Documents/code/magic-llm/data/gold/positions.jsonl`

- base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- adapter: `models/mtg-rules-adapter-v2-best`
- judge: `mlx-community/Qwen3-14B-4bit`

> **27/72 arm-position pairs went unjudged.** The judge carried roughly 271 characters of candidate text per batched call (~68 tokens) across 3 arms. If that is small against `--judge-max-tokens 4000`, the failure is in what the judge PRODUCES, not what it reads — a reasoning judge spends its budget thinking before it emits JSON.


| Arm | Blunder rate | Errors/answer | Correctness | Parsed ok | Actions/answer | All legal | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_open | 67% (n=15) | 1.33 | 2.00 | 100% | 2.3 | 71% | 0% |
| base_closed | 53% (n=15) | 0.80 | 2.73 | 100% | 2.5 | 75% | 0% |
| base_cards_open | 60% (n=15) | 1.13 | 2.47 | 100% | 1.7 | 62% | 0% |

## Gates

**NOT EVALUATED — the judge graded 45/72 arm answers (62%).** Gate verdicts are withheld rather than computed from what survived: the failures are not random, they track how much the prompt asks the judge to produce, so the graded remainder is selected by rubric size rather than by anything about the answers (Section 21.14).

> Most likely `--judge-max-tokens` is too small for this judge. A reasoning model spends its budget in `<think>` before emitting JSON; Qwen3-14B needs ~1,800 tokens for a **single** arm, so a three-arm call needs several times the 600-token default.
