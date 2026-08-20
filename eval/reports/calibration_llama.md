# Judge calibration — positive controls

99 questions from `gold_questions_eval.jsonl`, four known-quality candidates each.

- judge: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (prompt v3)
- **graded 94/99 questions (95%)** — the rest returned JSON the harness could not parse

| Candidate | Mean correctness | n | Expected |
| --- | --- | --- | --- |
| `oracle` (the reference itself) | **4.18** | 95 | near 5 |
| `partial` (half, by sentence) | 3.80 | 94 | middle |
| `wrong` (another question's answer) | 1.45 | 95 | near 1 |
| `refusal` | 1.19 | 95 | 1 |

- **Dynamic range (oracle − wrong): +2.74** on a 1–5 scale
- **Ordering accuracy: 94%** of questions rank oracle ≥ partial ≥ wrong
- **False errors on the oracle: 4%** — the reference answer cannot commit a listed common_error, so every one of these is a definitive false positive

## Trip-wires (STRUCTURAL_AUDIT.md)

- PASS — dynamic range ≥ 2.5 (measured +2.74)
- PASS — ordering accuracy ≥ 90% (measured 94%)
- PASS — false errors on oracle ≤ 5% (measured 4%)