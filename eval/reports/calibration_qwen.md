# Judge calibration — positive controls

99 questions from `gold_questions_eval.jsonl`, four known-quality candidates each.

- judge: `mlx-community/Qwen2.5-7B-Instruct-4bit` (prompt v3)
- **graded 97/99 questions (98%)** — the rest returned JSON the harness could not parse

| Candidate | Mean correctness | n | Expected |
| --- | --- | --- | --- |
| `oracle` (the reference itself) | **3.88** | 98 | near 5 |
| `partial` (half, by sentence) | 3.63 | 97 | middle |
| `wrong` (another question's answer) | 1.10 | 98 | near 1 |
| `refusal` | 1.07 | 98 | 1 |

- **Dynamic range (oracle − wrong): +2.77** on a 1–5 scale
- **Ordering accuracy: 93%** of questions rank oracle ≥ partial ≥ wrong
- **False errors on the oracle: 40%** — the reference answer cannot commit a listed common_error, so every one of these is a definitive false positive

## Trip-wires (STRUCTURAL_AUDIT.md)

- PASS — dynamic range ≥ 2.5 (measured +2.77)
- PASS — ordering accuracy ≥ 90% (measured 93%)
- **FAIL** — false errors on oracle ≤ 5% (measured 40%)

> The instrument does not yet separate known-good from known-bad to the standard the migration trip-wires require. A better model under test cannot be distinguished from a worse one until this passes, so hardware is not the binding constraint.
