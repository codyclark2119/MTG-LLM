# Judge calibration — positive controls

99 questions from `gold_questions_eval.jsonl`, four known-quality candidates each.

- judge: `mlx-community/Qwen2.5-32B-Instruct-4bit` (prompt v3)
- **graded 99/99 questions (100%)**

| Candidate | Mean correctness | n | Expected |
| --- | --- | --- | --- |
| `oracle` (the reference itself) | **4.60** | 99 | near 5 |
| `partial` (half, by sentence) | 3.90 | 99 | middle |
| `wrong` (another question's answer) | 1.00 | 99 | near 1 |
| `refusal` | 1.00 | 99 | 1 |

- **Dynamic range (oracle − wrong): +3.60** on a 1–5 scale
- **Ordering accuracy: 100%** of questions rank oracle ≥ partial ≥ wrong
- **False errors on the oracle: 0%** — the reference answer cannot commit a listed common_error, so every one of these is a definitive false positive

## Trip-wires (STRUCTURAL_AUDIT.md)

- PASS — dynamic range ≥ 2.5 (measured +3.60)
- PASS — ordering accuracy ≥ 90% (measured 100%)
- PASS — false errors on oracle ≤ 5% (measured 0%)