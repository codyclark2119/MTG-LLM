# Judge error-detection report — `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

24 records from `positions.jsonl`, **one arm**, prompt v3. Two candidates per record: the reference answer, which **cannot** commit a listed error, and one asserting a listed error **verbatim**, which definitionally does.

- **graded 48/48 (100%)**
- **24/24 records usable** (0 had no claim-form error; 0 entries skipped)
- quote drops: 0

| | fires at a **clean** answer | fires at a **1-error** answer |
| --- | --- | --- |
| rate | 58% (14/24) | 100% (24/24) |
| mean errors fired | 0.58 | 1.00 |
| caught the planted one | — | 100% |

## Separation: **+42%**

`P(fire | error) − P(fire | clean)`. **This is the number to read, and neither column means anything without the other** — a judge that never fires scores a perfect 0% on the left, one that always fires scores a perfect 100% on the right. Measured (positions, n=24, one arm): **32B +96, Llama-3.1-8B +42, Qwen2.5-7B +25** — and all three catch the planted error 24/24, so the right-hand column does not rank them at all (Sections 21.42, 21.43).

> **One arm.** Not comparable to the four-arm `false errors` trip-wire; arm count moves scores by as much as 23 points (Section 21.5). Compare only against another single-arm report on the same set.

> **This judge does not separate blundered from clean.** Blunder rate is defined on `errors_made` being non-empty, so it cannot support that metric however well it catches real errors.
