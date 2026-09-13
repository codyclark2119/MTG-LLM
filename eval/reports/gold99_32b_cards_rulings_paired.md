# Paired k-rules replication

- arm: `base_rag_cards_rulings`
- questions: 99
- k=0: `eval/runs/gold99_32b_cards_rulings_k0.jsonl`
- k=3: `eval/runs/gold99_32b_cards_rulings_k3.jsonl`

| Measure | Result |
| --- | ---: |
| k=0 wins | 14 |
| k=3 wins | 20 |
| ties | 65 |
| unjudged pairs | 0 |
| mean correctness delta (k=0 - k=3) | -0.131 |
| exact sign-test p (two-sided, ties excluded) | 0.391528 |
| fabricated citations: k=0 | 21/99 |
| fabricated citations: k=3 | 4/99 |
| answers citing at least one rule: k=0 | 85/99 |
| answers citing at least one rule: k=3 | 58/99 |

Interpret correctness and fabrication together. A correctness gain that is paid for by more invented rule citations is the exact trade-off this replication was designed to test.
