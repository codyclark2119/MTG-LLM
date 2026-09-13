# 32B k-rules gold-set replication — result

The precommitted replication in `k_rules_gold99_replication.md` is complete.

On the full 99-question gold set, the `base_rag_cards_rulings` arm produced:

- k=0: 14 paired wins, 20 losses, 65 ties versus k=3;
- mean correctness delta (k=0 - k=3): -0.131;
- two-sided exact sign-test p = 0.391528;
- fabricated citations: 21/99 at k=0 versus 4/99 at k=3;
- answers citing at least one rule: 85/99 at k=0 versus 58/99 at k=3.

The precommitted decision rule required both a significant positive paired
correctness effect for k=0 (p < 0.05) and no increase in fabricated citations.
The replication fails both conditions. The 53-question k=0 advantage therefore
does not replicate on the gold set, and dropping CR retrieval materially worsens
grounding.

**Serving decision:** restore flat `k_rules=3` as the shipped default for the
32B chat profile. Keep `--k-rules auto` available as an explicit experimental
override; the router remains useful for research, but it is no longer the
default supported by current evidence.

Source artifacts:

- `eval/runs/gold99_32b_cards_rulings_k0.jsonl`
- `eval/runs/gold99_32b_cards_rulings_k3.jsonl`
- `eval/reports/gold99_32b_cards_rulings_paired.md`
