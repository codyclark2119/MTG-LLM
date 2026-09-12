# 32B k-rules gold-set replication

## Question

Does the 32B card+rulings arm still benefit from dropping dense CR retrieval
(`k_rules=0`) when the 53-question card-ruling benchmark is replaced by the
full 99-question gold set?

This is the replication required by Section 21.165 before treating the shipped
`k_rules=auto` policy as settled evidence rather than a provisional judgement.

## Why this is precommitted

The existing 53-question result is ambiguous in exactly two dimensions:

- **correctness:** k=0 beat k=3 by +0.25 on average, with 15 wins / 6 losses /
  32 ties (two-sided exact sign-test p = 0.078);
- **grounding:** fabricated citations increased from 3/53 at k=3 to 7/53 at
  k=0.

Those are therefore the primary outcome and the safety outcome for the
replication. No new metric should be chosen after seeing the 99-question result.

The historical pair can be re-derived before running anything expensive:

```bash
python scripts/analyze_k_rules_pair.py \
  eval/runs/deconf32b_k0_card_ruling53.jsonl \
  eval/runs/deconf32b_k3_card_ruling53.jsonl
```

Expected historical result: 15 k=0 wins, 6 k=3 wins, 32 ties, p ~= 0.078;
fabricated citations 7/53 versus 3/53.

## Experimental design

Both runs use:

- gold set: `data/gold/gold_questions.jsonl` (99 questions);
- base model: `mlx-community/Qwen2.5-32B-Instruct-4bit`;
- `--base-only` so no incompatible 7B LoRA is loaded;
- `--with-cards --with-rulings`;
- `--no-plain-rag`, preserving the two-arm design and avoiding a duplicate
  empty-context RAG candidate when k=0;
- max generation tokens: 800;
- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`;
- judge prompt: V3 rubric judge;
- seed: 42.

The **only intended experimental variable is `--k-rules 0` versus
`--k-rules 3`**.

Run both sides and the paired analysis with:

```bash
bash scripts/run_k_rules_gold_replication.sh
```

`eval.py` writes an answers checkpoint before judging. If generation completed
but judging failed, rerunning the wrapper reuses that checkpoint instead of
regenerating 32B answers. Completed scored runs are never overwritten by the
wrapper.

## Outputs

- `eval/runs/gold99_32b_cards_rulings_k0.jsonl`
- `eval/reports/gold99_32b_cards_rulings_k0.md`
- `eval/runs/gold99_32b_cards_rulings_k3.jsonl`
- `eval/reports/gold99_32b_cards_rulings_k3.md`
- `eval/reports/gold99_32b_cards_rulings_paired.md`

The `.answers.jsonl` checkpoints beside each run are intermediate provenance and
should be retained at least until judging and analysis are complete.

## Analysis fixed before the run

Primary comparison: `base_rag_cards_rulings` correctness on matched question
IDs.

Report:

1. k=0 wins / k=3 wins / ties;
2. mean paired correctness delta (k=0 - k=3);
3. two-sided exact sign-test p-value with ties excluded;
4. fabricated-citation counts on both sides;
5. number of answers citing at least one rule on both sides;
6. unjudged pairs, if any.

The analyzer refuses different question sets, arm sets, judges, or judge
prompts so a provenance mismatch cannot silently become a k comparison.

## Decision rule

The existing `auto` policy should be called **supported by the replication**
only if both conditions hold:

1. k=0 has a positive paired correctness effect with two-sided exact sign-test
   `p < 0.05`; and
2. k=0 does **not** increase the number of fabricated-citation answers relative
   to k=3.

If either condition fails, the gold-set result does not support dropping CR
retrieval as the shipped default on card-resolved questions. That does not make
the experiment useless: the paired report should still be committed and the
serving decision documented from the observed correctness/grounding trade-off.

This rule is intentionally conservative. The service is a rules assistant, so
an apparent quality gain bought by more invented rule citations is not treated
as a clean win.
