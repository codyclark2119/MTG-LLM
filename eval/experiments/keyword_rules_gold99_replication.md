# 32B keyword-rule injection gold-set replication

## Question

On the current shipped retrieval baseline — Qwen2.5-32B, cards + official rulings,
flat `k_rules=3` — does keyword-rule injection materially reduce fabricated rule
citations without harming correctness on the full 99-question gold set?

The current service has keyword injection enabled, but the evidence behind that
choice came from a 7B gold run: fabrication moved from 5/99 to 1/99 while
correctness was effectively unchanged. That is only four prevented
fabrications, reported as p=0.125, and the serving model is now 32B. The k-rules
replication just demonstrated that retrieval behavior can reverse when moved to
the current model/full-gold setting, so this serving choice should be replicated
rather than inherited.

## Experimental design

Both runs use exactly:

- gold set: `data/gold/gold_questions.jsonl` (99 questions);
- base model: `mlx-community/Qwen2.5-32B-Instruct-4bit`;
- `--base-only`;
- `--with-cards --with-rulings`;
- flat `--k-rules 3` (the settled serving default);
- `--no-plain-rag` (same two-arm design on both sides);
- max generation tokens: 800;
- judge: `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`;
- judge prompt: V3 rubric judge;
- seed: 42.

The **only intended experimental variable** is keyword-rule injection:

- OFF: `--no-keyword-rules`
- ON: `--keyword-rules`

The runner writes a manifest beside each JSONL result and the analyzer refuses a
pair if any precommitted setting differs other than `keyword_rules`.

Run everything with:

```bash
bash scripts/run_keyword_rules_gold_replication.sh
```

The runner is restart-safe: if generation completed but judging did not, the
automatic `.answers.jsonl` checkpoint is reused. Completed scored runs are not
overwritten.

## Outputs

- `eval/runs/gold99_32b_k3_cards_rulings_keyword_off.jsonl`
- `eval/runs/gold99_32b_k3_cards_rulings_keyword_off.manifest.json`
- `eval/reports/gold99_32b_k3_cards_rulings_keyword_off.md`
- `eval/runs/gold99_32b_k3_cards_rulings_keyword_on.jsonl`
- `eval/runs/gold99_32b_k3_cards_rulings_keyword_on.manifest.json`
- `eval/reports/gold99_32b_k3_cards_rulings_keyword_on.md`
- `eval/reports/gold99_32b_k3_cards_rulings_keyword_paired.md`

## Analysis fixed before the run

The serving reason for keyword injection is grounding, so the primary safety
outcome is paired fabricated-citation behavior on `base_rag_cards_rulings`.
For each matched question, record whether fabrication appears with injection OFF
and ON. The exact McNemar/binomial test conditions on discordant pairs:

- **fixed by ON**: fabricated OFF, not fabricated ON;
- **introduced by ON**: not fabricated OFF, fabricated ON.

Also report paired rubric correctness:

1. ON wins / OFF wins / ties;
2. mean correctness delta (ON - OFF);
3. exact two-sided sign-test p-value;
4. unjudged pairs;
5. total fabricated-citation answers OFF and ON;
6. paired fabrication fixes vs introductions and exact p-value;
7. answers citing at least one rule OFF and ON.

## Decision rule

Keep keyword-rule injection enabled in the shipped profile only if both are true:

1. **grounding benefit replicates:** injection significantly reduces paired
   fabrication (`p < 0.05`, with more OFF→clean fixes than clean→fabricated
   introductions); and
2. **no demonstrated correctness harm:** the paired correctness comparison does
   not significantly favor OFF (`p < 0.05` with more OFF wins than ON wins).

If the grounding effect does not replicate, disable keyword injection by default
rather than carrying a serving behavior supported only by the earlier four-case
signal. If it reduces fabrication but significantly harms correctness, keep the
result as an explicit trade-off rather than calling either default settled.

This rule is fixed before the 32B results are observed.
