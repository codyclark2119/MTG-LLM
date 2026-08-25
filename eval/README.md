# eval/

Three roles, three directories. The split exists because 33 files at one level
with four competing naming schemes made it impossible to tell an input from an
output, or a report from the data it was built on.

```
sets/      inputs — the question sets fed to an eval run, plus their manifests
runs/      outputs — per-question scored results, one JSON object per question
reports/   outputs — human-readable summaries
```

**A report shares its run's stem.** `reports/cards_n100.md` summarizes
`runs/cards_n100.jsonl`. That was not previously true and it mattered: the old
`EVAL_REPORT_CARDS.md` documented `eval_results_cards_v2.jsonl`, while the
similarly-named `eval_results_cards.jsonl` was a *discarded* run — so pairing
them by name analyzed the wrong data.

## Suffixes

| Suffix | Means | Varied |
| --- | --- | --- |
| `_rejudged` | re-scored under the **v2 judge prompt** (Section 9.7) | the prompt |
| `_judge2` | re-scored by an **independent judge model** (Section 9.9) | the model |
| `_superseded_*` | kept for the record, do not analyze | — |

Both re-scoring suffixes mean *the answers are byte-identical and only the
judge changed*. That is the only design under which an agreement number means
anything, and it is what `--rescore-from` exists to do.

## What's here

| Run | Section | What it measured |
| --- | --- | --- |
| `rules_v1` | 9.5 | first adapter, 110 questions, 4 arms |
| `rules_v1_rejudged`, `rules_v2_rejudged` | 9.7 | both runs re-scored under the length-neutral v2 judge, which reversed a result |
| `rules_v2` | 9.6 | after the train/inference format fix |
| `cards_n60` | 13.5 | card-augmented arms, with a no-card control subset |
| `cards_n60_superseded_headerbug` | 13.5 | the run whose control read −0.34 — impossible with identical contexts, and the reason the control exists. Duplicated `"Rules text:"` header |
| `cards_n100`, `cards_n100_judge2` | 13.5 | the +0.69 card effect failing to replicate at n=100 |
| `*_cards_rulings` | new experiment | card lookup plus pinned official WotC rulings and rules retrieval |
| `pilot_v3`, `pilot_v3_judge2` | 14.5–14.6 | the rubric judge, and hand vs machine rubrics |
| `positions_seed*` | 16.11–16.12 | gameplay gates; the judge swap reverses two of three |

## Reading any of it

Numbers here are only as good as their judge. Two reasonable judges have
reversed an arm ranking (9.9) and two of three gameplay gates (16.12) on
identical answers. A single-judge report is a statement about that judge —
every report now names its base model, adapter and judge in the body, because
the most important variable of a two-judge study used to live only in the
filename.

## Official ruling experiment

Add the ruling-augmented arms explicitly:

```bash
python scripts/eval.py --with-cards --with-rulings \
	--gold-only --gold data/gold/gold_questions.jsonl \
	--out eval/runs/cards_rulings_n99.jsonl \
	--report-out eval/reports/cards_rulings_n99.md
```

This creates `base_rag_cards_rulings` and `finetuned_rag_cards_rulings` in
addition to the existing arms. It is a new six-arm experiment, not a result
that can be compared directly with a four-arm run: the batched judge scores
all candidates together, so use the same arm count for follow-up rescoring.
The option is disabled by default and does not change existing evaluations.
