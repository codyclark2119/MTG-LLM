# magic-llm

Teaching a local LLM to reason about Magic: The Gathering rules, on Apple Silicon, with MLX.

The system is a **retrieval-augmented rules assistant**: the Comprehensive Rules are parsed into a structured, cross-referenced corpus, retrieved per question, and answered by a 7B model — optionally with a LoRA adapter fine-tuned on synthesized rules Q&A. Card data and official rulings are retrieved separately and joined back to the rules that govern them.

> **Status: research in progress, not a finished product.** The evaluation work below found that fine-tuning as currently trained does *not* beat plain retrieval, and that two reasonable LLM judges disagree enough to reverse conclusions. Those results are documented rather than smoothed over — see [Honest results](#honest-results).
>
> Design rationale, experiment history, and the full record of what worked and what didn't live in **[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)**.

## Requirements

- Apple Silicon Mac (built and measured on an M3 Pro / 36GB, macOS Tahoe 26.x)
- Python 3.10+
- ~15GB free disk for models and card data
- No CUDA, no API keys, no network at inference time

## Setup

```bash
python3 -m venv mlx_env
source mlx_env/bin/activate
pip install -r requirements.txt
python -c "import mlx_lm; print('mlx-lm ready')"
```

## Quick start

The rules corpus is committed, so retrieval works immediately after building the vector index:

```bash
python scripts/rag.py index                        # embed 448 rules chunks (~1 min)
python scripts/rag.py query "when are state-based actions checked?" --k 3
```

The derived card corpus is committed too, so card lookup works immediately:

```bash
python scripts/retrieve_hybrid.py "Does [[Chatterfang]] double token creation?"

# Include pinned official WotC rulings for the named cards (opt-in experiment)
python scripts/retrieve_hybrid.py "Does [[Chatterfang]] double token creation?" --rulings
```

The `--rulings` route joins exact card-name matches to the separately pinned
official ruling corpus. It is disabled by default so existing evaluation arms
and prompt fingerprints remain comparable; use it for card-interaction
experiments before integrating ruling text into a training or evaluation arm.

Rebuilding it from source needs the raw Oracle dump (~200MB, not committed):

```bash
python scripts/fetch_cards.py --bulk oracle_cards   # -> data/cards/raw/oracle_cards.jsonl
python scripts/chunk_cards.py                       # -> card_chunks.jsonl (34,933)
```

## Pipeline

Each stage writes a committed artifact, so you can start anywhere.

| Stage | Command | Output |
| --- | --- | --- |
| Parse the rules | `python scripts/ingest.py` | `rules.jsonl` (3,162), `glossary.jsonl` (739) |
| Chunk for retrieval | `python scripts/chunk.py` | `chunks.jsonl` (448) |
| Build the index | `python scripts/rag.py index` | `chunk_embeddings.npz` (gitignored) |
| Generate SFT data | `python scripts/build_sft.py --grounded --freeze-eval` | `train.jsonl` (2,644), `valid.jsonl` (327) |
| Fine-tune | `python -m mlx_lm lora -c configs/phase1_lora_v3.yaml` | adapter in `models/` |
| Evaluate | `python scripts/eval.py --adapter-path models/mtg-rules-adapter-v2-best` | `eval/EVAL_REPORT*.md` |

Evaluation takes ~2.5 hours on an M3 Pro; both stages checkpoint incrementally. The v3 config runs `iters: 1322` — exactly one epoch at batch 2 over 2,644 lines, ~6 hours. `phase1_lora_v2.yaml` is kept for comparability but ran only **0.45 epochs**; use v3 for new runs.

**Why the eval command still names `v2-best` after run 3.** Run 3 removed the under-training confound and changed nothing: `finetuned_rag` moved +0.04 under Qwen and +0.02 under Llama, against control arms — byte-identical answers — that moved up to 0.23 on judge variance alone (§18.3). With no measured basis to prefer it, ckpt1322 was **not promoted** and `models/mtg-rules-adapter-v3-best` deliberately does not exist. `v2-best` is also bit-identical to iteration 600 of the v3 run, so the default is the earlier of two indistinguishable checkpoints on one curve, not an older experiment. Score run 3 explicitly with `--adapter-path models/mtg-rules-adapter-v3-ckpt1322`.

### Card data

```bash
python scripts/fetch_cards.py --bulk oracle_cards      # full Oracle pool via Scryfall bulk data
python scripts/chunk_cards.py                          # -> card_chunks.jsonl (34,933 playable)
python scripts/ingest_rulings.py                       # 77,918 official WotC rulings -> ruling_chunks.jsonl
```

Use the **full Oracle pool**, not a format subset: the Standard-only pool covered just 4% of the cards players actually ask about. `fetch_cards.py --format standard` still exists for format-scoped experiments, but `chunk_cards.py` refuses to overwrite the full corpus with a subset unless you pass `--force`.

### Gold set

Human-reviewed questions with rubric-based answers — the highest-trust data here, and the basis for deciding which automated judge to believe.

```bash
python scripts/fetch_rulesguru.py --all                # 1,402 verified Q&A from rulesguru.org
python scripts/rulesguru_to_gold.py                    # -> gold_candidates.jsonl (rubrics drafted, not final)
python scripts/rulesguru_to_gold.py --promote 3518     # after rewriting a rubric by hand
python scripts/validate_gold.py --to-eval
```

Candidates carry machine-drafted rubrics and are deliberately **not** part of the gold set until a human rewrites the rubric — that review is what the gold tier means. Rubric quality is not cosmetic: hand-authored rubrics roughly doubled inter-judge agreement (r +0.30 → +0.62).

Format, the four rubric-writing rules, and contribution guidance: [data/gold/SCHEMA.md](data/gold/SCHEMA.md).

## Corpora

| Dataset | Size | Source | Committed |
| --- | --- | --- | --- |
| Comprehensive Rules | 3,162 rules + 739 glossary | WotC, pinned 2026-08-07 | yes |
| Retrieval chunks | 448 (avg ~690 tok) | derived | yes |
| Cards | 34,933 playable | Scryfall Oracle | chunks only |
| Official rulings | 77,918 across 19,726 cards | WotC via Scryfall | chunks only |
| SFT training set | 2,644 train / 327 valid — but only **1,478 distinct** ([§21.13](DEVELOPMENT_PLAN.md)) | synthesized, RAG-grounded | yes |
| SFT set — verified | 1,001 train / 111 valid | human-written RulesGuru answers | yes |
| Eval — synthetic | 70 | generated from rules | yes |
| Eval — Reddit | 200 + 100 card-focused | r/MTGRules, LLM-filtered | yes |
| RulesGuru snapshot | 1,402 verified Q&A | rulesguru.org API | yes |
| RulesGuru candidates | 1,202 (drafted rubrics) | derived | yes |
| **Gold set** | **99** (human-reviewed rubrics) | RulesGuru + CR glossary | yes |
| **Positions** | **24** (all hand-adjudicated) | authored boards | yes |
| Judge worksheets | 39 scenarios | Competitive REL sims | yes |

## Scripts

| Script | Purpose |
| --- | --- |
| `ingest.py` | Comprehensive Rules → structured records |
| `chunk.py` | rules → retrieval chunks with cross-refs and glossary |
| `rag.py` | embed chunks; retrieve top-k for a query |
| `build_sft.py` | synthesize RAG-grounded training data |
| `eval.py` | multi-arm evaluation and LLM-judge scoring |
| `fetch_cards.py` / `chunk_cards.py` | Scryfall card pull → chunks linked to keyword rules |
| `card_lookup.py` | resolve card names by dictionary (99% on eval) |
| `retrieve_hybrid.py` | route cards, optional official rulings, and rules into separate retrieval budgets |
| `ingest_rulings.py` | official WotC rulings → chunks |
| `fetch_rulesguru.py` / `rulesguru_to_gold.py` | RulesGuru API → frozen snapshot → gold candidates |
| `ingest_qa_pastes.py` / `validate_gold.py` | gold set ingestion and validation |
| `ingest_judge_worksheets.py` | Competitive REL scenarios → structured records |
| `build_reddit_eval.py` | community Q&A → eval set, LLM-filtered for real rulings |
| `calibrate_judge.py` | **is the judge sound?** positive controls, and `--judge-report` |
| `adjudicate.py` | human ground truth on judge calls; scores any run against it |
| `audit_sft.py` | contamination check; exits non-zero. Run **before** training |
| `stamp_adapter.py` / `rescore_stored.py` | prompt-fingerprint an adapter; re-derive a run with no model |
| `webui.py` / `label_store.py` | local console: label, author records and positions, adjudicate, run scripts |
| `common.py` | shared prompts, CR pinning, rule-id patterns, canonical paths |
| `gameplay/actions.py` | the action grammar and its parser |
| `gameplay/positions.py` | board-position schema, validation, rendering |
| `gameplay/eval_positions.py` | blunder rate, legality, and the three gates |

### Choosing a judge

Every number here is read through an LLM judge, and **most judges do not work**.
Before trusting one, run both halves of the error-detection control:

```bash
python scripts/calibrate_judge.py --judge-report --gold data/gold/positions.jsonl \
    --judge-model mlx-community/Qwen2.5-32B-Instruct-4bit
```

It grades an answer that *cannot* commit a listed error and one that commits
exactly one, verbatim, and reports **`P(fire|error) − P(fire|clean)`**. There is
deliberately no flag for one half: a one-sided number reads as confident and was
published wrong three times (§21.43). Measured on positions, n=24, one arm:

| Judge | GB | separation | matched, n=22 |
| --- | --- | --- | --- |
| Qwen2.5-7B | 4.0 | +25% | +27% |
| Llama-3.1-8B | 4.2 | +42% | +45% |
| Qwen3-14B | 7.8 | +100% | **+100%** |
| **Qwen2.5-32B** | 17.6 | +96% | **+100%** |

`Qwen2.5-32B` is the default (`common.CALIBRATED_JUDGE_ID`) on **coverage** —
it grades 24/24 where Qwen3-14B grades 22/24 and runs ~60× slower. The two are
otherwise indistinguishable, so this is not a size result; a 7.8 GB judge
matches a 17.6 GB one. Llama-3.1-8B has the *best* sensitivity of the four and
fires at 58% of clean answers, which is why neither column ranks anything alone.

`common.py` is not a grab bag: the prompts, the pinned CR version, and the rule-id regexes were each duplicated across four to eight scripts, and a divergent copy of the prompt is exactly what caused the Section 8.7 fine-tune failure. Training data and evaluation must be built from the same strings.

## Gameplay (Phase 3)

Beyond explaining rules: give the model a board and let it choose a play. See [DEVELOPMENT_PLAN.md §16](DEVELOPMENT_PLAN.md).

```bash
python scripts/gameplay/test_actions.py                 # parser assertions (84)
python scripts/gameplay/positions.py                    # validate the 24-position set
python scripts/gameplay/positions.py --render pos-blocking-0001 --closed   # see the prompt
python scripts/gameplay/eval_positions.py --second-judge mlx-community/Meta-Llama-3.1-8B-Instruct-4bit

# promote a reviewed batch of drafts (dry run first — it renders every board)
python scripts/gameplay/positions.py --ingest drafts.jsonl --dry-run
python scripts/gameplay/positions.py --ingest drafts.jsonl --author "Your Name"
```

`data/gold/positions_seed.jsonl` holds 8 machine-drafted plumbing fixtures. They
are **not** gate evidence and the report warns when one is present in a scored
run; regenerate with `make_seed_positions.py`.

**Comparing runs requires the same arm count.** The judge grades every candidate
for a position in one batched call, so adding an arm changes every other arm's
score — measured in [§21.5](DEVELOPMENT_PLAN.md). Use `--arms` to match counts
rather than ignoring a column.

A position is a gold record whose question is a board, so `key_points` is the correct line and **`common_errors` is the blunder list** — `eval.py`'s rubric judge scores it unchanged, and blunder rate is just how often `errors_made` is non-empty. Author them at `#/position` in the web console; see [data/gold/POSITIONS.md](data/gold/POSITIONS.md).

**The judge defaults to Llama-3.1-8B, not the model under test** ([§21.29](DEVELOPMENT_PLAN.md)). It used to default to `--base-model`, i.e. the model graded itself. Positive controls then measured Qwen2.5-7B charging the *reference answer* with a `common_error` — which it definitionally cannot commit — on **40%** of questions, against Llama's 4%; blunder rate is defined on exactly that field, and the inflation reaches +25 points on identical answers. Pass the base model's id explicitly to reproduce a self-judged run.

`eval_positions.py` scores with **two judges** by default. On the seed set, swapping the judge reversed two of the three gates on byte-identical answers (kappa +0.48 on the blunder call), so a one-judge verdict is a statement about the judge.

**Gate 2 is measured per position on correctness** — what fraction separate the arms by ≥0.5 on the 1–5 scale, needing ≥50% — not on the spread of per-arm blunder rates ([§21.18](DEVELOPMENT_PLAN.md)). The old definition collapsed the score to a yes/no and then averaged per arm before comparing, losing separation twice; it read 9% under one judge and 26% under another on **identical answers**, which is a FAIL and a PASS. The per-position figure moves 2 points across the same swap. Blunder rate is still reported, as a diagnostic.

`data/gold/positions_seed.jsonl` is machine-drafted plumbing verification, kept separate from `data/gold/positions.jsonl`; Section 14.6's result says the gate needs hand-authored rubrics.

## Local web console

```bash
python scripts/webui.py --lan --author "judge:CC"   # prints laptop/phone/Bonjour URLs + token
```

Four views: `#/label`, `#/new`, `#/position`, `#/scripts`. Off loopback a token is generated and required. The script runner is an **allowlist** — the client names an action id and values for its declared args, and the command line is assembled server-side, never accepted from the client.

**This console is LAN-only and must never be deployed.** Its runner executes training and evaluation on the host and its store writes the gold set directly.

## Rubric form (deployable)

For contributors who know Magic, aren't on your network, and shouldn't have to install anything:

```bash
python scripts/author_rubrics.py --export-tasks      # -> tasks.json
python scripts/rubric_server.py --tasks data/gold/worksheets/tasks.json
```

A separate program with a deliberately small surface — it reads one exported task file, appends to one submissions log, and has no access to the gold set, the corpora, or any model. Dependencies are `fastapi` + `uvicorn`, so the container is small enough for a free tier. Category dropdown, per-author progress, read-only machine draft, and live rubric checks as you type.

Promotion into the gold set stays a local, reviewed step, which is what keeps *gold* meaning **a person reviewed this**:

```bash
curl -H "x-token: $RUBRIC_TOKEN" https://<app>/api/export > submissions.jsonl
python scripts/author_rubrics.py --ingest-submissions submissions.jsonl --dry-run
```

Deploying to fly.io: [deploy/README.md](deploy/README.md). Contributor guide to hand out: [data/gold/CONTRIBUTING.md](data/gold/CONTRIBUTING.md).

## Honest results

Measured on 110 questions (70 synthetic + 40 Reddit), four system arms, LLM-judged. Full detail in [DEVELOPMENT_PLAN.md §9](DEVELOPMENT_PLAN.md).

| Arm | Correctness (1–5) | Fabricated citations |
| --- | --- | --- |
| base | 3.04 | 33/110 |
| base + RAG | **3.75** | **1/110** |
| fine-tuned | 2.37 | 26/110 |
| fine-tuned + RAG | 3.22 | 5/110 |

**What holds up:**

- **Retrieval works.** RAG cuts fabricated rule citations from 33/110 to 1/110 — the clearest, most reproducible effect in the project.
- **Fine-tuning currently hurts.** Both judges agree; an independent Llama-3.1-8B judge scores `base_rag → finetuned_rag` at −0.61 (95% CI [−1.10, −0.12]). The likely cause is structural: the training data was generated *by* base+RAG, so the student can't exceed its teacher.

**What doesn't:**

- **Card-augmented retrieval showed +0.69 at n=16 and −0.01 at n=100.** It was noise.
- **The synthetic eval flatters RAG.** 65% of its questions retrieve their own source chunk — closer to a lookup test than a generalization test. On real questions all arms cluster within noise.
- **Judge choice can reverse the ranking.** Two independent judges agreed at only r = +0.43 (37% exact, mean disagreement 1.18 points on a 1–5 scale), which made effects below ~0.5 unmeasurable.

**What we did about the last one.** Scoring against an enumerated rubric — *which of these specific claims did the answer make?* — instead of against one prose reference, and then writing the rubrics carefully. On identical answers under identical judges, hand-authored rubrics lifted agreement from r = +0.30 to **+0.62** and cut mean disagreement from 1.26 to 0.80 points. Under those rubrics both judges finally rank the arms the same way.

Two things that fix is careful not to claim. It does **not** shrink the sample needed: per-question variance is unchanged, so resolving a 0.4-point effect still takes ~100–150 questions. And at **99 records** across eight categories the whole-set comparison is what the sample supports; per-category conclusions are not yet available.

**What the positive controls later established** ([§21.27](DEVELOPMENT_PLAN.md)). Scoring four candidates of *known* quality per question — the reference answer itself, half of it, a fluent answer to a different question, and a refusal — finally says what the judge does with answers whose quality is not in doubt:

| Trip-wire | Required | Qwen2.5-7B | Llama-3.1-8B |
| --- | --- | --- | --- |
| dynamic range (oracle − wrong) | ≥ 2.5 | +2.77 PASS | +2.74 PASS |
| ordering accuracy | ≥ 90% | 93% PASS | 94% PASS |
| false errors on the oracle | ≤ 5% | **40% FAIL** | **4% PASS** |

The **correctness scale is sound** — a 2.77-point separation between the reference answer and a well-written answer to a *different* question, so the judge is not being fooled by fluency, and the 0.5–0.8 point effects above are 18–29% of a measured range. The **error detection was not**: the oracle *is* the reference answer and cannot commit a listed `common_error`, yet one judge charged it with one on 40% of questions, costing correct answers 2.47 points apiece. Correctness now scores `points_hit` alone ([§21.28](DEVELOPMENT_PLAN.md)) and the calibrated judge is the default ([§21.29](DEVELOPMENT_PLAN.md)). `errors_made` is still reported — it is still what blunder rate means — it just no longer moves the score.

## Licensing

Comprehensive Rules text and card text are Wizards of the Coast IP, used here for personal research. Card data is courtesy of [Scryfall](https://scryfall.com). The Reddit-derived eval set comes from a CC-BY-SA-4.0 dataset. Confirm your own rights before redistributing anything built on these.
