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

Card-aware retrieval needs the card corpus (~25MB download, not committed):

```bash
python scripts/fetch_cards.py --format standard    # or: see Card data below
python scripts/chunk_cards.py
python scripts/retrieve_hybrid.py "Does [[Chatterfang]] double token creation?"
```

## Pipeline

Each stage writes a committed artifact, so you can start anywhere.

| Stage | Command | Output |
| --- | --- | --- |
| Parse the rules | `python scripts/ingest.py` | `rules.jsonl` (3,162), `glossary.jsonl` (739) |
| Chunk for retrieval | `python scripts/chunk.py` | `chunks.jsonl` (448) |
| Build the index | `python scripts/rag.py index` | `chunk_embeddings.npz` (gitignored) |
| Generate SFT data | `python scripts/build_sft.py --grounded --freeze-eval` | `train.jsonl` (2,644), `valid.jsonl` (327) |
| Fine-tune | `python -m mlx_lm lora -c configs/phase1_lora_v2.yaml` | adapter in `models/` |
| Evaluate | `python scripts/eval.py --adapter-path models/mtg-rules-adapter-v2-best` | `eval/EVAL_REPORT*.md` |

Fine-tuning takes ~2.5 hours and evaluation ~2.5 hours on an M3 Pro. Both checkpoint incrementally.

### Card data

```bash
python scripts/fetch_cards.py --format standard        # 4,887 Standard-legal cards via Scryfall search
python scripts/chunk_cards.py                          # -> card_chunks.jsonl
python scripts/ingest_rulings.py                       # 77,931 official WotC rulings -> ruling_chunks.jsonl
```

For the full card pool (recommended — the Standard-only pool covered just 4% of cards players actually ask about), download Scryfall's `oracle_cards` bulk file to `data/cards/raw/oracle_cards.jsonl` and run `chunk_cards.py --cards data/cards/raw/oracle_cards.jsonl`.

### Gold set

Human-authored questions with rubric-based answers — the highest-trust data here, and the basis for deciding which automated judge to believe.

```bash
python scripts/ingest_qa_pastes.py --from data/gold/pastes/*.txt --draft-rubric
python scripts/validate_gold.py --to-eval
```

Format and contribution guidance: [data/gold/SCHEMA.md](data/gold/SCHEMA.md).

## Corpora

| Dataset | Size | Source | Committed |
| --- | --- | --- | --- |
| Comprehensive Rules | 3,162 rules + 739 glossary | WotC, pinned 2026-08-07 | yes |
| Retrieval chunks | 448 (avg ~690 tok) | derived | yes |
| Cards | 34,933 playable | Scryfall Oracle | chunks only |
| Official rulings | 77,931 across 19,726 cards | WotC via Scryfall | chunks only |
| SFT training set | 2,644 train / 327 valid | synthesized, RAG-grounded | yes |
| Eval — synthetic | 70 | generated from rules | yes |
| Eval — Reddit | 200 + 100 card-focused | r/MTGRules, LLM-filtered | yes |
| **Gold set** | **6** | judge study site | yes |
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
| `retrieve_hybrid.py` | route cards and rules into separate retrieval budgets |
| `ingest_rulings.py` | official WotC rulings → chunks |
| `ingest_qa_pastes.py` / `validate_gold.py` | gold set ingestion and validation |
| `ingest_judge_worksheets.py` | Competitive REL scenarios → structured records |
| `build_reddit_eval.py` | community Q&A → eval set, LLM-filtered for real rulings |

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
- **Judge choice can reverse the ranking.** Two independent judges agree at only r = +0.43 (37% exact, mean disagreement 1.18 points on a 1–5 scale). Effects below ~0.5 are not currently measurable.

That last point gates everything else, which is why the gold set exists: 6 human-authored questions so far, against a target of 8–10 per category across eight categories.

## Licensing

Comprehensive Rules text and card text are Wizards of the Coast IP, used here for personal research. Card data is courtesy of [Scryfall](https://scryfall.com). The Reddit-derived eval set comes from a CC-BY-SA-4.0 dataset. Confirm your own rights before redistributing anything built on these.
