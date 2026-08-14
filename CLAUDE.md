# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A **research repository**, not a product. It trains and evaluates a local LLM
on Magic: The Gathering rules, on Apple Silicon via MLX. The deliverable is
*measurements you can trust* — several of which are negative, and stay negative
in the documentation on purpose.

Three documents carry the state, and they have distinct jobs:

| File | Job |
| --- | --- |
| `README.md` | how to run things; current artifact counts |
| `DEVELOPMENT_PLAN.md` | the experiment log — design rationale, every result, every failure. Sections are numbered and referenced from code comments |
| `data/gold/SCHEMA.md`, `data/gold/POSITIONS.md` | how to author the two kinds of hand-labeled data |

**Code comments cite plan sections** (`# see Section 8.7`). That is the
convention for explaining *why* something non-obvious is the way it is. Keep it.

## Non-negotiables

**Never commit.** The user commits; prepare a message and say it is ready.

**Never silently change a measurement.** `data/gold/gold_questions.jsonl`,
`eval/*.jsonl`, and the adapters in `models/` back published numbers. Before
touching any of them, check `git diff --quiet` on the gold set and say so.

**Report negative results as negative.** This project's value is that
"fine-tuning did not beat retrieval" and "two judges disagree enough to reverse
the ranking" are written down plainly. Do not soften a result, and do not
present a single-judge or n=8 number as settled.

**Verify by running, not by reading.** A `--help` that exits 0 proves almost
nothing — it has hidden a `NameError` here before. Import the module and
exercise the path.

## Setup

```bash
source mlx_env/bin/activate     # every command below assumes this
```

`requirements.txt` is a full `pip freeze`, not a curated list. Six packages are
imported directly: `mlx-lm`, `mlx-embeddings`, `numpy`, `datasets`, `fastapi`,
`uvicorn`.

No pytest. Tests are runnable scripts with plain asserts:
`python scripts/gameplay/test_actions.py`.

## Architecture

```
Comprehensive Rules ──ingest.py──> rules.jsonl ──chunk.py──> chunks.jsonl
                                                                  │
Scryfall Oracle ──fetch_cards.py──> chunk_cards.py ──> card_chunks.jsonl
                                                                  │
                                          rag.py (embeddings) ────┤
                                     card_lookup.py (exact name) ─┤
                                                                  ▼
                                                        retrieve_hybrid.py
                                                                  │
                                          build_sft.py ──> train/valid.jsonl
                                                                  │
                                             mlx_lm.lora ──> models/adapter
                                                                  ▼
                                                     eval.py / eval_positions.py
```

Cards and rules get **separate retrieval budgets**. Cards outnumber rules
chunks ~78:1, so merging them into one index lets card text win slots on pure
rules questions. Card names resolve by exact dictionary lookup (99%), never by
embedding.

### `scripts/common.py` is load-bearing

It holds the prompts, the pinned CR version, the rule-id regexes, the canonical
paths, and the jsonl helpers. Each moved there *after* a divergent copy caused
or nearly caused a real bug. **Add shared values here rather than re-declaring
them**, and never hardcode a path that already has a constant.

`REPO_ROOT` anchors every canonical path, so scripts run from any working
directory. Paths a user passes on the command line stay relative to their cwd.

### Prompts are versioned and adapter-coupled

`build_rag_messages()` is the single definition of a prompt's shape, used by
**both** training and inference. Changing a prompt in `common.py` invalidates
the trained adapter — an adapter is only valid for the format it saw. This is
the project's #1 documented failure mode (Section 8.7).

Three judge prompts exist. V2 (`judge_batch_anonymized`) is length-neutral and
anonymized; V3 (`judge_batch_rubric`) asks which enumerated claims an answer
made and computes the score in Python. `score_one_question` routes to V3 when a
record has `key_points`. V1 was removed in the Section 17 review.

## Evaluation — read this before trusting any number

**Two judges, always.** Two reasonable judges reversed the arm ranking on
identical answers (Section 9.9) and reversed two of three gameplay gates
(Section 16.12). Vary the judge and *nothing else* — that is what
`--rescore-from` is for. A number from one judge is a statement about the judge.

**The judge is deterministic.** Re-judging identical inputs reproduces exactly.
So a number that moved means something real changed; there is no sampling noise
to blame.

**Controls catch harness bugs.** A no-card control subset caught a
double-`"Rules text:"` prompt bug that had produced a convincing false result.
When adding an arm, add the subset where it should have *no* effect.

**Sample size:** ~100–150 questions to resolve a 0.4-point effect on the 1–5
scale; ~40 positions for blunder rate, because a proportion with a large
expected effect needs far fewer.

## Gameplay (`scripts/gameplay/`)

A **position** is a gold record whose question is a board. `key_points` is the
correct line, **`common_errors` is the blunder list**, and blunder rate is just
how often `errors_made` is non-empty — so the existing rubric judge scores
positions unchanged.

Positions live in `data/gold/positions.jsonl`, deliberately **not** in
`gold_questions.jsonl`: `label_store.CATEGORIES` drives stratified sampling for
the rules gold set, and an in-flight n≈100 comparison depends on its
composition.

`data/gold/positions_seed.jsonl` is machine-drafted plumbing verification, not
gate evidence. Hand-authored rubrics beat machine drafts by a wide margin
(r +0.30 → +0.62, Section 14.6).

The board is **rendered from structured state, never authored as text**, so a
renderer change applies retroactively.

## The web console

```bash
python scripts/webui.py --lan --author "judge:CC"
```

Four views: `#/label`, `#/new`, `#/position`, `#/scripts`. Off loopback it
generates a token.

**The script runner is an allowlist.** The client sends an action id and values
for that action's declared args; the command line is assembled server-side and
never accepted from the client. This server binds to the LAN, so a generic
"run a command" endpoint would be remote code execution. Adding a runnable
script means adding an entry to `ACTIONS` — never a passthrough. Jobs run one
at a time because several write the same files.

## Traps this repo has already fallen into

Each cost real time. They recur in new code, so they are worth knowing.

- **A stale default silently evaluating the wrong thing.** `ADAPTER_PATH`
  pointed at v1 long after v2 superseded it. `--judge-model` was accepted and
  then ignored on one code path. Defaults are overridable *and recorded in the
  output*.
- **The documented happy path being destructive.** The README's own quick start
  once overwrote a 34,933-chunk corpus with a 4% subset.
- **Config comments describing a plan that never ran.** `phase1_lora_v2.yaml`
  claimed ~1 epoch; it ran 0.45. Recompute `iters × batch / len(train)` before
  believing a comment.
- **A prompt inviting its own parse failure.** The action grammar wrote optional
  operands as `[TARGET <x>]`; the model copied the brackets and correct plays
  scored as illegal.
- **Substring matching on a keyword.** Verb detection with `startswith` turned
  "Blocking the Bears with Elves" into a BLOCK *with inverted operands*.
- **One name, two meanings.** `CROSS_REF_RE` meant both "find ids in prose" and
  "validate a whole string"; `JUDGE_SYSTEM_PROMPT` meant both "grade this
  answer" and "classify this comment".
- **A helper duplicated with a guard in only some copies.** Five jsonl readers,
  three of which crashed on a trailing blank line.

## Conventions

- Scripts are CLIs: `argparse` with `description=__doc__`, and a module
  docstring that explains *why*, with a usage block.
- Data is `.jsonl`, one record per line. Read with `common.read_jsonl`; write
  hand-authored data with `common.write_jsonl_atomic`.
- Destructive rebuilds refuse to shrink an existing corpus without `--force`.
- Snapshots of external APIs are **frozen and additive** — RulesGuru
  re-randomizes card and player names per request, so re-fetching a record
  would silently change it.
