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

**Never train on the eval set.** 90 of the 1,202 RulesGuru candidates reach
`gold_questions.jsonl`, and `build_sft_verified.py` excludes them on **three**
keys — `id`, `rulesguru_id`, and question-token overlap — because matching on
`id` alone missed 18 of them while asserting it had caught everything (Section
21.13). Run `python scripts/audit_sft.py <dataset-dir>` before any retrain; it
exits non-zero on contamination. Contamination is the one failure that
invalidates everything downstream and looks like an improvement while doing it,
so the check belongs before the training run, not after.

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

No pytest. Tests are runnable scripts with plain asserts, and none of them need
a GPU:

```bash
python scripts/test_imports.py                    # every script resolves every name it uses
python scripts/test_eval.py                       # the scoring arithmetic (113)
python scripts/test_docs.py                       # README's artifact counts match the artifacts
python scripts/gameplay/test_actions.py           # the action grammar (84)
python scripts/gameplay/test_eval_positions.py    # the gameplay gates (32)
```

`test_eval.py` covers the code that turns judge JSON into published numbers, and
`test_eval_positions.py` covers the gate arithmetic. Add a case whenever you
touch `rubric_correctness`, `verify_quoted_claims`, `score_citations`,
`judge_prompt_for`, `stratified_sample`, or `gate2_discrimination` — a bug in
those surfaces as a *plausible* number or a bold PASS rather than an obvious
failure, which is why they are tested more heavily than anything else here.
`test_eval_positions.py` also re-derives Section 21.18's published table from
the stored runs, so a gate change that contradicts the documentation fails.

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

`common.prompt_fingerprint()` hashes the three system prompts **and the
assembled message shape**; Section 8.7 was a shape change with every prompt
string untouched, so hashing the strings alone would pass through it. Stamp an
adapter right after training and `eval.py` will refuse to evaluate across a
prompt change:

```bash
python scripts/stamp_adapter.py models/mtg-rules-adapter-v4 --dataset data/datasets/verified
```

`--dataset` verifies the stamp against the training file rather than asserting
it — that is what makes the stamp evidence.

Three judge prompts exist. V2 (`judge_batch_anonymized`) is length-neutral and
anonymized; V3 (`judge_batch_rubric`) asks which enumerated claims an answer
made and computes the score in Python. `score_one_question` routes to V3 when a
record has `key_points`. V1 was removed in the Section 17 review.

**Correctness is `points_hit / n_points` and nothing else** (Section 21.28). The
old rule halved credit whenever `errors_made` was non-empty; the positive
controls measured the Qwen judge inventing an error against the *reference
answer* on 40% of questions, so that term was removing 2.47 points from correct
answers. `errors_made` is still extracted and is still what blunder rate is
defined on — it just no longer moves the score. Every row records a `scoring`
field, `halve_on_error=True` reproduces pre-21.28 numbers from the same stored
judge output, and `scripts/rescore_stored.py` re-derives any stored run without
a model.

## Evaluation — read this before trusting any number

**Two judges, always.** Two reasonable judges reversed the arm ranking on
identical answers (Section 9.9) and reversed two of three gameplay gates
(Section 16.12). Vary the judge and *nothing else* — that is what
`--rescore-from` is for. A number from one judge is a statement about the judge.

The judge-agreement report prints **both judges' gate verdicts side by side** and
flags two failures: a reversal, and — subtler — a gate that *agrees* while the
judges sit more than half the threshold apart (Section 21.19). Gate 3 currently
does the latter: 65% vs 27% against a 25% bar, so it agrees only because the
model is far from passing, and will start reversing when an arm gets close.
**Gate 1 is bit-identical under both judges** on every run pair, as it must be —
it comes from the action parser, which never sees the judge. That is the positive
control for the whole comparison.

**Comparing runs requires the same arm count.** `judge_batch_rubric` grades
every candidate for a question in ONE batched call — that is what makes the
anonymized A/B/C/D design work — so a fifth arm is not a fifth independent
grading, it is a fifth answer in the same prompt. Measured: adding one arm moved
a byte-identical arm's blunder rate 23 points (Section 21.5). An arm cannot be
added and compared against a run without it. Use `--arms` to match counts; never
run five and compare four of them against an old four-arm run.

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

**`webui.py` is LAN-only and must never be deployed.** That runner plus a store
that writes the gold set directly are fine behind a LAN token and are remote
code execution on a public URL.

## The rubric form (`scripts/rubric_server.py`)

The *deployable* half, and a separate program on purpose. It reads one
self-contained `tasks.json` (from `author_rubrics.py --export-tasks`), appends
to one `submissions.jsonl`, and reaches nothing else — no gold set, no
candidates, no corpora, no model. Its only project import is
`common.lint_common_errors`, which is why `common.py` must stay pure stdlib.

Two invariants worth keeping:

- **Promotion is local and reviewed.** The server never writes the gold set;
  `--ingest-submissions` does, after `--dry-run`. That is what keeps "gold"
  meaning *a person reviewed this*.
- **Attribution rides on each submission**, not on the import command, so one
  file holds several authors and `eval.py --compare` can break agreement down
  per author.

`deploy/` holds the Dockerfile, `fly.toml` and the three-line requirements. If
the Dockerfile's COPY list ever grows, that is the moment to ask whether the
new thing belongs on the public side.

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
  answer" and "classify this comment"; `PASS` meant both "the protocol
  terminator every answer must end with" and "the game action of passing
  priority", which failed Gate 1 on a mulligan — the one position where the two
  readings come apart (Section 16.13). The two meanings always agree until they
  suddenly don't, so the bug ships looking correct.
- **A rubric with no entry for the most likely wrong answer.** 69% of model
  answers to a board contain at most one action, so on a two-step line the
  likely failure is doing half of it. Four positions had no entry for that and
  carried 11 of 31 disputed judge calls; the one that had it drew zero. The
  lint cannot help — it checks the lines that are there, and this is a missing
  one.
- **Prose about a play parsing AS that play.** "Play Mountain first would strand
  Shock in hand" became a `PLAY` action with a garbage operand, silently, with
  no `ParseFailure`. Any arm asked to reason would have had its legality
  destroyed by its own explanation, and it would have read as "reasoning makes
  the model play worse". Reasoning is bounded by an `ACTIONS:` marker and
  `<think>` blocks now, but the general shape recurs: free text in a field that
  is later parsed.
- **A helper duplicated with a guard in only some copies.** Five jsonl readers,
  three of which crashed on a trailing blank line.
- **An identifier that changes when a record is promoted.** A RulesGuru
  candidate is `rg-1156`; promoted into the gold set it becomes
  `qa-amy-casts-assassin-s-trophy-...` and keeps `rulesguru_id: 1156`. The
  contamination filter compared `id` to `id`, so 16 eval questions passed
  straight through a check that *asserted* it had excluded them. Anything
  joining two files on an id must first ask whether the id survived the trip.
- **`pgrep -f` matching the shell that mentions the job.** A background shell
  whose command line contains `... --judge-model mlx-community/Qwen2.5-32B ...`
  is matched by `pgrep -f Qwen2.5-32B`, so "is the 32B run going?" answered yes
  while the model had not been loaded. It reads as a started job, and it also
  breaks the reverse case: a `while pgrep -f X; do sleep; done` waiter whose own
  loop condition contains `X` waits on itself forever. Match on the interpreter
  and script instead (`ps -eo pid,command | grep "Python.*eval\.py"`), or check
  for the artifact the job writes. This has produced a false reading twice.
- **A number that never was.** `data/datasets` was described as 2,644 training
  examples in three configs and two plan sections. It holds 1,478 distinct
  lines; the rest are exact duplicates, question and answer both. Nothing lied —
  nobody counted.
- **Valid JSON in an unexpected shape, read as absence.** Twice. The judge
  emitted `"points_hit": 3` where a list was expected (Section 21.12), and — for
  a single candidate — the entry *unwrapped*, without the `{"A": …}` around it
  (Section 21.39). A missing key and a key spelled differently both arrive as
  `None`, so a third of a benchmark went "ungraded" while the judge had answered
  every question. Both times the wrong number was **plausible** (a strict judge,
  an expensive prompt), and both times two hypotheses were reasoned out and
  tested before anyone printed the raw text — both wrong. **When a parse yields
  a surprising count, look at the bytes before theorizing about them.**

## Conventions

- Scripts are CLIs: `argparse` with `description=__doc__`, and a module
  docstring that explains *why*, with a usage block.
- Data is `.jsonl`, one record per line. Read with `common.read_jsonl`; write
  hand-authored data with `common.write_jsonl_atomic`.
- Destructive rebuilds refuse to shrink an existing corpus without `--force`,
  via `common.guard_shrink` — never a re-implementation. `min_ratio` is 1.0 for
  `rules.jsonl` (any shrink is a parse regression) and 0.5 for derived corpora,
  where the known failure is an order-of-magnitude subset. They write with
  `write_jsonl_atomic`, so a crash mid-write cannot leave a truncated corpus
  that reads as valid.
- **Corpora are content-pinned.** `rules.jsonl` and `glossary.jsonl` in
  `common.CR_PIN`; `card_chunks.jsonl` and `ruling_chunks.jsonl` in
  `common.CARD_PIN`, checked by `verify_card_pin` from `CardIndex.__init__` —
  the chokepoint eleven call sites reach card text through. Scryfall is a live
  API, so a re-fetch returns errata'd oracle text at an unchanged record count.
  Re-pin with `python scripts/chunk_cards.py --update-pin`. Both pins share
  `_verify_pin`; do not write a third copy. Note `ruling_chunks.jsonl` is
  currently read by no script.
- **`rules.jsonl` and `glossary.jsonl` are content-pinned** in `common.CR_PIN`,
  checked by `common.verify_cr_pin` from `load_rule_ids` (the chokepoint seven
  of nine readers reach) and directly from `chunk.py`/`chunk_cards.py`, which
  read structurally. This catches what `guard_shrink` and `git` cannot: a parse
  that keeps the rule count and changes the text. Only the *canonical* paths
  are checked — `--rules somewhere_else` is a deliberate act and is allowed.
  Re-pin with `python scripts/ingest.py --update-pin`, and treat needing to as
  a signal that every number downstream now describes a different corpus.
- Snapshots of external APIs are **frozen and additive** — RulesGuru
  re-randomizes card and player names per request, so re-fetching a record
  would silently change it.
- **Player normalization misses rather than guesses.** `common.find_players`
  should leave a raw name in place before it risks renaming something that
  isn't a player. A missed name is cosmetic — a question that reads
  "Player A ... Allison" still means what it said. A false positive rewrites
  the question into something it does not say, and every rubric, answer and
  score downstream then describes a different question. When adding a pattern:
  measure it across all of `gold_candidates.jsonl` first, require the
  grammatical role to be one a game term cannot occupy, and drop it if any hit
  is not a person. `_PLAYER_PREP` was retired under this rule — it reached 18
  records nothing else could, and turned "refers to Sand Warriors" into a
  player named Sand.
