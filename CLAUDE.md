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
python scripts/test_eval.py                       # the scoring arithmetic (272)
python scripts/test_docs.py                       # README's artifact counts match the artifacts
python scripts/test_webui.py                      # the served page's JavaScript actually parses
python scripts/test_deploy.py                     # what may leave the machine (83)
python scripts/gameplay/test_actions.py           # the action grammar (107)
python scripts/gameplay/test_eval_positions.py    # the gameplay gates (47)
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

**The gameplay prompt has its own fingerprint.** `prompt_fingerprint` covers the
rules track only, so until Section 21.60 an edit to `GAMEPLAY_SYSTEM_PROMPT` —
the grammar block that *is* the output contract — left no trace in any run file.
`common.gameplay_fingerprint()` is deliberately separate: the two tracks share no
prompt, and one digest would make a grammar edit invalidate a rules adapter that
never saw it. Every position run records it and `compare_judges` refuses to
interpret agreement across two values. Stored position runs are `5c196f40afd8`;
the current prompt is `d094e3934d2d`, so **no stored position number is
comparable to a new run** until the arms are regenerated.

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

**The fingerprint answers only half of 8.7.** It asks whether the prompts were
*edited* since training; it cannot ask whether the adapter ever *saw* the shape
an arm hands it. `data/datasets/verified` is 1,112 examples under
`SYSTEM_PROMPT` and **zero** under `RAG_SYSTEM_PROMPT` — deliberately, since
`build_sft_verified` attaches no retrieved context — so v4's fingerprint matches
perfectly while the `finetuned_rag` arm is evaluated on a shape those weights
never met. That is 8.7's actual mechanism reached with no edit at all.
`stamp_adapter.unseen_arms` reads the `dataset_prompt_counts` the stamp already
recorded, and `eval.py` prints it **twice** — before generation and in the report
body — because the warning and the number it qualifies sit at opposite ends of a run
(Section 21.50).

Three judge prompts exist. V2 (`judge_batch_anonymized`) is length-neutral and
anonymized; V3 (`judge_batch_rubric`) asks which enumerated claims an answer
made and computes the score in Python. `score_one_question` routes to V3 when a
record has `key_points`. V1 was removed in the Section 17 review.

**Correctness is `points_hit / n_points` and nothing else** (Section 21.28). The
old rule halved credit whenever `errors_made` was non-empty; the positive
controls measured **Qwen2.5-7B** inventing an error against the *reference
answer* on 40% of questions, so that term was removing 2.47 points from correct
answers. `errors_made` is still extracted and is still what blunder rate is
defined on — it just no longer moves the score. `halve_on_error=True` reproduces
pre-21.28 numbers from the same stored judge output, and
`scripts/rescore_stored.py` re-derives any stored run without a model.

Every row now records `judge_model`, `judge_prompt` and `scoring`, via
`eval.carry_diagnostics` — **one** copy list, because both writers previously
enumerated these by hand and neither listed `scoring` at all. Runs archived
before this carry none of it and are identified by filename only; five have
`scoring` because a different path wrote them.

**Six report writers exist, and a hardening has landed on a subset of them five
times** — `carry_diagnostics` (neither listed `scoring`),
`judge_model` on rescored rows (21.51), and the unjudged-coverage guard, which
lived in `rescore()` while the *first-pass* writer — where every coverage failure
this project has had actually happened — printed confident means with no warning
(21.53), the judge's *identity* in both `compare_judges` writers, whose only
subject is which judge said what (21.54), and scenario expansion, which reached
`eval_positions`' generation path and not its rescore path — so a run containing
turn steps could be produced and **never re-judged**, and `--rescore-from` is the
only way a second judge ever sees an answer (21.74). The direction flips each time, so the
rule is **check every other writer**, not "check `rescore`". Shared behaviour belongs in one function
(`carry_diagnostics`, `coverage_lines`, `turns.load_steps`) with a test that
asserts both callers exist, since no test over inputs and outputs can see a
missing caller — the generation path was correct and its tests passed, because
the bug lived entirely where those tests never went.

`--rescore-from` stamps the row too, which it did not until Section 21.51. It is
the one path where the judge is *guaranteed* to differ from the file it read —
that is what a rescore is for — so the stale label was wrong on every run worth
making and right only on the pointless ones. The derived report header was
correct throughout; the data file is what `--compare` reads.

**That 40% is a fact about the 7B, not about `errors_made`** (Section 21.40).
Name the judge whenever quoting it. On the same 24 positions, single-arm, with
the judge as the only variable, `CALIBRATED_JUDGE_ID` charges the reference
answer with an error on **1 of 24 against the 7B's 18**, and fires **1.12**
errors against an answer carrying exactly one where the 7B fires **2.75** — the
"every error at once" signature (21.26) measured directly. Blunder rate is a
working metric on the 32B and is not one on the 7B.

**Two judges of four pass; run `--judge-report` on any new one.** Positions,
n=24, single arm (Sections 21.42, 21.44). Read the *separation*, never either
column alone:

| Judge | GB | fires at **clean** | fires at **1-error** | separation | matched (n=22) |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | 4.0 | 18/24, mean **1.96** | 24/24, mean 2.75 | +25% | +27% |
| Llama-3.1-8B | 4.2 | 14/24 (58%) | 24/24, mean **1.00** | +42% | +45% |
| Qwen3-14B | 7.8 | 0/**22** | 23/**23** | +100% | **+100%** |
| **Qwen2.5-32B** | 17.6 | 1/24 (4%) | 24/24 | +96% | **+100%** |

Llama has the *best* sensitivity of the four and fires at 58% of clean answers —
which is why the middle two columns rank nothing on their own.

**Qwen3-14B ties the 32B, it does not beat it.** On the 22 positions every judge
graded they are identical (0/22, 22/22). The 32B's whole false-positive rate is
`pos-combat-math-0002`, the polarity case — which the 14B *failed to grade*.

**And Qwen3-14B is a single-arm control judge, not a gate judge** (21.44).
Rescoring the three-arm gate run it graded **0/72** at the default 600 tokens
and **45/72** at 4,000 — a reasoning model pays a fixed `<think>` cost *before*
output that then scales with arm count, so what works at one arm does not
survive three. The 32B is the default at both.

Not a size result: a 7.8 GB judge matches a 17.6 GB one. Every judge that passes
is a Qwen and every `base` arm is a Qwen, so **self-preference is still
untested** — that needs a different vendor, not a different generation.

**These are single-arm rates and roughly 17 points pessimistic** (Section
21.45). Holding the record set fixed, Llama fires at 0/41 clean answers on the
rules set at **four** arms and 7/42 at **one** — so fewer candidates in the
prompt means more false positives. Rubric type adds a further +41: positions are
genuinely harder to judge than rules questions, and that is most of Llama's 58%.

Two consequences. The matrix above is **internally valid** — every judge
measured identically — so the ranking holds; only absolute rates are
single-arm-specific, and never quote one against the four-arm trip-wire. And the
bias favours the published gates: the controls run at one arm, `pos_n24_32b`
runs at **three**, so the 32B's real false-positive rate where the gates were
measured is *below* the 4% the control reported.

Four plan sections investigated the 40% as a property of the field while the
calibration table already recorded the 32B at 0%. A rate measured on one judge
is a statement about that judge, and that applies to diagnosing a metric exactly
as it applies to ranking arms.

## Evaluation — read this before trusting any number

**Judge-vs-judge agreement is not correctness.** Measured: kappa **+0.47**
between two calibrated judges on the same answers, and **+0.17 to +0.23** between
each of them and a human on the same blunder calls (Section 21.57). They agree
with each other about twice as well as any agrees with the person. So two judges
is a floor, not a proof — and a constructed control cannot close the gap, because
its sensitivity half asserts a **listed** error while 12 of 17 real answers are
bad for a reason the rubric never enumerated. A control built from the rubric
cannot detect that the rubric is missing entries.

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

**One control measures one direction — this has produced three wrong published
conclusions** (Section 21.43). The oracle subset (an answer that *cannot* commit
an error) measures only **specificity**; the planted-error subset (an answer that
commits exactly one, verbatim) measures only **sensitivity**. Each alone is
uninterpretable, and each alone reads as confident:

- A judge that fires nothing scores a perfect 0% on the oracle control.
- A judge that fires everything scores a perfect 100% on the planted-error one.
- Llama-3.1-8B has the **best sensitivity of four judges** (mean 1.00) and fires
  at **58% of clean answers**. Ranked on either column alone it looks fine.

So `calibrate_judge.py` reports them **as a pair only**, and the headline is
`P(fire | error) − P(fire | clean)`, which cannot be computed from one half.
Never quote a judge quality number that has only one side. **A rate measured on
one half of a control is a statement about that half** — the same lesson as "a
number from one judge is a statement about the judge", one level up.

The corollary that already bit: "N fixed, **0 regressions**" from oracle-only
evidence means *no regressions were measurable* — an answer committing no error
cannot lose a true positive. Two sections published that; the negative control
found two lost.

**Sample size:** ~100–150 questions to resolve a 0.4-point effect on the 1–5
scale; ~40 positions for blunder rate, because a proportion with a large
expected effect needs far fewer.

## Gameplay (`scripts/gameplay/`)

**The gameplay layer supervises the judge layer, not the reverse.** Judge-vs-judge
kappa is +0.47 and judge-vs-**human** is **+0.06** (21.65) — a judge cannot
validate itself. But a board state is machine-checkable in a way a rules question
never is: **86% of position answers can be convicted with no judge at all**, and
`common.PROTOCOL_ERRORS` turns those seven checks into rubric entries a *parser*
confirms or refutes. That is the only per-error precision this project can
compute without a human (21.70).

**Measured, two judges, byte-identical answers: precision 37% (32B) and 34%
(Mistral-24B) — so 37% is the task, not the judge** (21.74). Do not read the
aggregate. Per class it is bimodal under *both* judges: entry 3 (*names a play
not in `legal_actions`*) scores 80–85%, and all six others score 18–43%. Entry 3
is the one entry `legal_actions` already checks mechanically, so **the judge is
reliable exactly where a parser makes it redundant** and unreliable on every
entry that exists because the parser could not see it. Take the mechanically
decidable entries from the parser; leave the judge the strategy entries, where
no check exists and 81% of adjudicated answers came back `not_covered`.

`PROTOCOL_ERRORS` is **append-only**. The judge returns error NUMBERS, so
strategy errors must keep `1..n` or every verdict already collected silently
changes meaning. Under-tapping (5) and over-tapping (6) are separate on purpose:
one makes the turn invalid, the other is legal and merely wasteful, and
`PROTOCOL_INVALIDATING` is what `valid_turn` reads. **Blunder rate is unchanged**
— it stays on `errors_made` per 21.28, with the strategy/protocol split stored
beside it, so Gate 3 keeps meaning "picked the wrong play".

**A turn scenario is a sequence, and it is teacher-forced** (`gameplay/turns.py`,
21.71). The board advances on the **reference** line, never on what the model
did: applying the model's own actions needs a rules engine this repo deliberately
is not, and errors would compound so that every later step re-measures step 1.
Each expanded step is a *complete position*, so `legality`, `protocol_findings`
and `judge_batch_rubric` work unchanged — do not add a second evaluation path. A
turn is valid only when **every** step is; averaging per-step validity lets four
good steps hide one that makes the turn illegal.

**An answer that declines to play beats one that plays.** `PASS` is the protocol
terminator, so it always matches `legal_actions` — an answer whose only action is
`PASS` scored `all_legal=True` on **69 of 69**, and the judge called it clean on
**63%**, because doing nothing commits none of the enumerated *strategies*.
`base_open` does this on 42% of positions and `base_cards_open` on 46%. It is
also where the two calibrated judges disagree most: 9% vs 57% blunder on
do-nothing answers against 49% vs 59% on answers that play — which is the whole
of 21.52's arm reversal, not a judge-quality difference. `only_pass` is measured
and reported; whether it becomes a gate is B3's call (Section 21.58).

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
renderer change applies retroactively. That is true of the *board*, not of the
answers: `render_position` is what the model reads, so changing it invalidates
every stored position answer for comparison. 16 of 24 positions state no
`mana_available` and the model infers the pool from the land list — making that
explicit is a deliberate decision with that cost, not a cleanup (Section 21.59).

`positions.py` validates a position three ways beyond parsing: `timing_problems`
(a sorcery-speed `legal_action` in a step that forbids it), `mana_problems` (one
the stated pool cannot pay for), and the name/rule-id checks. **Both of the first
two return `[]` when the card index is absent**, so the validator prints its
coverage — how many positions state a pool, and whether the index loaded — because
"no problems found" over a set nothing examined is the same defect as a gate
verdict with no coverage.

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
- **The form records what it showed.** `n_shown` rides on each verdict, because
  the rubric a reviewer saw is not necessarily the one the judge was given, and
  a judge charge against an entry nobody was offered is not a false positive
  (21.75). `score_run` restricts to it and prints `charges_not_shown` loudly —
  non-zero means the deployed form is behind the judge and must be re-exported.

`deploy/` holds the Dockerfile, `fly.toml` and the three-line requirements. If
the Dockerfile's COPY list ever grows, that is the moment to ask whether the
new thing belongs on the public side.

**Deploy only with the explicit config, never bare `fly launch`:**

```bash
fly launch --no-deploy --copy-config --config deploy/fly.toml --dockerfile deploy/Dockerfile
```

Bare `fly launch` auto-detects this as a generic Python app and writes its own
root `Dockerfile` with `COPY . .`, a root `fly.toml`, and a GitHub workflow that
deploys on push to `main`. That combination would put 5.0 GB on a public host —
the gold set, the adjudication queue, every eval run, 3.8 GB of adapters, and
`scripts/webui.py` with its script runner. It happened; nothing deployed only
because no build ran.

The root **`.dockerignore` is an allowlist**: it denies everything and re-admits
exactly the four files `deploy/Dockerfile` copies. This is separate from the
COPY list and both are needed — fly uploads the entire build context to its
remote builder *before* any COPY executes, so an over-broad context transmits
the gold set even when the image never contains it. `test_deploy.py` asserts the
context and the COPY list are the same four files, and fails if a root
`fly.toml` or `Dockerfile` reappears.

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
- **A rubric the judge was given and the human form was not.** `PROTOCOL_ERRORS`
  reached `eval_positions` and not `adjudicate.task_for`, so the reviewer had no
  checkbox for "taps six lands it does not control" and wrote it in the note
  instead — **30 of 33** `not_covered` notes name an entry the rubric already
  had, which is most of what 21.47 read as the *rubric* missing entries. Worse,
  `score_run` counts `judge - human` as false positives, and a charge the form
  never displayed can never be in `human`: 0% of charges on a pre-protocol run,
  **68%** on a protocol run, so precision read 2.2% vs 2.8% unfixed and 5.6% vs
  2.8% fixed — **the bug reversed the sign of the comparison**. Both sides are
  fixed and neither alone was enough. When a judge's rubric grows, the human
  form is a consumer of it (Section 21.75).
- **A control that is collected, stored, and read by nothing.** The adjudication
  form asks "Genuinely ambiguous — I could argue it either way", the server
  stores it, the ingest writes it to the gold-adjacent file, and `score_run`
  read `errors_present` and nothing else — so a reviewer who used the box
  changed no number, and their coin flip counted as a firm verdict. Every layer
  worked; the last one never looked. Worse than not having the control, because
  telling someone to use it makes the sample look cleaner than it is. When
  adding a field a human fills in, grep for it in the code that computes the
  headline before shipping the form (Section 21.55).
- **A prompt asking for new output, and every check that consumes it.** The
  gameplay prompt now requires `PHASE <step>` and `TAP <land> FOR <mana>` lines.
  `legality()` scored every parsed action against `legal_actions`, which
  enumerates *plays* — so the same correct answer, told the way the prompt
  demands, scored `all_legal=False` on every line it had been asked to add.
  Requiring verbosity would have collapsed Gate 1 across every arm and arrived
  as a finding: *"showing its working makes the model play worse."* Third time
  in this shape, after the copied `[TARGET x]` brackets and prose parsing as a
  play. `ParsedOutput.plays` (actions minus `DECLARATIONS`) is what legality,
  `only_pass` and the action count read now. **When a prompt starts asking for
  new output, audit every consumer of that output before running anything**
  (Section 21.61).
- **A validity check reading a list of alternatives as a list of permissions.**
  `legal_actions` enumerates every play legal *on its own*, so an answer that
  blocks one attacker with Fog Bank and then blocks a second with it matched
  twice and scored `all_legal=True` — 39 stored answers did. The inverse case
  works: a second land drop fails *because* it is not in the list. So the check
  is blind exactly to constraints that exist only **between** two individually
  legal choices, and it fails in the direction that reads as a pass. A human
  reviewer found it at n=12; nothing in the harness could (Section 21.49).
- **Prose about a play parsing AS that play.** "Play Mountain first would strand
  Shock in hand" became a `PLAY` action with a garbage operand, silently, with
  no `ParseFailure`. Any arm asked to reason would have had its legality
  destroyed by its own explanation, and it would have read as "reasoning makes
  the model play worse". Reasoning is bounded by an `ACTIONS:` marker and
  `<think>` blocks now, but the general shape recurs: free text in a field that
  is later parsed.
- **A helper duplicated with a guard in only some copies.** Five jsonl readers,
  three of which crashed on a trailing blank line.
- **An identifier that survives while its meaning changes.** An adjudication is
  keyed `record_id::arm`. Regenerate the arms — new adapter, new gameplay
  grammar — and the same key names a different answer, so 22 human verdicts
  would have scored against text their author never saw. Nothing fails: key,
  arm and position all match. The inverse of the promotion case below, and
  worse, because there the join silently missed and here it silently succeeds.
  Verdicts carry `answer_sha` now (Section 21.62).
- **An identifier that changes when a record is promoted.** A RulesGuru
  candidate is `rg-1156`; promoted into the gold set it becomes
  `qa-amy-casts-assassin-s-trophy-...` and keeps `rulesguru_id: 1156`. The
  contamination filter compared `id` to `id`, so 16 eval questions passed
  straight through a check that *asserted* it had excluded them. Anything
  joining two files on an id must first ask whether the id survived the trip.
- **A background job's progress lines sitting in a buffer.** `nohup python x.py
  > log 2>&1 &` block-buffers stdout at 8 KB when it is redirected, so `print()`
  progress never reaches the log until the buffer fills or the process exits —
  while tqdm, which writes to *stderr*, appears immediately. The log therefore
  looks alive and looks stalled at the same time: a run 36 minutes in had
  emitted its model-loading lines and not one of its `25/99` progress lines, and
  the obvious reading was that generation had hung. Launch with `python -u`, or
  monitor something the job cannot buffer — the artifact it writes, or its RSS
  (a 4 GB process is generating on the 7B; a 20 GB one has loaded the 32B judge
  and is scoring). Same family as the `pgrep` trap below: a monitoring signal
  that reads as one thing and means another.
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
- **A string in one language embedded in a file of another gets no checking
  from either.** `INDEX_HTML` in `webui.py` is a raw Python string holding the
  whole page. Two `#` comments were written inside it in Python style; Python
  does not strip them, so the browser received them as JavaScript, where `#` is
  a syntax error — and a syntax error is fatal to the entire `<script>`, so
  **all four views rendered blank**, including the three unrelated to the code
  the comments sat beside. It shipped for several commits. Nothing caught it:
  the module imports, every API endpoint returns 200 with correct JSON, and
  `--help` exits 0, because the server half was never broken. `test_webui.py`
  now parse-checks the served script with JavaScriptCore. **Testing the API of
  a page is not testing the page.**
- **Valid JSON in an unexpected shape, read as absence.** Twice. The judge
  emitted `"points_hit": 3` where a list was expected (Section 21.12), and — for
  a single candidate — the entry *unwrapped*, without the `{"A": …}` around it
  (Section 21.39). A missing key and a key spelled differently both arrive as
  `None`, so a third of a benchmark went "ungraded" while the judge had answered
  every question. Both times the wrong number was **plausible** (a strict judge,
  an expensive prompt), and both times two hypotheses were reasoned out and
  tested before anyone printed the raw text — both wrong. **When a parse yields
  a surprising count, look at the bytes before theorizing about them.**
- **A harness bug whose trigger rate depends on the condition under test.** The
  unwrapped-JSON bug above hit the V5 prompt on 8 of 24 calls and the V3 control
  on 0 — V5's added text mentions no output format, but naming a new per-entry
  field moved the judge off the `{"A": …}` wrapper. So the bug arrived wearing
  the shape of a result: *"the quote requirement costs a third of the coverage."*
  A bug that fires uniformly is visible as a bug; one correlated with the
  treatment is indistinguishable from a finding about the treatment. When an arm
  differs from its control on a **harness** metric — coverage, parse rate, drop
  count — rather than on the thing being measured, suspect the harness first.

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
