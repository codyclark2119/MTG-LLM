# Structural audit: what to fix before hardware matters

The question this document answers: **at what point does more memory become the
binding constraint rather than a more expensive way to be wrong?**

The honest current answer is "not yet." Thirteen defects surfaced in one working
session and none were resource-limited — four in judge and metric design, four
harness bugs, four in data and authoring, one in documentation. Meanwhile a 14B
model at 4-bit runs in 7.0 GB of 39 GB. The machine is not the limit; the
instrument is.

This is a plan to find out when that stops being true.

---

## The one test that decides it

Everything below is worth doing, but one measurement dominates: **the judge
cannot currently be shown to work.**

Every number in this project is read through a judge whose inter-judge kappa is
+0.24 and whose two instances disagree on 31 of 88 blunder calls. We have never
established what the judge does with answers of *known* quality. Without that,
"fine-tuned scores 1.88 and base_rag scores 2.38" is uninterpretable — the gap
might be real, or the whole scale might be noise.

### Positive controls

Build a calibration set from the existing gold questions. For each, four
synthetic candidates of known quality:

| Candidate | Construction | Expected |
| --- | --- | --- |
| `oracle` | the reference answer, verbatim | hits every key point, ~5 |
| `partial` | reference truncated to its first sentence | hits some, mid |
| `wrong` | an answer asserting a listed `common_error` | commits that error, ~1 |
| `refusal` | "The rules provided do not cover this." | hits nothing, 1 |

Run the judge over these exactly as it runs over real arms. Three numbers come
out, and they are the ones that matter:

- **Dynamic range** — `oracle` mean minus `wrong` mean. This is the scale on
  which every model difference is measured. If the range is 3.2 points, a
  0.5-point model gap is 16% of range and plausibly real. If the range is 0.8,
  nothing this project has measured means anything.
- **Ordering accuracy** — how often the judge ranks `oracle` > `partial` >
  `wrong` on a single question. A judge that inverts this on 30% of questions
  cannot support a 0.4-point claim.
- **Error detection** — the original plan was to build `wrong` so it asserts a
  specific enumerated error and count how often the judge reports *that* number.
  **Not implemented, deliberately.** Turning a rubric line written as a
  description of a mistake into a first-person answer asserting it means
  generating prose, so a low detection rate would be unattributable between "the
  judge cannot spot it" and "the generated sentence did not clearly assert it".
  `errors_made` is checked from the other side instead, which needs no
  construction: the oracle *cannot* commit a listed error, so every one reported
  against it is a definitive false positive. Measuring the true-positive side
  needs `common_errors` authored as assertions — a data change, not a harness one.

**Cost:** one eval run, no training, no new data authoring. The candidates are
constructed mechanically from records that already exist.

**This is the single highest-value thing in this document** and should be done
first. It is also the test that tells you whether to buy a computer: if the
judge separates known-good from known-bad cleanly, the instrument is sound and
model capability becomes the constraint. If it does not, a better model will
not show up in the numbers, and a bigger machine buys nothing.

---

## Layer-by-layer

Working from the inputs outward. For each layer: what could be silently wrong,
and what check would catch it.

### 1. Corpora — **done** (Section 21.16)

| Artifact | Guarded by |
| --- | --- |
| `rules.jsonl` / `glossary.jsonl` | `CR_PIN`, via `verify_cr_pin` from `load_rule_ids` |
| `card_chunks.jsonl` | `CARD_PIN`, via `verify_card_pin` from `CardIndex.__init__` |
| `ruling_chunks.jsonl` | `CARD_PIN` (opt-in second check) |

The *processed* corpora are pinned rather than the raw dumps, matching how
`rules.jsonl` is pinned rather than the raw CR text: it is the layer everything
downstream reads, and it catches a chunker change as well as a re-fetch.
Verified by simulating errata — one word changed at an unchanged record count is
refused, and a non-canonical `--chunks` path is still allowed.

**`ruling_chunks.jsonl` is read by no script.** 19,726 chunks ingested and never
wired into retrieval. It backs no published number; it is simply not doing
anything.

### 2. Derived artifacts — **already done; this entry was wrong**

The claim here was that nothing checks the embeddings against the chunks on
disk. That was written from a reading of the audit surface, not from the code:
`rag.fingerprint` has existed all along, `build_index` writes both
`chunks_fingerprint` and `model_id` into the `.npz`, and `rag.retrieve` raises
on either mismatch *before* loading the embedding model.

Verified by running rather than by reading a second time. A copy of
`chunks.jsonl` with one word appended to one chunk — 448 chunks either way, so
neither `guard_shrink` nor a line count would notice — is refused:

    real corpus      -> retrieval runs
    tampered corpus  -> ValueError: chunk_embeddings.npz is stale

The one residual gap is that both guards are conditional (`if stored_fp and
...`), so an `.npz` predating them would pass silently. The only index in the
repo carries both keys, so the guard is live today; a future rebuild that drops
them would be undetectable.

### 3. Prompts — **done** (Section 21.15)

`scripts/stamp_adapter.py` writes `prompt_fingerprint.json` beside the weights;
`eval.py` checks it before generating a single answer. A mismatch is fatal, a
missing stamp is a warning.

Two things worth keeping. The fingerprint covers the assembled **message
shape**, not just the three system strings — the Section 8.7 failure was a shape
change with every string untouched, so a hash of the strings alone would have
passed through the exact failure it was written for. And `--dataset` verifies the
stamp against the training data rather than asserting it, refusing to stamp a set
built under a prompt the code no longer defines.

The open question is also now answered: `data/datasets` and
`data/datasets/verified` contain **only current prompts, byte-identical**, so
runs 1–3 are valid for the prompts `eval.py` uses today. v2-best and
v3-ckpt1322 are stamped.

### 4. Training data — **done** (Section 21.13)

`audit_sft.py` reports refusals, inline citation, duplicates, length
distribution, and contamination — the last as an assertion that exits non-zero.

**It found contamination on its first run, and the builder's own assertion had
been passing.** 16 of the 99 gold eval questions were in the pending run-4
training set: they were promoted into gold under a `qa-*` id while keeping their
`rulesguru_id`, so `candidate["id"]` never equalled `gold["id"]`. Two more were
duplicate RulesGuru entries that no id comparison can catch. The builder now
filters on all three (72 + 16 + 2 = 90 excluded), and the set was rebuilt before
being trained on — which is the entire point of running this before a retrain.

The synthetic set behind the published v2 adapter was audited retrospectively
too: **44% duplicate lines** (1,478 unique behind 2,644) and three verbatim eval
questions, on which the fine-tuned arms scored *worse* than their own average,
so the A3 verdict is unchanged.

### 5. Harness — **done** (Section 21.12)

`test_eval.py` covers the scoring arithmetic at 96 assertions, in the priority
order this section originally set out: `rubric_correctness`, `score_citations`,
`verify_quoted_claims`, `judge_prompt_for`, plus `stratified_sample`,
`pearson_r`, `_author_of`, and `judge_batch_rubric` end to end with the model
stubbed out — which is what finally covers the **label mapping**, where an
off-by-one would attribute every arm's score to a different arm and look
entirely normal doing it.

Writing them found one defect: valid JSON of the wrong *shape*
(`"points_hit": 3` for `[3]`) parsed cleanly past both `json.loads` guards and
then raised `TypeError`, with no handler between it and `main()` in any of the
four callers — a multi-hour run dying at whatever question the judge fumbled.
Fixed in `_as_claim_list` / `_claim_index`, verified not to move any published
number: 1,190 + 821 exhaustive well-formed inputs and all 1,782 stored rubric
scores recompute identically.

### 6. Judge

Beyond the positive controls above:

- **V4 vs V3** — **ran, unusable** (Section 21.14). 86% of arm-answers came back
  as unparseable JSON: requiring a quote per claim makes the required output
  scale as arms × rubric items, and the judge runs out of tokens mid-JSON. Parse
  rate falls 37% → 8% as rubrics grow from 4 to 7 items, so the 14 questions
  that survived are selected by rubric size. Re-run with a much larger
  `--judge-max-tokens` before drawing anything from V4 — the question of whether
  a quote lifts kappa is still open, not answered.
- **Batching** — a fifth arm moved a byte-identical arm 23 points (Section
  21.5). Worth measuring whether scoring candidates *singly* changes the
  ranking. If it does not, singly is safer and removes the arm-count constraint
  entirely. If it does, that is a finding about the judge worth knowing.
- **Self-preference** — the judge is the same model as the `base` arm in every
  run to date. The Llama second judge partly covers this; a third family would
  settle it.
- **Rubrics are ruled out as a lever** (Section 21.20). Disagreement is diffuse,
  not localized: Gini 0.45–0.48 across rubric items, worst 10% carrying ~23% of
  disputes, and 91 of 99 rules records plus 19 of 22 positions carrying at least
  one. There is no handful of rubrics to rewrite. Section 16.12's contrary
  reading was n=8.

**This narrows the migration question.** The levers on judge agreement were
better rubrics, a better judge prompt, or a better judge model. Rubrics are ruled
out by measurement; the prompt was tried and V4 cost 60–85 points of coverage
without buying agreement (21.14). That leaves the judge model — which is exactly
the case this document names as the only genuine reason to migrate. It does not
settle it: the positive controls remain the deciding test, and a bigger judge may
disagree just as diffusely. But the cheap alternatives are gone.

### 7. Evaluation data — **done** (Sections 21.11 and 21.17)

Neither eval set is the constraint, which is the opposite of what this section
expected and of what 21.11 first concluded.

| Set | Separate the arms |
| --- | --- |
| rules gold (n=99), correctness | 74% by ≥1.0 point, mean spread 1.83 |
| positions (n=22), correctness | **82–86%** |
| positions (n=22), binary blunder | 45–55% |

The "10 of 24 positions discriminate" figure was measured on the **binary
blunder call**, which is Gate 2's metric — not on the positions. Scored on the
same 1–5 scale the rules set uses, the positions are marginally the *stronger*
instrument. Section 21.17 has the correction.

So the gap is in the metrics and the judge, not the data: rules eval is
constrained by judge agreement (r = +0.49 on identical answers), gameplay eval
by a gate defined on a binary that discards half the separation the judge
already produced.

---

## Order of work

Sequenced by (what it unblocks) × (what it costs):

1. **Positive controls on the judge** — decides whether anything else is
   measurable. One eval run. *(queued behind the Qwen3 position rerun)*
2. ~~**`test_eval.py`**~~ — **done**, Section 21.12. Found a run-killing shape
   bug; no published number moved.
3. ~~**Discrimination audit of the 99 rules rubrics**~~ — **done**, Section
   21.11. The rules set is healthy (74% separate the arms, mean spread 1.83);
   the *gameplay* set is the one that does not discriminate.
4. ~~**`audit_sft.py` with contamination as an assertion**~~ — **done**, Section
   21.13. Found 16 eval questions in the pending run-4 training set that the
   builder's own assertion had passed.
5. ~~**Embedding/chunk fingerprint**~~ — **already existed**; the audit entry was
   wrong. `rag.retrieve` refuses a stale index, verified by tampering with a
   copy of the corpus.
6. ~~**Prompt hash beside the adapter**~~ — **done**, Section 21.15. Also
   established that runs 1–3 were trained under the current prompts.
7. ~~**Card and ruling content pins**~~ — **done**, Section 21.16.

**Every item on this list is now closed except the judge work**, and none of
them needed new hardware. Between them they turned up four defects that no
amount of memory would have fixed:

- a run-killing crash on valid JSON of the wrong shape (21.12)
- 16 eval questions inside a pending retrain, past an assertion that said there
  were none (21.13)
- a report naming a judge prompt that had not run, and averaging 14 of 99
  questions without saying so (21.14)
- a live-API corpus with no content pin (21.16)

That leaves the judge, which was always the item that decides the migration
question.

---

## When memory becomes the constraint

Concrete trip-wires. Migrate when **both** of the following hold:

**A. The instrument is sound.** From the positive controls:

- dynamic range (`oracle` − `wrong`) **≥ 2.5 points** on the 1–5 scale
- ordering accuracy **≥ 90%** of questions
- inter-judge kappa on the blunder call **≥ 0.6** (currently **+0.24**)

Until these hold, a better model cannot be distinguished from a worse one, and
the extra memory is spent producing a more expensive wrong answer.

**B. The work actually needs the memory.** Specifically one of:

- **A judge that does not fit.** This is the likeliest real trigger. Judge
  quality caps every number in the project, and a 70B judge at 4-bit is ~40 GB,
  needing 64 GB+ to run alongside anything else. If the positive controls show
  the *judge model* is the limit rather than the judge prompt, this is the
  reason to move.
- **A base model that does not fit.** Not currently binding: Qwen3-14B runs in
  7 GB of 39 GB, and a 24B would fit. This binds at ~32B and above.
- **Training that does not fit.** Also not currently binding — `batch_size: 2`
  was chosen for throughput, not memory, at a 9.2 GB peak.

The asymmetry is worth stating plainly: **the case for more memory is about
grading models, not running them.** A migration justified by "we want a bigger
model under test" is premature at 7 GB of 39 GB. A migration justified by "the
judge is the measured bottleneck and the better judge needs 64 GB" is a real
argument — and the positive-control test is what would produce that evidence.
