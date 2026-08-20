# Structural audit: what to fix before hardware matters

The question this document answers: **at what point does more memory become the
binding constraint rather than a more expensive way to be wrong?**

The honest current answer is "not yet." Thirteen defects surfaced in one working
session and none were resource-limited — four in judge and metric design, four
harness bugs, four in data and authoring, one in documentation. Meanwhile a 14B
model at 4-bit runs in 7.8 GB of 36 GB. The machine is not the limit; the
instrument is.

This is a plan to find out when that stops being true.

> **Where it stands (Sections 21.31, 21.40, 21.41).** One thing did turn out to
> need a bigger model: **error detection works on a 32B judge and does not work
> on a 7B** — 1 false positive in 24 against 18, on identical positions and
> rubrics. Blunder rate is defined on that field, so half the instrument was
> judge-limited all along.
>
> That is still not a reason to migrate. **The judge that fixed it is 18 GB and
> the machine is 36.** It moves the answer from "no bigger judge has been shown
> to help" to "one did, and it fits" — which is a *stronger* not-yet, because the
> next size up (70B, 38.6 GB) is the first thing that would not fit, and the 32B
> already scores 0% false errors and 100% ordering accuracy. There is little left
> for a 70B to fix.
>
> The open item is no longer capacity. It is that **every gate verdict published
> so far is single-judge**, and Gates 2 and 3 have both reversed between judges
> on byte-identical answers.

---

## The one test that decides it — **it ran, and here is the answer**

> **RESULT (Section 21.27).** The positive controls ran under both judges, 98%
> and 95% coverage. The instrument splits cleanly in half: **the correctness
> scale works and the error detection does not**, and the broken half is
> judge-specific rather than fundamental.
>
> | Trip-wire | Required | Qwen2.5-7B | Llama-3.1-8B | **Qwen2.5-32B** |
> | --- | --- | --- | --- | --- |
> | dynamic range | ≥ 2.5 | +3.22 PASS | +2.66 PASS | **+3.60 PASS** |
> | ordering accuracy | ≥ 90% | 93% PASS | 94% PASS | **100% PASS** |
> | false errors on oracle | ≤ 5% | **40% FAIL** | 4% PASS | **0% PASS** |
> | credits the oracle / real answers | — | 86% / 43% | 81% / 63% | **90% / 8%** |
>
> **Llama passes all three; the 32B wins every one** (Section 21.31, under
> matched `points_only` arithmetic — the two older runs were inverted rather
> than re-run). So there is a working instrument on this hardware today, and the
> answer to "does more memory help" is no — the failing number was a property of
> one 7B model, not of the method, the prompt, or the machine.
>
> **The last row is why the 32B is not merely the best of three.** A judge that
> credits the *reference answer* with 90% of its own rubric understands the
> rubric, so crediting real answers at 8% is a statement about the answers. That
> ratio is 11.2× against Llama's 1.3× — one of these barely separates a model
> answer from the reference, the other separates them by an order of magnitude.
>
> Consequences, all applied: correctness now scores `points_hit` alone
> (Section 21.28), and the position eval no longer judges itself with the
> miscalibrated model (Section 21.29). The old rule is reproducible from the
> same stored judge output, so nothing in the record was lost.
>
> The rest of this section is the reasoning as written *before* the test, kept
> because it is what the test was designed against.

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
- **Error detection** — build `wrong` so it asserts a specific enumerated error
  and count how often the judge reports *that* number. **Deferred for a stated
  reason, and now implemented because the reason expired** (Section 21.40).

  The objection was that turning a rubric line written as a description of a
  mistake into an answer asserting it means generating prose, so a low detection
  rate would be unattributable between "the judge cannot spot it" and "the
  generated sentence did not clearly assert it". The precondition named here was
  `common_errors` authored as assertions — *a data change, not a harness one*.

  **Section 21.35 made that data change**, so the candidate is now the rubric
  line VERBATIM and no prose is generated. `calibrate_judge.py --sensitivity`
  runs it; `looks_like_behaviour` skips entries still in the old form rather
  than rewriting them, which is what keeps the objection answered rather than
  ignored. Positions are 24/24 usable, the rules gold set 42/99.

  **This mattered.** Specificity alone cannot distinguish a precise judge from a
  silent one, and two plan sections reported a judge prompt as "N fixed, 0
  regressions" on oracle-only evidence — where losing a true positive is
  unobservable by construction. It had lost two.

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
  that survived are selected by rubric size.
- **V5 — built, measured, and not adopted** (Sections 21.37–21.40). V4's cost
  was quotes on *everything*; V5 asks for one on `errors_made` only, plus a
  sentence telling the judge that an answer stating the OPPOSITE of an error has
  not committed it. On the 7B it removes 4 false positives, and the sensitivity
  control shows it **loses 2 true positives** doing so. On the 32B it does
  nothing measurable: **zero quote drops across all 96 gradings**, because that
  judge never fabricates an unquotable error claim in the first place. V3 stays
  the default; V5 stays in the code, tested, for a future judge that shows the
  7B's failure. **A better judge model turned out to dominate a better judge
  prompt**, which is the finding, not the prompt.
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

> **ANSWER, now that the controls have run: not yet, and for a reason that has
> nothing to do with memory.**
>
> Condition A is **partly met and the gap is not hardware-shaped.** With
> Llama-3.1-8B as judge, all three instrument trip-wires pass on the current
> machine. The failing measurement was Qwen2.5-7B inventing errors at 40%, and
> the fix was to stop using it for that — free, and applied.
>
> Condition B is **not met.** Qwen3-14B runs in 7.8 GB of 36 GB and is not
> better at the task (Section 21.25: best-arm blunder 53% → 50%, ~60× the wall
> time). A 70B judge does **not fit in 36 GB at all** — 38.6 GB of weights — so
> that is the only real trigger, and nothing yet says a 70B judge would help.
>
> **CORRECTION (Section 21.40).** This section previously read "a 32B judge fits
> in ~18 GB and *has not been shown necessary*." It has now been shown necessary.
> On the same 24 positions with the same rubrics, judge as the only variable:
>
> | | charges a clean answer | errors fired at an answer carrying one |
> | --- | --- | --- |
> | Qwen2.5-7B | 18/24 | 2.75 |
> | **Qwen2.5-32B** | **1/24** | **1.12** |
>
> Blunder rate is defined on `errors_made` being non-empty, so on the 7B that
> metric does not work at all and on the 32B it does. Judge scale **was** the
> lever, for this half of the instrument.
>
> **This strengthens the "do not migrate" answer rather than weakening it.** The
> question has moved from *"we do not know whether a bigger judge helps"* to
> *"a bigger judge helped, and the size that helped is 18 GB of 36."* The next
> trigger is unchanged and now sharper: a 70B judge is the only thing that would
> not fit, and there is still no evidence one is needed — the 32B already scores
> 0% false errors and 100% ordering, so there is little left for a 70B to fix.
>
> What is left before hardware could become the constraint:
>
> 1. **Kappa.** The blunder-call kappa of +0.24 was measured with the 40%-false-
>    positive judge on one side. It has not been re-measured with two calibrated
>    judges, and that number could move a long way on its own. This is now the
>    *only* open item of the three, and it is the one that matters most: the
>    published gate verdicts are single-judge until it is done.
> 2. ~~The 32B rescore~~ — **landed** (Sections 21.31, 21.40, 21.41).
> 3. **A third judge family**, to settle self-preference. Mistral-24B or
>    Gemma-27B both fit comfortably. Still open, and note that the judge is a
>    Qwen and every `base` arm is a Qwen.
>
> One further caveat, from Section 21.41: the 32B's clean rates are established
> at 122 characters (positions) and 236 (rules reference answers), and are
> **untested at the ~1,000 characters real rules answers run to**. An attempt to
> close this produced n=21 with a confound and no length trend — not an answer.
>
> All of these run on this machine. The reasoning below is preserved as written.

Concrete trip-wires. Migrate when **both** of the following hold:

**A. The instrument is sound.** From the positive controls:

- dynamic range (`oracle` − `wrong`) **≥ 2.5 points** on the 1–5 scale
- ordering accuracy **≥ 90%** of questions
- inter-judge kappa on the blunder call **≥ 0.6** (measured **+0.24**, but with
  a judge since shown to invent the blunder 40% of the time — Section 21.27.
  This needs re-measuring between two *calibrated* judges before it means
  anything, and it is the last of the three that is still genuinely open)

Until these hold, a better model cannot be distinguished from a worse one, and
the extra memory is spent producing a more expensive wrong answer.

**B. The work actually needs the memory.**

The machine is **36 GB**, not the 39 GB three earlier drafts of this document
said. Measured 4-bit footprints on disk: Qwen2.5-7B 4.0 GB, Llama-3.1-8B 4.2 GB,
Qwen3-14B 7.8 GB — about **0.55 GB per billion parameters**. Extrapolating:

| Judge | 4-bit weights | Fits in 36 GB? |
| --- | --- | --- |
| current (7–8B) | 4.0–4.2 GB | trivially |
| 14B | 7.8 GB | yes, measured |
| 24B | ~13 GB | yes |
| **32B** | **~17.6 GB** | **yes, with room to spare** |
| 70B | ~38.6 GB | **no — exceeds total RAM** |

This corrects the earlier claim that a base model "binds at ~32B and above". It
does not. Inference binds somewhere around 55–60B; 32B is comfortable.

**Which means the decisive experiment has not been run, and does not need new
hardware.** Section 21.20 ruled out rubrics as a lever and 21.14 ruled out the
judge prompt, leaving judge *model* as the only remaining candidate — and the
jump from an 8B judge to a 32B judge is a 4× scale increase available today, for
the cost of a download. The audit went straight from "8B judges disagree" to "a
70B judge needs 64 GB" and skipped the middle.

The decision logic is now clean, and it resolves either way:

- **A 32B judge materially improves agreement** → judge scale is the lever.
  Every number in the project improves immediately, *and* there is measured
  evidence that 70B (which genuinely needs 64 GB+) would be worth the migration.
- **A 32B judge does not improve agreement** → scale is not the lever. A 70B
  judge is unlikely to behave differently in kind, and the migration buys a more
  expensive version of the same disagreement.

Either outcome is worth more than migrating first and finding out after.

**Not currently binding, for completeness:**

- **A base model that does not fit.** Qwen3-14B runs in 7.8 GB of 36 GB.
- **Training that does not fit.** `batch_size: 2` was chosen for throughput, not
  memory, at a 9.2 GB peak.

The asymmetry still holds: **the case for more memory is about grading models,
not running them.** But the honest sequence is now 32B judge first, migration
only if that test comes back positive.
