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
- **Error detection** — `wrong` was built to assert a specific enumerated error.
  How often does the judge actually report *that* error number? This is a direct
  measure of whether `errors_made` means anything, and blunder rate is defined
  on it.

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

### 1. Corpora

| Artifact | Guarded? | Gap |
| --- | --- | --- |
| `rules.jsonl` / `glossary.jsonl` | **yes** — sha256 content pin, `verify_cr_pin` | — |
| `oracle_cards.jsonl` | count only | no content pin; a re-fetch can change card text silently |
| `rulings.jsonl` | count only | same |

**Action:** extend the `CR_PIN` pattern to cards and rulings. The rules pin
exists because a parse that keeps the rule count and changes the text is
invisible to `guard_shrink` and to git. Cards have exactly the same exposure and
34,933 of them.

### 2. Derived artifacts

`chunks.jsonl` → `chunk_embeddings.npz` is the dangerous edge. Nothing checks
that the embeddings were built from the chunks currently on disk. Rebuild the
chunks without re-embedding and retrieval silently returns vectors for text that
no longer exists — every RAG arm degrades, and it looks like a model result.

**Action:** write a fingerprint of the source chunks into the `.npz`, and have
`rag.retrieve` refuse to run when it does not match. This is the same reasoning
as the CR pin, applied one layer down.

### 3. Prompts

`build_rag_messages` is used by both training and inference, which is what stops
them drifting. But **nothing records which prompt an adapter was trained under**,
and a prompt change silently invalidates every adapter — the project's #1
documented failure mode (Section 8.7), which cost a full re-run.

**Action:** write a hash of the prompt-defining strings into the adapter
directory at training time, and check it at eval time. Cheap, and it converts
the single most expensive failure this project has had into an error message.

### 4. Training data

The 18% refusal rate (Section 21.6) was found by reading the file, not by any
check. Nothing audits training data before it is trained on.

**Action:** a `audit_sft.py` reporting, for any dataset directory:

- refusal-shaped targets (the regex already exists in `build_sft_verified.py`)
- targets citing a rule id inline
- duplicate and near-duplicate targets
- length distribution
- **contamination against every file in `eval/sets/` and the gold set**, by id
  and by question text

Contamination is the one that must be an assertion rather than a report. It is
the only failure here that invalidates everything downstream *and looks like an
improvement while doing it*.

### 5. Harness

`test_actions.py` covers the parser at 84 assertions. **`eval.py` has no tests
at all**, and it holds the code that turns judge output into numbers:
`rubric_correctness`, `verify_quoted_claims`, `score_citations`,
`judge_prompt_for`, the label mapping.

**Action:** `test_eval.py`, same style — plain asserts, runnable. Priority order,
by how badly a silent bug would corrupt results:

1. `rubric_correctness` — the arithmetic every rubric score comes from
2. `score_citations` — grounding, now a first-class metric
3. `verify_quoted_claims` — V4's whole mechanism
4. `judge_prompt_for` — must be byte-identical at four arms

### 6. Judge

Beyond the positive controls above:

- **V4 vs V3** — queued. Does requiring a quote lift kappa?
- **Batching** — a fifth arm moved a byte-identical arm 23 points (Section
  21.5). Worth measuring whether scoring candidates *singly* changes the
  ranking. If it does not, singly is safer and removes the arm-count constraint
  entirely. If it does, that is a finding about the judge worth knowing.
- **Self-preference** — the judge is the same model as the `base` arm in every
  run to date. The Llama second judge partly covers this; a third family would
  settle it.

### 7. Evaluation data

Positions: 10 of 24 discriminate (Section 21.9). The equivalent audit has
**never been run on the 99 rules rubrics** — we do not know how many of them
separate the arms either.

**Action:** run the same per-record discrimination analysis over `gold_n99`.
Expect it to reclassify some of the gold set as uninformative, and to say which
categories are worth growing. Same shape as the position finding, applied to the
larger and more consequential set.

---

## Order of work

Sequenced by (what it unblocks) × (what it costs):

1. **Positive controls on the judge** — decides whether anything else is
   measurable. One eval run.
2. **`test_eval.py`** — the measurement arithmetic is currently untested. Hours.
3. **Discrimination audit of the 99 rules rubrics** — pure analysis over stored
   runs, no GPU.
4. **`audit_sft.py` with contamination as an assertion** — before any retrain.
5. **Embedding/chunk fingerprint** — before any RAG number is trusted again.
6. **Prompt hash beside the adapter** — before the next fine-tune.
7. **Card and ruling content pins** — lowest urgency; these corpora change rarely.

Items 1–4 need no new hardware and no new data authoring.

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
