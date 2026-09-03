"""Positive controls: show what the judge does with answers of KNOWN quality.

    python scripts/calibrate_judge.py --limit 40
    python scripts/calibrate_judge.py --judge-model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit

WHY

Every number in this project is read through a judge whose inter-judge kappa is
+0.24 and whose two instances disagree on 31 of 88 blunder calls. What the judge
does with answers whose quality is known in advance has never been measured. Until
it is, "finetuned_rag 1.88 vs base_rag 2.38" is uninterpretable: the gap might be
real, or the scale might be noise.

This runs the judge over four candidates per question, built mechanically from
records that already exist — no authoring, no training, no model generation at
all. Four candidates, deliberately, so the judge prompt is byte-identical to
every published four-arm run.

    oracle    the reference answer, verbatim
    partial   the reference cut to roughly half, at a sentence boundary
    wrong     a DIFFERENT question's reference answer, same category
    refusal   "The rules provided do not cover this."

`wrong` is another real answer rather than gibberish on purpose. A judge that
separates the reference from noise but not from a fluent answer to a different
question is being fooled by fluency, which is exactly the failure mode a rubric
judge is supposed to remove.

WHAT COMES OUT, AND WHY EACH ONE MATTERS

  dynamic range     oracle mean minus wrong mean. This is the scale every model
                    comparison is measured on. A 0.5-point model gap on a
                    3.2-point range is plausibly real; on a 0.8-point range,
                    nothing this project has measured means anything.

  ordering accuracy how often oracle > partial > wrong holds on a single
                    question. Aggregate means can look healthy while individual
                    rankings are near-random.

  false errors      how often the judge says the ORACLE committed a listed
                    common_error. The oracle IS the reference answer, so it
                    cannot commit one; any hit here is a definitive false
                    positive. Section 21.3 caught a judge firing all three
                    errors at an answer that committed none, using the error
                    list to signal "this answer is bad" — blunder rate is
                    defined on exactly this field.

  refusal floor     the refusal should sit at the bottom of the scale. If it
                    does not, the scale has no floor.

These are what STRUCTURAL_AUDIT.md's migration trip-wires are stated against.

ERROR DETECTION: --judge-report (BOTH halves, never one)

Everything above measures SPECIFICITY. A judge that fires no errors at all
scores a perfect 0% on `false errors` and is worthless, so the four controls
alone cannot tell a precise judge from a silent one. Sections 21.38 and 21.39
reported a judge prompt as "N fixed, 0 regressions" on exactly this evidence,
where losing a true positive is unobservable by construction; the control below
found it had lost two (Section 21.40).

    python scripts/calibrate_judge.py --judge-report --gold data/gold/positions.jsonl

**There is deliberately no way to ask for one half.** Sections 21.40, 21.42 and
two STRUCTURAL_AUDIT edits were all published from a one-sided number and all
overturned. The failure is not carelessness — a one-sided report reads as
confident. All four judges tested catch the planted error 24/24; three of them
report a mean of 1.00–1.12 errors fired, which looks like precision. Llama-3.1-8B
has the BEST number in that column and fires at 58% of answers containing no
error at all.

So the headline is the **separation**, `P(fire | error) − P(fire | clean)`, which
cannot be computed from one half by construction:

    Qwen2.5-32B    +96      the only judge that separates them
    Llama-3.1-8B   +42
    Qwen2.5-7B     +25
    Qwen3-14B       ?       sensitivity 1.00, clean-answer rate unmeasured

Both halves run over the **same records**, because rates over different question
sets do not subtract.

This was previously declined, and the reason is worth keeping because it is what
changed. Building a candidate that commits a KNOWN error means turning a rubric
line into an answer asserting it — and while `common_errors` were written as
descriptions of behaviour ("Adds Centaur Courser to the block, spending a 3/3 to
save 3 life"), every such rewrite was prose I would be writing. A low detection
rate would then be unattributable between "the judge cannot spot the error" and
"my sentence did not clearly assert it": measuring my paraphrasing, not the
judge. The conclusion at the time was that this needed common_errors authored as
assertions in the first place — a data change, not a harness one.

**Section 21.35 made that data change.** A `common_errors` line is now required
to be a sentence a wrong answer could contain VERBATIM, so the candidate is the
rubric line itself and no paraphrase is involved. `common.looks_like_behaviour`
is the same check the authoring lint uses, and entries still in the old form are
SKIPPED rather than rewritten.

Coverage differs sharply between the two sets, and the difference is the whole
reason the report leads with it:

    positions.jsonl    83 errors,   0 behaviour-shaped     24/24 records usable
    gold_questions     264 errors, 214 behaviour-shaped    42/99 records usable

Positions were rewritten wholesale in 21.35. The rules gold set was NOT — 21.36
refuted the justification for converting it — so the 42 usable records are the
ones that happen to have been authored as claims. **That is a selected subset,
not a sample**, and a rate over it describes rubrics written in a particular
style rather than the gold set. The report says so whenever the drop is large;
treat 42/99 as a reason to widen the data, not as an n of 42.

Ground truth is then definitional: the answer contains the listed error word for
word, so a judge that does not fire it has a false negative. It also puts a
quote-verifying prompt (V5) on its easiest possible case, since the quote is
present exactly.

ONE ARM, AND WHY THAT IS NOT THE FOUR-ARM NUMBER

The report grades one candidate at a time, not four. Its two halves cannot be
folded into the four-arm run as extra arms: `judge_batch_rubric` grades every
candidate in ONE batched call, so a fifth arm changes all four others (23 points,
Section 21.5) and would invalidate the calibration against every published
four-arm run.

So a rate from this report is comparable to another rate from **this** report —
which is what makes the judge-vs-judge table in 21.42 sound — and is NOT
comparable to the `false errors` trip-wire above. That distinction is currently
load-bearing and unresolved: Llama-3.1-8B scores **4%** on the four-arm rules run
and **58%** here on positions. Arm count and rubric type both differ, and if arm
count turns out to be the cause then every number this report has produced is a
measurement of the harness.

Single-arm gradings also come back **unwrapped** from some judges, which read
silently as ungraded until Section 21.39 — that cost a third of a benchmark and
was diagnosed as a property of the prompt under test. `judge_batch_rubric`
handles the shape now, and coverage is printed here so a recurrence is visible
rather than averaged over.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    REPO_ROOT,
    looks_like_behaviour,
    read_jsonl,
    render_position,
)
from eval import (  # noqa: E402
    BASE_MODEL_ID,
    judge_batch_rubric,
    load_gold_questions,
    score_one_question,
)
from harness.core.calibration.controls import (  # noqa: E402
    half_answer,
    build_candidates,
    build_error_assertions,
    separation,
)

GOLD_EVAL_PATH = REPO_ROOT / "eval/sets/gold_questions_eval.jsonl"
REFUSAL = "The rules provided do not cover this, so I cannot answer the question."

# Appended so the candidate reads as an answer rather than a dangling fragment.
# Each asserts nothing on its own — the claim being detected is the rubric line
# in front of it, verbatim — and both are fixed constants rather than per-record
# prose, which is the whole point of the construction. Two of them only because
# "that is the play here" is nonsense attached to "is it a legal target for
# Flashback"; a tail that does not fit its question is prose that draws
# attention to itself, which is the thing being avoided.
ASSERTION_TAIL = ". That is the play here."
ASSERTION_TAIL_RULES = ". That is the ruling."


def question_text(rec: dict) -> str:
    """What the judge is shown as the question.

    A position's board is RENDERED, never stored as text (POSITIONS.md), so this
    has to branch rather than read a field. `battlefield` is the discriminator
    because every position has one and no rules question does.
    """
    if "battlefield" in rec:
        return render_position(rec)
    if rec.get("question"):
        return rec["question"]
    for m in rec.get("messages") or []:
        if m.get("role") == "user":
            return m["content"]
    return ""


def reference_answer(rec: dict) -> str:
    """The answer that definitionally commits no listed error.

    Three shapes, because the two gold files and the eval file differ:
    positions and `data/gold/gold_questions.jsonl` carry `answer`, while
    `eval/sets/gold_questions_eval.jsonl` carries only the assembled `messages`
    and the reference lives in the assistant turn. Returning "" for a record
    with no reference is deliberate — `run_judge_report` refuses to run rather
    than grading a blank as a clean answer, which would score as a perfect
    specificity result.
    """
    for key in ("answer", "reference"):
        if (rec.get(key) or "").strip():
            return rec[key]
    for m in reversed(rec.get("messages") or []):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            return m["content"]
    return ""


def pad_clean(rec: dict, answer: str, rule_text: dict) -> str:
    """Lengthen a clean answer without letting it become a wrong one.

    Section 21.41 could not settle whether the 32B's low false-positive rate
    survives answer-length text: its clean rates are measured at 122 characters
    (positions) and 236 (rules references) while real rules answers run to
    ~1,000. The attempt to test it used *model* answers that happened to credit
    every key point — which is not a clean control, because such an answer can
    state everything right and still say something wrong alongside it.

    This appends the **verbatim Comprehensive Rules text** of the rules the
    record already cites. That adds several hundred characters of prose which
    cannot introduce an error: it is the rulebook, quoted, on the rule the
    reference answer is about. So the answer stays definitionally clean and
    gains length, which is the one variable being tested.

    Returns the answer unchanged when no cited rule resolves — a record that
    cannot be padded must not silently count as a padded one.
    """
    ids = rec.get("rule_citations") or rec.get("supporting_rule_ids") or []
    quoted = [f"{r}. {rule_text[r]}" for r in ids if r in rule_text]
    if not quoted:
        return answer
    return answer.rstrip() + "\n\nFor reference, the rules text:\n" + "\n".join(quoted)


def _grade(lm_generate, model, tok, rec: dict, answer: str, args, rng) -> dict | None:
    """Grade one candidate against one record's rubric. None when ungraded."""
    out = judge_batch_rubric(
        lm_generate, model, tok, question_text(rec),
        rec["key_points"], rec["common_errors"], {"candidate": answer},
        args.judge_max_tokens, rng, judge_version=args.judge_prompt)
    d = out.get("candidate") or {}
    if d.get("errors_made") is None:
        return None
    fired = [e["n"] if isinstance(e, dict) else e for e in d["errors_made"]]
    return {"fired": fired, "n_fired": len(fired),
            "quote_drops": d.get("quote_drops") or 0}


def run_judge_report(args) -> None:
    """Both halves of the error-detection control, in one pass, on one set.

    Deliberately NOT two flags. A one-sided report is what produced Sections
    21.40, 21.42 and one STRUCTURAL_AUDIT edit, all three overturned.

    Both halves run over the SAME records — the ones carrying a plantable
    error — because `separation` subtracts one rate from the other, and rates
    over different question sets do not subtract. On positions the two sets
    coincide (24/24); on the rules gold set they do not (42 of 99), and taking
    specificity over all 99 while sensitivity covers 42 would silently compare
    a judge against itself on two different exams.
    """
    records = [r for r in read_jsonl(args.gold)
               if r.get("key_points") and r.get("common_errors")]
    if args.limit:
        records = records[:args.limit]
    # A single run reads records of ONE shape (positions XOR rules questions --
    # args.gold is one file), so the tail is decided once here rather than
    # per-record inside build_error_assertions, which only knows a caller's
    # generic assertion_tail, not this project's two-tail split.
    assertion_tail = (ASSERTION_TAIL if records and "battlefield" in records[0]
                      else ASSERTION_TAIL_RULES)
    # harness.core's is_usable_error(e) must return True for a USABLE (claim-
    # shaped) entry; looks_like_behaviour(e) returns True for the opposite
    # (a behaviour-shaped entry) -- passing it directly inverted the filter
    # (caught by comparing old-vs-new dry-run output: 4/5 usable, 8 skipped
    # became 5/5 usable, 5 skipped, on records this run never actually
    # changed).
    cases, skipped_entries, skipped_records = build_error_assertions(
        records, assertion_tail, is_usable_error=lambda e: not looks_like_behaviour(e))
    if not cases:
        raise SystemExit(
            f"{args.gold}: no claim-form common_errors to plant. Every entry reads as a "
            "description of behaviour, so asserting one verbatim would not produce an "
            "answer — see Section 21.35 and SCHEMA.md rule 5.")
    missing = [c["id"] for c in cases if not reference_answer(c["rec"]).strip()]
    if missing:
        raise SystemExit(
            f"{args.gold}: {len(missing)} records have no reference answer, so the clean "
            f"half cannot be built (first: {missing[0]}). Both halves are required.")
    print(f"{len(cases)}/{len(records)} records usable "
          f"({skipped_entries} behaviour-shaped entries skipped, "
          f"{skipped_records} records dropped entirely)")

    if args.dry_run:
        for c in cases[:2]:
            print(f"\n--- {c['id']} ---")
            print(f"  clean  : {reference_answer(c['rec'])[:110]}")
            print(f"  planted (error {c['n']}): {c['answer'][:110]}")
        return

    rule_text = {}
    if args.pad_clean:
        from common import RULES_PATH
        rule_text = {r["rule_id"]: r["text"] for r in read_jsonl(RULES_PATH)}
        padded = sum(1 for c in cases
                     if pad_clean(c["rec"], reference_answer(c["rec"]), rule_text)
                     != reference_answer(c["rec"]))
        print(f"--pad-clean: {padded}/{len(cases)} clean answers lengthened with "
              f"verbatim CR text")
        if padded < len(cases) * 0.5:
            raise SystemExit(
                f"only {padded}/{len(cases)} records have a resolvable rule citation, so "
                "most clean answers would be unpadded and the run would measure a mixture "
                "of two lengths rather than one.")

    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm
    print(f"loading {args.judge_model} as judge ...")
    model, tok = load_lm(args.judge_model)

    # One RNG per half, both seeded identically: the label shuffle must not
    # differ between the halves, or the two rates differ by prompt as well as
    # by candidate.
    rows: list[dict] = []
    for half, pick in (
            ("clean", lambda c: pad_clean(c["rec"], reference_answer(c["rec"]), rule_text)
                      if args.pad_clean else reference_answer(c["rec"])),
            ("planted", lambda c: c["answer"])):
        rng = random.Random(args.seed)
        for i, c in enumerate(cases, 1):
            g = _grade(lm_generate, model, tok, c["rec"], pick(c), args, rng)
            if g is None:
                continue      # ungraded; counted as coverage, never as a verdict
            rows.append({"id": c["id"], "half": half, "planted": c["n"],
                         "answer_chars": len(pick(c)),
                         "hit": c["n"] in g["fired"] if half == "planted" else None,
                         **g,
                         "judge_model": args.judge_model,
                         "judge_prompt": args.judge_prompt,
                         "arms": 1, "set": args.gold.name})
            if i % 10 == 0 or i == len(cases):
                print(f"  {half}: judged {i}/{len(cases)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    clean = [r for r in rows if r["half"] == "clean"]
    planted = [r for r in rows if r["half"] == "planted"]
    s = separation(clean, planted)          # raises if either half is empty
    cov = (len(clean) + len(planted)) / (2 * len(cases))
    drops = sum(r["quote_drops"] for r in rows)
    usable_frac = len(cases) / len(records) if records else 0.0

    L = [f"# Judge error-detection report — `{args.judge_model}`\n",
         f"{len(cases)} records from `{args.gold.name}`, **one arm**, prompt "
         f"{args.judge_prompt}. Two candidates per record: the reference answer, which "
         "**cannot** commit a listed error, and one asserting a listed error **verbatim**, "
         "which definitionally does.\n",
         f"- **graded {len(clean)+len(planted)}/{2*len(cases)} ({cov:.0%})**",
         f"- **{len(cases)}/{len(records)} records usable** "
         f"({skipped_records} had no claim-form error; {skipped_entries} entries skipped)",
         f"- quote drops: {drops}\n",
         "| | fires at a **clean** answer | fires at a **1-error** answer |",
         "| --- | --- | --- |",
         f"| rate | {s['p_fire_clean']:.0%} ({sum(1 for r in clean if r['fired'])}/{s['n_clean']}) "
         f"| {s['p_fire_error']:.0%} ({sum(1 for r in planted if r['fired'])}/{s['n_planted']}) |",
         f"| mean errors fired | {s['mean_fired_clean']:.2f} | {s['mean_fired_error']:.2f} |",
         f"| caught the planted one | — | {s['hit_planted']:.0%} |",
         "",
         f"## Separation: **{s['separation']:+.0%}**\n",
         "`P(fire | error) − P(fire | clean)`. **This is the number to read, and neither "
         "column means anything without the other** — a judge that never fires scores a "
         "perfect 0% on the left, one that always fires scores a perfect 100% on the right. "
         "Measured (positions, n=24, one arm): **32B +96, Llama-3.1-8B +42, "
         "Qwen2.5-7B +25** — and all three catch the planted error 24/24, so the right-hand "
         "column does not rank them at all (Sections 21.42, 21.43).\n",
         "> **One arm.** Not comparable to the four-arm `false errors` trip-wire; arm count "
         "moves scores by as much as 23 points (Section 21.5). Compare only against another "
         "single-arm report on the same set.\n"]
    if usable_frac < 0.90:
        L.append(
            f"> **{len(records) - len(cases)} of {len(records)} records were unusable, so "
            f"this is a selected subset rather than a sample of `{args.gold.name}`.** The "
            "survivors are those whose `common_errors` were authored as claims, and nothing "
            "says a judge behaves the same on the rest.\n")
    if s["separation"] < 0.50:
        L.append("> **This judge does not separate blundered from clean.** Blunder rate is "
                 "defined on `errors_made` being non-empty, so it cannot support that metric "
                 "however well it catches real errors.\n")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[5:]))
    print(f"\n-> {args.report_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=GOLD_EVAL_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge-model", default=BASE_MODEL_ID)
    ap.add_argument("--judge-prompt", choices=("v3", "v4", "v5"), default="v3")
    # Deliberately one flag, not two. There is no way to ask for half of this
    # (Section 21.43) — a one-sided error-detection number will be read
    # one-sided, and was, three times.
    ap.add_argument("--judge-report", action="store_true",
                    help="run the error-detection control instead: BOTH halves (a clean "
                         "answer and one asserting a listed error verbatim) and their "
                         "separation, which is the only number that means anything alone")
    ap.add_argument("--pad-clean", action="store_true",
                    help="lengthen the clean answer with the verbatim CR text of the rules "
                         "it already cites, to test whether specificity survives "
                         "answer-length input (Section 21.41)")
    ap.add_argument("--judge-max-tokens", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "eval/runs/calibration.jsonl")
    ap.add_argument("--report-out", type=Path,
                    default=REPO_ROOT / "eval/reports/calibration.md")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the candidates and show them; load no model")
    args = ap.parse_args()

    if args.judge_report:
        # Separate default paths, and the judge in the filename: a judge report
        # and a calibration run are different measurements, and two judges'
        # reports must not overwrite each other either.
        stem = args.judge_model.split("/")[-1]
        if args.out == ap.get_default("out"):
            args.out = REPO_ROOT / f"eval/runs/judge_report_{stem}.jsonl"
        if args.report_out == ap.get_default("report_out"):
            args.report_out = REPO_ROOT / f"eval/reports/judge_report_{stem}.md"
        run_judge_report(args)
        return

    questions = [q for q in load_gold_questions(args.gold, args.limit) if q.get("key_points")]
    if not questions:
        raise SystemExit(f"no rubric-bearing questions in {args.gold}")
    rng = random.Random(args.seed)
    cases = build_candidates(questions, rng, refusal=REFUSAL)
    print(f"{len(cases)} questions x 4 known-quality candidates")

    if args.dry_run:
        c = cases[0]
        print(f"\n--- {c['q'].get('gold_id')} ---")
        for name, text in c["candidates"].items():
            print(f"  {name:8} {text[:150]!r}")
        return

    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm
    print(f"loading {args.judge_model} as judge ...")
    judge_model, judge_tok = load_lm(args.judge_model)

    rows = []
    for i, case in enumerate(cases, 1):
        judged = score_one_question(lm_generate, judge_model, judge_tok, case["q"],
                                    case["candidates"], args.judge_max_tokens, rng,
                                    judge_version=args.judge_prompt)
        rows.append({"gold_id": case["q"].get("gold_id"),
                     "category": case["q"].get("category"),
                     "scores": {k: (judged.get(k) or {}).get("correctness") for k in case["candidates"]},
                     "errors": {k: (judged.get(k) or {}).get("errors_made") for k in case["candidates"]},
                     # Stored so a scoring change can be re-derived from this
                     # file instead of re-run. The Section 21.28 change had to
                     # be recovered by inverting the halving arithmetically
                     # because these two fields were not here.
                     "points_hit": {k: (judged.get(k) or {}).get("points_hit") for k in case["candidates"]},
                     "points_total": {k: (judged.get(k) or {}).get("points_total") for k in case["candidates"]},
                     "scoring": next((d.get("scoring") for d in judged.values() if d), None)})
        if i % 10 == 0 or i == len(cases):
            print(f"  judged {i}/{len(cases)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    arms = ["oracle", "partial", "wrong", "refusal"]
    means, counts = {}, {}
    for a in arms:
        vals = [r["scores"][a] for r in rows if r["scores"].get(a) is not None]
        means[a] = sum(vals) / len(vals) if vals else float("nan")
        counts[a] = len(vals)

    # COVERAGE, before anything else is read.
    #
    # This script produces the numbers STRUCTURAL_AUDIT.md's migration decision
    # is stated against, and it used to average over whatever the judge happened
    # to grade without saying how much that was. The V4 run (Section 21.14) is
    # what that failure looks like in practice: four confident means over 14 of
    # 99 questions, with the sample size a parenthetical beside them, and the
    # survivors selected by rubric size rather than by anything about the
    # answers. A calibration run has the same exposure and higher stakes.
    graded = min(counts.values()) if counts else 0
    coverage = graded / len(rows) if rows else 0.0

    ordered = [r for r in rows
               if all(r["scores"].get(a) is not None for a in ("oracle", "partial", "wrong"))]
    n_ok = sum(1 for r in ordered
               if r["scores"]["oracle"] >= r["scores"]["partial"] >= r["scores"]["wrong"]
               and r["scores"]["oracle"] > r["scores"]["wrong"])
    order_acc = n_ok / len(ordered) if ordered else float("nan")

    oracle_errs = [r["errors"]["oracle"] for r in rows if r["errors"].get("oracle") is not None]
    false_err = sum(1 for e in oracle_errs if e) / len(oracle_errs) if oracle_errs else float("nan")
    dyn = means["oracle"] - means["wrong"]

    L = ["# Judge calibration — positive controls\n",
         f"{len(rows)} questions from `{args.gold.name}`, four known-quality candidates each.\n",
         f"- judge: `{args.judge_model}` (prompt {args.judge_prompt})",
         f"- **graded {graded}/{len(rows)} questions ({coverage:.0%})**"
         + ("\n" if graded == len(rows)
            else " — the rest returned JSON the harness could not parse\n")]
    if coverage < 0.90:
        L.append(
            f"> **Coverage is {coverage:.0%}. Do not read the numbers below.** The judge failed "
            "to grade a large share of the set, and its failures are not random — they track "
            "how much the prompt asks it to produce, so what remains is selected by rubric "
            "size rather than by anything about the answers (Section 21.14). Re-run with a "
            "larger `--judge-max-tokens`. A calibration run is the input to the migration "
            "decision; a biased one is worse than none.\n")
    L += ["| Candidate | Mean correctness | n | Expected |",
          "| --- | --- | --- | --- |",
          f"| `oracle` (the reference itself) | **{means['oracle']:.2f}** | {counts['oracle']} | near 5 |",
          f"| `partial` (half, by sentence) | {means['partial']:.2f} | {counts['partial']} | middle |",
          f"| `wrong` (another question's answer) | {means['wrong']:.2f} | {counts['wrong']} | near 1 |",
          f"| `refusal` | {means['refusal']:.2f} | {counts['refusal']} | 1 |",
          "",
         f"- **Dynamic range (oracle − wrong): {dyn:+.2f}** on a 1–5 scale",
         f"- **Ordering accuracy: {order_acc:.0%}** of questions rank oracle ≥ partial ≥ wrong",
         f"- **False errors on the oracle: {false_err:.0%}** — the reference answer cannot "
         "commit a listed common_error, so every one of these is a definitive false positive",
         ""]

    # The trip-wires from STRUCTURAL_AUDIT.md, evaluated rather than left to the reader.
    checks = [("dynamic range ≥ 2.5", dyn >= 2.5, f"{dyn:+.2f}"),
              ("ordering accuracy ≥ 90%", order_acc >= 0.90, f"{order_acc:.0%}"),
              ("false errors on oracle ≤ 5%", false_err <= 0.05, f"{false_err:.0%}")]
    L.append("## Trip-wires (STRUCTURAL_AUDIT.md)\n")
    if coverage < 0.90:
        # A PASS computed on a selected subset is the worst possible output
        # here: it is the sentence that authorizes buying a computer. Withhold
        # the verdict rather than qualify it.
        L.append(f"**NOT EVALUATED — coverage {coverage:.0%}.** These three numbers decide "
                 "whether the instrument is sound, and they cannot be read off a subset the "
                 "judge selected by failing on the rest. Measured values below are recorded "
                 "for diagnosis only.\n")
        for label, _, got in checks:
            L.append(f"- (unevaluated) {label} — measured {got} on {graded}/{len(rows)}")
    else:
        for label, passed, got in checks:
            L.append(f"- {'PASS' if passed else '**FAIL**'} — {label} (measured {got})")
        if not all(p for _, p, _ in checks):
            L.append("\n> The instrument does not yet separate known-good from known-bad to "
                     "the standard the migration trip-wires require. A better model under "
                     "test cannot be distinguished from a worse one until this passes, so "
                     "hardware is not the binding constraint.\n")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[4:]))
    print(f"\n-> {args.report_out}")


if __name__ == "__main__":
    main()
