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
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import REPO_ROOT  # noqa: E402
from eval import BASE_MODEL_ID, load_gold_questions, score_one_question  # noqa: E402

GOLD_EVAL_PATH = REPO_ROOT / "eval/sets/gold_questions_eval.jsonl"
REFUSAL = "The rules provided do not cover this, so I cannot answer the question."

# Sentence boundaries that survive rule ids. "601.2h" and "3." both contain a
# period, so a naive split truncates mid-citation and would understate `partial`
# for reasons that have nothing to do with the judge: the boundary requires
# whitespace then a capital or an opening bracket.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def half_answer(text: str) -> str:
    """Roughly the first half of an answer, cut at a sentence boundary.

    Not the first SENTENCE. These answers open with a one-word verdict — "Yes."
    or "3." — so first-sentence truncation produces a four-character candidate
    and measures how the judge treats a bare verdict, not how it treats a
    half-complete answer. Half is the control that is actually wanted: it should
    state some of the key points and miss the rest, and a rubric judge scoring
    points-hit should land it in the middle.

    That also makes `partial` a length-neutrality check. V2 had a measured
    length bias (r = +0.21, Section 9.6) and V3 exists partly to remove it; a
    half-length answer stating half the points should score about half, not
    less.
    """
    text = (text or "").strip()
    parts = _SENT_RE.split(text)
    if len(parts) < 2:
        return text
    target, taken, out = len(text) / 2, 0, []
    for part in parts:
        out.append(part)
        taken += len(part) + 1
        if taken >= target:
            break
    return " ".join(out).strip()


def build_candidates(questions: list[dict], rng: random.Random) -> list[dict]:
    """Four known-quality candidates per question."""
    by_cat: dict[str, list[dict]] = {}
    for q in questions:
        by_cat.setdefault(q.get("category") or "?", []).append(q)

    out = []
    for q in questions:
        pool = [o for o in by_cat.get(q.get("category") or "?", []) if o is not q]
        # Same category keeps `wrong` topically plausible. Falling back to the
        # whole set only matters for a category with one member.
        if not pool:
            pool = [o for o in questions if o is not q]
        if not pool:
            continue
        other = rng.choice(pool)
        out.append({
            "q": q,
            "candidates": {
                "oracle": q["reference"],
                "partial": half_answer(q["reference"]),
                "wrong": other["reference"],
                "refusal": REFUSAL,
            },
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=GOLD_EVAL_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge-model", default=BASE_MODEL_ID)
    ap.add_argument("--judge-prompt", choices=("v3", "v4"), default="v3")
    ap.add_argument("--judge-max-tokens", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "eval/runs/calibration.jsonl")
    ap.add_argument("--report-out", type=Path,
                    default=REPO_ROOT / "eval/reports/calibration.md")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the candidates and show them; load no model")
    args = ap.parse_args()

    questions = [q for q in load_gold_questions(args.gold, args.limit) if q.get("key_points")]
    if not questions:
        raise SystemExit(f"no rubric-bearing questions in {args.gold}")
    rng = random.Random(args.seed)
    cases = build_candidates(questions, rng)
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
                     "errors": {k: (judged.get(k) or {}).get("errors_made") for k in case["candidates"]}})
        if i % 10 == 0 or i == len(cases):
            print(f"  judged {i}/{len(cases)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    arms = ["oracle", "partial", "wrong", "refusal"]
    means = {}
    for a in arms:
        vals = [r["scores"][a] for r in rows if r["scores"].get(a) is not None]
        means[a] = sum(vals) / len(vals) if vals else float("nan")

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
         f"- judge: `{args.judge_model}` (prompt {args.judge_prompt})\n",
         "| Candidate | Mean correctness | Expected |",
         "| --- | --- | --- |",
         f"| `oracle` (the reference itself) | **{means['oracle']:.2f}** | near 5 |",
         f"| `partial` (first half) | {means['partial']:.2f} | middle |",
         f"| `wrong` (another question's answer) | {means['wrong']:.2f} | near 1 |",
         f"| `refusal` | {means['refusal']:.2f} | 1 |",
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
    for label, passed, got in checks:
        L.append(f"- {'PASS' if passed else '**FAIL**'} — {label} (measured {got})")
    if not all(p for _, p, _ in checks):
        L.append("\n> The instrument does not yet separate known-good from known-bad to the "
                 "standard the migration trip-wires require. A better model under test "
                 "cannot be distinguished from a worse one until this passes, so hardware "
                 "is not the binding constraint.\n")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[4:]))
    print(f"\n-> {args.report_out}")


if __name__ == "__main__":
    main()
