"""Recompute correctness on stored runs under the current scoring, no judge.

    python scripts/rescore_stored.py eval/runs/gold_n99.jsonl --dry-run
    python scripts/rescore_stored.py eval/runs/*.jsonl --out-suffix _pointsonly

WHY THIS IS NOT A RE-JUDGE

Section 21.28 changed how `points_hit` and `errors_made` map to a 1-5 score. It
did not change what the judge extracts. Every stored run already carries
`points_hit`, `points_total` and `errors_made`, so the new number is *derivable*
from the old file — exactly, deterministically, with no model involved.

That distinction is the whole reason the change is auditable. A re-judge would
mix two variables: a new scoring rule and a fresh sample of judge behaviour. A
recompute holds the judge output fixed and moves one term, so every delta below
is attributable to the scoring change and nothing else.

Rows the judge never graded stay ungraded. A recompute cannot invent a score for
an answer that was never judged, and filling one in would quietly convert a
coverage problem into a data point.

WRITES A NEW FILE BY DEFAULT

The originals back published numbers. This writes `<name><suffix>.jsonl`
alongside them and prints the per-arm delta, so the old and new numbers can sit
side by side in the plan rather than one silently replacing the other. Pass
`--in-place` only when the intent is to move the record itself.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.core.io import read_jsonl, write_jsonl_atomic  # noqa: E402
from harness.core.eval.judge import SCORING, rubric_correctness  # noqa: E402


def recompute(rows: list[dict], halve: bool = False,
              rubrics: dict | None = None) -> tuple[list[dict], dict]:
    """Rescore in place on a copy. Returns (rows, per-arm before/after means).

    `rubrics` maps record id -> (n_key_points, n_common_errors), for runs that
    do not carry the rubric inline. Position runs are the case: they store
    `points_hit` but not `points_total`, because the rubric lives in
    `positions.jsonl` and is joined at report time.
    """
    rubrics = rubrics or {}
    stats: dict[str, list] = {}
    for r in rows:
        rid = r.get("id") or r.get("gold_id")
        for arm, d in (r.get("arms") or {}).items():
            before = d.get("correctness")
            if before is None or d.get("points_hit") is None:
                continue                      # never graded — leave it that way
            n_pts = d.get("points_total") or rubrics.get(rid, (None, None))[0]
            if not n_pts:
                continue
            # n_errors bounds which error indices are valid. Prefer the rubric;
            # fall back to the inline list, then to the largest index claimed.
            n_err = (len(r.get("common_errors") or [])
                     or rubrics.get(rid, (None, None))[1]
                     or max(d.get("errors_made") or [0]))
            after = rubric_correctness(d["points_hit"], d.get("errors_made") or [],
                                       n_pts, n_err, halve_on_error=halve)
            d["correctness"] = after["correctness"]
            d["scoring"] = after["scoring"]
            stats.setdefault(arm, []).append((before, after["correctness"]))
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--out-suffix", default="_pointsonly")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the run file. It backs published numbers — "
                         "only do this when moving the record is the intent.")
    ap.add_argument("--halve", action="store_true",
                    help="recompute under the OLD rule instead, to reproduce "
                         "numbers published through Section 21.27")
    ap.add_argument("--rubric", type=Path, default=None,
                    help="jsonl supplying key_points/common_errors by id, for runs "
                         "that do not carry the rubric inline (e.g. positions.jsonl)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rubrics = {}
    if args.rubric:
        for rec in read_jsonl(args.rubric):
            rubrics[rec.get("id") or rec.get("gold_id")] = (
                len(rec.get("key_points") or []), len(rec.get("common_errors") or []))
        print(f"rubrics for {len(rubrics)} records from {args.rubric.name}")

    for path in args.runs:
        rows = read_jsonl(path)
        if not rows or not (rows[0].get("arms")):
            print(f"{path.name}: not a run file, skipped")
            continue
        rows, stats = recompute(rows, halve=args.halve, rubrics=rubrics)
        if not stats:
            print(f"{path.name}: no rubric-scored rows, skipped")
            continue

        print(f"\n{path.name}   scoring -> {'halved_v3' if args.halve else SCORING}")
        print(f"  {'arm':<18} {'n':>4} {'before':>8} {'after':>8} {'delta':>8}  moved")
        for arm, pairs in stats.items():
            b = sum(x for x, _ in pairs) / len(pairs)
            a = sum(y for _, y in pairs) / len(pairs)
            moved = sum(1 for x, y in pairs if x != y)
            print(f"  {arm:<18} {len(pairs):>4} {b:>8.2f} {a:>8.2f} {a - b:>+8.2f}  "
                  f"{moved}/{len(pairs)} ({moved / len(pairs):.0%})")

        if args.dry_run:
            continue
        out = path if args.in_place else path.with_name(path.stem + args.out_suffix + path.suffix)
        write_jsonl_atomic(out, rows)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
