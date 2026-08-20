"""Human ground truth on error detection: is the judge RIGHT, not just consistent?

    python scripts/adjudicate.py --build --runs eval/runs/pos_n24_32b.jsonl --n 60
    python scripts/adjudicate.py --status
    python scripts/adjudicate.py --score eval/runs/pos_n24_32b.jsonl

WHY THIS AND NOT MORE JUDGE RUNS

Everything measured so far is one of two things, and neither is accuracy:

  judge vs judge      agreement. Two judges agreeing tells you nothing about
                      whether either is right — Section 21.42 found three of
                      four judges agreeing on a number (24/24 sensitivity) while
                      one of them fired at 58% of clean answers.
  judge vs control    the oracle CANNOT commit an error; the planted answer
                      commits exactly one, verbatim. Both are constructed
                      extremes.

**Real answers sit between those extremes and nothing has measured a judge on
one.** An arm answer commits some unknown number of listed errors, and the only
thing that knows which is a person reading it.

WHAT THIS PRODUCES

Per (record, answer) pair, a human says which of the numbered `common_errors`
that answer actually commits. From that, against any judge run:

  precision   of the errors the judge fired, how many were really there
  recall      of the errors really there, how many it fired
  blunder accuracy  did it get "this answer blundered at all" right — which is
                    the Gate 3 metric, defined on `errors_made` being non-empty

The set is judge-independent, so it scores **every future judge for free**. That
is the whole point: the human time is spent once and every later judge —
including ones not downloaded yet — is measured against it without asking for
more.

THE TASK IS BLIND TO THE JUDGE, DELIBERATELY

The queue never shows what any judge said. Showing it would anchor the answer to
the thing being tested, which is the same reason `judge_batch_anonymized` hides
arm identity and the rubric judge grades A/B/C/D rather than named arms. A
verdict collected while looking at the judge's output is not independent of it.

Consequence: the queue cannot be built to "review the judge's mistakes", because
that requires knowing which ones it got wrong. Disagreement between two judges
is allowed to drive SAMPLING — that is a property of the pair, not of either
one's correctness — but the reviewer is never told a disagreement exists.

WRITES ONE FILE

`data/gold/judge_adjudications.jsonl`, append-only. Never the gold set, never a
run file. A verdict is an opinion about a stored answer; it is not gold data and
must not silently become a rubric.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (  # noqa: E402
    GOLD_PATH,
    POSITIONS_PATH,
    REPO_ROOT,
    read_jsonl,
)

ADJUDICATIONS_PATH = REPO_ROOT / "data/gold/judge_adjudications.jsonl"
QUEUE_PATH = REPO_ROOT / "data/gold/adjudication_queue.json"


def _rubric_index() -> dict:
    """Every record that carries a rubric, by every id it might be found under.

    A position is keyed by `id`, a gold record by `gold_id` in the eval file and
    `id` in the source file — the identifier changes when a record is promoted,
    which is the trap CLAUDE.md records under "an identifier that changes when a
    record is promoted". Indexing both is cheaper than deciding which applies.
    """
    out = {}
    for path in (POSITIONS_PATH, GOLD_PATH,
                 REPO_ROOT / "eval/sets/gold_questions_eval.jsonl"):
        for r in read_jsonl(path):
            if not r.get("common_errors"):
                continue
            for key in ("id", "gold_id"):
                if r.get(key):
                    out.setdefault(r[key], r)
    return out


def build_queue(runs: list[Path], n: int, seed: int = 42) -> list[dict]:
    """Sample (record, arm, answer) triples for review.

    When two runs are given, pairs they disagree on are drawn FIRST. Sampling on
    disagreement is legitimate — it is a property of the pair, not a claim about
    either judge — and it concentrates human time where a tiebreaker is worth
    most. The reviewer is never shown that a disagreement exists.
    """
    rubrics = _rubric_index()
    loaded = [{r.get("id") or r.get("gold_id"): r for r in read_jsonl(p)} for p in runs]
    base = loaded[0]

    cands, disputed = [], set()
    for rid, row in base.items():
        rec = rubrics.get(rid)
        if not rec:
            continue
        for arm, d in (row.get("arms") or {}).items():
            if d.get("errors_made") is None or not (d.get("answer") or "").strip():
                continue
            key = (rid, arm)
            cands.append({
                "key": f"{rid}::{arm}",
                "record_id": rid,
                "arm": arm,
                "answer": d["answer"],
                "n_errors": len(rec["common_errors"]),
            })
            for other in loaded[1:]:
                od = ((other.get(rid) or {}).get("arms") or {}).get(arm) or {}
                if od.get("errors_made") is None:
                    continue
                if bool(od["errors_made"]) != bool(d["errors_made"]):
                    disputed.add(key)

    rng = random.Random(seed)
    rng.shuffle(cands)
    # Disputed first, then the rest — so a short session still spends its time
    # on the pairs where a human verdict decides something.
    cands.sort(key=lambda c: (c["key"] not in disputed,))
    return cands[:n]


def task_for(item: dict, rubrics: dict) -> dict | None:
    """The blind review task: board/question, the answer, the numbered errors."""
    rec = rubrics.get(item["record_id"])
    if not rec:
        return None
    from calibrate_judge import question_text
    return {
        "key": item["key"],
        "record_id": item["record_id"],
        "arm": item["arm"],
        "question": question_text(rec),
        "answer": item["answer"],
        "common_errors": rec["common_errors"],
        "key_points": rec.get("key_points") or [],
        "category": rec.get("category"),
    }


def append_verdict(verdict: dict, path: Path = ADJUDICATIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(verdict, ensure_ascii=False) + "\n")


def score_run(run: Path, verdicts: list[dict], rubrics: dict) -> dict:
    """Score one judge run against the human verdicts.

    Per-error precision/recall, plus blunder-call accuracy — the last is what
    Gate 3 is defined on, and it is not derivable from the other two: a judge
    can fire the wrong error on an answer that did blunder and still get the
    binary call right.
    """
    by_key = {}
    for row in read_jsonl(run):
        rid = row.get("id") or row.get("gold_id")
        for arm, d in (row.get("arms") or {}).items():
            if d.get("errors_made") is not None:
                by_key[f"{rid}::{arm}"] = [e["n"] if isinstance(e, dict) else e
                                           for e in d["errors_made"]]

    tp = fp = fn = 0
    b_right = b_total = 0
    fp_only = fn_only = 0
    for v in verdicts:
        if v["key"] not in by_key:
            continue
        human = set(v["errors_present"])
        judge = set(by_key[v["key"]])
        tp += len(human & judge)
        fp += len(judge - human)
        fn += len(human - judge)
        b_total += 1
        if bool(human) == bool(judge):
            b_right += 1
        elif judge and not human:
            fp_only += 1
        else:
            fn_only += 1

    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "run": run.name, "n": b_total,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan"),
        "blunder_accuracy": b_right / b_total if b_total else float("nan"),
        "false_blunder": fp_only, "missed_blunder": fn_only,
        "tp": tp, "fp": fp, "fn": fn,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build the review queue")
    ap.add_argument("--runs", type=Path, nargs="+",
                    help="run file(s); a second one drives disagreement-first sampling")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--status", action="store_true", help="how far through the queue")
    ap.add_argument("--score", type=Path, nargs="+",
                    help="score run(s) against the human verdicts collected so far")
    args = ap.parse_args()

    if args.build:
        if not args.runs:
            raise SystemExit("--build needs --runs")
        q = build_queue(args.runs, args.n, args.seed)
        QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        QUEUE_PATH.write_text(json.dumps(q, indent=1), encoding="utf-8")
        print(f"{len(q)} tasks -> {QUEUE_PATH}")
        return

    if args.status:
        q = json.loads(QUEUE_PATH.read_text()) if QUEUE_PATH.exists() else []
        done = {v["key"] for v in read_jsonl(ADJUDICATIONS_PATH)}
        print(f"queue     : {len(q)}")
        print(f"adjudicated: {len(done & {i['key'] for i in q})}/{len(q)}")
        print(f"total verdicts on file: {len(done)}")
        return

    if args.score:
        verdicts = read_jsonl(ADJUDICATIONS_PATH)
        if not verdicts:
            raise SystemExit(f"no verdicts in {ADJUDICATIONS_PATH} yet")
        rubrics = _rubric_index()
        print(f"scored against {len(verdicts)} human verdicts\n")
        print(f"{'run':40s} {'n':>4s} {'prec':>6s} {'recall':>7s} {'F1':>6s} "
              f"{'blunder acc':>12s} {'false':>6s} {'missed':>7s}")
        for r in args.score:
            s = score_run(r, verdicts, rubrics)
            print(f"{s['run']:40s} {s['n']:4d} {s['precision']:6.0%} {s['recall']:7.0%} "
                  f"{s['f1']:6.2f} {s['blunder_accuracy']:12.0%} "
                  f"{s['false_blunder']:6d} {s['missed_blunder']:7d}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
