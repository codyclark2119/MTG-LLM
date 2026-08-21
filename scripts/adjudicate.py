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
    cohens_kappa,
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


def export_tasks(queue: list[dict], out: Path) -> int:
    """Write a SELF-CONTAINED task file for the deployable form.

    The same contract `author_rubrics.py --export-tasks` follows, and for the
    same reason: `rubric_server.py` must be able to serve this with no access to
    the gold set, the runs, the corpora or any model. Every field the reviewer
    needs is inlined here, so the deployed container copies one file.

    Two things are deliberately absent. **No judge output** — not the errors it
    fired, not its score, not which run the answer came from — because a verdict
    collected while looking at the judge is not independent of it. And no record
    beyond what is being reviewed: no rubric ids that would let the form fetch
    more, since there is nothing on the far side to fetch from.
    """
    rubrics = _rubric_index()
    tasks = []
    for item in queue:
        t = task_for(item, rubrics)
        if not t:
            continue
        leaked = set(t) & {"errors_made", "judge_model", "fired", "blundered", "run"}
        assert not leaked, f"judge output would leak to the public form: {leaked}"
        tasks.append(t)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"kind": "adjudication", "tasks": tasks},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    return len(tasks)


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
    n_unsure = n_not_covered = n_unmatched = 0
    b_pairs: list[tuple[bool, bool]] = []
    for v in verdicts:
        if v["key"] not in by_key:
            n_unmatched += 1
            continue
        # "Genuinely ambiguous — I could argue it either way" is the reviewer
        # telling us their own call is a coin flip. Scoring it as a firm verdict
        # is the one thing the checkbox exists to prevent, and it was collected
        # by the form, stored on the submission, and read by nothing — so a
        # reviewer following the instruction to use it changed nothing, and a
        # guess entered the precision number wearing the shape of ground truth.
        # The whole point of human adjudication is that it is the tiebreaker;
        # a tiebreaker that silently includes coin flips is not one.
        if v.get("unsure"):
            n_unsure += 1
            continue
        if v.get("not_covered"):
            n_not_covered += 1
        human = set(v["errors_present"])
        judge = set(by_key[v["key"]])
        tp += len(human & judge)
        fp += len(judge - human)
        fn += len(human - judge)
        b_total += 1
        b_pairs.append((bool(human), bool(judge)))
        if bool(human) == bool(judge):
            b_right += 1
        elif judge and not human:
            fp_only += 1
        else:
            fn_only += 1

    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    # Raw agreement flatters a skewed call, and the human blunder calls ARE
    # skewed. eval_positions.compare_judges says so in its docstring and reports
    # kappa first; this — the judge-versus-HUMAN comparison, where it matters
    # more — reported the raw number alone (Section 21.57).
    kappa = cohens_kappa(b_pairs)
    return {
        "run": run.name, "n": b_total,
        # Reported, never silently dropped: each is a reason the denominator is
        # smaller than the queue, and a shrinking denominator nobody announced
        # is how a selected subset gets read as a sample (Section 21.14).
        "excluded_unsure": n_unsure,
        "unmatched": n_unmatched,
        # Not excluded — a reviewer saying "the rubric has no entry for what
        # this answer did" while listing no error IS the human agreeing that no
        # LISTED error occurred, which is exactly what blunder rate is defined
        # on (Section 21.49). Counted and reported because it means the metric
        # is narrower than "the answer was good", not because it is wrong.
        "not_covered": n_not_covered,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan"),
        "blunder_accuracy": b_right / b_total if b_total else float("nan"),
        "blunder_kappa": kappa,
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
    ap.add_argument("--export-tasks", type=Path, metavar="OUT",
                    help="write a self-contained task file for rubric_server.py "
                         "(no gold set, no runs, no judge output)")
    ap.add_argument("--ingest-submissions", type=Path, metavar="FILE",
                    help="merge verdicts collected by the deployed form")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true", help="how far through the queue")
    ap.add_argument("--score", type=Path, nargs="+",
                    help="score run(s) against the human verdicts collected so far")
    ap.add_argument("--form-version", type=int, default=None, metavar="N",
                    help="score only verdicts collected under this form wording. "
                         "v1 asked 'which of these does it commit?' and under-fires "
                         "on action-list answers; v2 says to judge the play rather "
                         "than the wording (Section 21.47). Pooling them mixes two "
                         "standards, so a mixed set is reported and can be split.")
    args = ap.parse_args()

    if args.build:
        if not args.runs:
            raise SystemExit("--build needs --runs")
        q = build_queue(args.runs, args.n, args.seed)
        QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        QUEUE_PATH.write_text(json.dumps(q, indent=1), encoding="utf-8")
        print(f"{len(q)} tasks -> {QUEUE_PATH}")
        return

    if args.export_tasks:
        q = json.loads(QUEUE_PATH.read_text()) if QUEUE_PATH.exists() else []
        if not q:
            raise SystemExit(f"no queue at {QUEUE_PATH}; run --build first")
        n = export_tasks(q, args.export_tasks)
        print(f"{n} self-contained tasks -> {args.export_tasks}")
        return

    if args.ingest_submissions:
        # Promotion is local and reviewed, the same invariant rubric_server
        # keeps for rubrics: the deployed form never writes this file itself.
        incoming = read_jsonl(args.ingest_submissions)
        have = {(v["key"], v.get("author")) for v in read_jsonl(ADJUDICATIONS_PATH)}
        new = [v for v in incoming if (v.get("key"), v.get("author")) not in have]
        print(f"{len(incoming)} submitted, {len(new)} new, "
              f"{len(incoming) - len(new)} already on file")
        by_author = {}
        for v in new:
            by_author[v.get("author", "?")] = by_author.get(v.get("author", "?"), 0) + 1
        for a, c in sorted(by_author.items()):
            print(f"  {a}: {c}")
        if args.dry_run:
            print("\nDRY RUN — re-run without --dry-run to append")
            return
        for v in new:
            append_verdict(v)
        print(f"appended {len(new)} -> {ADJUDICATIONS_PATH}")
        return

    if args.status:
        q = json.loads(QUEUE_PATH.read_text()) if QUEUE_PATH.exists() else []
        rows = read_jsonl(ADJUDICATIONS_PATH)
        done = {v["key"] for v in rows}
        # Rows and keys are different numbers once anything is re-adjudicated,
        # and reporting the key count as "verdicts on file" made three appended
        # rows look like nothing had happened. Same phantom-progress shape the
        # rubric server's task list already guards against.
        from collections import Counter
        vers = Counter(v.get("form_version", 1) for v in rows)
        print(f"queue      : {len(q)}")
        print(f"covered    : {len(done & {i['key'] for i in q})}/{len(q)} of the queue")
        print(f"verdicts   : {len(rows)} rows over {len(done)} distinct answers"
              + (f" ({len(rows) - len(done)} re-adjudicated)" if len(rows) > len(done) else ""))
        print(f"by wording : {', '.join(f'v{k}: {n}' for k, n in sorted(vers.items()))}")
        return

    if args.score:
        verdicts = read_jsonl(ADJUDICATIONS_PATH)
        if not verdicts:
            raise SystemExit(f"no verdicts in {ADJUDICATIONS_PATH} yet")
        rubrics = _rubric_index()

        vers = {}
        for v in verdicts:
            vers.setdefault(v.get("form_version", 1), []).append(v)
        if args.form_version is not None:
            verdicts = vers.get(args.form_version, [])
            if not verdicts:
                raise SystemExit(f"no verdicts at form_version {args.form_version}; "
                                 f"have {sorted(vers)}")
            print(f"scored against {len(verdicts)} verdicts at form_version "
                  f"{args.form_version}\n")
        else:
            # A re-adjudication supersedes the earlier one. The same key can be
            # answered twice — once under v1, once under v2 — and because the
            # author changed in between, dedupe on (key, author) keeps both.
            # Scoring both would count two contradictory verdicts for one
            # answer, so the newest wording wins: pos-combat-math-0004 went
            # [1,4,5] under v1 to [1] plus not_covered under v2, and only the
            # second is a verdict about the errors rather than about the answer
            # being bad.
            newest = {}
            for v in verdicts:
                k = v["key"]
                if k not in newest or v.get("form_version", 1) >= newest[k].get("form_version", 1):
                    newest[k] = v
            superseded = len(verdicts) - len(newest)
            verdicts = list(newest.values())
            print(f"scored against {len(verdicts)} human verdicts "
                  f"({', '.join(f'v{k}: {len(x)}' for k, x in sorted(vers.items()))}"
                  + (f"; {superseded} superseded by a re-adjudication)" if superseded else ")"))
            live = {v.get("form_version", 1) for v in verdicts}
            if len(live) > 1:
                print("  > MIXED WORDING. v1 under-fires on action-list answers, so a\n"
                      "  > judge's precision is understated where v1 dominates. Split\n"
                      "  > with --form-version N before quoting a number (21.47).")
            print()

        # Answers that are wrong in a way the rubric does not describe. No judge
        # number can see these: an answer committing no LISTED error scores as
        # clean, so a judge is credited for "correctly" finding nothing in an
        # answer that does nothing. This measures the RUBRIC, not the judge.
        nc = [v for v in verdicts if v.get("not_covered")]
        if nc:
            print(f"rubric coverage: {len(nc)}/{len(verdicts)} answers were bad for a "
                  f"reason no listed error describes")
            from collections import Counter
            for cat, n in Counter(v["record_id"].rsplit("-", 1)[0].replace("pos-", "")
                                  for v in nc).most_common(4):
                print(f"    {cat:24s} {n}")
            print()
        print(f"{'run':40s} {'n':>4s} {'prec':>6s} {'recall':>7s} {'F1':>6s} "
              f"{'blunder acc':>12s} {'kappa':>7s} {'false':>6s} {'missed':>7s}")
        scored = []
        for r in args.score:
            s = score_run(r, verdicts, rubrics)
            scored.append(s)
            print(f"{s['run']:40s} {s['n']:4d} {s['precision']:6.0%} {s['recall']:7.0%} "
                  f"{s['f1']:6.2f} {s['blunder_accuracy']:12.0%} "
                  f"{s['blunder_kappa']:+7.2f} "
                  f"{s['false_blunder']:6d} {s['missed_blunder']:7d}")
        # Every reason the denominator is smaller than the queue, next to the
        # numbers it qualifies. `n` alone reads as the sample size; it is the
        # sample size AFTER three different exclusions.
        for s in scored:
            bits = []
            if s["excluded_unsure"]:
                bits.append(f"{s['excluded_unsure']} marked genuinely ambiguous (excluded)")
            if s["unmatched"]:
                bits.append(f"{s['unmatched']} not graded by this judge (excluded)")
            if s["not_covered"]:
                bits.append(f"{s['not_covered']} flagged `not_covered` (counted — the human "
                            "agreed no LISTED error occurred, which is what blunder rate "
                            "measures; the answer may still be bad, Section 21.49)")
            if bits:
                print(f"\n{s['run']}: " + "; ".join(bits))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
