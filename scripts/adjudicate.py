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
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (  # noqa: E402
    GOLD_PATH,
    PROTOCOL_ERRORS,
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
                # Where this text came from and under which grammar. A verdict
                # is about a text (21.62); this makes the queue say which one
                # without needing the run file, so a regrade cycle is
                # self-describing after the run is archived or superseded.
                "source_run": runs[0].name,
                "gameplay_fingerprint": row.get("gameplay_fingerprint"),
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
    strategy = list(rec["common_errors"])
    return {
        "key": item["key"],
        "record_id": item["record_id"],
        "arm": item["arm"],
        "question": question_text(rec),
        "answer": item["answer"],
        # The SAME list the judge grades against, in the SAME order:
        # `eval_positions` passes `common_errors + PROTOCOL_ERRORS`, so strategy
        # entries keep 1..n and protocol entries take n+1..n+7. Concatenating in
        # any other order here would renumber every verdict silently.
        #
        # The form used to show `common_errors` alone while the judge was asked
        # about both. Two consequences, and each reads as a finding about
        # something else. The reviewer had no box for a protocol error, so they
        # wrote it in the note and ticked "not covered" — **30 of 33**
        # not-covered notes describe an entry that was already in
        # `PROTOCOL_ERRORS`, which is most of the 69% coverage gap 21.47 read as
        # the rubric missing entries. And `score_run` counted every protocol
        # charge as a false positive, because the human was never offered it:
        # 0% of charges on a pre-protocol run, **68%** on a protocol run, so the
        # damage arrives exactly when the thing under test changes (21.75).
        "common_errors": strategy + list(PROTOCOL_ERRORS),
        # Where the split falls, so the form can label the two groups and
        # `score_run` can tell a legacy verdict from a current one.
        "n_strategy": len(strategy),
        "key_points": rec.get("key_points") or [],
        "category": rec.get("category"),
        # `source_run` is deliberately NOT here. It is a filename like
        # `pos_n24_verbose_32b.jsonl`, which names the judge — and the blind-task
        # assertion in rubric_server lists `judge_model` precisely so a reviewer
        # cannot see which judge produced what they are grading. Provenance is
        # stamped at INGEST from the local queue instead, where it costs the
        # reviewer nothing and reaches no public URL.
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


def answer_sha(answer: str) -> str:
    """Short digest of the exact answer text a verdict was made against.

    A verdict is keyed `record_id::arm`, and that key is stable while the text
    behind it is not: regenerate the arms — a new adapter, a new gameplay
    grammar (21.60) — and the same key names a different answer. Nothing
    recorded which, so 22 human verdicts would have silently re-pointed at text
    their author never saw, and `--score` would have reported precision against
    answers nobody adjudicated.

    Exactly the identifier problem 21.13 documents one level down: anything
    joining two files on a key must first ask whether the key survived the trip.
    Here it survives and the *meaning* does not, which is worse, because nothing
    fails.
    """
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()[:12]


def score_run(run: Path, verdicts: list[dict], rubrics: dict) -> dict:
    """Score one judge run against the human verdicts.

    Per-error precision/recall, plus blunder-call accuracy — the last is what
    Gate 3 is defined on, and it is not derivable from the other two: a judge
    can fire the wrong error on an answer that did blunder and still get the
    binary call right.
    """
    by_key = {}
    by_sha = {}
    for row in read_jsonl(run):
        rid = row.get("id") or row.get("gold_id")
        for arm, d in (row.get("arms") or {}).items():
            if d.get("errors_made") is not None:
                by_key[f"{rid}::{arm}"] = [e["n"] if isinstance(e, dict) else e
                                           for e in d["errors_made"]]
                by_sha[f"{rid}::{arm}"] = answer_sha(d.get("answer") or "")

    tp = fp = fn = 0
    b_right = b_total = 0
    fp_only = fn_only = 0
    n_unsure = n_not_covered = n_unmatched = n_stale = n_not_shown = 0
    b_pairs: list[tuple[bool, bool]] = []
    for v in verdicts:
        if v["key"] not in by_key:
            n_unmatched += 1
            continue
        # The key matched; the TEXT may not have. A verdict carrying a digest
        # that disagrees with this run's answer was made against different text
        # and says nothing about this one.
        want = v.get("answer_sha")
        if want and want != by_sha.get(v["key"]):
            n_stale += 1
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
        # A charge the reviewer was never offered cannot be a false positive.
        # The form showed `common_errors` while the judge was given
        # `common_errors + PROTOCOL_ERRORS`, so every protocol charge landed in
        # `judge - human` by construction — 0% of charges on a pre-protocol run
        # and 68% on a protocol run. That is the 21.5-family trap in its worst
        # form: a harness error whose rate tracks the condition under test, so
        # it arrives wearing the shape of a result ("the protocol rubric made
        # the judge much less precise") rather than the shape of a bug.
        #
        # `n_shown` is recorded by the form from form_version 3. Verdicts
        # collected before it saw the strategy entries only, which is what the
        # rubric index still reports, so the fallback is exact rather than a
        # guess. Restricted charges are COUNTED and reported: a denominator that
        # quietly shrinks is how a subset gets read as a sample (21.14).
        shown = v.get("n_shown")
        if shown is None:
            # `record_id` is a separate field on a real verdict, but falling back
            # to it alone means a row missing it silently gets NO restriction —
            # the unsafe direction, since unrestricted is what the bug did. The
            # key is `record_id::arm` by construction, so split it.
            rid = v.get("record_id") or v.get("key", "").split("::")[0]
            rec = rubrics.get(rid) or {}
            shown = len(rec.get("common_errors") or [])
        if shown:
            beyond = {n for n in judge if n > shown}
            n_not_shown += len(beyond)
            judge -= beyond
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
        "stale": n_stale,
        # Judge charges against an entry the reviewer's form never displayed.
        # Non-zero means the form is behind the judge's rubric — re-export.
        "charges_not_shown": n_not_shown,
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
        # One deployment can now serve both forms (21.64), so the submissions
        # log holds rubric rows and adjudication rows together. Filter to ours
        # by the field only a verdict has — `author_rubrics.py` filters to its
        # own by `key_points`, symmetrically. Without this, a rubric submission
        # is appended to judge_adjudications.jsonl as a verdict with no
        # errors_present, which reads as "the human found no error".
        raw = read_jsonl(args.ingest_submissions)
        incoming = [v for v in raw if isinstance(v.get("errors_present"), list)]
        if len(incoming) != len(raw):
            print(f"  ignoring {len(raw) - len(incoming)} row(s) that are not "
                  "adjudication verdicts (the log is shared with the rubric form)")
        # Dedup on the TEXT as well as the key. `(key, author)` is the identifier
        # that survives a regeneration of the arms while its meaning changes
        # (21.62), so after a regrade every fresh verdict on an
        # already-adjudicated key collided with the old one and was dropped as
        # "already on file" — six of six, silently, while eleven unrelated older
        # rows appended in their place. Third home for the same assumption after
        # scoring and the done-set; this one discards the work rather than
        # mis-scoring it.
        #
        # A row with no digest keeps `None` as its third component, so a
        # pre-digest verdict and a fresh one on the same key are distinct and
        # both land.
        on_file = read_jsonl(ADJUDICATIONS_PATH)
        # Two dedup rules, because two kinds of row arrive.
        #
        # A verdict the SERVER stamped carries the digest of the exact answer it
        # was given about, so it is identified by (key, author, digest) and a
        # fresh verdict on a regenerated arm is correctly a different verdict.
        #
        # A verdict with NO digest predates server-side stamping, which means it
        # is already on file — and it must fall back to (key, author), or every
        # such row re-appends on every ingest, since the copy on file was
        # backfilled with a digest and would no longer match.
        #
        # Stamping from the queue was a ONE-TIME recovery for those rows and is
        # deliberately NOT done here: the queue now holds the regenerated
        # answers, so stamping an old submission from it would attribute a
        # verdict to text its author never saw — the exact error 21.62 exists to
        # prevent, committed by the fix for it (Section 21.65).
        exact = {(v.get("key"), v.get("author"), v.get("answer_sha")) for v in on_file}
        loose = {(v.get("key"), v.get("author")) for v in on_file}

        def seen(v):
            if v.get("answer_sha"):
                return (v.get("key"), v.get("author"), v["answer_sha"]) in exact
            return (v.get("key"), v.get("author")) in loose

        new = [v for v in incoming if not seen(v)]
        n_unstamped = sum(1 for v in new if not v.get("answer_sha"))
        print(f"{len(incoming)} submitted, {len(new)} new, "
              f"{len(incoming) - len(new)} already on file")
        if n_unstamped:
            print(f"  {n_unstamped} carry no answer digest — collected before the form "
                  "stamped them, so they cannot be checked against a run and will score "
                  "against any of them")
        regrades = sum(1 for v in new
                       if v.get("answer_sha") and (v.get("key"), v.get("author")) in loose)
        if regrades:
            print(f"  {regrades} are fresh verdicts on a key already adjudicated against "
                  "DIFFERENT text — the arms were regenerated, so both are kept")
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
        # Coverage against the CURRENT queue, not against any verdict ever given.
        # A key adjudicated under an earlier grammar is not covered now — its
        # verdict is about text the queue no longer serves, and --score excludes
        # it. Reporting 27/60 while 16 verdicts actually apply is the same
        # key-survives-meaning-changes error one more time (Section 21.62), here
        # inflating apparent progress by 11 tasks.
        live = {i["key"]: answer_sha(i.get("answer") or "") for i in q}
        done = {v["key"] for v in rows
                if v.get("answer_sha") and live.get(v["key"]) == v["answer_sha"]}
        stale_only = {v["key"] for v in rows} & set(live) - done
        # Rows and keys are different numbers once anything is re-adjudicated,
        # and reporting the key count as "verdicts on file" made three appended
        # rows look like nothing had happened. Same phantom-progress shape the
        # rubric server's task list already guards against.
        from collections import Counter
        vers = Counter(v.get("form_version", 1) for v in rows)
        print(f"queue      : {len(q)}")
        print(f"covered    : {len(done)}/{len(q)} of the queue")
        if stale_only:
            print(f"             ({len(stale_only)} more keys carry a verdict on an EARLIER "
                  "answer, which --score excludes)")
        # Rows and answers are different numbers, and mixing the all-time row
        # count with the current-queue key count made "32 re-adjudicated" out of
        # rows that mostly describe answers the queue no longer serves.
        n_answers = len({(v["key"], v.get("answer_sha")) for v in rows})
        print(f"verdicts   : {len(rows)} rows over {n_answers} distinct answers"
              + (f" ({len(rows) - n_answers} re-adjudicated)" if len(rows) > n_answers else ""))
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
            if s["stale"]:
                bits.append(f"{s['stale']} made against a DIFFERENT answer for the same key "
                            "(excluded — the arms were regenerated since, Section 21.62)")
            if s["not_covered"]:
                bits.append(f"{s['not_covered']} flagged `not_covered` (counted — the human "
                            "agreed no LISTED error occurred, which is what blunder rate "
                            "measures; the answer may still be bad, Section 21.49)")
            if bits:
                print(f"\n{s['run']}: " + "; ".join(bits))
            # Loud and separate, because this one is a defect in the FORM and
            # not a property of the sample: it means the judge was graded on
            # entries the reviewer could not see, and the precision above was
            # computed without them (Section 21.75).
            if s["charges_not_shown"]:
                print(f"  !! {s['charges_not_shown']} judge charges name an error the "
                      "reviewer's form never displayed, and are EXCLUDED from precision.\n"
                      "     The form is behind the judge's rubric. Re-export tasks "
                      "(`--export-tasks`) and redeploy, then re-adjudicate: until then "
                      "no verdict can confirm or refute those charges.")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
