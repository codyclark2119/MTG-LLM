"""Self-checking tests for the gameplay gate arithmetic.

    python scripts/gameplay/test_eval_positions.py

WHY

The three gates are the headline output of the gameplay track — "the protocol
works", "the eval discriminates", "blunder rate is under 25%" — and none of them
had a test. Gate 2 spent four revisions computing the spread of per-arm means
over a binary, losing separation twice over, inside a 200-line report builder
where nobody read it (Sections 21.17, 21.18).

A gate is a boolean printed in bold. There is no output that looks wrong when it
is computed wrong, which is the same reason `test_eval.py` exists.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_positions import (  # noqa: E402
    GATE2_MIN_FRACTION,
    GATE2_MIN_SPREAD,
    GATE3_MAX_BLUNDER,
    gate2_discrimination,
    gate3_blunder,
    unearned_action_points,
)

CHECKS_RUN = 0


def check(label: str, got, want) -> bool:
    global CHECKS_RUN
    CHECKS_RUN += 1
    if got == want:
        return True
    print(f"  FAIL {label}\n       got  {got!r}\n       want {want!r}")
    return False


def rows(*per_position):
    """Build results rows from [(arm_a_score, arm_b_score), ...]."""
    return [{"arms": {f"arm{i}": {"correctness": c} for i, c in enumerate(scores)}}
            for scores in per_position]


ARMS = ["arm0", "arm1"]


def test_gate2() -> int:
    failed = 0

    # --- the basic count ----------------------------------------------------
    r = rows((5.0, 1.0), (3.0, 3.0), (4.0, 2.0))
    failed += not check("2 of 3 separate", gate2_discrimination(r, ARMS)[:3], (2, 3, 2 / 3))
    failed += not check("...and that passes", gate2_discrimination(r, ARMS)[3], True)

    failed += not check("all tied -> 0, FAIL",
                        gate2_discrimination(rows((3.0, 3.0), (2.0, 2.0)), ARMS), (0, 2, 0.0, False))
    failed += not check("all separated -> 100%, PASS",
                        gate2_discrimination(rows((5.0, 1.0), (4.0, 2.0)), ARMS), (2, 2, 1.0, True))

    # --- the threshold is a boundary, and boundaries are where gates break ---
    # Exactly 0.5 apart must COUNT (>=), and exactly 50% must PASS (>=).
    failed += not check("a gap of exactly 0.5 counts",
                        gate2_discrimination(rows((3.0, 2.5)), ARMS)[0], 1)
    failed += not check("a gap just under 0.5 does not",
                        gate2_discrimination(rows((3.0, 2.51)), ARMS)[0], 0)
    half = rows((5.0, 1.0), (3.0, 3.0))
    failed += not check("exactly 50% passes", gate2_discrimination(half, ARMS)[3], True)
    just_under = rows((5.0, 1.0), (3.0, 3.0), (2.0, 2.0))
    failed += not check("just under 50% fails", gate2_discrimination(just_under, ARMS)[3], False)

    # --- an unjudged arm is EXCLUDED, never a tie ---------------------------
    # This is the case that made the n=9 Qwen3 run look like a discrimination
    # failure. Counting an unparseable judge response as "the arms scored the
    # same" lets judge capacity drive the gate and reports it as a fact about
    # the positions.
    mixed = rows((5.0, 1.0), (None, None), (4.0, 2.0))
    failed += not check("unjudged position dropped from both numerator and denominator",
                        gate2_discrimination(mixed, ARMS)[:2], (2, 2))
    failed += not check("...so it passes rather than being dragged down",
                        gate2_discrimination(mixed, ARMS)[3], True)
    partial = rows((5.0, 1.0), (3.0, None))
    failed += not check("one arm unjudged is enough to exclude the position",
                        gate2_discrimination(partial, ARMS)[:2], (1, 1))

    # A run where nothing was judged must not report 0% and FAIL as though the
    # positions had been measured and found flat.
    none_judged = rows((None, None), (None, None))
    failed += not check("nothing judged -> 0 of 0",
                        gate2_discrimination(none_judged, ARMS)[:2], (0, 0))
    failed += not check("...and does not pass on an empty denominator",
                        gate2_discrimination(none_judged, ARMS)[3], False)

    # --- a missing arm key is not a zero ------------------------------------
    missing = [{"arms": {"arm0": {"correctness": 5.0}}}]
    failed += not check("an arm absent from the record excludes the position",
                        gate2_discrimination(missing, ARMS)[:2], (0, 0))

    # --- more than two arms: the SPREAD is what matters, not adjacency ------
    three = ["arm0", "arm1", "arm2"]
    failed += not check("spread taken across all arms, not neighbours",
                        gate2_discrimination(
                            [{"arms": {"arm0": {"correctness": 3.0},
                                       "arm1": {"correctness": 3.2},
                                       "arm2": {"correctness": 3.6}}}], three)[0], 1)
    failed += not check("three arms all within 0.5 do not separate",
                        gate2_discrimination(
                            [{"arms": {"arm0": {"correctness": 3.0},
                                       "arm1": {"correctness": 3.2},
                                       "arm2": {"correctness": 3.4}}}], three)[0], 0)

    # --- empty input --------------------------------------------------------
    failed += not check("no results at all", gate2_discrimination([], ARMS), (0, 0, 0.0, False))

    # --- the thresholds are the documented ones -----------------------------
    # Section 21.18 states both in the plan and the README. A silent edit here
    # would leave three documents describing a gate that no longer exists.
    failed += not check("min spread is 0.5 points", GATE2_MIN_SPREAD, 0.5)
    failed += not check("min fraction is 50%", GATE2_MIN_FRACTION, 0.50)
    return failed


def brows(*specs):
    """Rows of (difficulty, blundered_per_arm) for Gate 3."""
    return [{"difficulty": d,
             "arms": {f"arm{i}": {"blundered": b} for i, b in enumerate(bs)}}
            for d, bs in specs]


def test_gate3() -> int:
    failed = 0

    # Gate 3 reads the BEST arm, over basic+intermediate only.
    r = brows(("basic", (True, False)), ("basic", (True, False)))
    failed += not check("best arm wins", gate3_blunder(r, ARMS), ("arm1", 0.0))

    r = brows(("basic", (True, True)), ("basic", (False, True)))
    failed += not check("best arm is the lower rate", gate3_blunder(r, ARMS), ("arm0", 0.5))

    # Advanced positions are excluded: the gate is about whether the model is
    # reliable on the ordinary boards, not the hard ones.
    r = brows(("basic", (False, False)), ("advanced", (True, True)))
    failed += not check("advanced excluded", gate3_blunder(r, ARMS)[1], 0.0)

    r = brows(("intermediate", (True, True)), ("advanced", (False, False)))
    failed += not check("intermediate included", gate3_blunder(r, ARMS)[1], 1.0)

    # An unjudged call is dropped, not counted clean. Counting it clean would
    # let a judge that failed to parse push the blunder rate DOWN and the gate
    # toward PASS — the direction that manufactures a good result.
    r = brows(("basic", (None, True)), ("basic", (True, True)))
    failed += not check("unjudged dropped, not scored clean",
                        gate3_blunder(r, ARMS), ("arm0", 1.0))

    # No usable positions at all must not report 0% and PASS.
    failed += not check("no easy positions -> no arm", gate3_blunder(brows(("advanced", (False,))), ARMS),
                        (None, 1.0))
    failed += not check("empty input -> no arm", gate3_blunder([], ARMS), (None, 1.0))

    failed += not check("threshold is 25%", GATE3_MAX_BLUNDER, 0.25)
    return failed



def test_unearned_action_points() -> int:
    """A credited key point naming an action the answer never took (21.56).

    Correctness on positions is `points_hit / n_points`, so a key point
    credited without cause inflates the headline directly. The action list
    comes from the parser, which never sees the judge, so this is checkable
    rather than a second opinion — measured 9 of 18 checkable points on the
    n=24 32B run, and NONE of the flagged answers so much as contained the word.

    The narrowings matter more than the check: a diagnostic that over-fires
    gets ignored, which is worse than not having it.
    """
    failed = 0
    kps = ["Block Centaur Courser: deathtouch trades up",
           "Cast Ambush Viper now, flash lets you",
           "Summoning sickness does not stop blocking"]

    # The real case, all three arms, all three judges.
    failed += not check("credits a BLOCK to an answer that only casts and passes",
                        unearned_action_points("CAST Ambush Viper\nPASS", kps, [1, 2, 3]),
                        [1])
    failed += not check("an answer that does block is not flagged",
                        unearned_action_points(
                            "CAST Ambush Viper\nBLOCK Ambush Viper -> Centaur Courser",
                            kps, [1, 2, 3]), [])
    # Only points that OPEN with a verb. Point 3 mentions blocking mid-sentence
    # and is explaining a rule; flagging it would fire on every correct answer.
    failed += not check("a rule explanation mentioning an action is not an instruction",
                        3 in unearned_action_points("CAST Ambush Viper\nPASS", kps, [3]),
                        False)
    # KEEP is implicit: not mulliganing IS keeping.
    failed += not check("KEEP is never flagged",
                        unearned_action_points("PLAY Forest\nPASS",
                                               ["Keep - three mana sources"], [1]), [])
    # An answer that parsed nothing is Gate 1's problem, not the judge's.
    failed += not check("an unparseable answer is not blamed on the judge",
                        unearned_action_points("I would block it", kps, [1, 2, 3]), [])
    # The judge emits both shapes for points_hit (Section 21.12).
    failed += not check("dict-shaped points_hit is handled",
                        unearned_action_points("CAST Ambush Viper\nPASS", kps,
                                               [{"n": 1}, {"n": 2}]), [1])
    # Out-of-range and non-integer indices must not raise or count.
    failed += not check("a point index past the rubric is ignored",
                        unearned_action_points("PASS", kps, [99, "x", None]), [])
    failed += not check("ungraded points_hit yields nothing",
                        unearned_action_points("PASS", kps, None), [])
    return failed

def test_gate2_against_stored_runs() -> int:
    """The numbers Section 21.18 publishes, recomputed from the stored runs.

    A regression here means the plan's table describes a gate the code no longer
    implements — the failure mode this repo has hit with config comments and
    report headers alike.
    """
    failed = 0
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common import REPO_ROOT, read_jsonl

    expected = {                    # (separating, judged) from Section 21.18
        "pos_qwen25_3arm": (18, 22),
        "pos_qwen25_3arm_judge2": (16, 19),
        "positions_n22": (19, 22),
        "positions_n22_think": (20, 22),
        "pos_qwen3_14b_3arm": (2, 9),
    }
    for tag, want in expected.items():
        p = REPO_ROOT / f"eval/runs/{tag}.jsonl"
        if not p.exists():
            print(f"  skip {tag} (not present)")
            continue
        results = read_jsonl(p)
        arms = list(results[0]["arms"])
        failed += not check(f"{tag}", gate2_discrimination(results, arms)[:2], want)
    return failed


def test_coverage_guard() -> int:
    """A run the judge could not grade must not produce gate verdicts.

    Qwen3-14B at the default 600-token budget graded 0 of 72 arm answers — its
    `<think>` block spent the budget before any JSON — and the report printed
    three bold FAILs anyway, including "0/0 positions (0%)" and "best arm None
    at 100%". Numbers with no denominator, in the exact shape a real run makes.
    """
    import tempfile
    from pathlib import Path as _P
    import eval_positions as ep
    from common import read_jsonl, POSITIONS_PATH

    failed = 0
    positions = read_jsonl(POSITIONS_PATH)[:4]
    arms = ["base_open", "base_closed"]

    def report_for(graded: bool) -> str:
        results = [{
            "id": p["id"], "category": p["category"], "difficulty": p["difficulty"],
            "arms": {a: {"answer": "PASS", "correctness": 3.0 if graded else None,
                         "errors_made": [] if graded else None,
                         "points_hit": [1] if graded else None,
                         "blundered": False if graded else None,
                         "n_actions": 1, "n_parse_failures": 0, "parsed_ok": True,
                         "degenerate": False, "repeats_collapsed": 0, "n_legal": 1,
                         "all_legal": True, "illegal": [], "actions": [],
                         "reasoning_chars": 0} for a in arms},
        } for p in positions]

        class A:
            judge_max_tokens = 600; max_tokens = 512; k_rules = 6
            positions = POSITIONS_PATH; no_retrieval = False; limit = 0; seed = 42
            second_judge = None; arms = None; adapter_path = None
            out = _P('/dev/null')
        out = _P(tempfile.mkdtemp()) / "r.md"
        ep._write_report(results, positions, arms, ["base_closed"], A(),
                         "base", None, "judge", out)
        return out.read_text()

    ungraded = report_for(False)
    failed += not check("ungraded run: verdicts withheld", "NOT EVALUATED" in ungraded, True)
    for g in ("Gate 1 —", "Gate 2 —", "Gate 3 —"):
        failed += not check(f"ungraded run: no {g.strip(' —')} verdict", g in ungraded, False)
    failed += not check("ungraded run: names the likely cause",
                        "judge-max-tokens" in ungraded, True)

    # And the guard must not fire on a healthy run, or it hides every result.
    ok = report_for(True)
    failed += not check("graded run: verdicts printed", "NOT EVALUATED" in ok, False)
    failed += not check("graded run: Gate 1 evaluated", "Gate 1 —" in ok, True)
    return failed


def main() -> None:
    failed = 0
    for name, fn in (("gate2_discrimination", test_gate2),
                     ("gate3_blunder", test_gate3),
                     ("coverage guard", test_coverage_guard),
                     ("gate2 vs stored runs", test_gate2_against_stored_runs),
                     ("unearned_action_points", test_unearned_action_points)):
        print(f"{name} ...")
        failed += fn()

    if failed:
        print(f"\n{failed} check(s) FAILED")
        raise SystemExit(1)
    print(f"\nall checks passed ({CHECKS_RUN} assertions)")


if __name__ == "__main__":
    main()
