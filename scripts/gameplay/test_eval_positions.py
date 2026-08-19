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
    gate2_discrimination,
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


def main() -> None:
    failed = 0
    for name, fn in (("gate2_discrimination", test_gate2),
                     ("gate2 vs stored runs", test_gate2_against_stored_runs)):
        print(f"{name} ...")
        failed += fn()

    if failed:
        print(f"\n{failed} check(s) FAILED")
        raise SystemExit(1)
    print(f"\nall checks passed ({CHECKS_RUN} assertions)")


if __name__ == "__main__":
    main()
