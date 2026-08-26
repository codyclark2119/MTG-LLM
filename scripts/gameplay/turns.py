"""Multi-step scenarios: play a whole turn, one decision at a time.

    python scripts/gameplay/turns.py --validate
    python scripts/gameplay/turns.py --render turn-payment-0001

WHY A SECOND SHAPE

A position is one board and one decision. That covers "which land do I play"
and "what do I block with", and it structurally cannot express "cast the trick,
then attack" or "play a whole turn" — the stages 4-7 of the gameplay curriculum.
Those are not harder positions, they are a different shape: several decisions in
sequence with the board changing between them.

A SCENARIO is a base position plus an ordered list of STEPS. Each step names its
phase, what is legal there, and what the right line is — the same rubric fields a
position carries, because `judge_batch_rubric` and every judge-free check already
work on that shape and should not be reimplemented.

TEACHER-FORCED, AND THAT IS A DELIBERATE LIMIT

The board advances on the REFERENCE line, never on what the model actually did.
Two reasons, and the second is the one that decides it:

  * Applying the model's own actions needs a rules engine — resolve the spell,
    update the battlefield, recompute what is legal. This repo is deliberately
    not one (see `rule_illegalities`), and a half-built engine would produce
    wrong board states that read as model errors.
  * Errors compound. If step 2 inherits step 1's mistake, every later step
    measures step 1 again, and a model that misplays the first decision scores
    zero on four independent skills it might have.

So each step is scored independently and the SEQUENCE is the unit: a turn is
valid when every step of it was. That is teacher forcing, it is standard for
sequence evaluation, and it is honest about what it does not test — recovery
from one's own mistake, which needs the engine and is a later problem.

WHAT THE MODEL IS TOLD

Each step renders the board for that step plus `prior_actions`, the line already
taken this turn. That is what a real player knows, and it carries the sequence
without the renderer having to derive one board from another.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import REPO_ROOT, read_jsonl, render_position  # noqa: E402

SCENARIOS_PATH = REPO_ROOT / "data/gold/turn_scenarios.jsonl"

# Fields a step may override on the base position. Anything else is inherited,
# so a scenario states what CHANGES between steps rather than restating a board
# five times — which is both less authoring and one fewer place to get it wrong.
STEP_OVERRIDES = ("phase", "battlefield", "players", "turn", "active_player",
                  "priority", "mana_available", "known_information")

# Fields a step must carry itself: they are what the step is testing.
STEP_REQUIRED = ("phase", "legal_actions", "key_points")


def scenario_from_submission(sub: dict) -> dict:
    """Build a scenario record from the authoring form's flat fields.

    The form authors each step as a whole board, because that is what a person
    can see and check. A scenario STORES only what changes between steps
    (`STEP_OVERRIDES`), because that is what keeps one board from being restated
    five times and drifting on the fourth.

    So the diff happens here rather than in the browser: step 1's board becomes
    the scenario's base, and each later step keeps only the fields that actually
    differ from it. A step that changes nothing carries no override, which is
    correct and common — the phase moves and the board does not.

    Server-side for the same reason `position_from_form` is: the browser never
    constructs the stored schema, so the form can change shape without the
    records changing shape (Section 21.96).
    """
    from common import position_from_form

    steps_in = sub.get("steps") or []
    if not steps_in:
        raise ValueError("a scenario needs at least one step")

    boards = [position_from_form(st) for st in steps_in]
    base = copy.deepcopy(boards[0])
    base["id"] = (sub.get("scenario_id") or "").strip()
    for key in ("category", "difficulty", "source", "rubric_source"):
        if sub.get(key):
            base[key] = sub[key]
    base.setdefault("source", "hand-authored")

    def rows(st, key):
        return [x.strip() for x in (st.get(key) or "").splitlines() if x.strip()]

    steps = []
    for i, (st, board) in enumerate(zip(steps_in, boards)):
        step = {
            "phase": board.get("phase") or "",
            "legal_actions": rows(st, "legal_actions"),
            "key_points": rows(st, "key_points"),
            "common_errors": rows(st, "common_errors"),
            "reference_actions": rows(st, "reference_actions"),
        }
        # Step 1 IS the base, so it never carries overrides. Later steps carry
        # only what differs — compared on the built structure, not on the form
        # text, so re-typing the same board in a different order is not a change.
        if i:
            for key in STEP_OVERRIDES:
                if key == "phase":
                    continue          # already on the step
                if board.get(key) != base.get(key):
                    step[key] = copy.deepcopy(board[key])
        steps.append(step)

    base["steps"] = steps
    # The base's own rubric fields describe step 1; the steps carry their own.
    base.pop("legal_actions", None)
    base["key_points"] = steps[0]["key_points"]
    base["common_errors"] = steps[0]["common_errors"]
    return base


def expand_steps(scenario: dict) -> list[dict]:
    """A scenario -> one position-shaped dict per step.

    Each is a complete position, so every existing check — `legality`,
    `protocol_findings`, `judge_batch_rubric` — works on it unchanged. Nothing
    downstream needs to know it came from a scenario.
    """
    steps = scenario.get("steps") or []
    out = []
    prior: list[str] = []
    for i, step in enumerate(steps):
        pos = copy.deepcopy({k: v for k, v in scenario.items() if k != "steps"})
        for key in STEP_OVERRIDES:
            if key in step:
                pos[key] = copy.deepcopy(step[key])
        pos["id"] = f"{scenario['id']}::step{i + 1}"
        pos["scenario_id"] = scenario["id"]
        pos["step_index"] = i + 1
        pos["step_count"] = len(steps)
        pos["legal_actions"] = step.get("legal_actions") or []
        pos["key_points"] = step.get("key_points") or []
        pos["common_errors"] = step.get("common_errors") or []
        # The step's own correct line, carried onto the expanded position under
        # the SAME name a position uses. It was read below for teacher forcing
        # and never set here, so a scenario step's reference was invisible to
        # the authoring form and was never run through `check_reference` — the
        # one field that claims to be verified was the one field nothing
        # verified (Section 21.90).
        pos["reference_actions"] = list(step.get("reference_actions") or [])
        # The line already taken this turn, as the model would know it. Shown
        # rather than derived: deriving one board from the previous one is the
        # rules engine this deliberately does not build.
        if prior:
            known = list(pos.get("known_information") or [])
            known.append("Already done this turn: " + "; ".join(prior))
            pos["known_information"] = known
        out.append(pos)
        # PHASE and END PHASE say WHERE the turn is, not what was done in it,
        # so they do not belong in "Already done this turn". TAP does: a land
        # tapped in step 1 is still tapped in step 2, which is exactly the state
        # the next step needs. Without this filter, requiring a PHASE line on
        # every reference (21.90) would have written "Already done this turn:
        # PHASE precombat main" into the board the model reads.
        prior.extend(a for a in (step.get("reference_actions") or [])
                     if not a.strip().upper().startswith(("PHASE ", "END PHASE")))
    return out


def validate_scenario(scenario: dict) -> list[str]:
    """Structural problems, in the spirit of `positions.validate_position`."""
    problems = []
    sid = scenario.get("id", "<no id>")
    steps = scenario.get("steps") or []
    if len(steps) < 2:
        problems.append(f"{sid}: a scenario needs at least 2 steps — one step is a position")
    for i, step in enumerate(steps, 1):
        for key in STEP_REQUIRED:
            if not step.get(key):
                problems.append(f"{sid} step {i}: missing {key!r}")
        # Without this the board never moves and every step reads identically,
        # which looks like a working scenario and tests one decision five times.
        if not step.get("reference_actions") and i < len(steps):
            problems.append(f"{sid} step {i}: no reference_actions, so the next step "
                            "would not know what had already happened")

    # A card the reference line CAST or PLAYED must leave the hand. `players` is
    # inherited unless a step overrides it, so forgetting the override leaves a
    # spell in hand after it was cast — and the model is then shown a card it
    # cannot have. That reads as the model ignoring a play when it is the
    # scenario lying to it, which is the worst class of harness bug: it
    # manufactures a model error out of an authoring slip.
    spent: list[str] = []
    for i, step in enumerate(steps, 1):
        # An EMPTY hand is a real value and the commonest one late in a turn, so
        # `step_hand or scenario_hand` fell through to the scenario's hand and
        # reported a card as unspent on every correctly-authored final step.
        # Presence of the override decides, never its truthiness — the same
        # falsy-value-reads-as-absent trap as `entry.get(k)` dropping False.
        step_hand = ((step.get("players") or {}).get("you") or {}).get("hand")
        base_hand = ((scenario.get("players") or {}).get("you") or {}).get("hand") or []
        hand = [c.lower() for c in (base_hand if step_hand is None else step_hand)]
        for card in spent:
            if card.lower() in hand:
                problems.append(
                    f"{sid} step {i}: {card!r} is still in hand after an earlier step "
                    "cast or played it — the step needs a `players` override")
        for act in step.get("reference_actions") or []:
            parts = act.split(None, 1)
            if len(parts) == 2 and parts[0].upper() in ("CAST", "PLAY"):
                spent.append(parts[1].split(" TARGET ")[0].strip())
    return problems


def load_steps(path: Path | None = None, *, validate: bool = True) -> list[dict]:
    """Every expanded step in a scenario file, position-shaped.

    Both of eval_positions' paths reach scenarios through here. They did not:
    generation expanded scenarios and rescore did not, so a run containing
    steps could be produced and never re-judged — which is the one thing a
    second judge needs (Section 9.9). That is the fifth one-of-two-paths bug
    in this repo, so the shared behaviour lives in one function and
    test_eval_positions asserts BOTH callers exist. No test over inputs and
    outputs can see a missing caller (Section 21.74).
    """
    scenarios = read_jsonl(path or SCENARIOS_PATH, missing_ok=True)
    if validate:
        bad = [problem for sc in scenarios for problem in validate_scenario(sc)]
        if bad:
            raise SystemExit("scenario problems:\n  " + "\n  ".join(bad))
    return [step for sc in scenarios for step in expand_steps(sc)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=Path, default=SCENARIOS_PATH)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--render", metavar="ID", help="print every step as the model sees it")
    args = ap.parse_args()

    scenarios = read_jsonl(args.scenarios, missing_ok=True)
    if not scenarios:
        raise SystemExit(f"no scenarios in {args.scenarios}")

    if args.render:
        sc = next((s for s in scenarios if s["id"] == args.render), None)
        if sc is None:
            raise SystemExit(f"no scenario {args.render!r}")
        for step in expand_steps(sc):
            print(f"\n{'=' * 62}\n{step['id']}  (step {step['step_index']}/{step['step_count']})")
            print(render_position(step))
            print(f"legal: {step['legal_actions']}")
        return

    problems = [p for s in scenarios for p in validate_scenario(s)]
    total_steps = sum(len(s.get("steps") or []) for s in scenarios)
    print(f"{len(scenarios)} scenarios, {total_steps} steps in {args.scenarios}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("all scenarios valid")


if __name__ == "__main__":
    main()
