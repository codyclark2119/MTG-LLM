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
        # The line already taken this turn, as the model would know it. Shown
        # rather than derived: deriving one board from the previous one is the
        # rules engine this deliberately does not build.
        if prior:
            known = list(pos.get("known_information") or [])
            known.append("Already done this turn: " + "; ".join(prior))
            pos["known_information"] = known
        out.append(pos)
        prior.extend(step.get("reference_actions") or [])
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
