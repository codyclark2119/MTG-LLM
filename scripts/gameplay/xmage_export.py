"""Emit an XMage JUnit test per position, to check `legal_actions` against a real
rules engine.

WHY THIS EXISTS

Every position in the set is machine-drafted (Section 21.98), and
`legal_actions` is the field most likely to be wrong: it is hand-enumerated,
`timing_problems` and `mana_problems` exist *because* hand-enumeration errs, and
nothing in this repo can settle it. This repo deliberately is not a rules engine.

XMage is one, it is open source, and its test framework builds a board
declaratively — `addCard(Zone.BATTLEFIELD, playerA, "Forest", 3)` — which is
close enough to a position record that the translation is mechanical. Its
`Player` interface exposes:

    List<ActivatedAbility> getPlayable(Game game, boolean hidden);
    PlayableObjectsList     getPlayableObjects(Game game, Zone zone);

So a position can be set up and the engine ASKED what is playable, rather than
only asserted against an outcome. That is the difference between checking that
each enumerated action is legal (a one-sided control — it cannot find a MISSING
action, and 21.43 says to say so) and checking the enumeration both ways.

WHAT THIS DOES AND DOES NOT CLAIM

It emits Java. It does not run it: there is no JVM here, and a generated test
that has never compiled is a draft, not a result. The output is meant to be
dropped into a `Mage.Tests` checkout and run there.

The API here is read from a checkout, not guessed. `CardTestPlayerBase`,
`addCard(Zone, player, name, count)`, `setLife`, `setStopAt(turn, PhaseStep.X)`
and the `currentGame` / `playerA` fields are all as the existing tests use them,
and `TestPlayer implements Player`, so `playerA.getPlayable(...)` is reachable
directly. Tapped is a FIFTH parameter to `addCard` rather than a separate call,
which is why grepping the tests for a `setTapped` helper finds nothing.

`PHASE_ALIASES` is the one concession to the data: the position set spells its
phases nine ways for fourteen closed steps, because it is generated (21.98).
Aliasing lets every position export while the run still names each one that
needed it, so the inconsistency stays visible rather than being absorbed here.

Usage:
    python scripts/gameplay/xmage_export.py --out /tmp/xmage
    python scripts/gameplay/xmage_export.py --position pos-combat-math-0005
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import POSITIONS_PATH  # noqa: E402
from positions import load_positions  # noqa: E402

# --- The two mappings this repo cannot verify -------------------------------
#
# Correct these once against a Mage.Tests checkout and every emitted test
# follows. They are here, together and named, so that "the export is wrong" has
# one place to look rather than being distributed through string templates.

# Our phase vocabulary -> XMage PhaseStep constants. XMage names steps, not
# phases, so "precombat main" is PRECOMBAT_MAIN and the combat steps are their
# own constants. `None` means "no confident mapping" and the emitter says so in
# the file rather than guessing.
PHASE_STEP = {
    "untap": "UNTAP",
    "upkeep": "UPKEEP",
    "draw": "DRAW",
    "precombat main": "PRECOMBAT_MAIN",
    "main": "PRECOMBAT_MAIN",
    "beginning of combat": "BEGIN_COMBAT",
    "declare attackers": "DECLARE_ATTACKERS",
    "declare blockers": "DECLARE_BLOCKERS",
    "combat damage": "COMBAT_DAMAGE",
    "end of combat": "END_COMBAT",
    "postcombat main": "POSTCOMBAT_MAIN",
    "end step": "END_TURN",
    "cleanup": "CLEANUP",
    # A mulligan happens before any step, so there is nothing to stop at. The
    # emitter refuses these rather than inventing a step.
    "opening hand": None,
}

# Verified against a checkout: tapped is a FIFTH parameter to addCard, not a
# separate call —
#     addCard(Zone gameZone, TestPlayer player, String cardName, int count,
#             boolean tapped)
# (`CardTestPlayerAPIImpl:742`). There is no `setTapped` helper, which is why
# grepping the tests for one found nothing.

# Our phase strings are not a vocabulary — 9 spellings for 14 closed steps,
# because the set is generated (21.98, 21.99). Aliases let all 32 export while
# `--report` still names every position that needed one, so the data problem
# stays visible instead of being absorbed here.
PHASE_ALIASES = {
    "pre-combat main phase": "precombat main",
    "post-combat main phase": "postcombat main",
    "opening hand, on the play": "opening hand",
    "opponent's declare attackers step": "declare attackers",
    "opponent's declare blockers step": "declare blockers",
}

_SAFE = re.compile(r"[^A-Za-z0-9]+")


def class_name(position_id: str) -> str:
    """`pos-combat-math-0005` -> `PosCombatMath0005Test`."""
    parts = [p for p in _SAFE.split(position_id) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) + "Test"


def _player(side: str) -> str:
    """Our sides are `you`/`opp`; XMage's fixtures are playerA/playerB."""
    return "playerA" if side == "you" else "playerB"


def emit(pos: dict) -> tuple[str, list[str]]:
    """Return (java source, problems). A problem means the test cannot be trusted.

    Problems are returned rather than raised so one unmappable position does not
    stop the export — the file is still written, with the problem stated inside
    it as well as reported. A generated test that looks runnable and is not is
    the failure mode worth avoiding.
    """
    problems: list[str] = []
    raw_phase = (pos.get("phase") or "").strip().lower()
    phase = PHASE_ALIASES.get(raw_phase, raw_phase)
    if phase != raw_phase:
        problems.append(f"phase {raw_phase!r} is not canonical; read as {phase!r}")
    step = PHASE_STEP.get(phase)
    if step is None:
        problems.append(f"no PhaseStep for phase {raw_phase!r}")

    lines: list[str] = []
    add = lines.append
    add("package org.mage.test.magicllm;")
    add("")
    add("import mage.constants.PhaseStep;")
    add("import mage.constants.Zone;")
    add("import org.junit.Test;")
    add("import org.mage.test.serverside.base.CardTestPlayerBase;")
    add("")
    add("/**")
    add(f" * {pos['id']} — {pos.get('category')}, {pos.get('difficulty')}")
    add(" *")
    add(" * Generated by scripts/gameplay/xmage_export.py. Do not hand-edit: fix the")
    add(" * position or the mapping tables in that file and re-export.")
    add(" *")
    add(" * Asks the engine what is playable and prints it. Compare against the")
    add(" * position's own legal_actions, listed at the bottom of this file.")
    for p in problems:
        add(f" *")
        add(f" * UNVERIFIED: {p}")
    add(" */")
    add(f"public class {class_name(pos['id'])} extends CardTestPlayerBase {{")
    add("")
    add("    @Test")
    add("    public void listPlayableActions() {")

    for side in ("you", "opp"):
        who = _player(side)
        info = (pos.get("players") or {}).get(side) or {}
        if "life" in info:
            add(f"        setLife({who}, {info['life']});")

    add("")
    # Battlefield, grouped so repeated basics become one call with a count.
    for side in ("you", "opp"):
        who = _player(side)
        perms = [b for b in (pos.get("battlefield") or [])
                 if b.get("controller") == side]
        # Grouped by (card, tapped): the same card untapped and tapped are two
        # calls, because `tapped` is a per-call flag and not per-permanent.
        counts: dict[tuple[str, bool], int] = {}
        order: list[tuple[str, bool]] = []
        for b in perms:
            card = b.get("card")
            if not card:
                continue
            key = (card, bool(b.get("tapped")))
            if key not in counts:
                order.append(key)
            counts[key] = counts.get(key, 0) + 1
        for card, is_tapped in order:
            n = counts[(card, is_tapped)]
            if is_tapped:
                add(f'        addCard(Zone.BATTLEFIELD, {who}, "{card}", {n}, true);')
            else:
                add(f'        addCard(Zone.BATTLEFIELD, {who}, "{card}"'
                    + (f", {n});" if n > 1 else ");"))

    add("")
    # Every test player loads "RB Aggro.dck" by default (CardTestPlayerAPIImpl:215),
    # so both start with a real hand and library that are not the position's. The
    # first run showed it immediately: the engine offered `Play Mountain` three
    # times from a hand this position never specified. Clear both first — a
    # position states its hand exactly, and anything else is the harness leaking
    # into the answer (Section 21.100).
    for side in ("you", "opp"):
        who = _player(side)
        add(f"        removeAllCardsFromHand({who});")
        add(f"        removeAllCardsFromLibrary({who});")

    add("")
    for card in ((pos.get("players") or {}).get("you") or {}).get("hand") or []:
        add(f'        addCard(Zone.HAND, playerA, "{card}");')

    # A library must exist or the engine decks the player on the next draw. The
    # count is what the position states; the CONTENT is not, so basics stand in
    # and are flagged — a position whose answer depends on library contents
    # cannot be checked this way.
    for side in ("you", "opp"):
        info = (pos.get("players") or {}).get(side) or {}
        n = info.get("library_count") or 0
        if n:
            filler = "Forest"
            add(f'        addCard(Zone.LIBRARY, {_player(side)}, "{filler}", {n});'
                "  // filler: count is real, contents are not")

    add("")
    if step:
        # Turn 1, NOT the position's own turn number. `setStopAt` SIMULATES up to
        # that turn rather than jumping to it, so `setStopAt(7, ...)` played six
        # turns of draws on top of the board and asked about the result. The
        # position's `turn` is descriptive — it says what turn the board
        # represents, not how many turns to play first (Section 21.100).
        add(f"        // position says turn {pos.get('turn')}; the board is placed"
            " directly, so stop at 1")
        add(f"        setStopAt(1, PhaseStep.{step});")
    else:
        add(f"        // no PhaseStep mapping for {phase!r} — set this by hand")
    add("        execute();")
    add("")
    add("        // What the ENGINE says is playable, versus what the position claims.")
    add("        // TestPlayer implements Player, so getPlayable is reachable directly.")
    add(f'        System.out.println("=== {pos["id"]} ===");')
    # Three calls, not one. `getPlayable` returns List<ActivatedAbility>, and
    # DECLARING AN ATTACK IS NOT AN ACTIVATED ABILITY — it is a turn-based
    # action — so attacks and blocks structurally cannot appear there. The first
    # clean run reported only `Cast Shock` for a board that also lists
    # `ATTACK Centaur Courser`, and that read as the position being wrong. It was
    # the query being one-sided: a control at DECLARE_ATTACKERS returned the same
    # single line, which is what exposed it (Section 21.100).
    #
    # `getAvailableAttackers` / `getAvailableBlockers` cover the other half.
    # Printed unconditionally and labelled: empty at the wrong step is itself
    # informative, and suppressing them by phase would hide exactly the case
    # where a position lists an ATTACK in a main phase.
    add("        playerA.getAvailableAttackers(currentGame)")
    add('                .forEach(p -> System.out.println("ATTACKER: " + p.getName()));')
    add("        playerA.getAvailableBlockers(currentGame)")
    add('                .forEach(p -> System.out.println("BLOCKER: " + p.getName()));')
    add("        playerA.getPlayable(currentGame, true).stream()")
    # Mana abilities are excluded: `legal_actions` enumerates PLAYS, and the
    # grammar handles mana separately through TAP lines. Leaving them in makes
    # every land a false difference — the first run reported `{T}: Add {G}.`
    # against a position that correctly does not list it.
    add('                .map(Object::toString)')
    add('                .filter(s -> !s.startsWith("{T}: Add"))')
    add('                .forEach(s -> System.out.println("PLAYABLE: " + s));')
    add("    }")
    add("}")
    add("")
    add("/* The position's own legal_actions, for comparison:")
    for a in pos.get("legal_actions") or []:
        add(f" *   {a}")
    if not (pos.get("legal_actions") or []):
        add(" *   (none — this board intentionally has no legal play)")
    add(" */")
    return "\n".join(lines) + "\n", problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", type=Path, default=POSITIONS_PATH)
    ap.add_argument("--out", type=Path, default=None,
                    help="directory to write one .java per position")
    ap.add_argument("--position", default=None, help="export just this id, to stdout")
    args = ap.parse_args()

    positions = load_positions(args.positions)
    if args.position:
        match = [p for p in positions if p["id"] == args.position]
        if not match:
            raise SystemExit(f"no position {args.position!r}")
        source, problems = emit(match[0])
        print(source)
        for p in problems:
            print(f"UNVERIFIED: {p}", file=sys.stderr)
        return

    if not args.out:
        raise SystemExit("--out DIR or --position ID")
    args.out.mkdir(parents=True, exist_ok=True)
    n_problem = 0
    for pos in positions:
        source, problems = emit(pos)
        (args.out / f"{class_name(pos['id'])}.java").write_text(source, encoding="utf-8")
        if problems:
            n_problem += 1
            print(f"  {pos['id']}: " + "; ".join(problems))
    print(f"{len(positions)} tests -> {args.out}")
    print(f"{n_problem} needed a phase alias — the position's own `phase` string is "
          "not one of the fourteen steps (21.99)")


if __name__ == "__main__":
    main()
