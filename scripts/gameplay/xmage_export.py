



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

`legal_actions` IS TURN-SCOPED, AND THIS QUERY MUST BE TOO

The single most important thing about the comparison, and it was wrong in the
first version. A position states one `phase`, but `legal_actions` enumerates the
plays available *over the turn* from that board, not the plays available in that
step — which is why the grammar has `PHASE` and `END PHASE` at all, why a
hand-authored reference on a `precombat main` board walks to declare attackers
before it attacks, and why `check_reference` accepts that line only because the
attack is in `legal_actions`.

Asking `getAvailableAttackers` at the STATED step therefore returns nothing on
every main-phase board, and the first run read that as eight positions being
wrong (Section 21.101). Attacks are declared in exactly one step, so the query
for them runs at `DECLARE_ATTACKERS` — a separate `@Test`, because `execute()`
runs once per method and each method gets a fresh game.

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

from common import (PHASE_ORDER, POSITIONS_PATH,  # noqa: E402
                    canonical_phase, phase_index, phase_step)
from positions import load_positions  # noqa: E402

# --- The two mappings this repo cannot verify -------------------------------
#
# Correct these once against a Mage.Tests checkout and every emitted test
# follows. They are here, together and named, so that "the export is wrong" has
# one place to look rather than being distributed through string templates.

# There is no phase mapping table here any more, and that is the point.
# `common.PHASE_STEPS` mirrors `mage.constants.PhaseStep`, so a position's phase
# resolves to an XMage constant by `common.phase_step()` — the same function the
# grammar and the validator use (Section 21.102). The two hand-written tables
# this file used to carry, one mapping our words to XMage's and one aliasing our
# nine spellings onto our own fourteen, both described a vocabulary problem that
# no longer exists.
#
# `phase_step` returns None for a mulligan (no step to stop at) and for `main`
# (an ambiguous alias for one of two steps). The emitter says so in the file
# rather than guessing.

# Verified against a checkout: tapped is a FIFTH parameter to addCard, not a
# separate call —
#     addCard(Zone gameZone, TestPlayer player, String cardName, int count,
#             boolean tapped)
# (`CardTestPlayerAPIImpl:742`). There is no `setTapped` helper, which is why
# grepping the tests for one found nothing.

_SAFE = re.compile(r"[^A-Za-z0-9]+")


def class_name(position_id: str) -> str:
    """`pos-combat-math-0005` -> `PosCombatMath0005Test`."""
    parts = [p for p in _SAFE.split(position_id) if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) + "Test"


def _player(side: str) -> str:
    """Our sides are `you`/`opp`; XMage's fixtures are playerA/playerB."""
    return "playerA" if side == "you" else "playerB"


def combat_reachable(phase: str) -> bool:
    """Is declare attackers still ahead of `phase` this turn?

    False for a step past it and for a phase with no place in the turn order at
    all — `main` (an alias for one of two steps) and `opening hand` (before the
    turn starts). Both cases mean the same thing here: do not ask the engine
    about attacks, because there is no honest step at which to ask.
    """
    here = phase_index(phase)
    return here is not None and here <= PHASE_ORDER.index("DECLARE_ATTACKERS")


def active_turn(pos: dict) -> int:
    """Which TEST turn belongs to the position's active player.

    playerA starts, so playerA's turns are odd and playerB's are even. A board
    on the opponent's turn — every blocking position is — must therefore run to
    turn 2, or the attack it describes cannot be scripted: only the active
    player declares attackers (508.1).

    This is NOT the position's own `turn`, which is descriptive. `setStopAt`
    SIMULATES, so `setStopAt(7, …)` played six turns on top of the board and
    asked about the result (Section 21.100). Two is the smallest number that
    makes the opponent active.
    """
    return 2 if pos.get("active_player") == "opp" else 1


def _stop_at(add, step: str | None, phase: str, pos: dict) -> None:
    """Emit the `setStopAt` / `execute` pair shared by both query methods."""
    if step:
        turn = active_turn(pos)
        add(f"        // position says turn {pos.get('turn')}; the board is placed"
            " directly, so the")
        add(f"        // number here is only whose turn it is — "
            f"{'playerB' if turn == 2 else 'playerA'} is active on turn {turn}.")
        add(f"        setStopAt({turn}, PhaseStep.{step});")
    else:
        add(f"        // no PhaseStep mapping for {phase!r} — set this by hand")
    add("        execute();")
    add("")


def combat_problems(pos: dict) -> list[str]:
    """Attacking permanents the rules do not allow this board to have.

    Only the ACTIVE player declares attackers (508.1). Computed separately from
    the emission below so the problem reaches the file's header, which is
    written before the board — a generated test that states its own defect is
    the point of returning problems rather than raising.
    """
    active = pos.get("active_player") or "you"
    return [f"{b.get('card')!r} is attacking but is controlled by "
            f"{b.get('controller')!r} while {active!r} is the active player — "
            "only the active player declares attackers (508.1)"
            for b in (pos.get("battlefield") or [])
            if b.get("attacking") and b.get("controller") != active]


def _combat(add, pos: dict) -> None:
    """Script the attacks the board says are already declared.

    Without this, a board at declare blockers exports with NOBODY attacking, so
    the engine is asked about a different board than the position describes —
    and the diff could not see it, because it does not compare blocks. Nine of
    thirty-two positions carry `attacking` permanents (Section 21.105).

    An attacker the active player does not control is skipped, not scripted:
    `attack()` would fail at runtime. `combat_problems` has already said so.
    """
    active = pos.get("active_player") or "you"
    attackers = [b for b in (pos.get("battlefield") or [])
                 if b.get("attacking") and b.get("controller") == active]
    if not attackers:
        return
    turn = active_turn(pos)
    attacker_side = _player(active)
    defender_side = _player("opp" if active == "you" else "you")
    add(f"        // {len(attackers)} attacking creature(s): DECLARED, not placed")
    add("        // as a flag — a board at declare blockers has to be in combat.")
    for b in attackers:
        add(f'        attack({turn}, {attacker_side}, "{b["card"]}", {defender_side});')
    add("")


def emit(pos: dict) -> tuple[str, list[str]]:
    """Return (java source, problems). A problem means the test cannot be trusted.

    Problems are returned rather than raised so one unmappable position does not
    stop the export — the file is still written, with the problem stated inside
    it as well as reported. A generated test that looks runnable and is not is
    the failure mode worth avoiding.
    """
    problems: list[str] = []
    raw_phase = (pos.get("phase") or "").strip().lower()
    phase = canonical_phase(raw_phase) or raw_phase
    if phase != raw_phase:
        problems.append(f"phase {raw_phase!r} is not canonical; read as {phase!r}")
    step = phase_step(phase)
    if step is None:
        problems.append(f"no PhaseStep for phase {raw_phase!r}")
    # `None` means "there is no step this turn at which to ask about attacks".
    combat_step = "DECLARE_ATTACKERS" if combat_reachable(phase) else None
    problems.extend(combat_problems(pos))

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

    # The board, built once and emitted into every test method. JUnit re-runs
    # setUp per method, so each gets its own fresh game and must place the board
    # itself; there is nothing to share by hoisting it.
    board: list[str] = []
    add = board.append

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
    _combat(add, pos)

    # --- the two query methods, each on its own copy of the board -----------
    add = lines.append

    add("    /** What is castable/activatable in the step the position states. */")
    add("    @Test")
    add("    public void listPlayableActions() {")
    lines.extend(board)
    _stop_at(add, step, phase, pos)
    add(f'        System.out.println("=== {pos["id"]} ===");')
    # `getPlayable` returns List<ActivatedAbility>, and DECLARING AN ATTACK IS
    # NOT AN ACTIVATED ABILITY — it is a turn-based action — so attacks cannot
    # appear here at any step. The first clean run reported only `Cast Shock`
    # for a board that also lists `ATTACK Centaur Courser`, and that read as the
    # position being wrong; it was the query being one-sided (21.100).
    # Mana abilities are excluded: `legal_actions` enumerates PLAYS, and the
    # grammar handles mana separately through TAP lines. Leaving them in makes
    # every land a false difference — the first run reported `{T}: Add {G}.`
    # against a position that correctly does not list it.
    #
    # Each playable is followed by its POSSIBLE TARGETS. `legal_actions` names
    # each targeting separately (`CAST Shock TARGET Grizzly Bears`), and
    # `match_to_legal` has no untargeted-CAST fallback the way it has for
    # ATTACK — so an untargeted entry would score every correct targeted cast
    # illegal, which is 21.61's shape a fourth time. The engine knows the legal
    # targets; guessing them here would be inventing a format again (21.102).
    add("        for (mage.abilities.ActivatedAbility ability "
        ": playerA.getPlayable(currentGame, true)) {")
    add("            String label = ability.toString();")
    add('            if (label.startsWith("{T}: Add")) {')
    add("                continue;")
    add("            }")
    add('            System.out.println("PLAYABLE: " + label);')
    add("            for (mage.target.Target target : ability.getTargets()) {")
    add("                for (java.util.UUID id : target.possibleTargets("
        "playerA.getId(), ability, currentGame)) {")
    add("                    String name = null;")
    add("                    mage.game.permanent.Permanent p = "
        "currentGame.getPermanent(id);")
    add("                    if (p != null) {")
    add("                        name = p.getName();")
    add("                    } else if (currentGame.getPlayer(id) != null) {")
    add("                        name = currentGame.getPlayer(id).getName();")
    add("                    } else if (currentGame.getCard(id) != null) {")
    add("                        name = currentGame.getCard(id).getName();")
    add("                    }")
    add("                    if (name != null) {")
    add('                        System.out.println("TARGET: " + label + " | " + name);')
    add("                    }")
    add("                }")
    add("            }")
    add("        }")
    # Not phase-sensitive — it returned the same creature at a main phase and at
    # declare blockers — so it says "could block something", not "blocking is
    # legal now". Printed for the record; `xmage_diff` does not compare it.
    add("        playerA.getAvailableBlockers(currentGame)")
    add('                .forEach(p -> System.out.println("BLOCKER: " + p.getName()));')
    add("    }")

    if combat_step is None:
        add("")
        add(f"    // No attacker query: {phase!r} is past declare attackers, so an")
        add("    // ATTACK in this position's legal_actions is genuinely unreachable")
        add("    // and the empty engine answer is the honest one.")
    else:
        add("")
        add("    /**")
        add("     * Who may attack. Declared in exactly ONE step, so this runs at")
        add("     * DECLARE_ATTACKERS regardless of the step the position states —")
        add("     * `legal_actions` is turn-scoped and this query must be too (21.101).")
        add("     */")
        add("    @Test")
        add("    public void listAvailableAttackers() {")
        lines.extend(board)
        _stop_at(add, combat_step, phase, pos)
        add(f'        System.out.println("=== {pos["id"]} @DECLARE_ATTACKERS ===");')
        add("        playerA.getAvailableAttackers(currentGame)")
        add('                .forEach(p -> System.out.println("ATTACKER: " + p.getName()));')
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
    n_phase = n_combat = 0
    for pos in positions:
        source, problems = emit(pos)
        (args.out / f"{class_name(pos['id'])}.java").write_text(source, encoding="utf-8")
        if problems:
            print(f"  {pos['id']}: " + "; ".join(problems))
            # Counted apart because they mean opposite things. A phase with no
            # PhaseStep is a limit of the EXPORT — a mulligan is not a step, and
            # never will be. An illegal attacker is a defect in the POSITION.
            # One number covering both would report a data defect as tooling
            # coverage, which is the direction that reads as fine.
            n_phase += any("PhaseStep" in p for p in problems)
            n_combat += any("508.1" in p for p in problems)
    print(f"{len(positions)} tests -> {args.out}")
    print(f"{n_phase} could not be mapped to an XMage PhaseStep. This was TEN "
          "before 21.102 canonicalized\nthe stored phases; what is left is the "
          "mulligan boards, which correctly have no step.")
    if n_combat:
        print(f"{n_combat} state an ILLEGAL board — an attacker the active player "
              "does not control.\nThat is a defect in the position, not in the "
              "export.")


if __name__ == "__main__":
    main()
