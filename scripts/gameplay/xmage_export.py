



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

Blocks are the same shape, one step later. `Player.getAvailableBlockers` only
answers "could this creature ever block something" and is not phase- or
attacker-sensitive (it returned the same creature at a main phase and at
declare blockers), so it cannot say a specific block is legal now — that is
still printed as `BLOCKER:` in `listPlayableActions`, unchanged, for the
record. `listAvailableBlocks` (Section 21.129) is a third `@Test`, at
`DECLARE_BLOCKERS`, that pairs each available blocker against each actually
declared attacker via `Permanent.canBlock(attackerId, game)` — the engine's
own evasion/restriction system, not a reimplementation of flying, menace, or
protection here.

A different timing gap runs the other direction: `getPlayable` at the STATED
step is under-inclusive for a sorcery-speed play that only opens up LATER
this turn — a land drop or sorcery listed on an upkeep or draw-step board,
castable once precombat main arrives but absent from the engine's answer
right now. `listPlayableAtMain` (Section 21.131) is a fourth `@Test`, run
only when the stated step is strictly before `PRECOMBAT_MAIN`, that asks the
identical `getPlayable`+targets question one step later. No position in the
set (the original 32 or the 71 recorded) currently has a land drop before
precombat main, so this closes a documented gap ahead of any position
actually hitting it, on purpose: the query is a mechanical repeat of one
already proven correct (`_emit_playable_query`, shared rather than
duplicated), not a guess about untested engine behavior.

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
                    canonical_phase, permanent_pt, phase_index, phase_step)
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


def blocks_reachable(phase: str) -> bool:
    """Is declare blockers still ahead of `phase` this turn?

    One step later than `combat_reachable`, for the same reason: a step past
    DECLARE_BLOCKERS, or a phase with no place in the turn order (`main`,
    `opening hand`), has no honest step at which to ask who may block whom.
    """
    here = phase_index(phase)
    return here is not None and here <= PHASE_ORDER.index("DECLARE_BLOCKERS")


def precombat_main_reachable(phase: str) -> bool:
    """Is `phase` strictly BEFORE precombat main this turn?

    The opposite direction from `combat_reachable`/`blocks_reachable`: those
    ask "is the decision still ahead", this asks "did the stated step's query
    already run at a point too early to see a sorcery-speed play that opens
    up later this same turn". `getPlayable` is step-scoped (Section 21.100),
    so a sorcery-speed spell or land drop listed on an upkeep or draw-step
    board — legitimately castable once precombat main arrives — would be
    absent from the engine's answer at the STATED step and read as a false
    disagreement. False for precombat main itself (already the stated-step
    query) and for anything at or after it (that window has already passed
    or IS the current one; there is nothing later this turn to ask about).
    """
    here = phase_index(phase)
    return here is not None and here < PHASE_ORDER.index("PRECOMBAT_MAIN")


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


# Keyword abilities a rebuilt token can carry, as the exact Java that creates
# one. Rule text -> expression, and fully qualified so no import management is
# needed in the emitted file.
#
# THIS IS A TABLE ON OUR SIDE, which the rest of this file avoids on principle
# (`CounterType.findByName`, `SubType.name()`, `common.PHASE_STEPS`). It is here
# because the engine offers no rule-text-to-ability lookup: there is no Keyword
# enum and no `Ability.byRule`. Verified against the checkout rather than
# assumed — every entry below was checked for `getInstance()`, and **Menace is
# not a singleton**, so it takes its constructor.
#
# Anything not listed REFUSES the board (Section 21.124). A token whose ability
# is real text — "{T}: Add one mana of any color" — cannot be rebuilt at all,
# and a vanilla stand-in for it would be the 21.113 mistake with a keyword
# instead of a token.
_KEYWORD_ABILITY = {
    "flying": "mage.abilities.keyword.FlyingAbility.getInstance()",
    "reach": "mage.abilities.keyword.ReachAbility.getInstance()",
    "trample": "mage.abilities.keyword.TrampleAbility.getInstance()",
    "vigilance": "mage.abilities.keyword.VigilanceAbility.getInstance()",
    "haste": "mage.abilities.keyword.HasteAbility.getInstance()",
    "lifelink": "mage.abilities.keyword.LifelinkAbility.getInstance()",
    "deathtouch": "mage.abilities.keyword.DeathtouchAbility.getInstance()",
    "first strike": "mage.abilities.keyword.FirstStrikeAbility.getInstance()",
    "double strike": "mage.abilities.keyword.DoubleStrikeAbility.getInstance()",
    "defender": "mage.abilities.keyword.DefenderAbility.getInstance()",
    "hexproof": "mage.abilities.keyword.HexproofAbility.getInstance()",
    "indestructible": "mage.abilities.keyword.IndestructibleAbility.getInstance()",
    "flash": "mage.abilities.keyword.FlashAbility.getInstance()",
    "shroud": "mage.abilities.keyword.ShroudAbility.getInstance()",
    "menace": "new mage.abilities.keyword.MenaceAbility()",
}


def token_keyword(rule: str) -> str | None:
    """The Java for a token's keyword ability, or None if it is not a keyword."""
    return _KEYWORD_ABILITY.get((rule or "").strip().lower().rstrip("."))


def token_class_name(card: str) -> str:
    """`Treefolk Token` -> `RecTreefolkToken`, a nested class name."""
    parts = [p for p in _SAFE.split(card) if p]
    return "Rec" + "".join(p[:1].upper() + p[1:] for p in parts)


def token_problems(pos: dict) -> list[str]:
    """Tokens this board carries that CANNOT be faithfully rebuilt.

    A token is not a card, so `addCard` cannot place one and the emitted test
    used to die with `Couldn't find a card: Everywhere` — 35 of 52 boards on the
    first recording, then 8 of 31 excluded on the second. Nor does the name
    identify it: 796 token classes exist and five are Treefolk, so
    "Treefolk Token" does not say which.

    So the collector records the characteristics and this rebuilds the token as
    a nested `TokenImpl` subclass (Section 21.113). Types, subtypes, colours and
    P/T reconstruct exactly — they are recorded as enum `name()`s, so there is no
    spelling table to drift.

    ABILITIES DO NOT. Nothing here can turn "{T}: Add {G}" back into Java, and a
    land token that taps for mana is precisely the case where a missing ability
    changes which spells are castable. So a token carrying rules text is
    REPORTED, exactly as before — the board is still not the position's board,
    and pretending otherwise would re-create the bug this fixes with a rebuilt
    token instead of an absent one.

    A token with no rules text rebuilds faithfully and is no longer a problem.
    """
    out = []
    for b in pos.get("battlefield") or []:
        if not b.get("token"):
            continue
        # Only rules that are NOT rebuildable keywords refuse. `flying`, `reach`
        # and `indestructible` are singletons the emitted class can add, and
        # those three alone accounted for 15 excluded boards in one game
        # (Section 21.124). A rule that is real text — "{T}: Add one mana of any
        # color" — still refuses, because a token missing it is a different
        # board and a vanilla stand-in looks correct.
        unknown = [r for r in (b.get("token_rules") or []) if not token_keyword(r)]
        if unknown:
            out.append(f"{b.get('card')!r} is a TOKEN with abilities this cannot "
                       f"rebuild ({'; '.join(unknown)[:80]}) — the engine sees a "
                       "different board")
        elif not b.get("token_types"):
            out.append(f"{b.get('card')!r} is a TOKEN recorded before its "
                       "characteristics were captured, so it cannot be rebuilt")
    return out


def rebuildable_tokens(pos: dict) -> list[dict]:
    """Token permanents this export can reconstruct, in board order."""
    out = []
    for b in pos.get("battlefield") or []:
        if not (b.get("token") and b.get("token_types")):
            continue
        rules = b.get("token_rules") or []
        if all(token_keyword(r) for r in rules):
            out.append(b)
    return out


def _token_classes(add, pos: dict) -> None:
    """One nested `TokenImpl` subclass per distinct rebuildable token.

    Nested rather than added to the checkout: a token shape belongs to the board
    that recorded it, and a shared class would need a registry keyed by a name
    that does not identify a token anyway.
    """
    seen: dict[str, dict] = {}
    for b in rebuildable_tokens(pos):
        seen.setdefault(b["card"], b)
    for card, b in seen.items():
        cls = token_class_name(card)
        add("")
        add(f"    /** Rebuilt from the recording: {card}. */")
        add(f"    public static final class {cls} extends TokenImpl {{")
        add(f"        public {cls}() {{")
        # The description is free text and XMage rejects one starting with an
        # indefinite article (TokenImpl's own constructor check), so it is built
        # from the P/T rather than copied from anywhere.
        tp, tt = permanent_pt(b)
        desc = f"{tp}/{tt} token"
        add(f'            super("{card}", "{desc}");')
        for t in b.get("token_types") or []:
            add(f"            cardType.add(CardType.{t});")
        for s in b.get("token_subtypes") or []:
            add(f"            subtype.add(SubType.{s});")
        for c in b.get("token_colors") or []:
            setter = {"W": "setWhite", "U": "setBlue", "B": "setBlack",
                      "R": "setRed", "G": "setGreen"}[c]
            add(f"            color.{setter}(true);")
        for r in b.get("token_rules") or []:
            add(f"            addAbility({token_keyword(r)});")
        add(f"            power = new MageInt({tp});")
        add(f"            toughness = new MageInt({tt});")
        add("        }")
        add(f"        private {cls}(final {cls} t) {{ super(t); }}")
        add(f"        public {cls} copy() {{ return new {cls}(this); }}")
        add("    }")


def _place_tokens(add, pos: dict) -> None:
    """Put the rebuilt tokens onto the battlefield, tapped/attacking as recorded."""
    toks = rebuildable_tokens(pos)
    if not toks:
        return
    add(f"        // {len(toks)} rebuilt token(s) — a token cannot be addCard'd")
    for b in toks:
        who = _player(b.get("controller") or "you")
        cls = token_class_name(b["card"])
        add(f"        new {cls}().putOntoBattlefield(1, currentGame, null, "
            f"{who}.getId(), {str(bool(b.get('tapped'))).lower()}, "
            f"{str(bool(b.get('attacking'))).lower()});")
    add("")


def permanent_state_problems(pos: dict) -> list[str]:
    """Board state `addCard(name)` cannot reproduce (Section 21.116).

    A permanent is not a card name plus tapped. Four kinds of state change what
    the engine will answer and none of them survive `addCard`:

    - a CHOSEN value — Secluded Courtyard's creature type decides which mana it
      makes, and the choice lives in the game state rather than on the permanent;
    - ATTACHMENTS — an Aura or Equipment grants abilities and changes P/T;
    - FACE-DOWN — a morph is a 2/2 with no abilities, not the card it will
      become;
    - PHASED OUT — the permanent is not on the battlefield at all.

    Counters are reported only when they are the reason the P/T diverges,
    because `putOntoBattlefield` cannot take counters either and a 4/4 rebuilt
    as a 2/2 is a different board.

    Reported, never silently rebuilt. That is the rule the token work settled
    (21.113): a wrong permanent looks correct and an absent one does not, so the
    only safe options are reproduce it exactly or refuse the board.
    """
    out = []
    for b in pos.get("battlefield") or []:
        card = b.get("card")
        if b.get("chosen"):
            out.append(f"{card!r} carries a CHOSEN value ({', '.join(b['chosen'])}) "
                       "which lives in the game state and cannot be replayed")
        if b.get("attachments"):
            out.append(f"{card!r} has {', '.join(b['attachments'])} attached; "
                       "attachments are not reproduced")
        # Counters are NOT refused. `addCounters(turn, step, player, cardName,
        # CounterType, count)` exists in the test API and `CounterType.findByName`
        # resolves the recorded name, so this rebuilds — which matters because
        # counters are the largest class by far: 10.6% of cards and **28% of
        # recorded permanents**. Refusing them would have cost more yield than
        # tokens did.
        #
        # Whether `findByName` knows a given name is not predictable from here
        # without copying the enum, and a copied enum is a table that drifts. So
        # the emitted Java calls it directly: an unknown name returns null and
        # the test fails, which the diff already treats as an excluded board.
        # Loud and self-correcting, rather than a second list to maintain.
        if b.get("face_down"):
            out.append(f"{card!r} is FACE DOWN — a 2/2 with no abilities, not the "
                       "card it will turn into")
        if b.get("phased_in") is False:
            out.append(f"{card!r} is PHASED OUT and is not on the battlefield")
    return out



def _counters(add, pos: dict) -> None:
    """Restore counters recorded on permanents.

    `addCounters` is scheduled like a player action — (turn, step, player, card,
    type, count) — so it runs during the simulated turn rather than at setup.
    That is why it goes in beside the combat declarations rather than with
    `addCard`.

    `CounterType.findByName` does the name lookup inside the engine, so there is
    no counter-name table on this side to drift (Section 21.117).
    """
    withc = [b for b in (pos.get("battlefield") or []) if b.get("counters")]
    if not withc:
        return
    step = phase_step(canonical_phase(pos.get("phase") or "") or "") or "PRECOMBAT_MAIN"
    turn = active_turn(pos)
    add(f"        // counters recorded on {len(withc)} permanent(s)")
    for b in withc:
        who = _player(b.get("controller") or "you")
        for name, count in b["counters"].items():
            add(f'        addCounters({turn}, PhaseStep.{step}, {who}, "{b["card"]}",')
            add(f'                CounterType.findByName("{name}"), {count});')
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


def _emit_playable_query(add) -> None:
    """`getPlayable` plus per-ability TARGET enumeration.

    Shared by `listPlayableActions` (queried at the stated step) and
    `listPlayableAtMain` (Section 21.131, queried at PRECOMBAT_MAIN for a
    board whose stated step is earlier this turn) — one copy rather than two,
    since two copies of the same query is exactly the kind of drift this
    project keeps finding costly (`common.py`'s whole reason for existing).

    `getPlayable` returns List<ActivatedAbility>, and DECLARING AN ATTACK IS
    NOT AN ACTIVATED ABILITY — it is a turn-based action — so attacks cannot
    appear here at any step. The first clean run reported only `Cast Shock`
    for a board that also lists `ATTACK Centaur Courser`, and that read as the
    position being wrong; it was the query being one-sided (21.100).
    Mana abilities are excluded: `legal_actions` enumerates PLAYS, and the
    grammar handles mana separately through TAP lines. Leaving them in makes
    every land a false difference — the first run reported `{T}: Add {G}.`
    against a position that correctly does not list it.

    Each playable is followed by its POSSIBLE TARGETS. `legal_actions` names
    each targeting separately (`CAST Shock TARGET Grizzly Bears`), and
    `match_to_legal` has no untargeted-CAST fallback the way it has for
    ATTACK — so an untargeted entry would score every correct targeted cast
    illegal, which is 21.61's shape a fourth time. The engine knows the legal
    targets; guessing them here would be inventing a format again (21.102).
    """
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
    blocks_step = "DECLARE_BLOCKERS" if blocks_reachable(phase) else None
    need_main_query = precombat_main_reachable(phase)
    problems.extend(combat_problems(pos))
    problems.extend(token_problems(pos))
    problems.extend(permanent_state_problems(pos))

    lines: list[str] = []
    add = lines.append
    add("package org.mage.test.magicllm;")
    add("")
    add("import mage.MageInt;")
    add("import mage.constants.CardType;")
    add("import mage.counters.CounterType;")
    add("import mage.constants.PhaseStep;")
    add("import mage.constants.SubType;")
    add("import mage.game.permanent.token.TokenImpl;")
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
        # Tokens excluded: `addCard` needs a card name and a token is not a
        # card. `token_problems` has already said so in the file's header.
        perms = [b for b in (pos.get("battlefield") or [])
                 if b.get("controller") == side and not b.get("token")]
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
    _place_tokens(add, pos)
    _counters(add, pos)
    _combat(add, pos)

    # --- the two query methods, each on its own copy of the board -----------
    add = lines.append

    add("    /** What is castable/activatable in the step the position states. */")
    add("    @Test")
    add("    public void listPlayableActions() {")
    lines.extend(board)
    _stop_at(add, step, phase, pos)
    add(f'        System.out.println("=== {pos["id"]} ===");')
    # The board AS THE ENGINE BUILT IT, so the rebuild can be checked against
    # the recording instead of trusted. Every fix in this thread — combat state,
    # tokens, counters — was a case of handing the engine a different board and
    # not finding out. One readback catches all of them and any future one
    # (Section 21.117).
    add("        currentGame.getBattlefield().getAllActivePermanents().forEach(p ->")
    add('                System.out.println("PERM: " + p.getName() + " "'
        ' + p.getPower().getValue() + "/" + p.getToughness().getValue()')
    add('                        + (p.isTapped() ? " tapped" : "")));')
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
    _emit_playable_query(add)
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

    if blocks_step is None:
        add("")
        add(f"    // No block query: {phase!r} is past declare blockers, so a")
        add("    // BLOCK in this position's legal_actions is genuinely unreachable")
        add("    // and the empty engine answer is the honest one.")
    else:
        add("")
        add("    /**")
        add("     * Which specific attacker each of your creatures may legally")
        add("     * block, at DECLARE_BLOCKERS regardless of the step the position")
        add("     * states (21.101's shape, for blocks). Unlike the BLOCKER: line")
        add("     * in listPlayableActions() — which only asks 'could this ever")
        add("     * block something' and is not phase- or attacker-sensitive —")
        add("     * this pairs each available blocker against each actually")
        add("     * declared attacker via Permanent.canBlock, which runs the")
        add("     * engine's own evasion/restriction system (flying, menace,")
        add("     * protection, ...) rather than reimplementing any of it here.")
        add("     */")
        add("    @Test")
        add("    public void listAvailableBlocks() {")
        lines.extend(board)
        _stop_at(add, blocks_step, phase, pos)
        add(f'        System.out.println("=== {pos["id"]} @DECLARE_BLOCKERS ===");')
        add("        java.util.List<mage.game.permanent.Permanent> attackers ="
            " new java.util.ArrayList<>();")
        add("        for (mage.game.permanent.Permanent p : "
            "currentGame.getBattlefield().getAllActivePermanents()) {")
        add("            if (p.isAttacking()) {")
        add("                attackers.add(p);")
        add("            }")
        add("        }")
        add("        for (mage.game.permanent.Permanent blocker : "
            "playerA.getAvailableBlockers(currentGame)) {")
        add("            for (mage.game.permanent.Permanent attacker : attackers) {")
        add("                if (blocker.canBlock(attacker.getId(), currentGame)) {")
        add('                    System.out.println("BLOCK: " + blocker.getName()'
            ' + " -> " + attacker.getName());')
        add("                }")
        add("            }")
        add("        }")
        add("    }")

    if need_main_query:
        add("")
        add("    /**")
        add("     * Sorcery-speed plays castable once precombat main arrives,")
        add("     * even though the stated step is earlier this turn (Section")
        add("     * 21.131). getPlayable is step-scoped, so the stated-step query")
        add("     * alone is under-inclusive here — a land drop or sorcery listed")
        add("     * on an upkeep or draw-step board would be absent from it and")
        add("     * read as a false disagreement (21.100's shape, for timing).")
        add("     */")
        add("    @Test")
        add("    public void listPlayableAtMain() {")
        lines.extend(board)
        _stop_at(add, "PRECOMBAT_MAIN", phase, pos)
        add(f'        System.out.println("=== {pos["id"]} @PRECOMBAT_MAIN ===");')
        _emit_playable_query(add)
        add("    }")

    _token_classes(add, pos)
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
    n_phase = n_combat = n_token = 0
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
            n_token += any("is a TOKEN" in p for p in problems)
    print(f"{len(positions)} tests -> {args.out}")
    print(f"{n_phase} could not be mapped to an XMage PhaseStep. This was TEN "
          "before 21.102 canonicalized\nthe stored phases; what is left is the "
          "mulligan boards, which correctly have no step.")
    if n_token:
        print(f"{n_token} carry a TOKEN the framework cannot place, so the engine sees a\n"
              "DIFFERENT board. Do not read their answers as being about the position.")
    if n_combat:
        print(f"{n_combat} state an ILLEGAL board — an attacker the active player "
              "does not control.\nThat is a defect in the position, not in the "
              "export.")


if __name__ == "__main__":
    main()
