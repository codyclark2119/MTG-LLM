"""Diff a position's `legal_actions` against what XMage says is legal.

Reads the surefire XML that `xmage_export.py`'s generated tests produce and
compares three engine answers per board:

    PLAYABLE:  Player.getPlayable(game, hidden)        -> casts and activations
    ATTACKER:  Player.getAvailableAttackers(game)      -> who may attack
    BLOCKER:   Player.getAvailableBlockers(game)       -> who may block

WHY THREE

`getPlayable` returns `List<ActivatedAbility>`, and declaring an attack is a
turn-based action rather than an activated ability, so attacks and blocks
CANNOT appear there. Querying only `getPlayable` reported one line for a board
that also lists an attack, and that read as the position being wrong when it was
the query being one-sided (Section 21.100) — 21.43's shape in a new place.

EACH AT THE STEP WHERE THE QUESTION HAS AN ANSWER

`legal_actions` enumerates the plays available over the TURN from a board, not
the plays available in the step the board states — that is why the grammar has
`PHASE`/`END PHASE`, and why a hand-authored reference on a `precombat main`
board walks to declare attackers before it attacks. So the engine is asked each
question where it can be answered: `getPlayable` at the stated step, and
`getAvailableAttackers` at DECLARE_ATTACKERS, in a second `@Test`.

Asking about attackers at the stated step instead returns nothing on every
main-phase board, and the first run read that as EIGHT positions being wrong
(Section 21.101). Both methods print into the same surefire report, so this
reads their union.

WHAT THE COMPARISON CAN AND CANNOT SAY

The engine names a spell once (`Cast Shock`); a position names each targeting of
it (`CAST Shock TARGET Grizzly Bears`, `CAST Shock TARGET opponent`). So the
match is on the CARD, not on the whole line — this checks whether a play is
available at all, never whether its target is legal.

`getAvailableBlockers` is NOT phase-sensitive: it returned the same creature at
a main phase and at declare blockers. So a BLOCKER line means "this creature
could block something", not "blocking is legal right now" — printed for the
record and deliberately not compared.

The `getPlayable` side stays step-scoped, so it is under-inclusive for a
sorcery-speed play listed on a pre-main board — castable later this turn, absent
from the engine's answer now. No position in the set does that (checked: no
non-main board lists a land drop), so it is a stated gap rather than a fixed
one; a position that does would need a third query at PRECOMBAT_MAIN.

Usage:
    python scripts/gameplay/xmage_diff.py --reports /path/to/Mage.Tests/target/surefire-reports
"""

import argparse
import collections
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import POSITIONS_PATH, read_jsonl, write_jsonl_atomic  # noqa: E402
from positions import load_positions  # noqa: E402
from xmage_export import class_name  # noqa: E402

_LINE = re.compile(r"(PLAYABLE|ATTACKER|BLOCKER): (.+)")
# `TARGET: <ability label> | <target name>` — the ability is repeated so a
# target can be tied back to the play it belongs to, since a board can offer two
# spells with different legal targets.
_TARGET = re.compile(r"TARGET: (.+?) \| (.+)")
# `PERM: <name> <p>/<t>[ tapped]` — the board AS THE ENGINE BUILT IT.
_PERM = re.compile(r"PERM: (.+?) (-?\d+)/(-?\d+)( tapped)?$")
# The engine labels a cast `Cast Shock`; the grammar's verb is `CAST`.
_ENGINE_VERB = {"cast": "CAST", "play": "PLAY", "activate": "ACTIVATE"}
# A position's line is `VERB Card TARGET ...`; the engine says `Cast Card`.
_ACTION = re.compile(r"^(CAST|PLAY|ACTIVATE|ATTACK|BLOCK)\s+([^,]+?)(?:\s+TARGET\s+|\s*->\s*|$)",
                     re.I)


def engine_answers(report: Path) -> dict[str, set[str]]:
    """{'PLAYABLE': {...}, 'ATTACKER': {...}, 'BLOCKER': {...}} from one report."""
    out = {"PLAYABLE": set(), "ATTACKER": set(), "BLOCKER": set()}
    try:
        text = ET.parse(report).getroot().itertext()
    except ET.ParseError:
        return out
    for chunk in text:
        for line in chunk.splitlines():
            m = _LINE.search(line)
            if m:
                out[m.group(1)].add(m.group(2).strip())
    return out


def engine_targets(report: Path) -> dict[str, list[str]]:
    """{'Cast Shock': ['Grizzly Bears', 'PlayerB', ...]} from one report."""
    out: dict[str, list[str]] = {}
    try:
        text = ET.parse(report).getroot().itertext()
    except ET.ParseError:
        return out
    for chunk in text:
        for line in chunk.splitlines():
            m = _TARGET.search(line)
            if m:
                out.setdefault(m.group(1).strip(), []).append(m.group(2).strip())
    return out


# The export maps our sides onto XMage's fixtures — `you` is always playerA
# (`xmage_export._player`) — so every engine report names players these two
# ways and no others. A target that is a PLAYER must be rendered the way the
# board names them, or `legal_actions` would carry a string the model never
# sees.
_ENGINE_PLAYERS = {"PlayerA": "you", "PlayerB": "opponent"}


def legal_actions_from_engine(eng: dict[str, set[str]],
                              targets: dict[str, list[str]]) -> list[str]:
    """The engine's answer, written in this project's action grammar.

    A DRAFT, and the distinction matters. `legal_actions` is a CURATED list —
    POSITIONS.md says *list only the plays that are genuinely available*, and a
    real board offers plays nobody would consider. The engine enumerates
    everything legal, so this is the raw material a person trims, not a finished
    field. An untrimmed list makes the closed arm a different task: the arm
    exists to separate *not knowing what is possible* from *not knowing what is
    good*, and handing it thirty options changes which of those is being asked.

    Blocks are not emitted: `getAvailableBlockers` is not phase-sensitive
    (see the module docstring), so it cannot say blocking is legal now.
    """
    out: list[str] = []
    for label in sorted(eng["PLAYABLE"]):
        head, _, rest = label.partition(" ")
        verb = _ENGINE_VERB.get(head.lower())
        if verb is None or not rest.strip():
            continue
        card = rest.strip()
        names = targets.get(label) or []
        if not names:
            out.append(f"{verb} {card}")
            continue
        for target in sorted(set(names)):
            # The board calls the other player "opponent"; the engine calls
            # them by name. Rendering "PlayerB" would put a string in
            # `legal_actions` that appears nowhere on the board the model reads.
            shown = _ENGINE_PLAYERS.get(target, target)
            out.append(f"{verb} {card} TARGET {shown}")
    for creature in sorted(eng["ATTACKER"]):
        out.append(f"ATTACK {creature}")
    return out




def _recorded_pt(perm: dict) -> str:
    """A permanent's P/T as the position stores it.

    A position stores `pt` as the STRING "1/1" and omits it entirely for a
    land, because `import_game` only writes it when non-zero — a rendered board
    must not claim a Mountain is a 0/0.

    Reading `power`/`toughness` with a default of 0 instead made every permanent
    compare as `0/0`, so the fidelity check reported 17 boards as mismatched on
    its first real run and named `Llanowar Elves 0/0` against the engine's
    `1/1`. A check that reads a key the writer never sets, and treats the
    default as a value, is 21.49's shape — except this one failed loudly rather
    than as a pass, which is the only reason it was caught the same hour
    (Section 21.123).
    """
    pt = perm.get("pt")
    if pt:
        return str(pt)
    if perm.get("power") is not None or perm.get("toughness") is not None:
        return f"{perm.get('power', 0)}/{perm.get('toughness', 0)}"
    return "0/0"          # a land, which the engine also reports as 0/0


def board_fidelity(report: Path, pos: dict) -> list[str]:
    """Ways the board the ENGINE built differs from the one recorded.

    The general form of every bug in this thread. Combat state (21.105), tokens
    (21.106), card faces (21.111) and permanent state (21.116) were all the same
    failure — the engine was handed a different board and nothing said so — and
    each was found only after it had cost something.

    Predicting which state classes matter is a list that grows every set. Asking
    the engine what it actually built is one check that covers all of them,
    including the ones nobody has thought of yet.

    Compares names, counts and P/T. A difference means the engine's answer is
    about a board the position does not describe, whatever the reason.
    """
    engine = collections.Counter()
    try:
        text = "".join(ET.parse(report).getroot().itertext())
    except ET.ParseError:
        return ["the engine report did not parse"]
    for line in text.splitlines():
        m = _PERM.search(line.strip())
        if m:
            engine[(m.group(1), f"{m.group(2)}/{m.group(3)}")] += 1
    if not engine:
        return []          # a run predating the readback; not a difference

    recorded = collections.Counter()
    for b in pos.get("battlefield") or []:
        recorded[(b.get("card"), _recorded_pt(b))] += 1

    out = []
    for key, n in (recorded - engine).items():
        out.append(f"recorded {n}x {key[0]!r} {key[1]} that the engine did not build")
    for key, n in (engine - recorded).items():
        out.append(f"the engine built {n}x {key[0]!r} {key[1]} that was not recorded")
    return out

def claimed(pos: dict) -> dict[str, set[str]]:
    """The position's own legal_actions, grouped the way the engine groups them."""
    out = {"PLAYABLE": set(), "ATTACKER": set(), "BLOCKER": set()}
    for action in pos.get("legal_actions") or []:
        m = _ACTION.match(action.strip())
        if not m:
            continue
        verb, name = m.group(1).upper(), m.group(2).strip()
        if verb in ("CAST", "PLAY", "ACTIVATE"):
            out["PLAYABLE"].add(name)
        elif verb == "ATTACK":
            out["ATTACKER"].add(name)
        elif verb == "BLOCK":
            out["BLOCKER"].add(name)
    return out


def emitter_blind_spot(pos: dict) -> str:
    """Why this board's `legal_actions` must NOT be generated, or "".

    The emitter can produce casts, activations and attacks. It cannot produce
    blocks (`getAvailableBlockers` is not phase-sensitive, so it cannot say
    blocking is legal now) or `ORDER TRIGGERS` (not an `ActivatedAbility`, so
    `getPlayable` never reports it) or a mulligan (before any step).

    On such a board the engine's answer is not *incomplete*, it is **empty** —
    and writing an empty `legal_actions` is far worse than writing none. The
    closed prompt then reads *"no legal plays are available; PASS is the only
    response"*, which is a falsehood about the board, and every correct block
    would score illegal against it. A gap that fills itself in with a confident
    wrong answer is the shape this project keeps paying for (21.49, 21.61), so
    the emitter refuses by name instead (Section 21.105).
    """
    from common import PRE_TURN_PHASE
    phase = (pos.get("phase") or "").lower()
    if PRE_TURN_PHASE in phase:
        return "a mulligan — before any step, so the engine has nothing to answer"
    if "declare blockers" in phase:
        return ("a blocking board — blocks are the whole answer here and the "
                "emitter cannot produce them")
    return ""


def emit_legal_actions(reports: Path, positions: Path, dry_run: bool = True) -> int:
    """Fill in `legal_actions` from the engine. Returns boards changed.

    The half `import_game.py` deliberately leaves empty. A recorded board has no
    author to enumerate its plays, and hand-enumeration is the thing this whole
    track exists because of — `timing_problems` and `mana_problems` were both
    written because a person got a list wrong.

    **Refuses a board that already has them.** Overwriting a hand-authored list
    with a raw engine dump would silently replace a curated field with an
    uncurated one, and the 32 stored positions would look untouched apart from a
    field nobody re-reads. `--positions` is a candidates file in normal use;
    pointing it at the gold set and having it quietly rewrite 31 of 32 boards is
    the accident worth making impossible rather than warning about.
    """
    rows = read_jsonl(positions, missing_ok=True)
    if not rows:
        raise SystemExit(f"no positions in {positions}")

    changed = skipped = no_report = 0
    refused: list[tuple[str, str]] = []
    for pos in rows:
        report = reports / f"TEST-org.mage.test.magicllm.{class_name(pos['id'])}.xml"
        if not report.exists():
            no_report += 1
            continue
        if pos.get("legal_actions"):
            skipped += 1
            continue
        blind = emitter_blind_spot(pos)
        if blind:
            refused.append((pos["id"], blind))
            continue
        actions = legal_actions_from_engine(engine_answers(report),
                                            engine_targets(report))
        pos["legal_actions"] = actions
        # Says where the list came from, so a reviewer knows it is an engine
        # dump rather than someone's judgement about what is worth offering.
        pos["legal_actions_source"] = "xmage-engine (untrimmed)"
        changed += 1
        print(f"  {pos['id']}: {len(actions)} action(s)")
        for a in actions[:6]:
            print(f"        {a}")
        if len(actions) > 6:
            print(f"        … {len(actions) - 6} more")

    print(f"\n{changed} board(s) filled, {skipped} already had a list "
          f"(left alone), {no_report} had no engine report")
    if refused:
        print(f"\n{len(refused)} REFUSED — the engine cannot answer for these, and an\nempty legal_actions would tell the closed arm that PASS is the only play:")
        for rid, why in refused:
            print(f"  {rid}\n      {why}")
    if changed:
        print("\nThis is an UNTRIMMED engine dump. `legal_actions` is a curated "
              "list —\nPOSITIONS.md: *list only the plays that are genuinely "
              "available* — and the\nclosed arm measures something different "
              "when handed thirty options.")
    if dry_run:
        print(f"\n(dry run — {positions} not written)")
        return changed
    write_jsonl_atomic(positions, rows)
    print(f"\n-> {positions}")
    return changed


def _why_uncompared(pos: dict) -> str:
    """Why a board contributed nothing to the comparison.

    Named reasons rather than a count, because the three are different
    problems: a mulligan is outside what an engine query can mean, blocks are a
    gap in the QUERY, and an empty list is a gap in the POSITION.
    """
    from common import PRE_TURN_PHASE
    actions = pos.get("legal_actions") or []
    if not actions:
        return "the position lists no legal actions at all"
    if PRE_TURN_PHASE in (pos.get("phase") or "").lower():
        return "a mulligan decision — before any step, so there is nothing to ask"
    verbs = {a.strip().split(" ", 1)[0].upper() for a in actions}
    if verbs <= {"BLOCK"}:
        return ("only BLOCK actions, and blocks are not compared "
                "(getAvailableBlockers is not phase-sensitive)")
    if verbs <= {"ORDER"}:
        return ("only ORDER TRIGGERS, which is not an ActivatedAbility so "
                "getPlayable cannot report it")
    return f"no comparable action ({', '.join(sorted(verbs))})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", type=Path, required=True)
    ap.add_argument("--positions", type=Path, default=POSITIONS_PATH)
    ap.add_argument("--emit-legal-actions", action="store_true",
                    help="write the engine's answer INTO --positions as "
                         "`legal_actions`. For imported drafts, which have "
                         "none; refuses a board that already carries them.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.emit_legal_actions:
        emit_legal_actions(args.reports, args.positions, dry_run=args.dry_run)
        return

    n_checked = n_clean = n_compared = 0
    findings: list[tuple[str, list[str]]] = []
    uncompared: list[tuple[str, str]] = []
    unfaithful: list[str] = []
    for pos in load_positions(args.positions):
        report = args.reports / f"TEST-org.mage.test.magicllm.{class_name(pos['id'])}.xml"
        if not report.exists():
            continue
        n_checked += 1
        eng, mine = engine_answers(report), claimed(pos)
        problems: list[str] = []

        # FIDELITY FIRST. If the engine built a different board, its answer is
        # about something else and comparing legal_actions to it is meaningless —
        # so this is reported and the board contributes to neither column, rather
        # than producing a legality "disagreement" that is really a setup bug.
        infidelity = board_fidelity(report, pos)
        if infidelity:
            findings.append((pos["id"], [f"BOARD DIFFERS: {x}" for x in infidelity]))
            unfaithful.append(pos["id"])
            continue

        # Casts: the engine prints "Cast Shock", we claim the card name.
        eng_cards = {re.sub(r"^(Cast|Play|Activate)\s+", "", s, flags=re.I).strip()
                     for s in eng["PLAYABLE"]}
        for card in sorted(mine["PLAYABLE"] - eng_cards):
            problems.append(f"claims {card!r} is castable; the engine does not offer it")
        for card in sorted(eng_cards - mine["PLAYABLE"]):
            problems.append(f"the engine offers {card!r}; the position does not list it")

        # Attacks ARE phase-sensitive, so a disagreement here is a real one.
        for card in sorted(mine["ATTACKER"] - eng["ATTACKER"]):
            problems.append(f"claims {card!r} may attack; the engine does not — "
                            f"at phase {pos.get('phase')!r}")

        if problems:
            findings.append((pos["id"], problems))
        else:
            n_clean += 1
        # COVERAGE, not agreement. A board whose `legal_actions` are all blocks
        # or a mulligan contributes nothing to either column: `claimed` files
        # them under BLOCKER or drops them, and neither is compared. It then
        # counts as "agrees", which is 21.49's shape — no problems found over a
        # board nothing examined (Section 21.105).
        if mine["PLAYABLE"] or mine["ATTACKER"] or eng_cards or eng["ATTACKER"]:
            n_compared += 1
        else:
            uncompared.append((pos["id"], _why_uncompared(pos)))

    print(f"{n_compared} of {n_checked} positions were actually COMPARED, "
          f"{n_clean - len(uncompared)} of those agree\n")
    for rid, problems in findings:
        print(f"  {rid}")
        for p in problems:
            print(f"      {p}")
    print(f"\n{len(findings)} disagree.")
    if unfaithful:
        print(f"\n{len(unfaithful)} of those are BOARD MISMATCHES, not legality\n"
              "disagreements — the engine built something the recording does not\n"
              "describe, so nothing about their legal_actions was tested.")

    # Coverage before caveats. "30 of 32 agree" counted every board with nothing
    # to compare as agreeing, which is the same defect as a gate verdict with no
    # coverage: no problems found over a set nothing examined (21.105). The
    # denominator is the finding here, so it goes first and by name.
    if uncompared:
        print(f"\n{len(uncompared)} contributed NOTHING to either column, and "
              "were previously counted\nas agreeing:")
        for rid, why in uncompared:
            print(f"  {rid}\n      {why}")

    print("\nWhat is out of scope, and why:")
    print("  blocks   — getAvailableBlockers is not phase-sensitive, so a BLOCKER")
    print("             line means 'could block something', not 'legal now'")
    print("  targets  — compared on the CARD only; the engine names a spell once")


if __name__ == "__main__":
    main()
