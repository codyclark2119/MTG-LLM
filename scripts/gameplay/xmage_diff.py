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
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import POSITIONS_PATH  # noqa: E402
from positions import load_positions  # noqa: E402
from xmage_export import class_name  # noqa: E402

_LINE = re.compile(r"(PLAYABLE|ATTACKER|BLOCKER): (.+)")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", type=Path, required=True)
    ap.add_argument("--positions", type=Path, default=POSITIONS_PATH)
    args = ap.parse_args()

    n_checked = n_clean = 0
    findings: list[tuple[str, list[str]]] = []
    for pos in load_positions(args.positions):
        report = args.reports / f"TEST-org.mage.test.magicllm.{class_name(pos['id'])}.xml"
        if not report.exists():
            continue
        n_checked += 1
        eng, mine = engine_answers(report), claimed(pos)
        problems: list[str] = []

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

    print(f"{n_checked} positions checked against the engine, {n_clean} agree\n")
    for rid, problems in findings:
        print(f"  {rid}")
        for p in problems:
            print(f"      {p}")
    print(f"\n{len(findings)} disagree.")
    print("Blocks are not compared: getAvailableBlockers is not phase-sensitive, so a")
    print("BLOCKER line means 'could block something', not 'blocking is legal now'.")
    print("Targets are not compared either — the engine names a spell once.")


if __name__ == "__main__":
    main()
