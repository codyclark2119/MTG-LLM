"""Validate decklists against a versioned format snapshot.

A decklist is not training truth until it is structurally legal. This validator
checks exact card identity against the snapshot, mainboard/sideboard sizes,
copy limits, and banned cards without asking a model to enforce constraints.

Usage:
    python scripts/decklist.py validate data/decks/example.json

Expected JSON:
    {
      "deck_id": "example",
      "format_snapshot": "data/manifests/formats/standard-2026-08-10.json",
      "mainboard": [{"name": "Island", "count": 24}],
      "sideboard": [{"name": "Negate", "count": 3}]
    }
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from format_snapshot import relative_path, verify_snapshot  # noqa: E402

BASIC_LAND_TYPES = {"Plains", "Island", "Swamp", "Mountain", "Forest"}


def entries_by_name(entries: list[dict], label: str) -> tuple[Counter, list[str]]:
    counts: Counter = Counter()
    problems = []
    for entry in entries:
        if not isinstance(entry, dict):
            problems.append(f"{label} entry must be an object: {entry!r}")
            continue
        name = entry.get("name")
        count = entry.get("count")
        if not isinstance(name, str) or not name.strip():
            problems.append(f"{label} entry has no card name")
            continue
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            problems.append(f"{label} {name!r} count must be a positive integer")
            continue
        counts[name.strip()] += count
    return counts, problems


def validate_decklist(deck: dict, snapshot: dict,
                      snapshot_ref: str | None = None) -> list[str]:
    """Return every machine-checkable legality problem in a decklist."""
    problems = []
    for field in ("deck_id", "format_snapshot", "mainboard"):
        if not deck.get(field):
            problems.append(f"missing required field: {field}")
    expected_path = snapshot_ref or snapshot.get("snapshot_id")
    if deck.get("format_snapshot") != expected_path:
        problems.append(f"format_snapshot must identify {expected_path!r}")

    main, main_problems = entries_by_name(deck.get("mainboard") or [], "mainboard")
    side, side_problems = entries_by_name(deck.get("sideboard") or [], "sideboard")
    problems.extend(main_problems + side_problems)
    legal = set(snapshot.get("card_names") or [])
    banned = set(snapshot.get("banned_cards") or [])
    for label, counts in (("mainboard", main), ("sideboard", side)):
        for name in counts:
            if name not in legal:
                problems.append(f"{label} card is not in snapshot: {name!r}")
            if name in banned:
                problems.append(f"{label} card is banned in snapshot: {name!r}")
    if sum(main.values()) != 60:
        problems.append(f"mainboard must contain exactly 60 cards, has {sum(main.values())}")
    if sum(side.values()) > 15:
        problems.append(f"sideboard must contain at most 15 cards, has {sum(side.values())}")
    for name, count in (main + side).items():
        if name not in BASIC_LAND_TYPES and count > 4:
            problems.append(f"card exceeds four-copy limit: {name!r} has {count}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("deck", type=Path)
    args = parser.parse_args()
    deck = json.loads(args.deck.read_text(encoding="utf-8"))
    snapshot_path = Path(deck.get("format_snapshot", ""))
    if not snapshot_path.is_absolute():
        snapshot_path = Path.cwd() / snapshot_path
    snapshot = verify_snapshot(snapshot_path)
    problems = validate_decklist(deck, snapshot, relative_path(snapshot_path))
    if problems:
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(f"{len(problems)} decklist problem(s)")
    print(f"valid decklist: {deck['deck_id']} against {snapshot['snapshot_id']}")


if __name__ == "__main__":
    main()
