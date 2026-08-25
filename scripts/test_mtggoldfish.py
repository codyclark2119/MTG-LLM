"""Test MTGGoldfish Arena export parsing and provenance preservation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mtggoldfish import import_deck, parse_arena_export  # noqa: E402
from format_snapshot import verify_snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data/manifests/formats/standard-2026-08-10.json"
EXPORT = """Name Mono-Green Landfall

Deck
4 Earthbender Ascension
3 Esper Origins
14 Forest
2 Icetill Explorer
2 Icetill Explorer
4 Llanowar Elves

Sideboard
3 Meltstrider's Resolve
2 Torpor Orb
"""


def main() -> None:
    name, main, side = parse_arena_export(EXPORT)
    assert name == "Mono-Green Landfall"
    assert {x["name"]: x["count"] for x in main}["Icetill Explorer"] == 4
    assert sum(x["count"] for x in side) == 5

    deck = import_deck(
        EXPORT, SNAPSHOT,
        "https://www.mtggoldfish.com/metagame/standard#paper",
        "https://www.mtggoldfish.com/deck/arena_download/7922863",
        "https://www.mtggoldfish.com/archetype/standard-mono-green-landfall-woe#paper",
    )
    assert deck["deck_id"] == "mono-green-landfall"
    assert deck["format_snapshot"] == "data/manifests/formats/standard-2026-08-10.json"
    assert deck["source"] == "mtggoldfish"
    print("MTGGoldfish parser and provenance test passed")


if __name__ == "__main__":
    main()
