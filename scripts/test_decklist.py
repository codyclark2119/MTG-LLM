"""Test deterministic Standard decklist legality checks.

    python scripts/test_decklist.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decklist import validate_decklist  # noqa: E402
from format_snapshot import verify_snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT / "data/manifests/formats/standard-2026-08-10.json"


def main() -> None:
    snapshot = verify_snapshot(SNAPSHOT_PATH)
    mainboard = [{"name": "Island", "count": 28},
                 {"name": "Mountain", "count": 20},
                 {"name": "Chart a Course", "count": 4},
                 {"name": "Shock", "count": 4},
                 {"name": "A Killer Among Us", "count": 4}]
    valid = {
        "deck_id": "sample-standard",
        "format_snapshot": "data/manifests/formats/standard-2026-08-10.json",
        "mainboard": mainboard,
        "sideboard": [{"name": "A Realm Reborn", "count": 3}],
    }
    assert validate_decklist(valid, snapshot, "data/manifests/formats/standard-2026-08-10.json") == []

    invalid = dict(valid)
    invalid["mainboard"] = mainboard + [{"name": "Island", "count": 1}]
    invalid["sideboard"] = [{"name": "Definitely Not A Card", "count": 16}]
    problems = validate_decklist(invalid, snapshot, "data/manifests/formats/standard-2026-08-10.json")
    assert any("exactly 60" in p for p in problems)
    assert any("not in snapshot" in p for p in problems)
    assert any("at most 15" in p for p in problems)
    print("decklist validation passed")


if __name__ == "__main__":
    main()
