"""Verify the dated MTGGoldfish Standard metagame index."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data/decks/standard/mtggoldfish_meta_2026-08-24.json"


def main() -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    assert index["source"] == "mtggoldfish"
    assert index["format"] == "standard"
    assert index["format_snapshot"] == "data/manifests/formats/standard-2026-08-10.json"
    archetypes = index["archetypes"]
    assert len(archetypes) == 10
    assert archetypes[0]["name"] == "Mono-Green Landfall"
    assert archetypes[0]["meta_percent"] == 11.8
    assert all(a["archetype_url"].startswith("https://www.mtggoldfish.com/")
               for a in archetypes)
    assert all(0 < a["meta_percent"] <= 100 and a["sample_size"] > 0
               for a in archetypes)
    print(f"verified Standard metagame index: {len(archetypes)} archetypes")


if __name__ == "__main__":
    main()
