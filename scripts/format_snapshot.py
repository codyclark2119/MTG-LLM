"""Create and validate a versioned format card-pool snapshot.

The format pool is separate from Oracle card data because legality changes over
 time. A decklist must name the snapshot it was checked against, not merely say
"Standard".

Usage:
    python scripts/format_snapshot.py create \
        --format standard \
        --cards data/cards/raw/standard_cards.jsonl \
        --effective-date 2026-08-10 \
        --out data/manifests/formats/standard-2026-08-10.json
    python scripts/format_snapshot.py verify data/manifests/formats/standard-2026-08-10.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import file_sha256, REPO_ROOT  # noqa: E402
from manifest import record_count, relative_path  # noqa: E402


def card_names(path: Path) -> list[str]:
    """Return sorted unique card names from a Scryfall JSONL snapshot."""
    names = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                card = json.loads(line)
                if card.get("name"):
                    names.add(card["name"])
    return sorted(names)


def build_snapshot(cards_path: Path, format_name: str, effective_date: str,
                   rotation_date: str | None = None,
                   banned_cards: list[str] | None = None) -> dict:
    """Build a format snapshot from an immutable card JSONL file."""
    if not cards_path.exists():
        raise FileNotFoundError(cards_path)
    names = card_names(cards_path)
    return {
        "schema_version": 1,
        "snapshot_id": f"{format_name}-{effective_date}",
        "format": format_name,
        "effective_date": effective_date,
        "rotation_date": rotation_date,
        "banned_cards": sorted(banned_cards or []),
        "cards_path": relative_path(cards_path),
        "card_count": len(names),
        "source_record_count": record_count(cards_path),
        "source_sha256": file_sha256(cards_path),
        "card_names": names,
    }


def verify_snapshot(path: Path) -> dict:
    """Verify the source file and card identities recorded by a snapshot."""
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    cards_path = REPO_ROOT / snapshot["cards_path"]
    actual = build_snapshot(
        cards_path, snapshot["format"], snapshot["effective_date"],
        snapshot.get("rotation_date"), snapshot.get("banned_cards"))
    for field in ("source_record_count", "source_sha256", "card_count", "card_names"):
        if actual[field] != snapshot.get(field):
            raise ValueError(f"format snapshot is stale: {path} differs in {field}")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--format", required=True)
    create.add_argument("--cards", type=Path, required=True)
    create.add_argument("--effective-date", required=True)
    create.add_argument("--rotation-date")
    create.add_argument("--banned-card", action="append", default=[])
    create.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("snapshot", type=Path)
    args = parser.parse_args()

    if args.command == "verify":
        snapshot = verify_snapshot(args.snapshot)
        print(f"verified {snapshot['snapshot_id']}: {snapshot['card_count']} cards")
        return

    snapshot = build_snapshot(args.cards, args.format, args.effective_date,
                              args.rotation_date, args.banned_card)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    temp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(args.out)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
