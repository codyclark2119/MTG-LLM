"""Import MTGGoldfish Arena exports into the canonical decklist shape.

The importer accepts the text copied from an MTGGoldfish Arena export. It keeps
source provenance and aggregates repeated card lines before validation.

Usage:
    python scripts/mtggoldfish.py import deck.txt \
        --archetype-url https://www.mtggoldfish.com/archetype/... \
        --export-url https://www.mtggoldfish.com/deck/arena_download/7922863 \
        --snapshot data/manifests/formats/standard-2026-08-10.json \
        --out data/decks/standard/mono-green-landfall-7922863.json
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decklist import validate_decklist  # noqa: E402
from format_snapshot import relative_path, verify_snapshot  # noqa: E402

CARD_LINE_RE = re.compile(r"^(?P<count>\d+)\s+(?P<name>.+?)\s*$")


def parse_arena_export(text: str) -> tuple[str, list[dict], list[dict]]:
    """Parse deck name, mainboard, and sideboard from Arena export text."""
    section = None
    deck_name = ""
    main: list[dict] = []
    side: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Name ") and not deck_name:
            deck_name = line[5:].strip()
            continue
        if line == "Deck":
            section = main
            continue
        if line == "Sideboard":
            section = side
            continue
        if section is None:
            continue
        match = CARD_LINE_RE.match(line)
        if not match:
            continue
        section.append({"name": match.group("name"), "count": int(match.group("count"))})
    if not main:
        raise ValueError("export contains no Deck entries")
    return deck_name, _aggregate(main), _aggregate(side)


def _aggregate(entries: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    order: list[str] = []
    for entry in entries:
        name = entry["name"].strip()
        if name not in counts:
            order.append(name)
            counts[name] = 0
        counts[name] += entry["count"]
    return [{"name": name, "count": counts[name]} for name in order]


def import_deck(text: str, snapshot_path: Path, source_url: str,
                export_url: str, archetype_url: str | None = None,
                archetype: str | None = None,
                meta_percent: float | None = None,
                sample_size: int | None = None,
                observed_at: str | None = None) -> dict:
    """Build a canonical deck record and validate it against a snapshot."""
    snapshot = verify_snapshot(snapshot_path)
    name, main, side = parse_arena_export(text)
    deck_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "mtggoldfish-deck"
    deck = {
        "deck_id": deck_id,
        "name": name or deck_id,
        "format": snapshot["format"],
        "format_snapshot": relative_path(snapshot_path),
        "source": "mtggoldfish",
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_url": source_url,
        "export_url": export_url,
        "archetype_url": archetype_url,
        "mainboard": main,
        "sideboard": side,
    }
    if archetype:
        deck["archetype"] = archetype
    if meta_percent is not None:
        deck["meta_percent"] = meta_percent
    if sample_size is not None:
        deck["metagame_sample_size"] = sample_size
    if observed_at:
        deck["metagame_observed_at"] = observed_at
    return deck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("source", type=Path)
    imp.add_argument("--snapshot", type=Path, required=True)
    imp.add_argument("--source-url", default="https://www.mtggoldfish.com/metagame/standard#paper")
    imp.add_argument("--export-url", required=True)
    imp.add_argument("--archetype-url")
    imp.add_argument("--archetype")
    imp.add_argument("--meta-percent", type=float)
    imp.add_argument("--sample-size", type=int)
    imp.add_argument("--observed-at")
    imp.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    snapshot_path = args.snapshot if args.snapshot.is_absolute() else Path.cwd() / args.snapshot
    deck = import_deck(args.source.read_text(encoding="utf-8"), snapshot_path,
                       args.source_url, args.export_url, args.archetype_url,
                       args.archetype, args.meta_percent, args.sample_size,
                       args.observed_at)
    snapshot = verify_snapshot(snapshot_path)
    problems = validate_decklist(deck, snapshot, relative_path(snapshot_path))
    if problems:
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(f"{len(problems)} decklist problem(s)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    temp.write_text(json.dumps(deck, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    os.replace(temp, args.out)
    print(f"imported {deck['name']}: {sum(x['count'] for x in deck['mainboard'])} main / "
          f"{sum(x['count'] for x in deck['sideboard'])} side -> {args.out}")


if __name__ == "__main__":
    main()
