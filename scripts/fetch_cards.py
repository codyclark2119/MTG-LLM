"""Pull card data from the Scryfall API, scoped to one format at a time.

Phase 2 kickoff (README Section 13): rather than pulling Magic's entire
~30,000-card database up front, start with the smallest actively-legal
format pool — Standard — to validate the ingestion pipeline cheaply
before expanding to larger formats.

Uses the /cards/search endpoint (not a full bulk-data download) since a
single format's legal pool is small enough that paginated search is the
lighter-weight option: ~2,000-3,000 Standard-legal cards is only a
dozen-odd requests at 175 cards/page.

Scryfall policy compliance: requires a descriptive User-Agent and Accept
header on every request (undocumented/missing headers get throttled),
and asks consumers to stay under 10 requests/second — this client sleeps
between pages well under that limit even though a single format's card
count rarely gets close to it.

Card data is Wizards of the Coast IP surfaced through Scryfall's free
API; Scryfall's terms ask for attribution ("Data courtesy of Scryfall")
and restrict bulk commercial redistribution — same licensing-awareness
note as the Comprehensive Rules pull (Section 3.1).

Usage:
    python scripts/fetch_cards.py --format standard
"""

import argparse
import gzip
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

API_ROOT = "https://api.scryfall.com"
BULK_URL = f"{API_ROOT}/bulk-data"
USER_AGENT = "MagicLLM-Phase2/0.1 (hobby research project)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
REQUEST_DELAY_SECONDS = 0.15  # comfortably under Scryfall's 10 req/s guidance
RATE_LIMIT_BACKOFF_SECONDS = 30  # Scryfall's documented cooldown after a 429


def download_bulk(bulk_type: str, dest: Path) -> str:
    """Download one of Scryfall's bulk files, e.g. `oracle_cards`, `rulings`.

    The search endpoint above is right for a single format's legal pool, but
    the full Oracle pool is a ~200MB download that Scryfall explicitly asks
    you to take from bulk data rather than by paginating search. This used to
    be a manual "download this file yourself" step in the README, which meant
    the *recommended* card path was the only unscripted one.

    Returns the upstream `updated_at` so callers can record provenance.
    """
    meta = json.loads(urlopen(Request(BULK_URL, headers=HEADERS)).read())
    entry = next((d for d in meta["data"] if d["type"] == bulk_type), None)
    if entry is None:
        available = ", ".join(sorted(d["type"] for d in meta["data"]))
        raise SystemExit(f"unknown bulk type {bulk_type!r}; Scryfall offers: {available}")

    print(f"downloading {bulk_type} bulk data (updated {entry['updated_at']}) ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    gz = dest.with_suffix(".jsonl.gz")
    with urlopen(Request(entry["jsonl_download_uri"], headers=HEADERS)) as r, gz.open("wb") as f:
        shutil.copyfileobj(r, f)
    with gzip.open(gz, "rt", encoding="utf-8") as fin, dest.open("w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line)
    gz.unlink()
    return entry["updated_at"]


def fetch_page(url: str, retries: int = 3) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request) as response:
                return json.loads(response.read())
        except HTTPError as e:
            if e.code == 429 and attempt < retries:
                print(f"  rate limited, backing off {RATE_LIMIT_BACKOFF_SECONDS}s...")
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            raise


def fetch_format(format_name: str) -> list[dict]:
    cards: list[dict] = []
    url = f"{API_ROOT}/cards/search?q=legal%3A{format_name}&unique=cards&order=name"
    page_num = 1
    while url:
        page = fetch_page(url)
        cards.extend(page["data"])
        print(f"  page {page_num}: {len(page['data'])} cards (running total {len(cards)})")
        url = page.get("next_page") if page.get("has_more") else None
        page_num += 1
        if url:
            time.sleep(REQUEST_DELAY_SECONDS)
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", default="standard", help="Scryfall format name, e.g. standard, pioneer, modern")
    parser.add_argument("--bulk", default=None, metavar="TYPE",
                        help="download a Scryfall bulk file instead of searching a format "
                             "(oracle_cards is the full pool chunk_cards.py expects)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/cards/raw"))
    args = parser.parse_args()

    if args.bulk:
        out_path = args.out_dir / f"{args.bulk}.jsonl"
        updated_at = download_bulk(args.bulk, out_path)
        count = sum(1 for _ in out_path.open(encoding="utf-8"))
        (args.out_dir / f"{args.bulk}_MANIFEST.md").write_text(
            f"# Scryfall `{args.bulk}` bulk snapshot\n\n"
            f"- Source: {BULK_URL} (type `{args.bulk}`)\n"
            f"- Upstream updated: {updated_at}\n"
            f"- Downloaded: {datetime.now(timezone.utc).isoformat()}\n"
            f"- Entries: {count}\n"
            f"- Card data courtesy of Scryfall (https://scryfall.com); card text and "
            f"templating are Wizards of the Coast IP.\n",
            encoding="utf-8",
        )
        print(f"\n{count} entries -> {out_path}")
        print("Next: python scripts/chunk_cards.py")
        return

    print(f"fetching cards legal in {args.format}...")
    cards = fetch_format(args.format)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.format}_cards.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")

    fetched_at = datetime.now(timezone.utc).isoformat()
    manifest_path = args.out_dir / f"{args.format}_MANIFEST.md"
    manifest_path.write_text(
        f"# {args.format.title()} card pool snapshot\n\n"
        f"- Source: Scryfall API (`{API_ROOT}/cards/search?q=legal:{args.format}`)\n"
        f"- Fetched: {fetched_at}\n"
        f"- Card count: {len(cards)}\n"
        f"- Format legality is a moving target (rotation, bans) — this is a point-in-time "
        f"snapshot, not a live view. Re-fetch and re-date this file before relying on it "
        f"for anything legality-sensitive.\n"
        f"- Card data courtesy of Scryfall (https://scryfall.com); underlying card text "
        f"and templating are Wizards of the Coast IP. See Scryfall's API terms before any "
        f"redistribution: https://scryfall.com/docs/api\n",
        encoding="utf-8",
    )

    print(f"\n{len(cards)} {args.format}-legal cards -> {out_path}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
