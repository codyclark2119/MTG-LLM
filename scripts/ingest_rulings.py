"""Ingest official WotC card rulings and join them to cards and rules.

The single most authoritative corpus available to this project. Scryfall
publishes ~78,000 rulings, almost all issued by Wizards of the Coast,
covering ~19,800 cards. They are short (median ~180 chars), and they
address exactly the edge-case interactions players actually ask about —
which is what the rest of our data is weakest on:

  - the synthetic set is machine-generated from rules text
  - the reddit set is community prose, with only 21% citing any rule

Rulings are neither. They are first-party answers to interaction
questions, so they can serve as retrieval context, as SFT targets, and —
most valuably — as trustworthy eval references.

Each ruling is joined to its card via oracle_id, and any comprehensive
rule numbers it mentions are extracted and validated against the pinned
CR, so a ruling can be traced into the rules corpus the same way card
chunks are (see chunk_cards.py).

Usage:
    python scripts/ingest_rulings.py [--refresh]
"""

import argparse
import gzip
import json
import re
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

CROSS_REF_RE = re.compile(r"\b\d{3}\.\d+[a-z]?\b")
BULK_URL = "https://api.scryfall.com/bulk-data"
HEADERS = {"User-Agent": "MagicLLM-Phase2/0.1 (hobby research project)", "Accept": "application/json"}


def download_rulings(dest: Path) -> None:
    meta = json.loads(urllib.request.urlopen(urllib.request.Request(BULK_URL, headers=HEADERS)).read())
    entry = next(d for d in meta["data"] if d["type"] == "rulings")
    print(f"downloading rulings bulk data (updated {entry['updated_at']}) ...")
    req = urllib.request.Request(entry["jsonl_download_uri"], headers=HEADERS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    gz = dest.with_suffix(".jsonl.gz")
    with urllib.request.urlopen(req) as r, gz.open("wb") as f:
        shutil.copyfileobj(r, f)
    with gzip.open(gz, "rt", encoding="utf-8") as fin, dest.open("w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line)
    gz.unlink()
    print(f"-> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rulings", type=Path, default=Path("data/cards/raw/rulings.jsonl"))
    parser.add_argument("--card-chunks", type=Path, default=Path("data/cards/processed/card_chunks.jsonl"))
    parser.add_argument("--rules", type=Path, default=Path("data/processed/rules.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/cards/processed/ruling_chunks.jsonl"))
    parser.add_argument("--refresh", action="store_true", help="re-download even if the file exists")
    parser.add_argument("--wotc-only", action="store_true", default=True, help="keep only WotC-issued rulings")
    args = parser.parse_args()

    if args.refresh or not args.rulings.exists():
        download_rulings(args.rulings)

    valid_rule_ids = {json.loads(l)["rule_id"] for l in args.rules.open(encoding="utf-8")}

    cards_by_oracle: dict[str, dict] = {}
    with args.card_chunks.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("oracle_id"):
                cards_by_oracle[c["oracle_id"]] = c

    grouped: dict[str, list[dict]] = defaultdict(list)
    total = kept = 0
    with args.rulings.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            total += 1
            if args.wotc_only and r.get("source") != "wotc":
                continue
            kept += 1
            grouped[r["oracle_id"]].append(r)

    chunks = []
    orphans = 0
    for oracle_id, rulings in grouped.items():
        card = cards_by_oracle.get(oracle_id)
        if card is None:
            # Rulings exist for cards filtered out of the playable pool
            # (tokens, art series) or not present in the oracle dump.
            orphans += 1
            continue

        rulings.sort(key=lambda r: r.get("published_at") or "")
        body = "\n".join(f"- {r['comment']}" for r in rulings)
        text = f"Official rulings for {card['name']}:\n{body}"

        cited = sorted({m for r in rulings for m in CROSS_REF_RE.findall(r["comment"])} & valid_rule_ids)

        chunks.append(
            {
                "chunk_id": f"rulings:{card['name']}",
                "kind": "rulings",
                "name": card["name"],
                "oracle_id": oracle_id,
                "type_line": card.get("type_line"),
                "n_rulings": len(rulings),
                "latest_published": rulings[-1].get("published_at"),
                "cited_rule_ids": cited,
                "keyword_rule_ids": card.get("keyword_rule_ids", []),
                "text": text,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    lengths = sorted(len(c["text"]) for c in chunks)
    with_rules = sum(1 for c in chunks if c["cited_rule_ids"])
    print(f"{total} rulings read, {kept} from WotC")
    print(f"{len(chunks)} cards with rulings -> {args.out}")
    print(f"  {orphans} oracle_ids had no matching playable card (skipped)")
    print(f"  {with_rules} carry a rule number that resolves against the pinned CR")
    print(f"  chars per chunk: median {lengths[len(lengths) // 2]}, p90 {lengths[int(len(lengths) * 0.9)]}, max {lengths[-1]}")


if __name__ == "__main__":
    main()
