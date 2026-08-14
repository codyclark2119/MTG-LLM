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
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import CARD_CHUNKS_PATH, RULES_PATH, RULING_CHUNKS_PATH, load_rule_ids
from common import RULE_ID_RE as CROSS_REF_RE

# Scryfall bulk downloading lives in fetch_cards.py — this was a second,
# near-identical copy of it.
from fetch_cards import download_bulk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rulings", type=Path, default=Path("data/cards/raw/rulings.jsonl"))
    parser.add_argument("--card-chunks", type=Path, default=CARD_CHUNKS_PATH)
    parser.add_argument("--rules", type=Path, default=RULES_PATH)
    parser.add_argument("--out", type=Path, default=RULING_CHUNKS_PATH)
    parser.add_argument("--refresh", action="store_true", help="re-download even if the file exists")
    # `store_true` with `default=True` can never be false, so --wotc-only was a
    # no-op that read like a toggle. The filter is the intended behaviour, so
    # the opt-out is what gets a flag.
    parser.add_argument("--include-non-wotc", action="store_true",
                        help="also keep rulings not issued by Wizards of the Coast")
    args = parser.parse_args()
    wotc_only = not args.include_non_wotc

    if args.refresh or not args.rulings.exists():
        download_bulk("rulings", args.rulings)

    valid_rule_ids = load_rule_ids(args.rules)

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
            if wotc_only and r.get("source") != "wotc":
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
