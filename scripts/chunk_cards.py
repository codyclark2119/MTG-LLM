"""Turn a Scryfall card pool into retrieval chunks linked to the rules.

Phase 2 prep (README Section 13). Each card becomes one self-contained
chunk — cards are already small and semantically atomic, so the
overflow//bundling machinery that rules chunking needs (Section 5) doesn't
apply here.

The part that matters for this project is the **rules bridge**: Scryfall
tags each card with the keywords it uses, and the CR stores keyword
definitions under rule headers whose text is just the keyword name
("702.33. Kicker"). That lets every card carry the rule IDs governing its
own mechanics, which is exactly the "map card text to the rules mechanics
learned in Phase 1" step Section 12 describes — and it means a card
retrieved for a question can be traced back into the rules corpus.

Ability words (Landfall, Delirium) and token types (Treasure, Food)
deliberately do NOT resolve: rule 207.2c says ability words have no rules
meaning, so there is no rule to point at. ~84% of keyword *instances* on
Standard cards resolve; the remainder are these by-design misses.

Written to a SEPARATE output/index from the rules chunks so it can be
evaluated as its own arm rather than silently changing the rules-only
retrieval baseline mid-experiment.

Usage:
    python scripts/chunk_cards.py [--cards PATH] [--out PATH]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (CARD_CHUNKS_PATH, GLOSSARY_PATH, ORACLE_CARDS_PATH, RULES_PATH,
                    guard_shrink, iter_jsonl, read_jsonl, verify_cr_pin,
                    write_jsonl_atomic)
from common import RULE_ID_RE as CROSS_REF_RE

KEYWORD_RULE_RE = re.compile(r"70[12]\.\d+")

# Scryfall's oracle dump includes entries that aren't playable cards. They
# carry names and text that would otherwise compete with real cards during
# retrieval, so they're dropped rather than embedded.
EXCLUDED_LAYOUTS = {"art_series", "token", "double_faced_token", "emblem"}
EXCLUDED_SET_TYPES = {"memorabilia", "token"}


def is_playable(card: dict) -> bool:
    return (
        card.get("layout") not in EXCLUDED_LAYOUTS
        and card.get("set_type") not in EXCLUDED_SET_TYPES
    )


def build_keyword_rule_map(rules_path: Path, glossary_path: Path) -> dict[str, str]:
    """keyword name (lowercased) -> the rule ID defining it.

    Primary source is the CR's own keyword rule headers, which parse into
    records whose text is just the keyword name. The glossary is a fallback
    for keywords whose header didn't survive as a clean record.
    """
    mapping: dict[str, str] = {}
    for r in iter_jsonl(rules_path):
        if KEYWORD_RULE_RE.fullmatch(r["rule_id"]):
            text = r["text"].strip().rstrip(".")
            if len(text) < 60 and "\n" not in text:
                mapping.setdefault(text.lower(), r["rule_id"])

    for g in iter_jsonl(glossary_path):
        term = g["term"].lower()
        if term in mapping:
            continue
        refs = CROSS_REF_RE.findall(g["definition"])
        if refs:
            mapping[term] = refs[0]
    return mapping


def face_lines(face: dict) -> list[str]:
    lines = [face["name"]]
    if face.get("mana_cost"):
        lines[0] += f"  {face['mana_cost']}"
    if face.get("type_line"):
        lines.append(face["type_line"])
    if face.get("oracle_text"):
        lines.append(face["oracle_text"])
    if face.get("power") is not None and face.get("toughness") is not None:
        lines.append(f"{face['power']}/{face['toughness']}")
    if face.get("loyalty") is not None:
        lines.append(f"Loyalty: {face['loyalty']}")
    if face.get("defense") is not None:
        lines.append(f"Defense: {face['defense']}")
    return lines


def render_card(card: dict) -> str:
    if "card_faces" in card:
        # Split/transform/adventure cards: render each face, since a question
        # about one face shouldn't retrieve text with the other face missing.
        blocks = ["\n".join(face_lines(f)) for f in card["card_faces"]]
        body = "\n//\n".join(blocks)
    else:
        body = "\n".join(face_lines(card))
    footer = f"({card.get('set_name', '?')}, {card.get('rarity', '?')})"
    return f"{body}\n{footer}"


def build_card_chunk(card: dict, keyword_rules: dict[str, str]) -> dict:
    keywords = card.get("keywords", [])
    linked = {k: keyword_rules[k.lower()] for k in keywords if k.lower() in keyword_rules}

    text = render_card(card)
    if linked:
        text += "\n\nRules for this card's keywords:\n" + "\n".join(
            f"- {kw}: see rule {rule_id}" for kw, rule_id in sorted(linked.items())
        )

    return {
        "chunk_id": f"card:{card['name']}",
        "kind": "card",
        "name": card["name"],
        "oracle_id": card.get("oracle_id"),
        "type_line": card.get("type_line"),
        "mana_cost": card.get("mana_cost"),
        "cmc": card.get("cmc"),
        "colors": card.get("color_identity", []),
        "keywords": keywords,
        "keyword_rule_ids": sorted(set(linked.values())),
        "set_name": card.get("set_name"),
        "rarity": card.get("rarity"),
        "layout": card.get("layout"),
        "scryfall_uri": card.get("scryfall_uri"),
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults to the FULL Oracle pool, not a format subset. This used to
    # default to standard_cards.jsonl while writing over the Oracle-derived
    # card_chunks.jsonl, so the documented `python scripts/chunk_cards.py`
    # replaced 34,933 chunks with 4,887 Standard ones — silently reverting to
    # the 4% card coverage Section 13.5 was written to fix, and breaking card
    # resolution for card_lookup, retrieve_hybrid, and gold-set validation.
    parser.add_argument("--cards", type=Path, default=ORACLE_CARDS_PATH)
    parser.add_argument("--rules", type=Path, default=RULES_PATH)
    parser.add_argument("--glossary", type=Path, default=GLOSSARY_PATH)
    parser.add_argument("--out", type=Path, default=CARD_CHUNKS_PATH)
    parser.add_argument("--force", action="store_true",
                        help="write even if it would shrink an existing corpus by more than half")
    args = parser.parse_args()

    if not args.cards.exists():
        raise SystemExit(
            f"{args.cards} not found.\n"
            f"  Full Oracle pool (recommended):  python scripts/fetch_cards.py --bulk oracle_cards\n"
            f"  Or point at a format subset:     python scripts/chunk_cards.py --cards <file>"
        )

    # Same reason as chunk.py: read structurally, so the pin check that lives
    # in load_rule_ids never runs here. The keyword -> rule mapping below is
    # only as good as the parse it comes from.
    verify_cr_pin(args.rules, args.glossary)
    keyword_rules = build_keyword_rule_map(args.rules, args.glossary)
    print(f"{len(keyword_rules)} keyword -> rule mappings available")

    raw = read_jsonl(args.cards, missing_ok=False)
    cards = [c for c in raw if is_playable(c)]
    print(f"{len(raw)} entries, {len(cards)} playable after dropping tokens/art-series/memorabilia")

    chunks = [build_card_chunk(c, keyword_rules) for c in cards]

    # A large shrink almost always means a format subset is being written over
    # the full pool. Downstream card resolution degrades quietly when that
    # happens, so make it an explicit choice rather than a silent one. The
    # check moved to common.guard_shrink so ingest.py and chunk.py share it
    # rather than each deciding again.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    guard_shrink(args.out, len(chunks), args.force, what="chunks",
                 hint="this usually means --cards points at a format subset rather "
                      "than the full Oracle pool.")
    write_jsonl_atomic(args.out, chunks)

    linked = sum(1 for c in chunks if c["keyword_rule_ids"])
    with_kw = sum(1 for c in chunks if c["keywords"])
    lengths = sorted(len(c["text"]) for c in chunks)
    print(f"{len(chunks)} card chunks -> {args.out}")
    print(f"  {with_kw} cards have keywords; {linked} carry at least one linked rule ID")
    print(f"  chars per chunk: median {lengths[len(lengths) // 2]}, max {lengths[-1]} (~{lengths[-1] // 4} tokens)")


if __name__ == "__main__":
    main()
