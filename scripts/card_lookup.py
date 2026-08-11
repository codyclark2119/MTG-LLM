"""Resolve card names in a question to card chunks, without embeddings.

Semantic retrieval is the wrong tool for a named card. Players write the
name explicitly — usually in [[double brackets]] — so this is a dictionary
lookup, not a similarity problem. Measured against the reddit eval set,
matching bracketed names to the Oracle pool gives 63% exact, 90% once
names are normalized; the rest are short names ("Chatterfang" for
"Chatterfang, Squirrel General"), apostrophe variants ("Bolas' Citadel"
vs "Bolas's Citadel"), and outright typos ("Vorinclex, Monstrous Rider").
Hence the layered resolution below.

Doing this by lookup instead of embedding also keeps the 35k card chunks
out of the rules retrieval pool, where they would swamp 448 rules chunks
by sheer volume and turn a question like "when are SBAs checked" into a
card search.

Usage:
    python scripts/card_lookup.py "Does [[Chatterfang]] work with [[Parallel Lives]]?"
"""

import argparse
import difflib
import json
import re
from functools import lru_cache
from pathlib import Path

BRACKET_RE = re.compile(r"\[\[(.*?)\]\]")
CARD_CHUNKS = Path("data/cards/processed/card_chunks.jsonl")


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class CardIndex:
    def __init__(self, chunks_path: Path = CARD_CHUNKS):
        self.by_name: dict[str, dict] = {}
        self.by_norm: dict[str, dict] = {}
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                name = c["name"]
                self.by_name[name] = c
                self.by_norm.setdefault(normalize(name), c)
                # Index each face of a split/transform card under its own
                # name too — players cite "Fire" not "Fire // Ice".
                for face in name.split(" // "):
                    self.by_norm.setdefault(normalize(face), c)
        self._norm_keys = list(self.by_norm)

    @lru_cache(maxsize=4096)
    def resolve(self, name: str) -> tuple[dict | None, str]:
        """Return (card chunk, how it matched). None if unresolvable."""
        if name in self.by_name:
            return self.by_name[name], "exact"

        n = normalize(name)
        if n in self.by_norm:
            return self.by_norm[n], "normalized"

        # Short name: "Chatterfang" -> "Chatterfang, Squirrel General".
        # Only accept an unambiguous prefix hit, so "Ajani" (dozens of
        # cards) resolves to nothing rather than to an arbitrary Ajani.
        prefixes = [k for k in self._norm_keys if k.startswith(n)]
        if len(prefixes) == 1:
            return self.by_norm[prefixes[0]], "prefix"
        if len(prefixes) > 1:
            return None, "ambiguous-prefix"

        # Typos / apostrophe variants. Cutoff is deliberately high: a wrong
        # card is worse than no card, since it would feed confidently
        # incorrect text into an answer.
        close = difflib.get_close_matches(n, self._norm_keys, n=1, cutoff=0.9)
        if close:
            return self.by_norm[close[0]], "fuzzy"

        return None, "unresolved"

    def find_in_text(self, text: str, max_cards: int = 5) -> list[dict]:
        """Cards named in a question, bracketed refs first."""
        found: list[dict] = []
        seen: set[str] = set()
        for raw in BRACKET_RE.findall(text):
            card, how = self.resolve(raw.strip())
            if card and card["name"] not in seen:
                seen.add(card["name"])
                found.append({**card, "matched_as": raw.strip(), "match_type": how})
            if len(found) >= max_cards:
                break
        return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", help="question text possibly containing [[card names]]")
    parser.add_argument("--chunks", type=Path, default=CARD_CHUNKS)
    args = parser.parse_args()

    index = CardIndex(args.chunks)
    hits = index.find_in_text(args.text)
    if not hits:
        print("no card names resolved")
        return
    for h in hits:
        print(f"=== {h['name']}  (matched '{h['matched_as']}' via {h['match_type']}) ===")
        print(h["text"])
        print()


if __name__ == "__main__":
    main()
