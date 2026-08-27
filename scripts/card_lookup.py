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
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import CARD_CHUNKS_PATH as CARD_CHUNKS
from common import iter_jsonl, verify_card_pin

BRACKET_RE = re.compile(r"\[\[(.*?)\]\]")

# The `how` values that mean *this string legitimately names this card*, as
# opposed to *I guessed which card you meant*. Nine call sites gate on
# exactness; this is one tuple so a tenth cannot be written that disagrees, and
# so a fix lands on all of them at once — a hardening reaching a subset of its
# callers is this repo's most repeated bug (21.53, 21.54, 21.74).
#
# `face` earns its place here and `normalized` does not, and the difference is
# not strictness. A `normalized` hit means the author wrote a real name badly:
# the exact string exists, so asking for it costs nothing. A `face` hit means
# they wrote a real, complete, PRINTED card name that happens to be one side of
# a double-faced card. "Correcting" `Gollum, Silent Slinker` to
# `Gollum, Silent Slinker // Meager Meal` would put a string on the board that
# no game client, no deck list and no player ever writes (Section 21.104).
CERTAIN_MATCHES = ("exact", "face")


def names_a_card(how: str) -> bool:
    """Did the string name this card, rather than merely point at it?"""
    return how in CERTAIN_MATCHES


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class CardIndex:
    def __init__(self, chunks_path: Path = CARD_CHUNKS):
        # The chokepoint for the card corpus, the way `load_rule_ids` is for
        # rules: eleven call sites reach card text through this constructor, so
        # one check covers them. Only the canonical path is verified, so
        # `--chunks somewhere_else` stays a deliberate act.
        verify_card_pin(chunks_path)
        self.by_name: dict[str, dict] = {}
        self.by_norm: dict[str, dict] = {}
        # Faces of split/transform cards, kept in their OWN index rather than
        # folded into `by_norm`. Both used to answer `normalized`, so a caller
        # could not tell "you wrote the name sloppily" from "you wrote one side
        # of a two-sided card" — one name for two meanings, and the gates that
        # reject `normalized` were silently rejecting every DFC (21.104).
        self.by_face: dict[str, dict] = {}
        face_owners: dict[str, set[str]] = {}
        for c in iter_jsonl(chunks_path):
            name = c["name"]
            self.by_name[name] = c
            self.by_norm.setdefault(normalize(name), c)
            if " // " in name:
                # Players cite "Fire", not "Fire // Ice".
                for face in name.split(" // "):
                    key = normalize(face)
                    face_owners.setdefault(key, set()).add(name)
                    self.by_face.setdefault(key, c)
        # Two faces claimed by different cards resolve to NEITHER: "Fire" is a
        # face of both Fire // Ice and Start // Fire, and picking one silently
        # would put the wrong card's text in front of the model. Measured: 2 of
        # 1,771 face names. The same discipline as the ambiguous-prefix rule
        # below, and as `find_players` missing rather than guessing.
        self._ambiguous_faces = {k for k, v in face_owners.items() if len(v) > 1}
        for key in self._ambiguous_faces:
            self.by_face.pop(key, None)
        self._norm_keys = list(self.by_norm)

    @lru_cache(maxsize=4096)
    def resolve(self, name: str) -> tuple[dict | None, str]:
        """Return (card chunk, how it matched). None if unresolvable."""
        if name in self.by_name:
            return self.by_name[name], "exact"

        n = normalize(name)
        if n in self.by_norm:
            return self.by_norm[n], "normalized"

        # A printed face. Checked AFTER whole names, so the 25 faces that are
        # also real single-faced cards — "Brainstorm", "Ancestral Recall",
        # "Bind" — resolve to the card actually named rather than to the
        # two-sided one that happens to share a side's name.
        if n in self.by_face:
            return self.by_face[n], "face"
        if n in self._ambiguous_faces:
            return None, "ambiguous-face"

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
