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


# Whether the SECOND printed name on a multi-face card is itself a legal play.
# Measured over the pinned corpus: 885 cards carry a `//` name across six
# layouts, and they do not behave alike (Section 21.111).
#
#   split      137  either half is cast from hand
#   adventure  170  the creature or its Adventure
#   prepare     55  the spell half is cast as a copy while the creature is
#                   prepared — Secrets of Strixhaven, two sets old
#   modal_dfc  100  you choose a face when playing it; the back is often a LAND,
#                   so it is played rather than cast, but it is still a play
#   transform  401  the back is reached by transforming and is never played
#   flip        22  Role tokens here; the back is never played
#
# `mana_cost` looks like it should decide this and does not: it is `'{2}{B} // {B}'`
# for adventure and **None** for both modal_dfc and transform, because per-face
# costs live in `card_faces[]` — which `chunk_cards.py` flattens away. So the
# answer has to come from the layout.
#
# OPEN ON PURPOSE. `prepare` did not exist when this project started and is
# already 55 cards; a seventh layout will arrive. An unknown `//` layout returns
# None — "cannot decide" — rather than defaulting, because both defaults are
# wrong somewhere: treating a transform back as playable invents a legal play,
# and treating a modal_dfc back as unplayable removes a real one.
BACK_FACE_IS_A_PLAY = {
    "split": True, "adventure": True, "prepare": True, "modal_dfc": True,
    "transform": False, "flip": False,
}


def back_face_is_a_play(card: dict) -> bool | None:
    """Can the second printed name be played, or only reached in play?

    Layout first, then a TYPE-SHAPE fallback for a layout this table has never
    seen — `prepare` is two sets old and already 55 cards, so the table WILL go
    stale.

    The fallback rule: a creature front with an instant or sorcery back means
    the back is playable. Measured over 886 multi-face game cards, that shape is
    `adventure` (147), `prepare` (55) and `modal_dfc` (8) — 210 cards, no
    `transform`, no `flip`.

    Note what that does and does not say. The shape does **not** identify the
    layout; three layouts share it. It settles the VERDICT, because all three
    agree the back is playable. Shape settles the verdict for **56%** of
    multi-face cards and cannot for the rest — the largest conflicted shape is
    `Creature -> Creature` at 223 cards (`transform` 191, `modal_dfc` 17,
    `flip` 15), where the same two type lines mean "you transform into it" and
    "you may play it" depending only on the layout.

    `Creature -> Land` is conflicted too (`modal_dfc` 13, `transform` 6) and is
    deliberately NOT decided here: a creature that transforms into a land and a
    creature you may instead play as a land are the same shape, and guessing
    would invent six legal plays.

    None means neither could decide, and the caller must not guess. Same
    discipline as `protocol_findings` returning None for an undecidable entry:
    a check that cannot run must not be reported as an answer.
    """
    name = card.get("name") or ""
    if " // " not in name:
        return None
    known = BACK_FACE_IS_A_PLAY.get(card.get("layout"))
    if known is not None:
        return known
    # `chunk_cards` flattens `card_faces` away but joins `type_line` with the
    # same ` // `, so the per-face types survive even though the per-face costs
    # do not.
    parts = (card.get("type_line") or "").split(" // ")
    if len(parts) != 2:
        return None
    front, back = parts
    if "Creature" in front and ("Instant" in back or "Sorcery" in back):
        return True
    return None


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

    def find_in_text(self, text: str, max_cards: int = 5,
                     scan_prose: bool = False) -> list[dict]:
        """Cards named in a question, bracketed refs first.

        `scan_prose` adds the fallback this docstring has implied since it was
        written: match card names written as ordinary prose, with no brackets.
        It is OPT-IN because bracket-only is what every stored eval number was
        produced under, and because prose matching is only safe under the two
        rules below — a caller that wants it should say so.

        Section 21.136 fixed the eval path by resolving from a record's own
        verified `cards` field. A live user has no such field: they type
        "Does Lightning Bolt kill a Grizzly Bears?" and, bracket-only, resolve
        NOTHING — which drops the question onto the rules-only arm, measured at
        2.91-3.41 and BELOW no-retrieval at both model sizes (21.141). So the
        serving path needs to read prose or it cannot reach the best arm at all.
        """
        found: list[dict] = []
        seen: set[str] = set()
        for raw in BRACKET_RE.findall(text):
            card, how = self.resolve(raw.strip())
            if card and card["name"] not in seen:
                seen.add(card["name"])
                found.append({**card, "matched_as": raw.strip(), "match_type": how})
            if len(found) >= max_cards:
                break
        if scan_prose and len(found) < max_cards:
            found.extend(self._scan_prose(text, max_cards - len(found), seen))
        return found

    # Phrases that are card names AND ordinary rules language. Measured, not
    # guessed: across all 1,202 hand-labelled RulesGuru questions these three
    # are 19 of the 35 total false positives ("The End" 11x, "The Command Zone"
    # 5x, "Deal Damage" 3x). "Red Mana" joins them for the same reason.
    PROSE_STOPLIST = frozenset({"the end", "the command zone", "deal damage", "red mana"})

    def _scan_prose(self, text: str, budget: int, seen: set[str]) -> list[dict]:
        """Unbracketed card names, longest first, MULTI-WORD ONLY.

        Two rules, both measured across `gold_candidates.jsonl` before shipping,
        the discipline `common.find_players` is held to:

        * **at least two words.** Eighteen ordinary rules words are also card
          names — `exile`, `lifelink`, `regeneration`, `fear`, `shock`, `fog`,
          `counterspell` among them — so "How does lifelink work?" would
          otherwise inject a card into a pure rules question. Requiring two
          words removes ALL eighteen and still covers 92% of the index. A
          single-word card is still reachable by bracketing it.
        * **longest match wins, and its span is masked**, so "Genesis" cannot
          also match inside "Genesis Wave" and spend a card slot on the wrong
          card.

        Measured precision: 1,961 correct against 35 false positives (98.2%),
        and 15 (99.2%) with the stoplist. It MISSES far more than it invents,
        which is the right direction: a miss is a question answered with less
        context, an invention is a question answered about the wrong card.
        """
        if not hasattr(self, "_prose_names"):
            self._prose_names = sorted(
                ((n.lower(), n) for n in self.by_name if len(n.split()) >= 2),
                key=lambda p: -len(p[0]))
        hay = re.sub(r"\s+", " ", " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " ")
        out: list[dict] = []
        for low, name in self._prose_names:
            if len(out) >= budget:
                break
            if low in self.PROSE_STOPLIST or name in seen:
                continue
            if " " + low + " " in hay:
                card, how = self.resolve(name)
                if card and card["name"] not in seen:
                    seen.add(card["name"])
                    out.append({**card, "matched_as": name, "match_type": how + "+prose"})
                hay = hay.replace(low, " " * len(low))
        return out


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
