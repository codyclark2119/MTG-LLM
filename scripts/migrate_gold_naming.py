"""Normalize player names and add card slots to the gold set. One-time.

WHY BOTH AT ONCE
----------------
RulesGuru randomizes player names *and cards* per request (see
`fetch_rulesguru.py`). Two consequences the gold set has been living with:

  * Player names are arbitrary labels. The same question already disagrees
    with itself — gold record `qa-amy-...` says "Amy casts Assassin's
    Trophy" while the snapshot for the same rulesguru id says "Alaia". 18 of
    38 linkable records differ from the snapshot this way.
  * The cards are the *interesting* variable. Refetching an id returns the
    same ruling instantiated on different cards, which is the makings of a
    real generalization test: same rule, different cards, so an answer that
    only works for one instantiation is memorization rather than rules
    knowledge.

Rewriting question and answer text invalidates comparability with runs
generated against the old text (`pilot_v3`, `pilot_v3_judge2`). That cost
is paid once whether we do one migration or two, so both changes go
together.

WHAT IT DOES
------------
1. Players -> `Player A`, `Player B`, ... in order of first appearance,
   across question and answer.
2. `card_slots: {"card1": "<name>", ...}` added from the record's `cards`,
   which matches the snapshot's `includedCards` exactly.
3. Hand-authored rubrics get card names replaced with `[[card1]]`-style
   templates so one rubric can score every instantiation of its question.
   Brackets, not braces: Magic writes mana costs as `{b}`, and one gold
   rubric already contains exactly that.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
Templates never enter `question` or `answer`. Those are what the model
reads, and meta-syntax in a prompt gets copied: `ACTION_GRAMMAR` once wrote
optional operands as `[TARGET <x>]`, the model reproduced the brackets, and
correct plays scored as illegal. Slots live in structured metadata and in
the rubric only; the authoring form shows the mapping.

Players are normalized rather than templated because, unlike cards, they
carry no rules-relevant text — once they are `Player A`, they are stable
across every variant and a rubric can name them directly.

Usage:
    python scripts/migrate_gold_naming.py              # dry run, shows every change
    python scripts/migrate_gold_naming.py --apply
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (GOLD_PATH, is_hand_authored, read_jsonl, templatize,
                    write_jsonl_atomic)

# A capitalized word directly in front of a verb only a player performs. Same
# grammar test `common.stray_names` uses, and for the same reason: matching on
# capitalization alone cannot tell "Nylah" from "Swamp".
# Only verbs that REQUIRE a player subject. The first draft also carried
# "is/are/do/does/will/would" and turned "What are the characteristics of..."
# into "Player B are the characteristics of...", corrupting the question. A
# generic copula is not evidence of a player.
PLAYER_VERBS = (
    "controls control casts cast has have had plays play played attacks attack "
    "blocks block targets target draws draw discards discard sacrifices sacrifice "
    "activates activate taps tap owns own gains gain loses lose wants want "
    "responds respond chooses choose declares declare puts put moves move "
    "exiles exile destroys destroy counters counter reveals reveal wins win "
    "searches search passes pass concedes concede attempts attempt tries try"
).split()
PLAYER_RE = re.compile(r"\b([A-Z][a-z]{2,})\s+(?:" + "|".join(PLAYER_VERBS) + r")\b")
LABELS = [f"Player {c}" for c in "ABCDEFGH"]

# The [[cardN]] form and the reason for it live in common.py, alongside the
# inverse the authoring form needs.


def card_words(cards: list[str]) -> set[str]:
    """Every capitalized token appearing inside a card name.

    A card called "Alesha, Who Smiles at Death" would otherwise have "Alesha"
    read as a player. Card names are known exactly, so exclude their tokens.
    """
    out: set[str] = set()
    for name in cards or []:
        for tok in re.split(r"[^A-Za-z]+", name):
            if tok and tok[0].isupper():
                out.add(tok)
    return out


def find_players(question: str, answer: str, cards: list[str]) -> list[str]:
    """Player names, in order of first appearance in the question then answer."""
    banned = card_words(cards)
    order: list[str] = []
    for text in (question or "", answer or ""):
        for name in PLAYER_RE.findall(text):
            if name not in banned and name not in order:
                order.append(name)
    return order


def rename(text: str, mapping: dict[str, str]) -> str:
    if not text or not mapping:
        return text
    # Longest first so a name that is a prefix of another cannot half-match.
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in
                                           sorted(mapping, key=len, reverse=True)) + r")\b")
    return pattern.sub(lambda m: mapping[m.group(1)], text)


def migrate_record(rec: dict) -> tuple[dict, list[str]]:
    """Return (new record, human-readable notes). Never mutates the input."""
    out = dict(rec)
    notes: list[str] = []
    cards = rec.get("cards") or []

    players = find_players(rec.get("question", ""), rec.get("answer", ""), cards)
    if len(players) > len(LABELS):
        notes.append(f"!! {len(players)} players found, only {len(LABELS)} labels — skipped")
        return rec, notes
    mapping = dict(zip(players, LABELS))
    if mapping:
        out["question"] = rename(rec.get("question", ""), mapping)
        out["answer"] = rename(rec.get("answer", ""), mapping)
        out["paraphrases"] = [rename(p, mapping) for p in rec.get("paraphrases") or []]
        # The rubric names players too. Renaming only question and answer left
        # a key point reading "At the moment [[card2]] enters Alex controls no
        # Swamps" against a question that no longer mentions Alex.
        for field in ("key_points", "common_errors"):
            out[field] = [rename(x, mapping) for x in rec.get(field) or []]
        notes.append("players: " + ", ".join(f"{k} -> {v}" for k, v in mapping.items()))
    else:
        notes.append("players: none found")

    if cards:
        slots = {f"card{i}": name for i, name in enumerate(cards, 1)}
        out["card_slots"] = slots
        notes.append("slots: " + ", ".join(f"{k}={v}" for k, v in slots.items()))
        # Only hand-authored rubrics are templated. A machine draft is going to
        # be rewritten from scratch anyway, and templating it would just make
        # the text the author is meant to replace harder to read.
        if is_hand_authored(rec):
            for field in ("key_points", "common_errors"):
                before = out.get(field) or []          # already player-renamed
                after = [templatize(x, slots) for x in before]
                out[field] = after
                for b, a in zip(before, after):
                    if b != a:
                        notes.append(f"  {field}: {a}")
        else:
            notes.append("  (machine draft — rubric left alone, it gets rewritten)")
    return out, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=GOLD_PATH)
    ap.add_argument("--apply", action="store_true", help="write. Without this, dry run.")
    ap.add_argument("--limit", type=int, default=None, help="show only the first N records")
    args = ap.parse_args()

    rows = read_jsonl(args.gold, missing_ok=False)
    migrated, changed, skipped = [], 0, 0
    shown = 0
    for rec in rows:
        new, notes = migrate_record(rec)
        migrated.append(new)
        if new != rec:
            changed += 1
        if any(n.startswith("!!") for n in notes):
            skipped += 1
        if args.limit is None or shown < args.limit:
            shown += 1
            print(f"\n### {rec['id']}")
            if new.get("question") != rec.get("question"):
                print(f"  -  {rec['question'][:110]}")
                print(f"  +  {new['question'][:110]}")
            for n in notes:
                print(f"     {n}")

    print(f"\n{changed}/{len(rows)} records change, {skipped} skipped")
    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return

    backup = args.gold.with_name(
        f"{args.gold.stem}.backup-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}{args.gold.suffix}")
    shutil.copy2(args.gold, backup)
    write_jsonl_atomic(args.gold, migrated)
    print(f"\nbackup -> {backup.name}")
    print(f"written -> {args.gold}")
    print("\nNext: python scripts/validate_gold.py --to-eval")


if __name__ == "__main__":
    main()
