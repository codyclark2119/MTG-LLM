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
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (GOLD_PATH, load_glossary_terms, normalize_record, read_jsonl,
                    write_jsonl_atomic)

# The normalizer itself now lives in common.py — `migrate_record` was copied
# into the task export and the promotion path, and a third copy is how the
# player-verb lists drifted apart in the first place. This script is the
# one-time driver; the logic is shared. See `common.normalize_record`.
migrate_record = normalize_record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=GOLD_PATH)
    ap.add_argument("--apply", action="store_true", help="write. Without this, dry run.")
    ap.add_argument("--limit", type=int, default=None, help="show only the first N records")
    args = ap.parse_args()

    rows = read_jsonl(args.gold, missing_ok=False)
    # Supplying the glossary switches on possessive and prepositional discovery
    # ("Nyla's hand", "from Braylen"), which the verb test alone cannot see.
    # The glossary is what keeps those loose patterns from renaming "to Devour"
    # or "with Cascade" — see common.load_glossary_terms.
    terms = load_glossary_terms()
    migrated, changed, skipped = [], 0, 0
    shown = 0
    for rec in rows:
        new, notes = migrate_record(rec, magic_terms=terms)
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
