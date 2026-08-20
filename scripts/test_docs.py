"""Check that README's artifact counts match the artifacts.

    python scripts/test_docs.py

WHY

CLAUDE.md gives README one job besides "how to run things": **current artifact
counts**. They drift, silently, because nothing reads them. Three were wrong on
the same day — the gold set listed as 39 records when it held 99, the position
set as 22 when it held 24, and the verified SFT split as 1,019 when a
contamination fix had cut it to 1,001.

None of those are dangerous on their own. Together they are the same failure the
plan keeps recording in the code: a number written from intent and never
compared to the thing it describes.

Matching is on the DATA INVENTORY TABLE ROW LABEL, not on free text. The first
draft of this check used a loose pattern and matched "19,726 cards" out of the
*rulings* row while looking for the card-chunk count — reporting the README
stale when it was correct. A check that cries wolf gets deleted, so it reads the
one place the counts are declared and nowhere else.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (  # noqa: E402
    CARD_CHUNKS_PATH,
    CHUNKS_PATH,
    GLOSSARY_PATH,
    GOLD_CANDIDATES_PATH,
    GOLD_PATH,
    POSITIONS_PATH,
    REPO_ROOT,
    RULING_CHUNKS_PATH,
)

README = REPO_ROOT / "README.md"

# row label in the inventory table -> path whose line count it states
ROWS = {
    "Cards": CARD_CHUNKS_PATH,
    "Retrieval chunks": CHUNKS_PATH,
    "RulesGuru candidates": GOLD_CANDIDATES_PATH,
    "**Gold set**": GOLD_PATH,
    "**Positions**": POSITIONS_PATH,
}


def count(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def stated_in_row(text: str, label: str) -> int | None:
    """The first number in the table row beginning with `label`."""
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 2 and cells[1] == label:
            m = re.search(r"[\d,]+", cells[2])
            return int(m.group(0).replace(",", "")) if m else None
    return None


def main() -> None:
    text = README.read_text(encoding="utf-8")
    failed = 0
    for label, path in ROWS.items():
        if not path.exists():
            print(f"  skip  {label}: {path} missing")
            continue
        actual, stated = count(path), stated_in_row(text, label)
        if stated is None:
            print(f"  FAIL  no inventory row labelled {label!r} in README.md")
            failed += 1
        elif stated != actual:
            print(f"  FAIL  {label}: README says {stated:,}, {path.name} holds {actual:,}")
            failed += 1

    # Counts quoted in prose and commands, where they are easy to forget.
    for pattern, actual, what in (
            (r"validate the (\d+)-position set", count(POSITIONS_PATH), "positions.py usage line"),
            (r"card_chunks\.jsonl \(([\d,]+)", count(CARD_CHUNKS_PATH), "chunk_cards.py output"),
            (r"([\d,]+) glossary", count(GLOSSARY_PATH), "glossary count"),
            (r"across ([\d,]+) cards", count(RULING_CHUNKS_PATH), "rulings coverage")):
        for m in re.finditer(pattern, text):
            stated = int(m.group(1).replace(",", ""))
            if stated != actual:
                print(f"  FAIL  {what}: README says {stated:,}, actual {actual:,}")
                failed += 1

    if failed:
        print(f"\n{failed} stale count(s) in README.md")
        raise SystemExit(1)
    print(f"README artifact counts match ({len(ROWS)} inventory rows + prose)")


if __name__ == "__main__":
    main()
