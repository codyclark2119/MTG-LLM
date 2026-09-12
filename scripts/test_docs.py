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


def test_suite_lists_agree() -> list[str]:
    """The three places the test list is written must name the same files.

    CLAUDE.md lists the suite in prose, `scripts/run_tests.sh` lists it as the
    thing that actually runs, and the files themselves are on disk. Three
    copies of one list is three chances to drift, and this repo has already
    paid for exactly that: CLAUDE.md's list "was seven for a while, with four
    stale assertion counts, so 'I ran the tests' meant half of them."

    A test file on disk that the runner does not run is the dangerous
    direction — it looks like coverage and is not — so it is a failure here
    rather than a note.
    """
    problems = []
    scripts_dir = REPO_ROOT / "scripts"
    on_disk = {str(q.relative_to(REPO_ROOT)) for q in scripts_dir.rglob("test_*.py")}

    runner = (scripts_dir / "run_tests.sh").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\s*(scripts/(?:gameplay/)?test_\w+\.py)\s*$",
                            runner, re.M))

    for missing in sorted(on_disk - listed):
        problems.append(f"{missing} exists but scripts/run_tests.sh does not run it")
    for ghost in sorted(listed - on_disk):
        problems.append(f"scripts/run_tests.sh lists {ghost}, which is not on disk")

    claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for path in sorted(on_disk):
        if path not in claude:
            problems.append(f"{path} is not listed in CLAUDE.md's suite table")
    return problems


def test_current_state_claims() -> list[str]:
    """Claims about how the system behaves NOW, checked against the code.

    Prose that describes current behaviour goes stale silently -- 21.163 moved
    the serving model and left five files saying 7B, and CLAUDE.md described
    `ruling_chunks.jsonl` as read by nothing for long after ten scripts read
    it. Where a claim can be checked mechanically it is checked here instead of
    being restated and re-synchronised by hand.

    Historical notes are deliberately NOT in scope: DEVELOPMENT_PLAN.md is an
    experiment log and a 2024 observation stays true of 2024. Only files that
    describe the present are read.
    """
    from common import CHAT_SERVING_PROFILE

    problems = []

    # 1. `ruling_chunks.jsonl` is on the serving path. The claim that went
    #    stale was that nothing read it, so assert the readership itself.
    hybrid = (REPO_ROOT / "scripts" / "retrieve_hybrid.py").read_text(encoding="utf-8")
    if "RULING_CHUNKS_PATH" not in hybrid:
        problems.append(
            "retrieve_hybrid.py no longer reads RULING_CHUNKS_PATH -- if that is "
            "deliberate, DEVELOPMENT_PLAN 21.16's superseded note and CLAUDE.md's "
            "conventions section both need revisiting")

    # 2. No current-state file may claim the chat surface serves the 7B.
    #    The serving model is whatever the profile says; these files must not
    #    hold a second copy of that answer.
    serves_7b = ("the 7B this serves", "the 7B this service actually serves",
                 "the 7B this project\nserves", "THE 7B THIS PROJECT",
                 "7B this service")
    for name in ("README.md", "scripts/chat_server.py", "scripts/serve_chat.sh",
                 "scripts/common.py", "deploy/README.md"):
        path = REPO_ROOT / name
        if not path.exists():
            continue
        body = path.read_text(encoding="utf-8")
        for claim in serves_7b:
            if claim in body:
                problems.append(f"{name} still says the chat surface serves the 7B "
                                f"({claim!r}); it serves {CHAT_SERVING_PROFILE.model_id}")

    # 3. The profile's own pin. Stated here as well as in test_chat_server so a
    #    docs-only run still catches a configuration that moved silently.
    from common import CHAT_SERVING_PROFILE_FINGERPRINT
    if CHAT_SERVING_PROFILE.fingerprint() != CHAT_SERVING_PROFILE_FINGERPRINT:
        problems.append(
            f"CHAT_SERVING_PROFILE changed without its pin: "
            f"{CHAT_SERVING_PROFILE.fingerprint()} vs "
            f"{CHAT_SERVING_PROFILE_FINGERPRINT}. Bump the profile_id if this is "
            f"a new serving configuration, so stored ratings stay separable.")

    # 4. The documented export flow must not tell anyone to use the contributor
    #    token on /api/export -- it is refused now.
    for name in ("README.md", "deploy/README.md"):
        path = REPO_ROOT / name
        if path.exists() and "x-token: $RUBRIC_TOKEN" in path.read_text(encoding="utf-8"):
            if "/api/export" in path.read_text(encoding="utf-8"):
                problems.append(
                    f"{name} documents exporting with the contributor token; "
                    f"/api/export requires RUBRIC_EXPORT_TOKEN via x-export-token")
    return problems


def main() -> None:
    text = README.read_text(encoding="utf-8")
    failed = 0

    for problem in test_suite_lists_agree() + test_current_state_claims():
        print(f"  FAIL  {problem}")
        failed += 1
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
        print(f"\n{failed} stale claim(s) — documentation does not match the code")
        raise SystemExit(1)
    print(f"README artifact counts match ({len(ROWS)} inventory rows + prose); "
          f"suite lists and current-state claims agree")


if __name__ == "__main__":
    main()
