"""Mine XMage's own JUnit test suite for candidate ruling/rules-interaction questions.

WHY THIS EXISTS

The card/ruling benchmark (Section 21.132) and any rulings-training-data
experiment (PLAN_NEXT.md item 3's third bullet) both need real, verified
scenarios — and hand-authoring each one from a ruling's prose, one at a time,
is exactly the "loose logic, hard to teach or scale" problem this was built
to get out from under (Section 21.135).

XMage ships ~1,800 JUnit tests under `Mage.Tests/src/test/java/org/mage/test/cards/`,
written by its own contributor community to verify their card implementations,
and continuously run in XMage's own CI. Each one is already a verified
(setup, action, assertion) triple for some rules interaction — the exact
shape a benchmark question needs, except already checked by a real engine
instead of by one person's reading of a ruling. This is the leverage-what's-
built alternative to writing a new probe by hand per question: read what
already exists and passes, rather than construct something new and hope it
does.

WHAT THIS DOES AND DOES NOT DO

It extracts CANDIDATES — card names, setup/action calls, and assertions,
scraped with regexes rather than a real Java parser, deliberately, since the
goal is triage material for a person to read the actual file from, not fully
automated question generation. A wrong or missed extraction costs nothing;
it just means a human looks at the source file directly, which the report
always names.

It does not run the tests. Confirming a specific test still passes IN THIS
CHECKOUT (not just that it once did) is a separate, deliberate step —
`--verify` compiles and runs the named tests via the same `mvn clean test`
pattern this repo's other XMage tooling already uses, because a test that
does not currently pass is not a fact about the current engine no matter how
promising its assertions look.

It does not write the gold set, or any candidates file, or any question.
This is upstream of that.

Usage:
    python scripts/mine_xmage_tests.py --mage-home ~/Documents/personal_code/mage
    python scripts/mine_xmage_tests.py --category triggers --min-cards 2 --limit 20
    python scripts/mine_xmage_tests.py --verify --limit 10
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Directories under org/mage/test/cards/ that hold multi-concept rules-
# interaction tests, as opposed to `single/`, which is mostly one-card
# sanity checks (does this creature have the keyword it says it has). Not
# excluding `single` — just deprioritizing it, since a `single` test CAN
# still cover a real interaction (an ETB trigger with a cost, say).
INTERACTION_DIRS = ("triggers", "cost", "continuous", "replacement", "rules",
                    "targets", "damage", "prevention", "protection",
                    "restriction", "requirement", "asthough", "copy",
                    "dynamicvalue", "conditional")

_PACKAGE_RE = re.compile(r"^package\s+([\w.]+);", re.MULTILINE)
_TEST_METHOD_RE = re.compile(
    r"@Test\s*(?:\([^)]*\))?\s*\n\s*public\s+void\s+(\w+)\s*\([^)]*\)\s*\{")
_CLASS_RE = re.compile(r"public\s+class\s+(\w+)")
# addCard/castSpell/activateAbility/attack/block calls, whose first or second
# string argument is usually a card name — good enough for triage, not
# claimed to be exact.
_CALL_RE = re.compile(
    r"\b(addCard|castSpell|activateAbility|attack|block|setChoice|"
    r"checkPlayableAbility|addTarget)\s*\(([^;]*?)\);", re.DOTALL)
_ASSERT_RE = re.compile(r"\b(assert\w+)\s*\(([^;]*?)\);", re.DOTALL)
_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
# `/* ... */` and `// ...` comments — stripped before any other extraction
# regex runs, so a disabled test (an @Test method whose entire body is
# commented out) naturally falls out via the existing "no calls, no asserts"
# skip below, rather than being read as a real one. Not comment-aware of
# string literals containing `//` (a URL, say) — no such literal exists in
# this test suite's actual usage, and the cost of being wrong is the same as
# any other extraction miss here: nothing, since a person reads the source
# file directly before trusting a candidate.
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)


def _strip_java_comments(text: str) -> str:
    return _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub(" ", text))

# Path components and filename patterns that mark an implementation-detail
# regression test (client/server sync, a Java exception that once crashed
# something) rather than a rules interaction a person could turn into a
# question. Matched against the full relative path, since some of these are
# nested under an otherwise-legitimate category (`cost/modaldoublefaced/`).
NOISY_PATH_PARTS = ("modaldoublefaced",)
NOISY_NAME_RE = re.compile(
    r"Exception|Crash|Regression|ConcurrentModification|NullPointer|Freeze",
    re.I)

# A "card name" extracted from an assertion MESSAGE string, not a card-lookup
# argument, is usually an English sentence fragment describing what is being
# checked ("client must ignore side 2", "before last cast 1", "Phantasmal
# Image should be a Rogue") rather than a card, even when it starts with a
# real card name. Real card names Title Case every word and never contain
# these; reject anything that does, rather than try to parse where the card
# name ends and the description begins.
_PROSE_RE = re.compile(
    r"\b(should|must|can|can't|cannot|before|after|client|server|is|are|"
    r"was|were|not|use|ignore|have|has)\b", re.I)


def _looks_like_a_card(name: str) -> bool:
    if not name or not name[0].isupper():
        return False
    if " - " in name or "@" in name or name.startswith("[") or "=" in name:
        return False
    if _PROSE_RE.search(name):
        return False
    return True


def _find_matching_brace(text: str, open_idx: int) -> int:
    """Index just after the `}` matching the `{` at `open_idx`."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def extract_tests(path: Path) -> list[dict]:
    """One dict per `@Test` method in a file: calls, assertions, card names."""
    text = _strip_java_comments(path.read_text(encoding="utf-8", errors="replace"))
    pkg_m = _PACKAGE_RE.search(text)
    cls_m = _CLASS_RE.search(text)
    package = pkg_m.group(1) if pkg_m else "?"
    class_name = cls_m.group(1) if cls_m else path.stem

    out = []
    for m in _TEST_METHOD_RE.finditer(text):
        method_name = m.group(1)
        body_start = m.end() - 1  # the '{' the regex matched up to
        body_end = _find_matching_brace(text, body_start)
        body = text[body_start:body_end]

        calls = [(kind, args.strip()) for kind, args in _CALL_RE.findall(body)]
        asserts = [(kind, args.strip()) for kind, args in _ASSERT_RE.findall(body)]
        cards = sorted({s for _, args in calls + asserts for s in _STRING_RE.findall(args)
                        if _looks_like_a_card(s)})

        if not calls and not asserts:
            continue
        out.append({
            "package": package, "class_name": class_name, "method": method_name,
            "source_file": str(path), "cards": cards,
            "setup_actions": [f"{k}({a})" for k, a in calls],
            "assertions": [f"{k}({a})" for k, a in asserts],
            "n_cards": len(cards), "n_asserts": len(asserts),
        })
    return out


def cards_with_rulings() -> dict[str, int]:
    """{card name: number of official rulings} from the pinned corpus.

    A different field (`n_rulings`) than
    `validate_card_ruling_benchmark.load_ruling_chunk_ids_by_card` reads
    (chunk ids, always one per card in this corpus's shape, so useless as a
    ruling COUNT) — not reused, since reuse would mean changing what that
    function returns for its own, different, purpose.
    """
    from common import RULING_CHUNKS_PATH, read_jsonl
    return {rec["name"]: rec.get("n_rulings", 0) for rec in read_jsonl(RULING_CHUNKS_PATH)}


def score(t: dict) -> tuple:
    """Rank candidates for triage, not a quality gate.

    A high raw card/assertion count is as often noisy extraction (an
    assertion message string quoting several words) as it is a genuinely
    rich interaction — capped at 8 real cards, since a scenario needing more
    than that is likely too sprawling to state as one clean question even
    when the extraction is accurate. Above the cap, MORE cards is a
    complexity penalty, not a bonus.
    """
    n = t["n_cards"]
    band = n if n <= 8 else max(0, 16 - n)
    return (band, t["n_asserts"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mage-home", type=Path,
                    default=Path.home() / "Documents/personal_code/mage")
    ap.add_argument("--category", choices=INTERACTION_DIRS, default=None,
                    help="restrict to one org/mage/test/cards/<category> directory; "
                         "default scans all of INTERACTION_DIRS plus single/")
    ap.add_argument("--include-single", action="store_true",
                    help="also scan cards/single/ (940 files, mostly one-card checks)")
    ap.add_argument("--min-cards", type=int, default=2,
                    help="drop candidates naming fewer than this many distinct cards")
    ap.add_argument("--min-rulings", type=int, default=0,
                    help="drop candidates where no named card has at least this many "
                         "official rulings on file — the filter to use when hunting for "
                         "a ruling_relevant_insufficient benchmark example specifically, "
                         "since that category needs a real ruling to be relevant AND "
                         "insufficient, not merely a rules interaction with no ruling at all")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--verify", action="store_true",
                    help="compile and run the top --limit candidates via mvn, to "
                         "confirm each still passes in this checkout right now")
    args = ap.parse_args()

    base = args.mage_home / "Mage.Tests/src/test/java/org/mage/test/cards"
    if not base.is_dir():
        raise SystemExit(f"no such directory: {base} — is --mage-home a real checkout?")

    dirs = [args.category] if args.category else list(INTERACTION_DIRS)
    if args.include_single or args.category == "single":
        dirs = dirs + ["single"] if args.category != "single" else ["single"]

    candidates: list[dict] = []
    scanned_files = skipped_noisy = 0
    for d in dirs:
        d_path = base / d
        if not d_path.is_dir():
            continue
        for f in d_path.rglob("*.java"):
            rel = str(f.relative_to(base))
            if any(part in rel for part in NOISY_PATH_PARTS) or NOISY_NAME_RE.search(f.name):
                skipped_noisy += 1
                continue
            scanned_files += 1
            candidates.extend(t for t in extract_tests(f) if t["n_cards"] >= args.min_cards)

    rulings_by_card = cards_with_rulings()
    for t in candidates:
        t["ruling_cards"] = {c: rulings_by_card[c] for c in t["cards"] if c in rulings_by_card}
    if args.min_rulings:
        candidates = [t for t in candidates
                      if any(n >= args.min_rulings for n in t["ruling_cards"].values())]

    candidates.sort(key=score, reverse=True)
    top = candidates[:args.limit]

    print(f"scanned {scanned_files} files across {len(dirs)} categor{'y' if len(dirs)==1 else 'ies'} "
          f"({skipped_noisy} skipped as implementation-detail regression tests), "
          f"{len(candidates)} candidate(s) with >= {args.min_cards} distinct card names"
          + (f" and >= {args.min_rulings} ruling(s) on a named card" if args.min_rulings else "")
          + "\n")

    for t in top:
        rel_dir = Path(t["source_file"]).parent.name
        print(f"=== [{rel_dir}] {t['class_name']}.{t['method']}"
              f" ({t['n_cards']} cards, {t['n_asserts']} assertions) ===")
        print(f"    cards: {', '.join(t['cards'])}")
        if t["ruling_cards"]:
            print(f"    HAS RULINGS: "
                  + ", ".join(f"{c} ({n})" for c, n in t["ruling_cards"].items()))
        print(f"    file:  {t['source_file']}")
        for line in t["setup_actions"][:6]:
            print(f"    setup: {line[:100]}")
        for line in t["assertions"][:6]:
            print(f"    ASSERT: {line[:100]}")
        print()

    if args.verify:
        if not top:
            return
        classes = ",".join(f"{t['class_name']}" for t in top)
        print(f"==> mvn clean test -Dtest={classes} (confirming these still pass) ...")
        result = subprocess.run(
            ["mvn", "clean", "test", "-pl", "Mage.Tests", "-am",
             f"-Dtest={classes}", "-Dsurefire.failIfNoSpecifiedTests=false"],
            cwd=args.mage_home)
        if result.returncode != 0:
            print("\nmvn reported failures — check "
                  f"{args.mage_home}/Mage.Tests/target/surefire-reports before trusting "
                  "any candidate whose class is among the ones above.")


if __name__ == "__main__":
    main()
