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
    text = path.read_text(encoding="utf-8", errors="replace")
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
                        if s and not s.startswith("{") and len(s) > 1})

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


def score(t: dict) -> tuple:
    """Rank candidates: more distinct cards and more assertions is more
    likely to be a genuine INTERACTION rather than a single-card sanity
    check, but this is a heuristic ranking for triage, not a quality gate."""
    return (t["n_cards"], t["n_asserts"])


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
    scanned_files = 0
    for d in dirs:
        d_path = base / d
        if not d_path.is_dir():
            continue
        for f in d_path.rglob("*.java"):
            scanned_files += 1
            candidates.extend(t for t in extract_tests(f) if t["n_cards"] >= args.min_cards)

    candidates.sort(key=score, reverse=True)
    top = candidates[:args.limit]

    print(f"scanned {scanned_files} files across {len(dirs)} categor{'y' if len(dirs)==1 else 'ies'}, "
          f"{len(candidates)} candidate(s) with >= {args.min_cards} distinct card names\n")

    for t in top:
        rel_dir = Path(t["source_file"]).parent.name
        print(f"=== [{rel_dir}] {t['class_name']}.{t['method']}"
              f" ({t['n_cards']} cards, {t['n_asserts']} assertions) ===")
        print(f"    cards: {', '.join(t['cards'])}")
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
