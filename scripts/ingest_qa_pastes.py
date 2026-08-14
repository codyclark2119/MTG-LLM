"""Parse pasted rules Q&A into gold-schema records.

Input format (repeatable, blank line between entries):

    Question
    "..."
    Answer
    "... (603.3,117.2a) ..."
    Cards Mentioned:
    [Card One, Card Two]

This is the best-shaped source the project has for Phase 1: unlike the
judge worksheets (tournament policy, needs IPG/MTR) and unlike the reddit
set (only 21% cite any rule), these are Comprehensive Rules answers that
already carry their citations inline.

That means two of the three expensive gold fields come for free —
`rule_citations` is extracted from the answer text and validated against
the pinned CR, and `cards` is resolved against the Oracle pool. What
still needs a human is `key_points`: the rubric that makes scoring
length-neutral and reproducible (see data/gold/SCHEMA.md for why a prose
reference alone caused the judge problems in Sections 9.6-9.9). Records
are emitted with `key_points: []` and `needs_rubric: true` so they are
easy to queue for a reviewer, and `--draft-rubric` proposes a starting
rubric by sentence-splitting the answer for a human to edit rather than
write from scratch.

Usage:
    python scripts/ingest_qa_pastes.py --from data/gold/pastes/*.txt
    python scripts/ingest_qa_pastes.py --from paste.txt --draft-rubric
"""

import argparse
import json
import re
import sys
from pathlib import Path

from common import CR_VERSION, GOLD_PATH, RULES_PATH, load_rule_ids
from common import RULE_ID_RE as CROSS_REF_RE

# Labels vary across sources ("Question" vs "Question:"), and pasted text
# picks up smart quotes from wherever it was copied, so both are optional
# rather than assumed. A source that silently parses to zero entries is
# worse than one that errors, so main() reports a per-file count.
ENTRY_RE = re.compile(
    # Each field ends at the closing quote that precedes the NEXT section
    # label, not at the first quote encountered. Answers legitimately
    # contain quoted text — '...subtypes: "Plains", "Island", ...' — and
    # stopping at the first inner quote silently truncated the answer to
    # its opening clause, discarding every citation and the card list with
    # it. The lookahead makes the section boundary the terminator.
    r"Question:?[ \t]*\n[ \t]*[\"“](?P<q>.*?)[\"”][ \t]*\n[ \t]*(?=Answer)"
    r"Answer:?[ \t]*\n[ \t]*[\"“](?P<a>.*?)[\"”][ \t]*\n?[ \t]*"
    r"(?=Cards\s+Mentioned|Question:?[ \t]*\n|\Z)"
    r"(?:Cards\s+Mentioned:?[ \t]*\n?[ \t]*\[(?P<cards>.*?)\])?",
    re.S | re.I,
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Citation parentheticals appear in several shapes across sources:
# "(614.4)", "(603.3,117.2a)", slash-separated "(105.2c/105.1/105.4)", and
# the RulesGuru API's bracketed "([603.3], [117.2a])". These are pure
# citations and are dropped from a rubric, which scores claims.
CITATION_GROUP_RE = re.compile(
    r"\(\s*\[?\d{3}\.\d+[a-z]?\]?(?:\s*[,/]\s*\[?\d{3}\.\d+[a-z]?\]?)*\s*\)"
)
# A bracketed rule number *inside* a sentence is different: it is part of
# the assertion ("None of the exceptions in [601.3] apply here"). Deleting
# it leaves "None of the exceptions in apply here", so unbracket instead.
INLINE_BRACKET_RE = re.compile(r"\[(\d{3}\.\d+[a-z]?)\]")


def slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:maxlen].rstrip("-")


def draft_rubric(answer: str) -> list[str]:
    """Propose key points by splitting the answer into assertions.

    Deliberately crude: this is a starting point for a human to edit, not
    a substitute for judgement. Rule citations are stripped from each
    point because the rubric checks *claims*, while citations are scored
    separately.
    """
    sentences = SENTENCE_RE.split(answer.strip())
    points = []
    for i, sentence in enumerate(sentences):
        cleaned = CITATION_GROUP_RE.sub("", sentence).strip(" .,;")
        cleaned = INLINE_BRACKET_RE.sub(r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        # Removing a mid-sentence citation leaves an orphaned space before
        # the following punctuation ("the stack , so ...").
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned).strip(" .,;")
        if not cleaned:
            continue
        # The verdict is usually the first sentence and usually short
        # ("Yes.", "No.", "Tapped.") — a length filter drops exactly the
        # most important assertion in the rubric, so the leading sentence
        # is always kept regardless of length.
        if i == 0 or len(cleaned) > 12:
            points.append(cleaned)
    return points

CARD_ITEM_RE = re.compile(r"""\s*(?:'([^']*)'|"([^"]*)"|([^,]+))\s*(?:,|$)""")


def split_bracket_cards(raw: str) -> list[str]:
    """Split a `Cards Mentioned` list, respecting quotes.

    Card names contain commas ("Urborg, Tomb of Yawgmoth"), and sources
    quote those entries to disambiguate — sometimes mixing quoted and bare
    items in one list. Splitting on commas alone turns one legendary land
    into two nonexistent cards.
    """
    out = []
    for m in CARD_ITEM_RE.finditer(raw):
        value = next((g for g in m.groups() if g is not None), "").strip()
        if value:
            out.append(value)
    return out


def parse_pastes(text: str) -> tuple[list[dict], int]:
    """Return (entries, blank_stub_count).

    Collection files accumulate empty templates at the end — a contributor
    leaves several `Question: ""` blocks ready to fill in. Those parse
    perfectly well into entries with no content, which then collapse to a
    single id-less record that looks like a parser bug. Drop them here and
    report the count so an unexpectedly large number is still visible.
    """
    entries = []
    blank = 0
    for m in ENTRY_RE.finditer(text):
        question = re.sub(r"\s+", " ", m.group("q")).strip()
        answer = re.sub(r"\s+", " ", m.group("a")).strip()
        if not question or not answer:
            blank += 1
            continue
        cards = split_bracket_cards(m.group("cards") or "")
        entries.append({"question": question, "answer": answer, "cards": cards})
    return entries, blank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="sources", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=GOLD_PATH)
    parser.add_argument("--rules", type=Path, default=RULES_PATH)
    parser.add_argument("--cr-version", default=CR_VERSION)
    parser.add_argument("--source-label", default="rules-qa-paste")
    parser.add_argument("--category", default="interaction puzzle", help="default category; review per record")
    parser.add_argument("--difficulty", default="intermediate")
    parser.add_argument("--draft-rubric", action="store_true", help="propose key_points for a human to edit")
    parser.add_argument("--skip-cards", action="store_true")
    args = parser.parse_args()

    valid_rule_ids = load_rule_ids(args.rules)

    card_index = None
    if not args.skip_cards:
        sys.path.insert(0, str(Path(__file__).parent))
        from card_lookup import CardIndex

        # Reused rather than reimplemented: the worksheet ingester already
        # solves rejoining comma-split card names against the Oracle pool.
        from ingest_judge_worksheets import split_card_list

        card_index = CardIndex()

    existing: dict[str, dict] = {}
    if args.out.exists():
        with args.out.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                existing[r["id"]] = r
    print(f"{len(existing)} gold records already present")

    entries = []
    empty_files = []
    blank_stubs = 0
    for path in args.sources:
        found, blank = parse_pastes(path.read_text(encoding="utf-8"))
        blank_stubs += blank
        # A format the regex doesn't recognize yields zero entries, which
        # would otherwise look identical to an empty file and silently drop
        # real data. Surface it per file instead.
        if not found:
            empty_files.append(path)
        entries.extend(found)
    print(f"parsed {len(entries)} entries from {len(args.sources)} file(s)")
    if blank_stubs:
        print(f"skipped {blank_stubs} blank template stub(s)")
    if empty_files:
        print(f"WARNING: {len(empty_files)} file(s) parsed to ZERO entries — check the label format:")
        for p in empty_files:
            print(f"  - {p}")

    new = skipped = 0
    warnings: list[str] = []
    for e in entries:
        cited = sorted(set(CROSS_REF_RE.findall(e["answer"])))
        good = [c for c in cited if c in valid_rule_ids]
        bad = [c for c in cited if c not in valid_rule_ids]

        # The `Cards Mentioned` list is authoritative about which cards the
        # question concerns, and it disambiguates names that collide with
        # CR keywords: "Flashback" listed here means the card Flashback
        # ("Target instant or sorcery card in your graveyard gains
        # flashback"), not the keyword ability. Free-text questions have no
        # such signal, so this list should be preferred over scanning the
        # question body whenever it is present.
        # Quoting handles "'Urborg, Tomb of Yawgmoth'", but sources don't
        # always quote — an unquoted "Gideon, Ally of Zendikar" arrives here
        # already split in two. Rejoin runs that only resolve when combined.
        names = e["cards"]
        if card_index is not None:
            names = split_card_list(", ".join(names), card_index)

        resolved_cards = []
        for name in names:
            if card_index is None:
                resolved_cards.append(name)
                continue
            card, how = card_index.resolve(name)
            if card is None:
                warnings.append(f"card {name!r} did not resolve — left as written")
                resolved_cards.append(name)
            else:
                if how != "exact":
                    warnings.append(f"card {name!r} resolved via {how} -> {card['name']!r}")
                resolved_cards.append(card["name"])

        rid = f"qa-{slugify(e['question'])}"
        if rid in existing:
            skipped += 1
            continue

        if not good:
            warnings.append(f"[{rid}] answer cites no valid CR rule — needs a citation before use as gold")
        if bad:
            warnings.append(f"[{rid}] cites {bad} which do not resolve against the pinned CR")

        existing[rid] = {
            "id": rid,
            "question": e["question"],
            "paraphrases": [],
            "answer": e["answer"],
            "key_points": draft_rubric(e["answer"]) if args.draft_rubric else [],
            "common_errors": [],
            "rule_citations": good,
            "rule_citations_unresolved": bad,
            "cards": resolved_cards,
            "category": args.category,
            "difficulty": args.difficulty,
            "source": args.source_label,
            "cr_version": args.cr_version,
            # Everything above can be derived; the rubric and the category
            # are the parts that still need a human before this is gold.
            "needs_rubric": not args.draft_rubric,
            "needs_review": True,
        }
        new += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in existing.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with_cites = sum(1 for r in existing.values() if r.get("rule_citations"))
    print(f"{new} new, {skipped} already present, {len(existing)} total -> {args.out}")
    print(f"{with_cites}/{len(existing)} carry at least one valid CR citation")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings[:20]:
            print(f"  - {w}")
    print("\nNext: fill in key_points/common_errors, set category per record, then run:")
    print("  python scripts/validate_gold.py --to-eval")

if __name__ == "__main__":
    main()
