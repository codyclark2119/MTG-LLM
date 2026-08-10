"""Parse the raw Comprehensive Rules text into structured JSONL records.

Implements README Section 4: preserves the CR's section/rule/subrule
hierarchy (4.1), cleans and cross-reference-annotates the text (4.2),
and validates the parse before anything downstream depends on it (4.3).

Usage:
    python scripts/ingest.py [--raw PATH] [--out-dir DIR]
"""

import argparse
import json
import re
import sys
from pathlib import Path

SECTION_RE = re.compile(r"^([1-9])\.\s+(.+)$")
GROUP_RE = re.compile(r"^(\d{3})\.\s+([A-Za-z].+)$")
# Handles both "509.1. Text" and typo'd variants missing the trailing
# period ("606.5 Text") or adding one after the subrule letter ("119.1d. Text").
RULE_RE = re.compile(r"^(\d{3}\.\d+)([a-z]*)\.?\s+(.+)$")
CROSS_REF_RE = re.compile(r"\b\d{3}\.\d+[a-z]?\b")
REDIRECT_RE = re.compile(r"^See [^.]+\.$")

HEADING_START = "1. Game Concepts"
GLOSSARY_HEADING = "Glossary"
CREDITS_HEADING = "Credits"

WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(s: str) -> str:
    # The source occasionally embeds stray U+2028 line separators and runs
    # of U+00A0 non-breaking spaces mid-sentence (a PDF-conversion artifact).
    # Python's \s matches both, so this collapses them to a single space.
    return WHITESPACE_RE.sub(" ", s).strip()


def find_body_bounds(lines: list[str]) -> tuple[int, int, int]:
    """Locate the real section/glossary/credits boundaries, skipping the
    table of contents copies of the same headings near the top of the file."""
    toc_credits = lines.index(CREDITS_HEADING)  # the ToC entry comes first
    body_start = lines.index(HEADING_START, toc_credits)
    glossary_start = lines.index(GLOSSARY_HEADING, body_start)
    credits_start = lines.index(CREDITS_HEADING, glossary_start)
    return body_start, glossary_start, credits_start


def parse_rules(lines: list[str]) -> list[dict]:
    records: list[dict] = []
    by_id: dict[str, dict] = {}
    current_section = None
    current_section_title = None
    current: dict | None = None

    for raw_line in lines:
        line = normalize_whitespace(raw_line)
        if not line:
            continue

        m = SECTION_RE.match(line)
        if m:
            current_section, current_section_title = m.group(1), m.group(2)
            current = None
            continue

        if GROUP_RE.match(line):
            # Rule-group header (e.g. "509. Combat Phase"); not a record
            # itself, just context for the rules that follow it.
            current = None
            continue

        m = RULE_RE.match(line)
        if m:
            number, letters, text = m.groups()
            rule_id = number + letters
            parent_rule = number if letters else number.split(".")[0]
            record = {
                "rule_id": rule_id,
                "section": current_section,
                "section_title": current_section_title,
                "parent_rule": parent_rule,
                "text": text,
                "cross_refs": [],  # filled in after all rules are known
                "glossary_terms": [],  # filled in once the glossary is parsed
            }
            records.append(record)
            by_id[rule_id] = record
            current = record
            continue

        # Continuation line (e.g. an "Example:" line following its rule).
        if current is not None:
            current["text"] += "\n" + line
        else:
            print(f"warning: unclassified line with no active rule: {line[:80]!r}", file=sys.stderr)

    return records


def parse_glossary(lines: list[str]) -> list[dict]:
    entries: list[dict] = []
    block: list[str] = []

    def flush(block: list[str]) -> None:
        if not block:
            return
        term, *definition_lines = block
        entries.append({"term": term, "definition": " ".join(definition_lines)})

    for raw_line in lines:
        line = normalize_whitespace(raw_line)
        if not line:
            flush(block)
            block = []
            continue
        block.append(line)
    flush(block)

    return entries


def annotate_cross_refs(rules: list[dict]) -> None:
    for record in rules:
        refs = {m for m in CROSS_REF_RE.findall(record["text"]) if m != record["rule_id"]}
        record["cross_refs"] = sorted(refs)


def annotate_glossary_terms(rules: list[dict], glossary: list[dict]) -> None:
    # Pure "See Other Term." redirects add no independent definition, and
    # the short/common ones (e.g. "If" -> "See Intervening 'If' Clause.")
    # collide with ordinary English words, tagging a huge fraction of rules
    # for no benefit. Exclude redirects as tag anchors; they still appear
    # in glossary.jsonl for lookup.
    taggable = [g["term"] for g in glossary if not REDIRECT_RE.match(g["definition"])]
    # Longest term first so multi-word terms ("Activated Ability") win over
    # single-word substrings ("Ability") when both would match.
    terms = sorted(taggable, key=len, reverse=True)
    pattern = re.compile(
        "|".join(rf"\b{re.escape(t)}\b" for t in terms), re.IGNORECASE
    )
    canonical = {t.lower(): t for t in terms}
    for record in rules:
        found = {m.group(0) for m in pattern.finditer(record["text"])}
        record["glossary_terms"] = sorted({canonical[f.lower()] for f in found})

    for entry in glossary:
        entry["related_rules"] = sorted(
            {m for m in CROSS_REF_RE.findall(entry["definition"])}
        )


def validate(rules: list[dict], glossary: list[dict], sections: set[str]) -> list[str]:
    problems: list[str] = []
    by_id = {r["rule_id"]: r for r in rules}

    if len(sections) != 9:
        problems.append(f"expected 9 top-level sections, found {len(sections)}: {sorted(sections)}")

    missing_parents = [
        r["rule_id"] for r in rules
        if r["rule_id"][-1].isalpha() and r["parent_rule"] not in by_id
    ]
    if missing_parents:
        problems.append(f"{len(missing_parents)} subrules reference a missing parent, e.g. {missing_parents[:5]}")

    orphan_refs = set()
    for r in rules:
        for ref in r["cross_refs"]:
            if ref not in by_id:
                orphan_refs.add(ref)
    if orphan_refs:
        problems.append(
            f"{len(orphan_refs)} distinct cross-referenced rule IDs don't resolve to a parsed rule, "
            f"e.g. {sorted(orphan_refs)[:5]} (some may point at the Tournament Rules doc, not the CR)"
        )

    if not glossary:
        problems.append("glossary parsed to zero entries")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/MagicCompRules_20260807.txt"),
        help="Path to the raw Comprehensive Rules .txt file",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/processed"),
        help="Directory to write rules.jsonl and glossary.jsonl into",
    )
    args = parser.parse_args()

    text = args.raw.read_text(encoding="utf-8")
    lines = text.split("\n")

    body_start, glossary_start, credits_start = find_body_bounds(lines)

    rules = parse_rules(lines[body_start:glossary_start])
    glossary = parse_glossary(lines[glossary_start + 1 : credits_start])

    annotate_cross_refs(rules)
    annotate_glossary_terms(rules, glossary)

    sections = {r["section"] for r in rules}
    problems = validate(rules, glossary, sections)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rules_path = args.out_dir / "rules.jsonl"
    glossary_path = args.out_dir / "glossary.jsonl"

    with rules_path.open("w", encoding="utf-8") as f:
        for r in rules:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with glossary_path.open("w", encoding="utf-8") as f:
        for g in glossary:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print(f"parsed {len(rules)} rules across {len(sections)} sections -> {rules_path}")
    print(f"parsed {len(glossary)} glossary entries -> {glossary_path}")
    if problems:
        print("\nvalidation issues:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)
    else:
        print("validation passed: sections complete, all subrules parented, no orphaned cross-refs")


if __name__ == "__main__":
    main()
