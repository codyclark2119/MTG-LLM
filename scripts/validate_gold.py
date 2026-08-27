"""Validate the judge-authored gold set and convert it to eval format.

The gold set is the only human-authored data in the project, so mistakes
in it are expensive: a wrong rule citation here silently becomes the
"correct" answer everything else is scored against. This checks the parts
a machine can check — rule IDs resolve against the pinned CR, card names
resolve against the Oracle pool, categories are spelled the way the rest
of the pipeline expects, ids are unique — and reports rows by id so they
can be handed straight back to a contributor.

It also converts to the record shape scripts/eval.py consumes, expanding
`paraphrases` into their own eval rows that share the source question's
rubric (see data/gold/SCHEMA.md).

Usage:
    python scripts/validate_gold.py
    python scripts/validate_gold.py --to-eval
    python scripts/validate_gold.py --from-csv intake.csv   # judge spreadsheet -> jsonl
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from common import (GOLD_PATH, REPO_ROOT, RULES_PATH, SLOT_RE, SYSTEM_PROMPT,
                    load_rule_ids, untemplatize)
from common import RULE_ID_EXACT_RE as CROSS_REF_RE

# Re-exported, not redefined (Section 21.97). Previously a SET here and a LIST
# in label_store — same members, two shapes, two places to edit.
from common import CATEGORIES, DIFFICULTIES  # noqa: E402,F401
from card_lookup import names_a_card  # noqa: E402
REQUIRED = ["id", "question", "answer", "key_points", "rule_citations", "category", "difficulty", "source", "cr_version"]
LIST_FIELDS = ["paraphrases", "key_points", "common_errors", "rule_citations", "cards"]


def from_csv(csv_path: Path, out_path: Path) -> None:
    """Convert a judge-filled spreadsheet to JSONL.

    Semicolons separate list items — commas appear inside card names and
    rules prose too often to be safe as a delimiter.
    """
    rows = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            rec = {}
            for k, v in raw.items():
                if k is None:
                    continue
                key = k.strip()
                val = (v or "").strip()
                if key in LIST_FIELDS:
                    rec[key] = [p.strip() for p in val.split(";") if p.strip()]
                elif val:
                    rec[key] = val
            if rec.get("id"):
                rows.append(rec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"converted {len(rows)} rows -> {out_path}")


def validate(records: list[dict], valid_rule_ids: set[str], card_index) -> list[str]:
    problems: list[str] = []
    seen_ids: set[str] = set()

    for i, r in enumerate(records, 1):
        rid = r.get("id") or f"<row {i}, no id>"

        for field in REQUIRED:
            if not r.get(field):
                problems.append(f"[{rid}] missing required field: {field}")

        if r.get("id"):
            if r["id"] in seen_ids:
                problems.append(f"[{rid}] duplicate id")
            seen_ids.add(r["id"])

        for field in LIST_FIELDS:
            if field in r and not isinstance(r[field], list):
                problems.append(f"[{rid}] {field} must be a list")

        cat = r.get("category")
        if cat and cat not in CATEGORIES:
            problems.append(f"[{rid}] unknown category {cat!r} (expected one of: {', '.join(sorted(CATEGORIES))})")

        diff = r.get("difficulty")
        if diff and diff not in DIFFICULTIES:
            problems.append(f"[{rid}] difficulty must be one of {sorted(DIFFICULTIES)}, got {diff!r}")

        for rule in r.get("rule_citations", []) or []:
            if not CROSS_REF_RE.match(rule):
                problems.append(f"[{rid}] malformed rule id {rule!r} (expected like 704.5g)")
            elif rule not in valid_rule_ids:
                problems.append(f"[{rid}] rule {rule} does not exist in the pinned CR — check the number")

        if card_index is not None:
            for name in r.get("cards", []) or []:
                card, how = card_index.resolve(name)
                if card is None:
                    problems.append(f"[{rid}] card {name!r} did not resolve ({how})")
                elif not names_a_card(how):
                    problems.append(f"[{rid}] card {name!r} resolved only via {how} -> {card['name']!r}; use the exact name")

        kp = r.get("key_points") or []
        if kp and len(kp) < 2:
            problems.append(f"[{rid}] only {len(kp)} key_point — rubric scoring needs at least 2 to be meaningful")

        # A slot with no entry in card_slots survives expansion and reaches the
        # judge as literal "[[card2]]", which no answer can ever match. That
        # silently costs the record a key point rather than failing loudly, so
        # it is caught here instead.
        slots = r.get("card_slots") or {}
        dangling = {m for line in kp + (r.get("common_errors") or [])
                    for m in SLOT_RE.findall(line)} - set(slots)
        if dangling:
            problems.append(f"[{rid}] rubric references {sorted(dangling)} with no card_slots entry "
                            "— it would reach the judge as literal text")

    return problems


def to_eval_records(records: list[dict]) -> list[dict]:
    """Gold records in the shape scripts/eval.py consumes.

    **Slots are expanded here.** The gold set stores rubrics templated —
    `"[[card2]] enters tapped"` — so one rubric can score every instantiation
    RulesGuru builds of the same ruling. The judge is a different audience: it
    is asked which of these claims the answer made, and an answer says "Urborg,
    Tomb of Yawgmoth enters tapped". A claim naming `[[card2]]` matches nothing
    an answer can contain, so leaving the template in deflates the score of a
    correct answer — silently, and only on the 35 of 39 records that carry a
    slot.

    This is the `[TARGET <x>]` failure again (Section 16.x): meta-syntax handed
    to a model that reads it as literal text. `--ingest-submissions` was fixed
    for the same reason; this path was missed because a templated rubric is
    still valid JSON and still scores — just lower.

    `card_slots` rides along on the eval row so a future variant run can
    re-template against a *different* instantiation's cards, which is the whole
    point of storing slots rather than names.
    """
    out = []
    for r in records:
        slots = r.get("card_slots") or {}
        key_points = [untemplatize(x, slots) for x in r.get("key_points") or []]
        common_errors = [untemplatize(x, slots) for x in r.get("common_errors") or []]
        # Each paraphrase becomes its own eval row sharing the rubric, so
        # wording robustness is measured without re-authoring judgement.
        for variant, q in enumerate([r["question"], *(r.get("paraphrases") or [])]):
            out.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": r["answer"]},
                    ],
                    "source": f"gold:{r['source']}",
                    "gold_id": r["id"],
                    "variant": "canonical" if variant == 0 else f"paraphrase-{variant}",
                    "category": r["category"],
                    "difficulty": r["difficulty"],
                    "key_points": key_points,
                    "common_errors": common_errors,
                    "card_slots": slots,
                    "supporting_rule_ids": r.get("rule_citations", []),
                    "cited_rule_ids": r.get("rule_citations", []),
                    "retrieved_rule_ids": [],
                    "cards": r.get("cards", []),
                    "cr_version": r["cr_version"],
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--rules", type=Path, default=RULES_PATH)
    parser.add_argument("--from-csv", type=Path, default=None, help="convert a judge spreadsheet to JSONL first")
    parser.add_argument("--to-eval", action="store_true", help="write eval-format records")
    parser.add_argument("--eval-out", type=Path, default=REPO_ROOT / "eval/sets/gold_questions_eval.jsonl")
    parser.add_argument("--skip-cards", action="store_true", help="skip card-name validation (faster)")
    args = parser.parse_args()

    if args.from_csv:
        from_csv(args.from_csv, args.gold)

    if not args.gold.exists():
        print(f"{args.gold} does not exist yet — see data/gold/SCHEMA.md", file=sys.stderr)
        sys.exit(1)

    with args.gold.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        print(f"{args.gold} is empty", file=sys.stderr)
        sys.exit(1)

    valid_rule_ids = load_rule_ids(args.rules)

    card_index = None
    if not args.skip_cards and any(r.get("cards") for r in records):
        sys.path.insert(0, str(Path(__file__).parent))
        from card_lookup import CardIndex

        card_index = CardIndex()

    problems = validate(records, valid_rule_ids, card_index)

    from collections import Counter

    cats = Counter(r.get("category") for r in records)
    diffs = Counter(r.get("difficulty") for r in records)
    n_para = sum(len(r.get("paraphrases") or []) for r in records)

    print(f"{len(records)} gold records ({n_para} paraphrases -> {len(records) + n_para} eval rows)")
    print("by category:")
    for c in sorted(CATEGORIES):
        n = cats.get(c, 0)
        flag = "  <- under target (8)" if n < 8 else ""
        print(f"  {c}: {n}{flag}")
    print("by difficulty:", dict(diffs))

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    print("\nvalidation passed")

    if args.to_eval:
        rows = to_eval_records(records)
        args.eval_out.parent.mkdir(parents=True, exist_ok=True)
        with args.eval_out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(rows)} eval rows -> {args.eval_out}")

if __name__ == "__main__":
    main()
