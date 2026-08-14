"""Convert the RulesGuru snapshot into gold-schema candidate records.

What this data is, precisely — because the distinction decides where it
may be used:

  the ANSWER is human-authored and rules-verified. RulesGuru publishes
    only questions that clear a review queue (the site reports ~1,500
    finished against ~5,600 pending), and 89% ship CR citations inline.
    This is genuinely better ground truth than anything else here except
    the judge-authored records.

  the RUBRIC is not. `key_points` is drafted by sentence-splitting the
    answer, and `category` is inferred from the site's topic tags. Both
    are machine guesses at the two fields the gold set exists to get
    right.

So these land in a separate candidate file rather than in
`gold_questions.jsonl`. Promoting one is a human act: read the drafted
rubric, fix it, then `--promote`. Bulk-importing them as gold would
dissolve the one property that makes the gold set worth having — that a
person vouched for every rubric it scores against.

Even unpromoted, the candidates are immediately useful as an eval set:
1,400 human-written questions with verified citations is a far better
test than the synthetic set (65% of which retrieves its own source chunk)
or the Reddit set (only 21% of which cites any rule at all).

Usage:
    python scripts/rulesguru_to_gold.py
    python scripts/rulesguru_to_gold.py --promote 3518 6 102
"""

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from common import CR_VERSION, GOLD_CANDIDATES_PATH, GOLD_PATH, REPO_ROOT, RULESGURU_SNAPSHOT, RULES_PATH, load_rule_ids
from common import RULE_ID_EXACT_RE as CROSS_REF_RE

sys.path.insert(0, str(Path(__file__).parent))
from ingest_qa_pastes import draft_rubric  # noqa: E402  (same rubric drafting as the paste path)

# Tag -> category, most diagnostic first: questions carry several tags and
# the first match wins. "Layers" on a combat question still makes it a
# layer question; "Combat" on a layers question does not make it a combat
# question. Ordering encodes that asymmetry.
CATEGORY_RULES: list[tuple[str, set[str]]] = [
    ("state-based actions", {"State-based actions"}),
    (
        "layer-system question",
        {
            "Layers", "Dependency", "Dependency loops", "Type-changing effects",
            "Text-changing effects", "Color-changing effects", "Control-changing effects",
            "Characteristic-defining abilities", "Continuous effects",
        },
    ),
    (
        "turn-structure walkthrough",
        {"Turn structure", "Turn-based actions", "Cleanup Step", "Starting the game"},
    ),
    (
        "priority reasoning",
        {
            "Timing and priority", "The stack", "Resolving objects", "Special actions",
            "Intervening If", "Delayed Triggers",
        },
    ),
    (
        "zone transition",
        {"Zone-changes", "Graveyard", "Libraries", "Leaving the game", "Phasing", "Discard"},
    ),
    (
        "templating/keyword meaning",
        {
            "Evergreen keywords", "Non-evergreen keywords", "Reading cards",
            "Card text interpretation", "Card names", "Regenerate", "Banding", "Bestow",
            "Numbers and symbols", "Mana value", "Subtypes", "Types", "Variables",
        },
    ),
]
DEFAULT_CATEGORY = "interaction puzzle"

# The site's own difficulty label. Levels 0-3 are described as intro /
# common-tournament / rarer / very rare, with "Corner Case" above them.
LEVEL_TO_DIFFICULTY = {
    "0": "basic",
    "1": "intermediate",
    "2": "intermediate",
    "3": "advanced",
    "Corner Case": "advanced",
}


def categorize(tags: list[str]) -> tuple[str, bool]:
    """Return (category, matched_a_tag). Unmatched falls to the default."""
    tagset = set(tags)
    for category, triggers in CATEGORY_RULES:
        if tagset & triggers:
            return category, True
    return DEFAULT_CATEGORY, False


def difficulty_for(level: str, complexity: str) -> str:
    difficulty = LEVEL_TO_DIFFICULTY.get(level, "intermediate")
    # Complexity is about how many objects must be tracked, not how obscure
    # the rule is; a "Complicated" board state is not a basic question even
    # when the underlying rule is.
    if complexity == "Complicated" and difficulty == "basic":
        difficulty = "intermediate"
    return difficulty

PLAYER_NAME_RE = re.compile(r"\b[A-Z][a-z]+\b")


def dedupe_key(text: str, card_names: list[str]) -> str:
    """Strip the parts the API randomizes so the same ruling compares equal.

    Card names and player names are re-rolled per request, so two fetches
    of one question share only their skeleton. Masking both is what lets a
    hand-pasted record be matched against its API-fetched twin.
    """
    masked = text
    for name in sorted(card_names, key=len, reverse=True):
        masked = masked.replace(name, " CARD ")
    masked = PLAYER_NAME_RE.sub(" NAME ", masked)
    masked = re.sub(r"[^a-z ]+", " ", masked.lower())
    return re.sub(r"\s+", " ", masked).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, default=RULESGURU_SNAPSHOT)
    parser.add_argument("--out", type=Path, default=GOLD_CANDIDATES_PATH)
    parser.add_argument("--needs-work", type=Path, default=REPO_ROOT / "data/gold/rulesguru/needs_work.jsonl")
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--rules", type=Path, default=RULES_PATH)
    parser.add_argument("--cr-version", default=CR_VERSION)
    parser.add_argument("--min-key-points", type=int, default=2, help="validator requires 2+ to score a rubric")
    parser.add_argument("--promote", type=int, nargs="+", default=[], help="RulesGuru ids to append to the gold set")
    parser.add_argument("--skip-cards", action="store_true")
    args = parser.parse_args()

    if not args.snapshot.exists():
        raise SystemExit(f"{args.snapshot} not found — run scripts/fetch_rulesguru.py first")

    with args.snapshot.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    print(f"{len(questions)} questions in the snapshot")

    valid_rule_ids = load_rule_ids(args.rules)

    card_index = None
    if not args.skip_cards:
        from card_lookup import CardIndex

        card_index = CardIndex()

    existing_gold: list[dict] = []
    if args.gold.exists():
        with args.gold.open(encoding="utf-8") as f:
            existing_gold = [json.loads(line) for line in f if line.strip()]
    gold_keys = {dedupe_key(r["answer"], r.get("cards") or []): r["id"] for r in existing_gold}

    candidates, needs_work = [], []
    unresolved_cards: Counter = Counter()
    dropped_citations = 0
    overlaps: list[tuple[str, str]] = []

    for q in questions:
        qid = q["id"]
        answer_cited = (q.get("answerSimpleCited") or "").strip()
        answer_plain = (q.get("answerSimple") or "").strip()
        question_text = (q.get("questionSimple") or "").strip()
        if not question_text or not answer_cited:
            continue

        cited = [r for r in (q.get("citedRules") or {}) if CROSS_REF_RE.match(r)]
        good = sorted({r for r in cited if r in valid_rule_ids})
        bad = sorted({r for r in cited if r not in valid_rule_ids})
        dropped_citations += len(bad)

        cards = []
        for c in q.get("includedCards") or []:
            name = c.get("name")
            if not name:
                continue
            if card_index is not None:
                card, how = card_index.resolve(name)
                if card is None:
                    unresolved_cards[name] += 1
                    continue  # a name the eval can't ground is worse than no name
                name = card["name"]
            cards.append(name)

        category, tag_matched = categorize(q.get("tags") or [])
        key_points = draft_rubric(answer_plain or answer_cited)

        # Flag anything that already exists in the hand-curated gold set, so
        # the same ruling is not scored twice under two different card names.
        key = dedupe_key(answer_plain or answer_cited, cards)
        for gk, gid in gold_keys.items():
            if difflib.SequenceMatcher(None, key, gk).ratio() >= 0.72:
                overlaps.append((f"rg-{qid}", gid))
                break

        rec = {
            "id": f"rg-{qid}",
            "question": question_text,
            "paraphrases": [],
            "answer": answer_cited,
            "key_points": key_points,
            "common_errors": [],
            "rule_citations": good,
            "rule_citations_unresolved": bad,
            "cards": cards,
            "category": category,
            "difficulty": difficulty_for(q.get("level", ""), q.get("complexity", "")),
            "source": f"rulesguru:{qid}",
            "cr_version": args.cr_version,
            # Provenance kept so a reviewer can open the question and so the
            # inferred category can be re-derived if the mapping changes.
            "rulesguru_id": qid,
            "rulesguru_url": q.get("rulesguru_url", f"https://rulesguru.org/?{qid}"),
            "rulesguru_level": q.get("level"),
            "rulesguru_complexity": q.get("complexity"),
            "rulesguru_tags": q.get("tags") or [],
            "category_inferred": True,
            "category_tag_matched": tag_matched,
            # The answer is verified upstream; the rubric and category are not.
            "needs_rubric_review": True,
            "needs_review": True,
        }

        # The validator refuses a one-point rubric because a single point
        # cannot distinguish a partially correct answer from a wrong one.
        if len(key_points) < args.min_key_points or not good:
            rec["blocked_because"] = (
                "no valid CR citation" if not good else f"only {len(key_points)} drafted key point(s)"
            )
            needs_work.append(rec)
        else:
            candidates.append(rec)

    for path, rows in ((args.out, candidates), (args.needs_work, needs_work)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{len(candidates)} candidates -> {args.out}")
    print(f"{len(needs_work)} held back -> {args.needs_work}")
    if needs_work:
        reasons = Counter(r["blocked_because"] for r in needs_work)
        for reason, n in reasons.most_common():
            print(f"    {n}: {reason}")

    cats = Counter(r["category"] for r in candidates)
    print("\nby category (inferred — review before trusting):")
    for c, _ in CATEGORY_RULES:
        print(f"  {c}: {cats.get(c, 0)}")
    print(f"  {DEFAULT_CATEGORY}: {cats.get(DEFAULT_CATEGORY, 0)}")
    untagged = sum(1 for r in candidates if not r["category_tag_matched"])
    print(f"  ({untagged} fell through to the default rather than matching a tag)")
    print("by difficulty:", dict(Counter(r["difficulty"] for r in candidates)))

    if dropped_citations:
        print(f"\n{dropped_citations} cited rule(s) did not resolve against the pinned CR {args.cr_version}")
    if unresolved_cards:
        print(f"{sum(unresolved_cards.values())} card mention(s) across {len(unresolved_cards)} name(s) "
              f"did not resolve and were dropped, e.g. {[n for n, _ in unresolved_cards.most_common(5)]}")
    if overlaps:
        print(f"\n{len(overlaps)} candidate(s) look like rulings already in the hand-curated gold set:")
        for a, b in overlaps[:10]:
            print(f"  - {a} ~ {b}")

    if args.promote:
        by_id = {r["rulesguru_id"]: r for r in candidates + needs_work}
        have = {r["id"] for r in existing_gold}
        promoted = 0
        for qid in args.promote:
            rec = by_id.get(qid)
            if rec is None:
                print(f"  cannot promote {qid}: not in the snapshot")
                continue
            if rec["id"] in have:
                print(f"  {rec['id']} already in the gold set")
                continue
            existing_gold.append(rec)
            promoted += 1
        with args.gold.open("w", encoding="utf-8") as f:
            for r in existing_gold:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\npromoted {promoted} record(s) -> {args.gold}")
        print("Review key_points/common_errors on each, then: python scripts/validate_gold.py --to-eval")
    else:
        print("\nNext:")
        print(f"  python scripts/validate_gold.py --gold {args.out} \\")
        print("      --to-eval --eval-out eval/sets/rulesguru_candidates.jsonl")

if __name__ == "__main__":
    main()
