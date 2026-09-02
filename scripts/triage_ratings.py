"""Classify low-rated chat answers by WHICH half failed, and queue them.

Phase 2 step 4. A rating on its own is not actionable: "the bot was wrong"
does not say whether retrieval failed to surface the governing rule or the
model had it and reasoned badly, and those have opposite fixes. That is the
distinction Sections 21.139-21.144 exist to draw, and `chat_server.py` stores
the retrieved context precisely so it can be drawn after the fact.

The classification uses the context that was actually used:

  no_context            nothing was retrieved at all
  nonexistent_citation  cited a rule id that is not in the CR
  unretrieved_citation  cited a REAL rule that was NOT in the context
  reasoning_miss        every cited rule WAS in the context
  no_citation           made no rule citation at all

`unretrieved_citation` is the category `eval.score_citations` cannot express,
and it is not academic. A real user question ("5/5 deathtouch trample into a
4/4") produced an answer citing **702.7a (Deathtouch)** and **702.13a
(Trample)** with quoted text for both. Both ids are real — they are FIRST
STRIKE and INTIMIDATE — so a fabrication check keyed on existence reports
nothing wrong. A citation that points at a real rule and misstates it is
worse than an invented number, because it survives a spot check.

`reasoning_miss` is the one that must NOT be answered with a fine-tune. Six
attempts scored below the base model (21.139-21.140), and few-shot failed too
(21.143), so the standing response to a reasoning miss is to record it as a
benchmark candidate, not to queue training. Retrieval misses are the ones
with a known lever.

Usage:
    python scripts/triage_ratings.py
    python scripts/triage_ratings.py --show down
    python scripts/triage_ratings.py --export-queue data/chat/review_queue.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import RULE_ID_RE, REPO_ROOT, load_rule_ids, write_jsonl_atomic

RATINGS_PATH = REPO_ROOT / "data" / "chat" / "ratings.jsonl"

CATEGORY_ACTION = {
    "no_context": "retrieval — nothing was retrieved; check the question type",
    "nonexistent_citation": "corpus/prompt — the model invented a rule id",
    "unretrieved_citation": "retrieval — cited a real rule that was never shown to it",
    "reasoning_miss": "benchmark — it HAD the rules and reasoned wrong; do NOT fine-tune",
    "no_citation": "prompt — the answer cited nothing at all",
}


def read_ratings(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def classify(row: dict, valid_rule_ids: set[str]) -> dict:
    """Which half failed, decided from the answer and the stored context.

    Ordered most-diagnostic first: a nonexistent citation is a stronger signal
    than an unretrieved one, and both are stronger than "it had everything and
    still got it wrong".
    """
    answer = row.get("answer", "")
    context = row.get("context", "") or ""
    cited = sorted(set(RULE_ID_RE.findall(answer)))
    nonexistent = [r for r in cited if r not in valid_rule_ids]
    # A rule counts as retrieved only if its id appears in the text the model
    # was actually handed — the same string comparison the live diagnosis used.
    unretrieved = [r for r in cited if r in valid_rule_ids and r not in context]

    if not context:
        category = "no_context"
    elif nonexistent:
        category = "nonexistent_citation"
    elif unretrieved:
        category = "unretrieved_citation"
    elif cited:
        category = "reasoning_miss"
    else:
        category = "no_citation"

    return {
        "category": category,
        "action": CATEGORY_ACTION[category],
        "cited": cited,
        "nonexistent": nonexistent,
        "unretrieved": unretrieved,
        "context_chars": len(context),
        "cards": row.get("cards", []),
        "k_rules": (row.get("config") or {}).get("k_rules"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", type=Path, default=RATINGS_PATH)
    ap.add_argument("--show", choices=("down", "up", "all"), default=None,
                    help="print each matching answer with its diagnosis")
    ap.add_argument("--export-queue", type=Path, default=None,
                    help="write down-rated rows plus their diagnosis for review")
    args = ap.parse_args()

    rows = read_ratings(args.ratings)
    if not rows:
        print(f"no ratings in {args.ratings}")
        return
    valid = load_rule_ids()

    up = [r for r in rows if r.get("rating") == "up"]
    down = [r for r in rows if r.get("rating") == "down"]
    print(f"{len(rows)} rating(s): {len(up)} up, {len(down)} down "
          f"({len(down) / len(rows):.0%} negative)")

    diagnosed = [(r, classify(r, valid)) for r in down]
    if diagnosed:
        print("\ndown-rated by failure type:")
        for cat, n in Counter(d["category"] for _, d in diagnosed).most_common():
            print(f"  {n:3d}  {cat:22s} {CATEGORY_ACTION[cat]}")

    by_k = Counter((r.get("config") or {}).get("k_rules") for r in rows)
    if len(by_k) > 1:
        print("\nby k_rules (each answer records the config it was generated under):")
        for k, n in sorted(by_k.items(), key=lambda kv: (kv[0] is None, kv[0])):
            d = sum(1 for r in rows
                    if (r.get("config") or {}).get("k_rules") == k
                    and r.get("rating") == "down")
            print(f"  k={k}: {n} rated, {d} down ({d / n:.0%})")

    if args.show:
        want = {"down": ["down"], "up": ["up"], "all": ["up", "down"]}[args.show]
        for row in [r for r in rows if r.get("rating") in want]:
            d = classify(row, valid)
            print(f"\n{'=' * 72}\n[{row.get('rating')}] {d['category']}  "
                  f"k={d['k_rules']}  context={d['context_chars']}ch  cards={d['cards']}")
            print(f"  -> {d['action']}")
            print(f"Q: {row.get('question', '')[:300]}")
            if d["nonexistent"]:
                print(f"  INVENTED rule ids: {d['nonexistent']}")
            if d["unretrieved"]:
                print(f"  cited but NEVER SHOWN to it: {d['unretrieved']}")
            if row.get("note"):
                print(f"  user note: {row['note']}")

    if args.export_queue:
        out = [{**row, "diagnosis": d} for row, d in diagnosed]
        write_jsonl_atomic(args.export_queue, out)
        print(f"\nwrote {len(out)} row(s) to {args.export_queue}")
        print("  Review, then promote genuinely-hard questions into the benchmark by hand.")
        print("  A reasoning_miss is NOT a training signal — see the module docstring.")


if __name__ == "__main__":
    main()
