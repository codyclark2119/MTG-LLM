"""Classify low-rated chat answers by WHICH half failed, and queue them.

Phase 2 step 4. A rating on its own is not actionable: "the bot was wrong"
does not say whether retrieval failed to surface the governing rule or the
model had it and reasoned badly, and those have opposite fixes. That is the
distinction Sections 21.139-21.144 exist to draw, and `chat_server.py` stores
the retrieved context precisely so it can be drawn after the fact.

The classification uses the context that was actually used:

  no_context            nothing was retrieved at all
  invented_citation     cited a rule id that is not in the CR
  ungrounded_citation   cited a REAL rule that was NOT in the context
  grounded              every cited rule WAS in the context
  no_citation           made no rule citation at all

Those are OBSERVATIONS, not verdicts. The same observation means opposite
things depending on the rating — `grounded` on a down-rated answer is a
reasoning failure, and on an up-rated one it is the system working exactly as
designed. `ACTIONS` maps (observation, rating) to what to do, so the category
never has to carry a judgement it cannot support.

`ungrounded_citation` is the category `eval.score_citations` cannot express,
and it is not academic. A real user question ("5/5 deathtouch trample into a
4/4") produced an answer citing **702.7a (Deathtouch)** and **702.13a
(Trample)** with quoted text for both. Both ids are real — they are FIRST
STRIKE and INTIMIDATE — so a fabrication check keyed on existence reports
nothing wrong. A citation that points at a real rule and misstates it is
worse than an invented number, because it survives a spot check.

A down-rated `grounded` answer is the one that must NOT be answered with a
fine-tune. Six
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
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Imported, not reimplemented: this file had its own copy of read_ratings,
# which is the trap CLAUDE.md records as five jsonl readers with the
# blank-line guard in only three of them. Both copies happened to be
# correct, which is luck rather than structure.
from chat_common import read_ratings
from common import RULE_ID_RE, REPO_ROOT, load_rule_ids, write_jsonl_atomic

RATINGS_PATH = REPO_ROOT / "data" / "chat" / "ratings.jsonl"

# What the parser OBSERVED about an answer. Deliberately neutral names: the
# same observation means opposite things depending on the rating, and naming
# the observation after one of them is this repo's "one name, two meanings"
# trap (CROSS_REF_RE, JUDGE_SYSTEM_PROMPT, PASS). `grounded` was originally
# called `reasoning_miss`, which read as a defect — and then labelled an
# UP-rated answer a reasoning miss on the second rating ever collected.
OBSERVATIONS = {
    "no_context": "nothing was retrieved",
    "invented_citation": "cited a rule id that is not in the CR",
    "ungrounded_citation": "cited a real rule that was never shown to it",
    "grounded": "every cited rule was in the retrieved context",
    "no_citation": "cited no rule at all",
}

# The interpretation is (observation, rating) -> action. Only the down-rated
# half names a defect; an up-rated `grounded` answer is the system working.
# Named RATING_ACTIONS, not ACTIONS: `webui.py` already owns that name for
# its script-runner allowlist, and test_imports refuses a vocabulary with
# two homes — the same guard that exists because CROSS_REF_RE and
# JUDGE_SYSTEM_PROMPT each meant two things.
RATING_ACTIONS = {
    ("no_context", "down"): "retrieval — nothing was retrieved; check the question type",
    ("invented_citation", "down"): "corpus/prompt — the model invented a rule id",
    ("ungrounded_citation", "down"): "retrieval — cited a real rule never shown to it",
    ("grounded", "down"): "benchmark — it HAD the rules and reasoned wrong; do NOT fine-tune",
    ("no_citation", "down"): "prompt — the answer cited nothing at all",
    ("grounded", "up"): "working as intended — grounded and useful",
    ("no_context", "up"): "answered correctly from parametric knowledge, unverifiable",
    ("invented_citation", "up"): "WARNING: liked, but the citation is fabricated",
    ("ungrounded_citation", "up"): "liked, but the citation was not shown to it — unverified",
    ("no_citation", "up"): "liked, but uncited — nothing to check it against",
}


def action_for(observation: str, rating: str) -> str:
    return RATING_ACTIONS.get((observation, rating), OBSERVATIONS.get(observation, "?"))


def classify(row: dict, valid_rule_ids: set[str]) -> dict:
    """What the parser can OBSERVE, plus the action its rating implies.

    Ordered most-diagnostic first: an invented citation is a stronger signal
    than an ungrounded one, and both are stronger than "everything it cited was
    in front of it".
    """
    answer = row.get("answer", "")
    context = row.get("context", "") or ""
    cited = sorted(set(RULE_ID_RE.findall(answer)))
    invented = [r for r in cited if r not in valid_rule_ids]
    # A rule counts as retrieved only if its id appears in the text the model
    # was actually handed — the same string comparison the live diagnosis used.
    ungrounded = [r for r in cited if r in valid_rule_ids and r not in context]

    if not context:
        observation = "no_context"
    elif invented:
        observation = "invented_citation"
    elif ungrounded:
        observation = "ungrounded_citation"
    elif cited:
        observation = "grounded"
    else:
        observation = "no_citation"

    return {
        "observation": observation,
        "action": action_for(observation, row.get("rating", "down")),
        "cited": cited,
        "invented": invented,
        "ungrounded": ungrounded,
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
        for obs, n in Counter(d["observation"] for _, d in diagnosed).most_common():
            print(f"  {n:3d}  {obs:21s} {action_for(obs, 'down')}")

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
            print(f"\n{'=' * 72}\n[{row.get('rating')}] {d['observation']}  "
                  f"k={d['k_rules']}  context={d['context_chars']}ch  cards={d['cards']}")
            print(f"  -> {d['action']}")
            print(f"Q: {row.get('question', '')[:300]}")
            if d["invented"]:
                print(f"  INVENTED rule ids: {d['invented']}")
            if d["ungrounded"]:
                print(f"  cited but NEVER SHOWN to it: {d['ungrounded']}")
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
