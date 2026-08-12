"""Fetch rules Q&A from the RulesGuru public API into a frozen snapshot.

RulesGuru (https://rulesguru.org) is a curated database of human-authored
MTG rules questions, each verified before publication and each carrying
its Comprehensive Rules citations inline. It is the source the pasted
questions in `data/gold/pastes/` were being copied from by hand; this
script replaces the copy-paste round trip with the site's documented
public API (https://rulesguru.org/api/documentation/).

Why this data is worth the plumbing: it is the only corpus in the project
that is simultaneously human-written, rules-verified, citation-bearing,
and *labeled* — every question ships a difficulty level, a complexity
rating, and a topic tag set. Those labels map onto the eval categories
directly, which means the long-standing turn-structure blind spot can be
filled by querying for it rather than hoping it turns up.

THE SNAPSHOT IS FROZEN, AND THAT IS NOT AN OPTIMIZATION
-------------------------------------------------------
The API randomizes player names *and the cards themselves* on every
request. Refetching question #3518 twice returns the same ruling about
two different cards:

    Nickolas controls Trinisphere. Amiya casts Surgical Extraction ...
    Nikolas   controls Trinisphere. Alice casts Mental Misstep      ...

Question ids are stable; question *text* is not. So a record is written
once and never rewritten. If refetching were allowed to overwrite, any
rubric a human had already authored against the stored wording would
silently start describing a question that no longer exists — the exact
failure the gold set is meant to be immune to. Re-runs are additive only.

Rate limiting: the API documents one request per 2 seconds, and notes
that matching questions is computationally expensive. `--count` batches
up to 250 per request, so the whole ~1,500-question corpus costs well
under a minute of the host's time. Be a good guest; this is a hobby site.

Usage:
    python scripts/fetch_rulesguru.py --max 200
    python scripts/fetch_rulesguru.py --all
    python scripts/fetch_rulesguru.py --tags "Turn structure" "Cleanup Step" --max 60
"""

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_URL = "https://rulesguru.org/api/questions/"
# The API asks callers to identify themselves. Doing so is free and it is
# how the maintainer learns the endpoint is worth keeping up.
APP_NAME = "magic-llm-research"
USER_AGENT = f"{APP_NAME}/0.1 (personal MTG rules research)"

ALL_LEVELS = ["0", "1", "2", "3", "Corner Case"]
ALL_COMPLEXITIES = ["Simple", "Intermediate", "Complicated"]

# Excluded from a rules corpus on purpose:
#   Unsupported answers — the site's own marker for answers the CR does
#     not settle. Unusable as ground truth by definition.
#   Silly / Trick question — deliberately misleading or joke framings.
#   Trivia — tests knowledge of Magic history, not the rules.
# Applied client-side rather than server-side because the API accepts a
# single `tagsConjunc` for the whole `tags` list, so an OR-query for a
# topic cannot simultaneously carry a NOT-query for these.
DEFAULT_EXCLUDE_TAGS = ["Unsupported answers", "Silly", "Trick question", "Trivia"]


# Card fields worth freezing alongside the question. The API embeds the
# full MTGJSON object per card — every printing, every set code, every
# format legality — which is ~60% of the payload and entirely redundant
# with the Oracle pool already in data/cards/. The rest is dropped:
# questionHTML/answerHTML are markup variants of the Simple fields, and
# citedRules caches CR text this repo has pinned locally. Full responses
# stay in --raw-dir for anyone who wants them.
CARD_FIELDS = {
    "name", "manaCost", "manaValue", "type", "types", "subtypes", "supertypes",
    "text", "rulesText", "power", "toughness", "loyalty", "layout", "keywords",
}


def slim_question(q: dict) -> dict:
    q = dict(q)
    q["includedCards"] = [
        {k: v for k, v in c.items() if k in CARD_FIELDS} for c in q.get("includedCards") or []
    ]
    cited = q.get("citedRules")
    # Reduce to rule ids; downstream iterates this, which works the same
    # for a list of ids as it did for the dict keyed by them.
    q["citedRules"] = sorted(cited.keys()) if isinstance(cited, dict) else (cited or [])
    q.pop("questionHTML", None)
    q.pop("answerHTML", None)
    return q


def api_get(settings: dict, timeout: int = 120) -> list[dict]:
    query = urllib.parse.quote(json.dumps(settings, separators=(",", ":")))
    req = urllib.request.Request(f"{API_URL}?json={query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_batch(settings: dict, retries: int = 3, delay: float = 2.5) -> list[dict]:
    """One API call, backing off on rate limiting and transient errors."""
    for attempt in range(retries):
        try:
            return api_get(settings)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            if e.code == 429 or e.code >= 500:
                wait = delay * (attempt + 2)
                print(f"  HTTP {e.code} ({body}); backing off {wait:.1f}s")
                time.sleep(wait)
                continue
            # 400s are our bug (a bad parameter), not the server's mood —
            # retrying an identical malformed query just wastes their CPU.
            raise SystemExit(f"API rejected the request: HTTP {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError) as e:
            wait = delay * (attempt + 2)
            print(f"  network error ({e}); retrying in {wait:.1f}s")
            time.sleep(wait)
    return []


def build_settings(args) -> dict:
    settings = {
        "count": args.count,
        "level": args.level,
        "complexity": args.complexity,
        "legality": args.legality,
        "from": APP_NAME,
    }
    if args.tags:
        # Topic-targeted pull: OR across the requested tags.
        settings["tags"] = args.tags
        settings["tagsConjunc"] = args.tags_conjunc
    else:
        # Untargeted pull: let the server drop the unusable ones so they
        # do not consume the batch budget.
        settings["tags"] = args.exclude_tags
        settings["tagsConjunc"] = "NOT"
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/gold/rulesguru/questions.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/gold/rulesguru/raw"))
    parser.add_argument("--count", type=int, default=100, help="questions per request (API tolerates 250)")
    parser.add_argument("--max", type=int, default=200, help="stop after this many NEW questions")
    parser.add_argument("--all", action="store_true", help="walk the whole matching set")
    parser.add_argument("--start-after", type=int, default=1, help="previousId cursor; must be >= 1")
    parser.add_argument("--level", nargs="+", default=ALL_LEVELS)
    parser.add_argument("--complexity", nargs="+", default=ALL_COMPLEXITIES)
    parser.add_argument("--legality", default="all")
    parser.add_argument("--tags", nargs="*", default=[], help="topic tags to target (OR by default)")
    parser.add_argument("--tags-conjunc", default="OR", choices=["AND", "OR", "NOT"])
    parser.add_argument("--exclude-tags", nargs="*", default=DEFAULT_EXCLUDE_TAGS)
    parser.add_argument("--delay", type=float, default=2.5, help="seconds between requests (API asks for >= 2)")
    parser.add_argument("--no-raw", action="store_true", help="skip archiving raw API responses")
    args = parser.parse_args()

    if args.delay < 2.0:
        raise SystemExit("--delay below 2.0s violates the documented API rate limit")

    # Existing ids are load-bearing: they are what makes re-runs additive
    # instead of destructive, given the server re-randomizes question text.
    existing: dict[int, dict] = {}
    if args.out.exists():
        with args.out.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                existing[r["id"]] = r
    print(f"{len(existing)} questions already in the snapshot")

    settings = build_settings(args)
    exclude = set(args.exclude_tags)
    target = float("inf") if args.all else args.max

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    cursor = args.start_after
    new = 0
    filtered = 0
    refetched = 0
    empty_rounds = 0
    seen_this_run: set[int] = set()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    while new < target:
        batch = fetch_batch(dict(settings, previousId=cursor), delay=args.delay)
        if not batch:
            print("empty response; stopping")
            break

        if not args.no_raw:
            stamp = f"{int(time.time())}_{cursor}"
            (args.raw_dir / f"batch_{stamp}.json").write_text(
                json.dumps(batch, ensure_ascii=False), encoding="utf-8"
            )

        batch_new = 0
        for q in batch:
            qid = q["id"]
            seen_this_run.add(qid)
            if set(q.get("tags", [])) & exclude:
                filtered += 1
                continue
            if qid in existing:
                # Do NOT overwrite: the stored wording may already have a
                # human-authored rubric attached to it downstream.
                refetched += 1
                continue
            q = slim_question(q)
            q["fetched_at"] = fetched_at
            q["rulesguru_url"] = f"https://rulesguru.org/?{qid}"
            existing[qid] = q
            batch_new += 1
            new += 1
            if new >= target:
                break

        cursor = batch[-1]["id"]
        print(f"  cursor -> {cursor}: +{batch_new} new ({new} total new, {refetched} already held)")

        # The cursor wraps to the start of the matching set once exhausted,
        # so "a whole batch contained nothing new" is the natural terminator.
        empty_rounds = empty_rounds + 1 if batch_new == 0 else 0
        if empty_rounds >= 2:
            print("two consecutive batches with nothing new — matching set exhausted")
            break

        if new < target:
            time.sleep(args.delay)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for qid in sorted(existing):
            f.write(json.dumps(existing[qid], ensure_ascii=False) + "\n")

    print(f"\n{new} new, {refetched} already held, {filtered} filtered by tag -> {args.out}")
    print(f"{len(existing)} questions in the snapshot")

    if existing:
        from collections import Counter

        levels = Counter(r["level"] for r in existing.values())
        cx = Counter(r["complexity"] for r in existing.values())
        tags = Counter(t for r in existing.values() for t in r.get("tags", []))
        cited = sum(1 for r in existing.values() if r.get("citedRules"))
        print(f"level: {dict(sorted(levels.items()))}")
        print(f"complexity: {dict(cx)}")
        print(f"{cited}/{len(existing)} carry at least one CR citation")
        print("top tags:", ", ".join(f"{t}({n})" for t, n in tags.most_common(10)))

    print("\nNext: python scripts/rulesguru_to_gold.py")


if __name__ == "__main__":
    main()
