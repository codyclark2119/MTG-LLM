"""Snapshot the MTG Wiki's rules portal as an annotated reference layer.

    python scripts/fetch_wiki.py --dry-run        # what would be fetched
    python scripts/fetch_wiki.py                  # -> data/raw/wiki_pages.jsonl
    python scripts/fetch_wiki.py --refresh        # re-fetch, keeping the old snapshot

WHY A SECOND RULES CORPUS

`rules.jsonl` is the Comprehensive Rules: authoritative, complete, and written
for judges. It says what the rule IS and never what it MEANS. A question like
"can a creature with summoning sickness block?" is answered by 302.6, but only
if you already know to look there.

The wiki pages linked from Portal:Rules are the other half — Object, Zone,
Timing and priority, one page per step of the turn — prose that explains the
same rules in the vocabulary players actually use. That is exactly the text a
retrieval index wants, because a player's question resembles the explanation far
more than it resembles the rule.

THIS IS NOT AUTHORITATIVE AND MUST NEVER BE CITED AS THE CR

Community-edited, so it is a *gloss*, not a source. The project's whole citation
discipline rests on a rule id resolving against the pinned CR (`verify_cr_pin`),
and a corpus that reads like rules text but is not the rules text is the fastest
way to break that. Hence:

  * a separate file, never merged into `rules.jsonl` or `chunks.jsonl`;
  * every record carries `authority: "unofficial"`;
  * every record carries the revision it came from, so a claim can be traced to
    an exact version of an editable page.

LICENSING — READ BEFORE ANY TRAINING USE

Fandom serves this under **CC BY-NC-SA 2.5**, which the API reports per page and
which this script stores on every record. Three consequences:

  BY    attribution is required; `title`, `revision`, `url` and `license` are
        stored on every record so any downstream use can attribute correctly.
  NC    noncommercial only.
  SA    derivative works inherit the licence.

Retrieval at inference time quotes a source. **Training on it arguably makes the
adapter a derivative work**, which is a licensing decision and not a technical
one — so nothing here is wired into `build_sft*.py`, and that stays a deliberate
choice rather than a default.

FETCHING POLITELY

The site is behind Cloudflare and plain page scraping is challenged; the
MediaWiki API is the interface the operator publishes for this, so that is what
this uses. Titles are batched (the API's own limit for extracts is 20) and a
delay separates requests. A snapshot is FROZEN and additive, like the RulesGuru
one: `--refresh` writes a new file rather than editing the old, because a page
edited between runs would otherwise silently change a record that downstream
numbers were computed from.
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import REPO_ROOT, write_jsonl_atomic  # noqa: E402

API = "https://mtg.fandom.com/api.php"
PORTAL = "Portal:Rules"
WIKI_RAW_PATH = REPO_ROOT / "data/raw/wiki_pages.jsonl"
# ASCII only: HTTP headers are latin-1, and an em-dash here raised
# UnicodeEncodeError from http.client before the first request went out.
USER_AGENT = "magic-llm-research/0.1 (personal research corpus; contact: repository owner)"
# ONE title per request, not twenty.
#
# `exlimit` documents a maximum of 20, and the first version of this used it —
# and got back 3 pages of 60, because TextExtracts only honours exlimit > 1 when
# `exintro` is set. Ask for FULL text and it silently serves one page per
# request and drops the rest, with no error and no `continue`: 57 pages simply
# were not there. The comment that used to sit here warned about exactly this
# and the code did it anyway.
#
# Intro-only extracts would batch, but the explanatory body is the reason this
# corpus exists, so the cost is 60 requests at DELAY_S instead of 3.
BATCH = 1
DELAY_S = 1.0

# Namespaces that are not article text. Files and categories carry no prose, and
# following them would turn a 63-page snapshot into a crawl of the whole wiki.
SKIP_PREFIXES = ("File:", "Category:", "Template:", "Help:", "Special:",
                 "Talk:", "User:", "Portal:", "MediaWiki:")


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def portal_links(portal: str = PORTAL) -> list[str]:
    """Article titles linked from the portal, in page order, minus non-article namespaces."""
    out, cont = [], {}
    while True:
        data = _get({"action": "query", "prop": "links", "titles": portal,
                     "pllimit": "max", **cont})
        for page in data.get("query", {}).get("pages", {}).values():
            for link in page.get("links") or []:
                t = link.get("title", "")
                if t and not t.startswith(SKIP_PREFIXES) and t not in out:
                    out.append(t)
        if "continue" not in data:
            return out
        cont = data["continue"]
        time.sleep(DELAY_S)


def fetch_pages(titles: list[str]) -> list[dict]:
    """One record per page: plaintext plus the provenance needed to cite it."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    out: list[dict] = []
    # target title -> the portal links that redirect to it. Recorded so a lookup
    # for "Upkeep step" can still find the text, which lives under its parent.
    aliases: dict[str, list[str]] = {}
    seen: set[str] = set()
    for i in range(0, len(titles), BATCH):
        chunk = titles[i:i + BATCH]
        data = _get({"action": "query", "prop": "extracts|info|revisions",
                     "explaintext": "1", "exlimit": "1",
                     "rvprop": "ids|timestamp", "inprop": "url",
                     # Nine of the sixty portal links are redirects to a section
                     # anchor: "Upkeep step" -> "Beginning phase#Upkeep step".
                     # Unresolved they look like missing pages; resolved they are
                     # aliases for text already captured, and following them
                     # blindly would store the parent page nine extra times.
                     "redirects": "1",
                     "titles": "|".join(chunk)})
        for r in data.get("query", {}).get("redirects") or []:
            aliases.setdefault(r["to"], []).append(
                r["from"] + ("#" + r["tofragment"] if r.get("tofragment") else ""))
        for page in data.get("query", {}).get("pages", {}).values():
            if "missing" in page:
                continue
            if page["title"] in seen:
                continue        # several portal links can redirect to one page
            text = (page.get("extract") or "").strip()
            if not text:
                continue
            seen.add(page["title"])
            rev = (page.get("revisions") or [{}])[0]
            out.append({
                "title": page["title"],
                "pageid": page["pageid"],
                # The exact version this text came from. A wiki page is
                # editable, so "the MTG Wiki says X" is not a citation and
                # "revision 564208 says X" is.
                "revision": rev.get("revid"),
                "revised": rev.get("timestamp"),
                "url": page.get("fullurl"),
                "text": text,
                "chars": len(text),
                "aliases": [],   # filled after the loop; see below
                # Never merge this with the CR. See the module docstring.
                "authority": "unofficial",
                "source": "mtg.fandom.com",
                "license": "CC BY-NC-SA 2.5",
                "fetched_at": fetched_at,
            })
        print(f"  fetched {min(i + BATCH, len(titles))}/{len(titles)}")
        if i + BATCH < len(titles):
            time.sleep(DELAY_S)

    # Attached HERE, not while building each record. At one title per request a
    # redirect is usually resolved AFTER its target has already been written, so
    # filling this inside the loop left the target's `aliases` empty and the
    # redirect still counted as a missing page.
    for rec in out:
        rec["aliases"] = sorted(aliases.get(rec["title"], []))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--portal", default=PORTAL)
    ap.add_argument("--out", type=Path, default=WIKI_RAW_PATH)
    ap.add_argument("--dry-run", action="store_true",
                    help="list the pages that would be fetched and stop")
    ap.add_argument("--refresh", action="store_true",
                    help="overwrite an existing snapshot. Without it an existing "
                         "file is left alone, because a re-fetch of an edited page "
                         "silently changes a record numbers were computed from.")
    args = ap.parse_args()

    if args.out.exists() and not args.refresh and not args.dry_run:
        raise SystemExit(f"{args.out} already exists — pass --refresh to replace it, "
                         "and expect every number derived from it to describe a "
                         "different snapshot afterwards")

    print(f"reading links from {args.portal} ...")
    titles = portal_links(args.portal)
    print(f"{len(titles)} article pages linked")
    if args.dry_run:
        for t in titles:
            print("   ", t)
        return

    pages = fetch_pages(titles)
    # A snapshot that quietly holds a third of what was asked for is worse than
    # a failed one: it looks like a corpus. See the note above BATCH.
    captured = {p["title"] for p in pages}
    aliased = {a.split("#")[0] for p in pages for a in p.get("aliases") or []}
    missing = [t for t in titles if t not in captured and t not in aliased]
    if missing:
        # Distinct from "not fetched": TextExtracts returns an empty extract for
        # a page that is entirely template or infobox, which three of these are.
        # Reported so the count reconciles — 51 captured + 6 reached by alias +
        # 3 with no prose = the 60 the portal links.
        print(f"\n{len(missing)} of {len(titles)} linked pages yielded no plain text "
              "(template-only pages, not fetch failures):")
        for t in missing[:10]:
            print("     ", t)
        if len(missing) > len(titles) // 4:
            raise SystemExit("more than a quarter of the pages are missing — refusing "
                             "to write a snapshot this incomplete")
    n_alias = sum(len(p["aliases"]) for p in pages)
    print(f"\n{len(pages)} pages captured, {n_alias} further titles reached by redirect")
    write_jsonl_atomic(args.out, pages)
    chars = sum(p["chars"] for p in pages)
    print(f"\n{len(pages)} pages, {chars:,} chars -> {args.out}")
    print("  licence CC BY-NC-SA 2.5, authority 'unofficial' — see the module docstring "
          "before any training use")


if __name__ == "__main__":
    main()
