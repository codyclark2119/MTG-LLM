"""Chunk the wiki snapshot into a retrievable, content-pinned corpus.

    python scripts/chunk_wiki.py                 # -> data/processed/wiki_chunks.jsonl
    python scripts/chunk_wiki.py --update-pin    # re-pin after a deliberate refresh

Splits on the plaintext section headings TextExtracts emits, so a chunk is a
section of a page rather than an arbitrary window — the same reason `chunk.py`
splits the CR on rule boundaries. Sections longer than the target are split on
paragraph breaks; short trailing sections are merged forward, because a 40-char
chunk retrieves on noise.

Every chunk keeps `authority: "unofficial"` and the page revision it came from.
The gloss must stay separable from the rules text at every stage, not only in
the file it was fetched into (see `fetch_wiki.py`).

NOT WIRED INTO RETRIEVAL. `retrieve_hybrid.py` does not read this. Whether wiki
prose competes with CR text for slots is a measurement, not a default — cards
already needed a separate budget because they outnumber rules chunks 78:1, and
wiki prose resembles a player's question more closely than the rule that answers
it, so it would win slots on phrasing.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import REPO_ROOT, read_jsonl, write_jsonl_atomic  # noqa: E402

WIKI_RAW_PATH = REPO_ROOT / "data/raw/wiki_pages.jsonl"
WIKI_CHUNKS_PATH = REPO_ROOT / "data/processed/wiki_chunks.jsonl"
TARGET = 3000          # matches the CR chunker's ~3,000-char median
MIN_CHUNK = 400        # below this a chunk retrieves on noise; merged forward

# TextExtracts renders headings as "== Name ==" / "=== Name ===" on their own line.
_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)

# Sections that are not rules content. TextExtracts already strips Fandom's page
# furniture — there is no ad markup, no cookie banner, no navigation; the two
# hits for "advertisement" and "subscribe" in the snapshot are article prose
# about token cards and judge fees. What it does NOT strip is the article's own
# apparatus, and that is most of what a plain scrape drags in:
#
#   References      41 of 51 pages, and EMPTY after extraction (the citations
#                   are markup), so it contributes a bare heading
#   External links  16 pages, a list of URLs
#   Trivia          9, See also 8, Gallery 5, Notes 4
#
# Dropped by heading rather than by pattern-matching the body, because a body
# filter would have to guess and this does not: the wiki labels these itself.
# `History` is KEPT — rule changes over time are rules content — and so is
# anything unlisted, so a new heading is included by default rather than
# silently dropped.
_DROP_SECTIONS = {
    "references", "external links", "see also", "gallery", "notes", "trivia",
    "sources", "further reading", "flavor", "flavour",
}


def split_sections(text: str) -> list[tuple[str, str]]:
    """[(heading, body)] for a page. The lead paragraph gets heading ''."""
    marks = [(m.start(), m.end(), m.group(2)) for m in _HEADING_RE.finditer(text)]
    if not marks:
        return [("", text.strip())]
    out = [("", text[:marks[0][0]].strip())]
    for i, (_, end, name) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append((name, text[end:stop].strip()))
    return [(h, b) for h, b in out
            if b and h.strip().lower() not in _DROP_SECTIONS]


def _split_long(body: str, target: int = TARGET) -> list[str]:
    """Split a section too long to be one chunk, on the best break available.

    Paragraphs first. TextExtracts renders lists with SINGLE newlines, though,
    so a long list section contains no blank line at all and paragraph splitting
    returned it whole — one chunk came out at 7,235 characters against a 3,000
    target, with `'\n\n' count=0`. Falls back to single newlines, then to a hard
    character cut, so no input shape can defeat the target.
    """
    if len(body) <= target:
        return [body]
    for sep in ("\n\n", "\n"):
        if sep not in body:
            continue
        out, cur = [], ""
        for part in body.split(sep):
            if cur and len(cur) + len(part) + len(sep) > target:
                out.append(cur.strip())
                cur = part
            else:
                cur = f"{cur}{sep}{part}" if cur else part
        if cur.strip():
            out.append(cur.strip())
        if all(len(x) <= target for x in out):
            return out
    # No break small enough exists — an unbroken wall of text. Cut it.
    return [body[i:i + target].strip() for i in range(0, len(body), target)]


def chunk_page(page: dict) -> list[dict]:
    pieces: list[tuple[str, str]] = []
    for heading, body in split_sections(page["text"]):
        for part in _split_long(body):
            pieces.append((heading, part))

    # Merge a too-short piece into the previous one rather than emitting it.
    # Merge a too-short piece into the previous one, or — for a page's LEAD
    # paragraph, which has no previous — forward into the next. Leaving it alone
    # emitted 27 chunks under the floor, the shortest 50 characters
    # ("A permanent is a card or token on the battlefield."): a crisp definition,
    # and far too little context to retrieve on without matching everything.
    # Merging must not undo the split: absorbing a short tail into a full-size
    # chunk produced a 4,687-character chunk against a 3,000 target, because the
    # merge ran after _split_long and never re-checked the size. A merge happens
    # only where it fits; where it does not, the short piece stands, since an
    # oversized chunk costs retrieval more than a small one does.
    # A tiny tail may overflow the target rather than stand alone: two chunks
    # came out at 37 characters when the merge was refused on size, and a
    # 37-character chunk retrieves on nothing. The bound is TARGET + MIN_CHUNK,
    # so "soft target" stays a stated number rather than an unbounded drift.
    MAX_CHUNK = TARGET + MIN_CHUNK

    def _fits(a: str, b: str) -> bool:
        return len(a) + len(b) + 2 <= MAX_CHUNK

    merged: list[tuple[str, str]] = []
    for heading, body in pieces:
        if merged and len(body) < MIN_CHUNK and _fits(merged[-1][1], body):
            ph, pb = merged[-1]
            # The merged piece keeps the PREVIOUS chunk's `heading`, so its own
            # label would be lost — and the text prefix only names one heading.
            # Carried inline so the chunk stays self-describing, which is the
            # whole reason the title is prefixed in the first place.
            tail = f"{heading}\n{body}" if heading and heading != ph else body
            merged[-1] = (ph, f"{pb}\n\n{tail}")
        else:
            merged.append((heading, body))
    # A page's LEAD has no previous chunk to merge into, so it merges forward.
    if len(merged) > 1 and len(merged[0][1]) < MIN_CHUNK and _fits(merged[0][1], merged[1][1]):
        (_, lead), (h2, b2) = merged[0], merged[1]
        merged[0:2] = [(h2, f"{lead}\n\n{b2}")]   # the lead has no heading to keep

    out = []
    for i, (heading, body) in enumerate(merged):
        out.append({
            "chunk_id": f"wiki-{page['pageid']}-{i:03d}",
            "title": page["title"],
            "heading": heading,
            # Prefixed so a retrieved chunk carries its own provenance into the
            # prompt. A bare paragraph of wiki prose is indistinguishable from
            # rules text once it is in a context window.
            "text": f"{page['title']}"
                    f"{' — ' + heading if heading else ''}\n{body}",
            "chars": len(body),
            "pageid": page["pageid"],
            "revision": page["revision"],
            "url": page["url"],
            "authority": "unofficial",
            "source": page["source"],
            "license": page["license"],
        })
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=Path, default=WIKI_RAW_PATH)
    ap.add_argument("--out", type=Path, default=WIKI_CHUNKS_PATH)
    ap.add_argument("--update-pin", action="store_true",
                    help="print the WIKI_PIN literal for common.py after writing")
    args = ap.parse_args()

    pages = read_jsonl(args.pages)
    if not pages:
        raise SystemExit(f"no pages in {args.pages} — run scripts/fetch_wiki.py first")

    chunks = [c for p in pages for c in chunk_page(p)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.out, chunks)

    lens = sorted(c["chars"] for c in chunks)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    print(f"  chars: median {lens[len(lens) // 2]}, min {lens[0]}, max {lens[-1]}")
    print(f"  -> {args.out}")
    print(f"  every chunk marked unofficial: {all(c['authority'] == 'unofficial' for c in chunks)}")

    if args.update_pin:
        print("\nWIKI_PIN = " + json.dumps({
            "wiki_chunks_sha256": sha256_file(args.out),
            "n_wiki_chunks": len(chunks),
            "n_pages": len(pages),
            "source": "mtg.fandom.com Portal:Rules",
            "license": "CC BY-NC-SA 2.5",
        }, indent=4))


if __name__ == "__main__":
    main()
