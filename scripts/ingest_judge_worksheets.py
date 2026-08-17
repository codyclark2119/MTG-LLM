"""Ingest Judge Simulation Worksheet scenarios into structured records.

These are Competitive-REL judge-training scenarios: real situations
written by judges, each with an authoritative ruling. They are the
highest-quality human-authored MTG data this project has access to.

**They carry two distinct layers, and only one is in Phase 1 scope.**

  policy layer — the infraction and fix (GRV, FMGS, HCE, backups,
    penalties). This lives in the Infraction Procedure Guide and Magic
    Tournament Rules, NOT the Comprehensive Rules. Verified directly: the
    CR contains zero occurrences of "Game Rule Violation", "Failure to
    Maintain", or "Competitive REL". Answering these requires adding
    IPG/MTR to the corpus — a documented scope change (README Section 0
    puts tournament policy out of Phase 1).

  rules layer — the Comprehensive Rules question the scenario is built
    on: Bone to Ash can only target creature spells; protection from blue
    prevents targeting; Tarmogoyf's power comes from graveyard card types.
    This IS in Phase 1 scope, and it is precisely the "interaction puzzle"
    category the current eval is thinnest on (11 of 70 synthetic
    questions) — except written by judges from real situations rather
    than generated from rules text.

So each scenario is stored once with both layers marked, rather than
being forced into one track. `--emit-rules-layer` writes only the
scenarios whose embedded rules question can be evaluated today.

The upstream generator returns a random selection per request, so
`--fetch N` pulls repeatedly and deduplicates by question id.

Usage:
    python scripts/ingest_judge_worksheets.py --fetch 20
    python scripts/ingest_judge_worksheets.py --from-file worksheet.txt
"""

import argparse
import json
import re
import time
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO_ROOT, iter_jsonl  # noqa: E402

RECORD_RE = re.compile(r"^Question ID:\s*(\d+)\s*$", re.M)
FIELD_RE = re.compile(r"^(Infractions|Cards|Description|Answer):\s*$|^(Infractions|Cards):\s*(.*)$", re.M)

BASE_URL = "http://simulations.matthew.ath.cx/worksheet.php/create"
INFRACTIONS = [
    "GPE-FMGS", "GPE-GRV", "GPE-HCE", "GPE-LEC", "GPE-MPE", "GPE-MT",
    "NO", "TE-CPV", "TE-DLP", "TE-DP", "TE-MC", "TE-OA",
]
# Expansions for the infraction codes, so records are readable without
# needing the IPG open alongside them.
INFRACTION_NAMES = {
    "GPE-GRV": "Game Play Error — Game Rule Violation",
    "GPE-FMGS": "Game Play Error — Failure to Maintain Game State",
    "GPE-HCE": "Game Play Error — Hidden Card Error",
    "GPE-LEC": "Game Play Error — Looking at Extra Cards",
    "GPE-MPE": "Game Play Error — Mulligan Procedure Error",
    "GPE-MT": "Game Play Error — Missed Trigger",
    "TE-CPV": "Tournament Error — Communication Policy Violation",
    "TE-DP": "Tournament Error — Deck Problem",
    "TE-DLP": "Tournament Error — Decklist Problem",
    "TE-MC": "Tournament Error — Marked Cards",
    "TE-OA": "Tournament Error — Outside Assistance",
    "NO": "No infraction",
}


def build_url() -> str:
    params = [("outputmode", "txt"), ("questions", "")]
    params += [(f"infraction[{i}]", code) for i, code in enumerate(INFRACTIONS)]
    return f"{BASE_URL}?{urllib.parse.urlencode(params)}"


def fetch_once(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "MagicLLM-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def split_card_list(raw: str, card_index=None) -> list[str]:
    """Split the Cards field without breaking names that contain commas.

    Legendary names are the problem: "Gideon, Ally of Zendikar" and
    "Kalitas, Traitor of Ghet" split on commas into halves that resolve to
    nothing. Greedily prefer the longest comma-joined run that resolves to
    a real card, falling back to plain splitting when no index is
    available.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return []
    if card_index is None:
        return parts

    out: list[str] = []
    i = 0
    while i < len(parts):
        matched = None
        # Try the longest join first so "Gideon, Ally of Zendikar" wins over "Gideon".
        for j in range(min(len(parts), i + 3), i, -1):
            candidate = ", ".join(parts[i:j])
            card, how = card_index.resolve(candidate)
            if card is not None and how in ("exact", "normalized"):
                matched = (card["name"], j)
                break
        if matched:
            out.append(matched[0])
            i = matched[1]
        else:
            out.append(parts[i])
            i += 1
    return out


def parse_worksheet(text: str, card_index=None) -> list[dict]:
    """Split on 'Question ID:' rather than blank lines — several scenarios
    contain blank lines inside their Description or Answer body."""
    records = []
    starts = [m.start() for m in RECORD_RE.finditer(text)]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[start:end]

        qid = RECORD_RE.match(block).group(1)
        infractions_m = re.search(r"^Infractions:\s*(.*)$", block, re.M)
        cards_m = re.search(r"^Cards:\s*(.*)$", block, re.M)
        desc_m = re.search(r"^Description:\s*\n(.*?)(?=^Answer:\s*$)", block, re.M | re.S)
        ans_m = re.search(r"^Answer:\s*\n(.*)$", block, re.M | re.S)

        def clean(s: str | None) -> str:
            if not s:
                return ""
            lines = [ln.strip() for ln in s.strip().split("\n")]
            return "\n".join(ln for ln in lines).strip()

        infractions = [c.strip() for c in (infractions_m.group(1) if infractions_m else "").split(",") if c.strip()]
        cards = split_card_list(cards_m.group(1) if cards_m else "", card_index)

        records.append(
            {
                "id": f"jsw-{qid}",
                "worksheet_question_id": int(qid),
                "rel": "Competitive",
                "infractions": infractions,
                "infraction_names": [INFRACTION_NAMES.get(c, c) for c in infractions],
                "no_infraction": infractions == ["NO"],
                "cards": cards,
                "scenario": clean(desc_m.group(1) if desc_m else ""),
                "ruling": clean(ans_m.group(1) if ans_m else ""),
                "layers": ["policy", "rules"],
                "policy_source": "IPG/MTR (not in the Comprehensive Rules corpus)",
                "source": "judge-simulation-worksheet",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", type=int, default=0, help="number of times to pull the randomized export")
    parser.add_argument("--from-file", type=Path, nargs="*", default=[], help="parse local worksheet text files")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/judge_worksheets/scenarios.jsonl")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/judge_worksheets/raw")
    parser.add_argument("--delay", type=float, default=1.5, help="seconds between fetches — be polite to a hobby host")
    args = parser.parse_args()

    # Keep anything already collected; the generator is random, so each run
    # is additive rather than a fresh snapshot.
    existing: dict[str, dict] = {}
    if args.out.exists():
        for r in iter_jsonl(args.out):
            existing[r["id"]] = r
    print(f"{len(existing)} scenarios already collected")

    texts: list[str] = []
    for p in args.from_file:
        texts.append(p.read_text(encoding="utf-8"))

    if args.fetch:
        url = build_url()
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.fetch):
            try:
                t = fetch_once(url)
            except Exception as e:
                print(f"  fetch {i + 1} failed: {e}")
                continue
            texts.append(t)
            (args.raw_dir / f"worksheet_{int(time.time())}_{i}.txt").write_text(t, encoding="utf-8")
            if i + 1 < args.fetch:
                time.sleep(args.delay)
        print(f"fetched {len(texts)} worksheet(s)")

    # Card names in the Cards field can contain commas; resolving against
    # the Oracle pool is what makes them splittable correctly.
    card_index = None
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from card_lookup import CardIndex
        card_index = CardIndex()
    except Exception as e:
        print(f"card index unavailable ({e}); card names will be split naively")

    new = 0
    for t in texts:
        for rec in parse_worksheet(t, card_index):
            if rec["id"] not in existing:
                new += 1
            existing[rec["id"]] = rec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in sorted(existing.values(), key=lambda r: r["worksheet_question_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter

    codes = Counter(c for r in existing.values() for c in r["infractions"])
    incomplete = [r["id"] for r in existing.values() if not r["scenario"] or not r["ruling"]]
    print(f"{new} new, {len(existing)} total -> {args.out}")
    print("infraction coverage:", dict(codes.most_common()))
    if incomplete:
        print(f"WARNING: {len(incomplete)} records missing scenario or ruling: {incomplete[:5]}")


if __name__ == "__main__":
    main()
