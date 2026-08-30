"""Validate the card/ruling benchmark and measure retrieval recall (PLAN_NEXT.md item 4).

WHY THIS EXISTS

PLAN_NEXT.md's card/ruling benchmark item says to "measure retrieval recall
before generation quality" — this is that measurement, and it is a distinct
question from whether an authored question is well-formed. A question can
have a perfectly correct hand-authored answer and still be useless as a
retrieval-sensitive benchmark item if the current pipeline never actually
surfaces the ruling or rule it depends on.

TWO SEPARATE CHECKS

  grounding  — is the record's own metadata TRUE? Do the named cards resolve
              against the pinned card index, does each `official_rulings_used`
              chunk id actually exist for one of those cards, is each
              `cr_rule_citations` entry a real rule id in the pinned CR, and
              is `ruling_sufficiency` consistent with whether a ruling is even
              cited. This catches an authored record that cites something
              that does not exist — a chunk id typo, a rule id that was
              renumbered, a card name that does not resolve.

  retrieval recall — if this question were asked for real, would
              `card_lookup.CardIndex` resolve the same cards, would
              `retrieve_hybrid.RulingIndex` surface the SAME ruling this
              record depends on, and would `rag.retrieve` surface a chunk
              containing the cited CR rule id. Grounding can be perfect while
              recall is 0% — that would mean the benchmark item is testing
              something the current retrieval pipeline cannot even find,
              which is a finding about the PIPELINE, not a defect in the
              question.

Never promotes anything. Every record here needs `needs_review: true` and a
human pass before it is anything more than a draft (the same rule every other
gold-adjacent file in this repo follows).

Usage:
    python scripts/validate_card_ruling_benchmark.py
    python scripts/validate_card_ruling_benchmark.py --path data/gold/card_ruling_candidates.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import REPO_ROOT, RULING_CHUNKS_PATH, load_rule_ids, read_jsonl  # noqa: E402

DEFAULT_PATH = REPO_ROOT / "data/gold/card_ruling_candidates.jsonl"
VALID_SUFFICIENCY = {"ruling_sufficient", "ruling_relevant_insufficient", "no_ruling_needed_cr_only"}


def load_ruling_chunk_ids_by_card() -> dict[str, set[str]]:
    """{card name: {chunk_id, ...}} straight from the pinned ruling corpus."""
    out: dict[str, set[str]] = {}
    for rec in read_jsonl(RULING_CHUNKS_PATH):
        out.setdefault(rec["name"], set()).add(rec["chunk_id"])
    return out


def validate_grounding(rows: list[dict], card_index, rule_ids: set[str],
                       ruling_ids_by_card: dict[str, set[str]]) -> list[str]:
    """Metadata-level checks. Returns one problem string per issue found."""
    problems = []
    for r in rows:
        rid = r["id"]
        for card in r.get("cards") or []:
            if card not in card_index.by_name and card not in card_index.by_norm:
                problems.append(f"{rid}: card {card!r} does not resolve against the pinned card index")
        for chunk_id in r.get("official_rulings_used") or []:
            name = chunk_id.split("rulings:", 1)[-1]
            if chunk_id not in ruling_ids_by_card.get(name, set()):
                problems.append(f"{rid}: official_rulings_used chunk {chunk_id!r} does not exist "
                                f"in the pinned ruling corpus")
        for cr in r.get("cr_rule_citations") or []:
            if cr not in rule_ids:
                problems.append(f"{rid}: cr_rule_citations entry {cr!r} is not a rule id in the pinned CR")
        suff = r.get("ruling_sufficiency")
        if suff not in VALID_SUFFICIENCY:
            problems.append(f"{rid}: ruling_sufficiency {suff!r} is not one of {sorted(VALID_SUFFICIENCY)}")
        has_ruling = bool(r.get("official_rulings_used"))
        if suff == "no_ruling_needed_cr_only" and has_ruling:
            problems.append(f"{rid}: marked no_ruling_needed_cr_only but official_rulings_used is non-empty")
        if suff in ("ruling_sufficient", "ruling_relevant_insufficient") and not has_ruling:
            problems.append(f"{rid}: marked {suff} but official_rulings_used is empty")
        if not r.get("needs_review", True):
            problems.append(f"{rid}: needs_review is not true — nothing in this file should skip human review")
    return problems


def measure_retrieval_recall(rows: list[dict], card_index, ruling_index, embed_model) -> None:
    """For each record, does the CURRENT retrieval pipeline actually find what it depends on."""
    from rag import retrieve

    card_hits = ruling_hits = rule_hits = 0
    n_with_ruling = n_with_rule = 0
    for r in rows:
        rid = r["id"]
        resolved = card_index.find_in_text(r["question"])
        resolved_names = {c["name"] for c in resolved}
        expected = set(r.get("cards") or [])
        card_ok = expected <= resolved_names
        card_hits += card_ok
        line = f"{rid}: cards {'OK' if card_ok else 'MISS'} ({sorted(resolved_names) or 'none resolved'})"

        if r.get("official_rulings_used"):
            n_with_ruling += 1
            found = ruling_index.find_for_cards(resolved)
            got = {f["chunk_id"] for f in found}
            want = set(r["official_rulings_used"])
            ok = want <= got
            ruling_hits += ok
            line += f" | ruling {'OK' if ok else 'MISS'}"

        if r.get("cr_rule_citations"):
            n_with_rule += 1
            hits = retrieve(r["question"], k=3, model_and_tokenizer=embed_model)
            retrieved_rule_ids = {rid_ for h in hits for rid_ in h.get("rule_ids", [])}
            ok = any(cr in retrieved_rule_ids for cr in r["cr_rule_citations"])
            rule_hits += ok
            line += f" | CR rule {'OK' if ok else 'MISS'} (top-3 chunks: {[h['chunk_id'] for h in hits]})"

        print(line)

    n = len(rows)
    print()
    print(f"card name recall : {card_hits}/{n}")
    if n_with_ruling:
        print(f"ruling recall    : {ruling_hits}/{n_with_ruling} (of records citing a ruling)")
    if n_with_rule:
        print(f"CR rule recall   : {rule_hits}/{n_with_rule} (of records citing a CR rule, top-3 chunks)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--no-recall", action="store_true",
                    help="grounding checks only, skip loading the embedding model")
    args = ap.parse_args()

    rows = read_jsonl(args.path)
    if not rows:
        raise SystemExit(f"no records in {args.path}")

    from card_lookup import CardIndex
    card_index = CardIndex()
    rule_ids = load_rule_ids()
    ruling_ids_by_card = load_ruling_chunk_ids_by_card()

    problems = validate_grounding(rows, card_index, rule_ids, ruling_ids_by_card)
    print(f"{len(rows)} candidate(s) in {args.path}")
    if problems:
        print(f"\n{len(problems)} grounding problem(s):")
        for p in problems:
            print(f"  {p}")
    else:
        print("grounding: clean — every card, ruling, and rule citation resolves against pinned data")

    if args.no_recall:
        return

    print()
    from mlx_embeddings import load as load_embedder
    from rag import MODEL_ID as EMBED_MODEL_ID
    from retrieve_hybrid import RulingIndex
    print(f"loading {EMBED_MODEL_ID} for retrieval recall measurement ...")
    embed_model = load_embedder(EMBED_MODEL_ID)
    ruling_index = RulingIndex()
    measure_retrieval_recall(rows, card_index, ruling_index, embed_model)

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
