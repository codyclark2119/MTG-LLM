"""Routed retrieval: named cards and rulings by lookup, rules by embedding.

Section 9.5 showed both RAG arms scoring *worse* on card-referencing
questions than on pure-rules ones. The cause is structural rather than a
tuning problem: the RAG system prompt says "use ONLY the provided rules
text", but the rules corpus can never contain card text, so a card
question is unanswerable under its own instructions.

The fix is not to dump 35k card chunks into the rules index. Cards
outnumber rules chunks ~78:1 there, so volume alone would let them win
retrieval slots for questions like "when are state-based actions
checked". Instead each source is retrieved on its own terms and given its
own budget:

  - cards: exact-name lookup (scripts/card_lookup.py), 99% resolution on
    the eval set, costs no similarity budget and cannot drift to a
    near-miss card
  - rules: the existing embedding index, unchanged, so the rules-only
    baseline stays exactly comparable to previous runs

Card chunks already carry the rule IDs for their own keywords (see
chunk_cards.py), so a retrieved card also points back into the rules
corpus.

Official rulings are an opt-in third source. They are keyed by the same exact
card name as the Oracle index and are returned separately, so enabling them is
an explicit experiment rather than a silent change to existing evaluation
prompts or arm outputs.

Usage:
    python scripts/retrieve_hybrid.py "Does [[Chatterfang]] double tokens?" [--k 3]
    python scripts/retrieve_hybrid.py "Does [[Chatterfang]] double tokens?" --rulings
    python scripts/retrieve_hybrid.py "When are state-based actions checked?" --k auto
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from card_lookup import CARD_CHUNKS, CardIndex
from common import (AUTO_K_RULES, RULING_CHUNKS_PATH, iter_jsonl, k_rules_arg,
                    route_k_rules, verify_card_pin)
from rag import CHUNKS_PATH, INDEX_PATH, retrieve


class RulingIndex:
    """Resolve official rulings for cards named in a question.

    Rulings are grouped by card, so exact and normalized card-name resolution
    is safer than embedding a 19k-record corpus into the rules index.
    """

    def __init__(self, chunks_path: Path = RULING_CHUNKS_PATH):
        verify_card_pin(chunks_path, ruling_chunks_path=chunks_path)
        self.by_name: dict[str, dict] = {}
        for ruling in iter_jsonl(chunks_path):
            name = ruling["name"]
            self.by_name[name] = ruling

    def find_for_cards(self, cards: list[dict]) -> list[dict]:
        """Return rulings for resolved cards, preserving card order."""
        found = []
        for card in cards:
            ruling = self.by_name.get(card["name"])
            if ruling:
                found.append(ruling)
        return found


def build_context(
    question: str,
    card_index: CardIndex,
    embed_model=None,
    k_rules: int | str = 3,
    max_cards: int = 3,
    ruling_index: RulingIndex | None = None,
    chunks_path: Path = CHUNKS_PATH,
    index_path: Path = INDEX_PATH,
    card_names: list[str] | None = None,
    scan_prose: bool = False,
) -> dict:
    # `card_names` bypasses `find_in_text`, which resolves ONLY `[[bracket]]`
    # syntax and therefore resolves nothing at all on either real gold corpus
    # — 0/99 on gold_questions.jsonl and 0/1202 on gold_candidates.jsonl,
    # because neither ever adopted that convention in its question text
    # (Section 21.136). Callers that already know which cards a record is
    # about — every gold record carries a hand-verified `cards` field — pass
    # them here and get exact/normalized dictionary resolution instead of
    # text-scanning. Same technique `build_sft_verified._record_context`
    # already uses; shared here rather than reimplemented so the two cannot
    # drift into disagreeing about what "the cards for this question" means.
    if card_names:
        cards, seen = [], set()
        for name in card_names:
            card, how = card_index.resolve(name)
            if card and card["name"] not in seen:
                seen.add(card["name"])
                # `matched_as`/`match_type` are part of find_in_text's contract
                # and downstream metadata reads them; omitting them here made
                # build_context raise KeyError on the very first call.
                cards.append({**card, "matched_as": name, "match_type": how})
            if len(cards) >= max_cards:
                break
    else:
        # scan_prose is how a LIVE question reaches the card arm at all: a user
        # types "Does Lightning Bolt kill a Grizzly Bears?" and, bracket-only,
        # resolves nothing (Section 21.154).
        cards = card_index.find_in_text(question, max_cards=max_cards,
                                        scan_prose=scan_prose)
    rulings = ruling_index.find_for_cards(cards) if ruling_index else []
    # Resolved here and not by the caller: routing keys off the cards this call
    # actually resolved, which no caller knows until the lookup above has run.
    # Also note the comparison below is `k_rules > 0`, which raises TypeError on
    # a str -- so "auto" must become an int before it reaches that line.
    routed = k_rules == AUTO_K_RULES
    k_rules = route_k_rules(len(cards)) if routed else k_rules
    # k_rules=0 omits the dense-retrieved CR section entirely. Not a
    # micro-optimisation: Section 21.141 measured rules-only RAG scoring BELOW
    # no-retrieval, with cited-rule recall stuck near 25% and no available fix
    # (k, BM25, finer chunks, query rewriting and a second embedding model were
    # all tested and all failed). But that was measured only WHERE CARD TEXT WAS
    # ALREADY PRESENT; on card-free questions the same retrieval is worth +0.65
    # (21.155). So k=0 is the right setting for one half of the traffic and the
    # wrong one for the other half, which is what AUTO exists to decide.
    rules_hits = retrieve(
        question, k=k_rules, chunks_path=chunks_path, index_path=index_path,
        model_and_tokenizer=embed_model,
    ) if k_rules > 0 else []

    parts = []
    if cards:
        parts.append("Cards referenced:\n" + "\n\n".join(c["text"] for c in cards))
    if rulings:
        parts.append("Official rulings:\n" + "\n\n".join(r["text"] for r in rulings))
    if rules_hits:
        parts.append("Rules text:\n" + "\n\n".join(h["text"] for h in rules_hits))

    # Returned as METADATA ONLY — these ids are not injected into `context`.
    # The original comment here claimed a card's keyword rules were "worth
    # surfacing even when the question's phrasing didn't retrieve them
    # semantically", which describes a feature that was never wired up:
    # nothing downstream reads this field except the debug print in main().
    #
    # Injecting the rule TEXT for these ids is a genuine retrieval
    # improvement to try — a card with trample would arrive with 702.19
    # attached regardless of phrasing — but it changes what every card arm
    # sees, so it must be run as a measured change against the current
    # baseline rather than switched on quietly.
    card_rule_ids = sorted({r for c in cards for r in c["keyword_rule_ids"]})

    return {
        "context": "\n\n".join(parts),
        "card_names": [c["name"] for c in cards],
        "card_match_types": [c["match_type"] for c in cards],
        "ruling_card_names": [r["name"] for r in rulings],
        "card_keyword_rule_ids": card_rule_ids,
        "rules_chunk_ids": [h["chunk_id"] for h in rules_hits],
        # `k_rules` stops being a run-level constant under AUTO, so the value
        # that ACTUALLY applied has to ride on the record. A rating or a row
        # stamped `k_rules: "auto"` and nothing else is uninterpretable: it
        # names the policy, not the treatment the answer received.
        "k_rules_used": k_rules,
        "k_rules_routed": routed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--k", type=k_rules_arg, default=3,
                        help='CR chunks to retrieve, or "auto" to route on '
                             'whether a card resolved (Section 21.156)')
    parser.add_argument("--max-cards", type=int, default=3)
    parser.add_argument("--card-chunks", type=Path, default=CARD_CHUNKS)
    parser.add_argument("--ruling-chunks", type=Path, default=RULING_CHUNKS_PATH)
    parser.add_argument("--rulings", action="store_true",
                        help="include official rulings for resolved cards")
    args = parser.parse_args()

    card_index = CardIndex(args.card_chunks)
    ruling_index = RulingIndex(args.ruling_chunks) if args.rulings else None
    result = build_context(args.question, card_index, k_rules=args.k,
                           max_cards=args.max_cards, ruling_index=ruling_index)
    print(result["context"])
    print("\n---")
    print("cards:", result["card_names"], result["card_match_types"])
    print("ruling cards:", result["ruling_card_names"])
    print("card keyword rules:", result["card_keyword_rule_ids"])
    print("rules chunks:", result["rules_chunk_ids"])
    print("k_rules used:", result["k_rules_used"],
          "(routed)" if result["k_rules_routed"] else "(fixed)")


if __name__ == "__main__":
    main()
