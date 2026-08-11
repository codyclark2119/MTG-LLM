"""Routed retrieval: named cards by lookup, rules by embedding.

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

Usage:
    python scripts/retrieve_hybrid.py "Does [[Chatterfang]] double tokens?" [--k 3]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from card_lookup import CARD_CHUNKS, CardIndex
from rag import CHUNKS_PATH, INDEX_PATH, retrieve


def build_context(
    question: str,
    card_index: CardIndex,
    embed_model=None,
    k_rules: int = 3,
    max_cards: int = 3,
    chunks_path: Path = CHUNKS_PATH,
    index_path: Path = INDEX_PATH,
) -> dict:
    cards = card_index.find_in_text(question, max_cards=max_cards)
    rules_hits = retrieve(
        question, k=k_rules, chunks_path=chunks_path, index_path=index_path,
        model_and_tokenizer=embed_model,
    )

    parts = []
    if cards:
        parts.append("Cards referenced:\n" + "\n\n".join(c["text"] for c in cards))
    parts.append("Rules text:\n" + "\n\n".join(h["text"] for h in rules_hits))

    # A card's own keyword rules are worth surfacing even when the question's
    # phrasing didn't retrieve them semantically.
    card_rule_ids = sorted({r for c in cards for r in c["keyword_rule_ids"]})

    return {
        "context": "\n\n".join(parts),
        "card_names": [c["name"] for c in cards],
        "card_match_types": [c["match_type"] for c in cards],
        "card_keyword_rule_ids": card_rule_ids,
        "rules_chunk_ids": [h["chunk_id"] for h in rules_hits],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-cards", type=int, default=3)
    parser.add_argument("--card-chunks", type=Path, default=CARD_CHUNKS)
    args = parser.parse_args()

    result = build_context(args.question, CardIndex(args.card_chunks), k_rules=args.k, max_cards=args.max_cards)
    print(result["context"])
    print("\n---")
    print("cards:", result["card_names"], result["card_match_types"])
    print("card keyword rules:", result["card_keyword_rule_ids"])
    print("rules chunks:", result["rules_chunk_ids"])


if __name__ == "__main__":
    main()
