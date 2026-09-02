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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from card_lookup import CARD_CHUNKS, CardIndex
from common import (AUTO_K_RULES, RULES_PATH, RULING_CHUNKS_PATH, iter_jsonl,
                    k_rules_arg, route_k_rules, verify_card_pin, verify_cr_pin)
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


# A card chunk's `keyword_rule_ids` are PARENT ids, and a parent's own text is
# just the keyword's NAME -- rule 702.19 is the single word "Trample". The
# substance lives in 702.19a-702.19g, so injecting the parent alone injects a
# label and nothing else.
#
# Expanding it by `startswith` is wrong and would look right: "702.19" also
# prefixes 702.190 through 702.195, which are entirely different keywords
# (Boast and friends). A subrule is the parent followed by LETTERS ONLY. Third
# appearance of this repo's substring trap, after the BLOCK verb and the
# `[TARGET x]` brackets -- measured here as over-counting 0 on both benchmarks,
# which is luck, not safety.
_SUBRULE_RE = re.compile(r"(\d{3}\.\d+)([a-z]+)")


class KeywordRuleIndex:
    """CR text for the rules a resolved card's own keywords already name.

    This is a THIRD retrieval source, and deliberately not a similarity one.
    Section 21.141 killed five attempts to make embedding bridge card-scenario
    language to abstract rules language -- higher k, BM25, finer chunks, query
    rewriting, a second embedding model -- and BM25's failure is the instructive
    one: its union with dense retrieval added **zero** questions, because two
    similarity methods find the same things. This adds 12 of 99 on the gold set
    (17% -> 29% cited-rule recall, Section 21.159) precisely because it is not
    searching at all. A card with trample carries 702.19 in its own chunk, so
    the governing rule arrives however the question was phrased -- the same
    reason exact-name card lookup resolves 99% where embedding could not.
    """

    def __init__(self, rules_path: Path = RULES_PATH):
        verify_cr_pin(rules_path)
        self.text: dict[str, str] = {}
        self.subrules: dict[str, list[str]] = {}
        for rule in iter_jsonl(rules_path):
            rid = rule["rule_id"]
            self.text[rid] = rule["text"]
            m = _SUBRULE_RE.fullmatch(rid)
            if m:
                self.subrules.setdefault(m.group(1), []).append(rid)
        for group in self.subrules.values():
            group.sort()

    def expand(self, keyword_rule_ids: list[str], exclude: set[str] = frozenset(),
               max_rules: int = 20) -> list[tuple[str, str]]:
        """(rule_id, text) for each keyword parent and its lettered subrules.

        `exclude` drops rules the dense retriever already put in the context, so
        the two sources cannot restate each other. `max_rules` bounds the worst
        case: the median question injects 4 rules / 317 chars against an ~8,600
        char k=3 context, but the maximum is 24 rules / 5,597 chars, and 20
        truncates only 2 of 99.
        """
        out: list[tuple[str, str]] = []
        taken: set[str] = set()
        for parent in sorted(set(keyword_rule_ids)):
            for rid in [parent] + self.subrules.get(parent, []):
                if rid in exclude or rid in taken or rid not in self.text:
                    continue
                taken.add(rid)
                out.append((rid, self.text[rid]))
                if len(out) >= max_rules:
                    return out
        return out


def build_context(
    question: str,
    card_index: CardIndex,
    embed_model=None,
    k_rules: int | str = 3,
    max_cards: int = 3,
    ruling_index: RulingIndex | None = None,
    keyword_rule_index: "KeywordRuleIndex | None" = None,
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
    # (21.155), which is what AUTO exists to decide.
    #
    # AUTO is a 32B setting. On the 7B, k=0 on card questions is worth -0.02 and
    # costs five fabricated citations where there were none (21.158), so the
    # served default is a flat k=3 and this branch is off by default.
    rules_hits = retrieve(
        question, k=k_rules, chunks_path=chunks_path, index_path=index_path,
        model_and_tokenizer=embed_model,
    ) if k_rules > 0 else []

    # Was returned as METADATA ONLY and read by nothing but a debug print. The
    # old comment here said injecting the rule TEXT was "a genuine retrieval
    # improvement to try ... but it changes what every card arm sees, so it must
    # be run as a measured change". That is what `keyword_rule_index` does, and
    # it stays opt-in for exactly that reason.
    card_rule_ids = sorted({r for c in cards for r in c["keyword_rule_ids"]})
    keyword_rules = []
    if keyword_rule_index is not None and card_rule_ids:
        # Excluded against what dense retrieval already placed, so the two
        # sources cannot restate each other and inflate the context.
        already = {i for h in rules_hits for i in h.get("rule_ids", [])}
        keyword_rules = keyword_rule_index.expand(card_rule_ids, exclude=already)

    parts = []
    if cards:
        parts.append("Cards referenced:\n" + "\n\n".join(c["text"] for c in cards))
    if rulings:
        parts.append("Official rulings:\n" + "\n\n".join(r["text"] for r in rulings))
    # ONE "Rules text:" section, not two. The system prompt already says to use
    # "the provided card text and rules text", so these are rules text and need
    # no prompt change -- which means no prompt_fingerprint change and no
    # adapter invalidation (Section 8.7). Keyword rules go FIRST because they
    # are deterministic (the card's own chunk names them) while the dense hits
    # carry the ~25% recall from 21.141; that ordering is a deliberate choice
    # and an untested one.
    rules_blocks = []
    if keyword_rules:
        rules_blocks.append("\n".join(f"{rid}. {text}" for rid, text in keyword_rules))
    if rules_hits:
        rules_blocks.append("\n\n".join(h["text"] for h in rules_hits))
    if rules_blocks:
        parts.append("Rules text:\n" + "\n\n".join(rules_blocks))

    return {
        "context": "\n\n".join(parts),
        "card_names": [c["name"] for c in cards],
        "card_match_types": [c["match_type"] for c in cards],
        "ruling_card_names": [r["name"] for r in rulings],
        "card_keyword_rule_ids": card_rule_ids,
        "keyword_rules_injected": [rid for rid, _ in keyword_rules],
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
