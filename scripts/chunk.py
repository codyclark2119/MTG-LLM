"""Group parsed rules into self-contained, reference-aware training/retrieval chunks.

Implements README Section 5: keeps a rule and its subrules together as the
base unit (5.2), splits oversized rule groups on rule boundaries rather
than mid-sentence, appends directly cross-referenced rule text within a
bounded budget so a chunk stands alone, and injects short glossary
definitions for the terms a chunk actually uses.

Usage:
    python scripts/chunk.py [--rules PATH] [--glossary PATH] [--out PATH]
                             [--target-tokens N] [--ref-budget-tokens N] [--max-refs N]
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def estimate_tokens(text: str) -> int:
    # Rough, tokenizer-agnostic heuristic (~4 chars/token for English) —
    # the base model isn't chosen yet (Section 1.3), so an exact count
    # would be false precision. Good enough to hit "a few hundred to
    # ~1,000 tokens" (Section 5.3).
    return max(1, len(text) // 4)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def group_rules(rules: list[dict]) -> "OrderedDict[str, list[dict]]":
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for rule in rules:
        group_id = rule["rule_id"].split(".")[0]
        groups.setdefault(group_id, []).append(rule)
    return groups


def render_rule(rule: dict) -> str:
    return f"{rule['rule_id']}. {rule['text']}"


def split_on_rule_boundaries(rules: list[dict], target_tokens: int) -> list[list[dict]]:
    """Greedily bin-pack a rule group's members into chunks. Each rule
    record is already the atomic unit (a subrule is never split mid-text),
    so this satisfies "split on subrule boundaries, never mid-sentence."""
    parts: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0
    for rule in rules:
        rule_tokens = estimate_tokens(render_rule(rule))
        if current and current_tokens + rule_tokens > target_tokens:
            parts.append(current)
            current, current_tokens = [], 0
        current.append(rule)
        current_tokens += rule_tokens
    if current:
        parts.append(current)
    return parts


def build_chunk(
    part: list[dict],
    chunk_id: str,
    by_id: dict[str, dict],
    glossary_by_term: dict[str, str],
    ref_budget_tokens: int,
    max_refs: int,
    glossary_budget_tokens: int,
) -> dict:
    member_ids = {r["rule_id"] for r in part}
    body = "\n".join(render_rule(r) for r in part)

    # Reference-aware bundling: pull in directly cross-referenced rules
    # (in the order first referenced) up to a bounded token budget, so a
    # chunk that says "see rule 508.1" doesn't force a second retrieval.
    referenced_ids: list[str] = []
    seen_refs = set(member_ids)
    for rule in part:
        for ref in rule["cross_refs"]:
            if ref not in seen_refs and ref in by_id:
                referenced_ids.append(ref)
                seen_refs.add(ref)

    appended, ref_tokens_used = [], 0
    for ref in referenced_ids[:max_refs]:
        ref_rule = by_id[ref]
        rendered = render_rule(ref_rule)
        cost = estimate_tokens(rendered)
        if ref_tokens_used + cost > ref_budget_tokens:
            break
        appended.append(ref_rule)
        ref_tokens_used += cost

    text = body
    if appended:
        text += "\n\nRelated rules:\n" + "\n".join(render_rule(r) for r in appended)

    # Glossary injection: short definitions for terms used in this chunk's
    # own rules (not the appended related rules). Keyword-ability rule
    # groups (e.g. 702.x) can each match 40+ terms, so this is budgeted
    # like the related-rules appendix rather than dumped in unbounded.
    all_terms = sorted({t for r in part for t in r["glossary_terms"]})
    included_terms, gloss_tokens_used = [], 0
    for t in all_terms:
        definition = glossary_by_term.get(t)
        if definition is None:
            continue
        cost = estimate_tokens(f"- {t}: {definition}")
        if gloss_tokens_used + cost > glossary_budget_tokens:
            break
        included_terms.append(t)
        gloss_tokens_used += cost
    if included_terms:
        text += "\n\nGlossary:\n" + "\n".join(
            f"- {t}: {glossary_by_term[t]}" for t in included_terms
        )

    first = part[0]
    return {
        "chunk_id": chunk_id,
        "section": first["section"],
        "section_title": first["section_title"],
        "rule_group": first["rule_id"].split(".")[0],
        "rule_ids": sorted(member_ids, key=lambda r: (len(r), r)),
        "cross_ref_rule_ids": [r["rule_id"] for r in appended],
        "glossary_terms": included_terms,
        "text": text,
        "approx_tokens": estimate_tokens(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=Path("data/processed/rules.jsonl"))
    parser.add_argument("--glossary", type=Path, default=Path("data/processed/glossary.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/chunks.jsonl"))
    # Tuned so body + related-rules + glossary together land near the
    # README's "a few hundred to ~1,000 tokens" target (Section 5.3)
    # rather than each budget stacking independently toward 1,500+.
    parser.add_argument("--target-tokens", type=int, default=600)
    parser.add_argument("--ref-budget-tokens", type=int, default=200)
    parser.add_argument("--max-refs", type=int, default=5)
    parser.add_argument("--glossary-budget-tokens", type=int, default=200)
    args = parser.parse_args()

    rules = load_jsonl(args.rules)
    glossary = load_jsonl(args.glossary)
    by_id = {r["rule_id"]: r for r in rules}
    glossary_by_term = {g["term"]: g["definition"] for g in glossary}

    chunks: list[dict] = []
    for group_id, group_rules_list in group_rules(rules).items():
        parts = split_on_rule_boundaries(group_rules_list, args.target_tokens)
        for i, part in enumerate(parts):
            chunk_id = group_id if len(parts) == 1 else f"{group_id}-{i + 1}"
            chunks.append(
                build_chunk(
                    part, chunk_id, by_id, glossary_by_term,
                    args.ref_budget_tokens, args.max_refs, args.glossary_budget_tokens,
                )
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    token_counts = [c["approx_tokens"] for c in chunks]
    print(f"built {len(chunks)} chunks from {len(rules)} rules -> {args.out}")
    print(
        f"approx tokens per chunk: min={min(token_counts)} "
        f"max={max(token_counts)} avg={sum(token_counts) / len(token_counts):.0f}"
    )
    full_budget = args.target_tokens + args.ref_budget_tokens + args.glossary_budget_tokens
    over_budget = sum(1 for t in token_counts if t > full_budget)
    if over_budget:
        print(f"note: {over_budget} chunks exceed target+ref budget (single oversized rules)")


if __name__ == "__main__":
    main()
