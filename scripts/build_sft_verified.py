"""Build an SFT set from human-verified answers instead of model-generated ones.

    python scripts/build_sft_verified.py --dry-run
    python scripts/build_sft_verified.py --out-dir data/datasets/verified

WHY THIS EXISTS

`build_sft.py` synthesizes training targets with Qwen2.5-7B — the same model the
adapter is trained on top of. A student cannot exceed its teacher, and Section
21.4 measured what that cost:

    18% of the 2,971 SFT targets are REFUSALS

i.e. the adapter was explicitly trained to answer "The rules provided do not
contain specific information about ... Therefore, I cannot answer." Roughly one
example in five taught it to decline. That is a sufficient explanation on its
own for the fine-tuned arm scoring lowest on correctness and producing the
shortest answers in every run since Section 9.

`data/gold/gold_candidates.jsonl` holds RulesGuru records whose answers were
written and verified by people. Measured against the same refusal test:

    1,130 uncontaminated pairs, 0.1% refusals, 100% cite a rule id inline

Grounding is the largest clean effect in the project (86/99 vs 45/99, Section
19.1), and inline citation is exactly what these carry and the synthetic set
cannot.

CONTAMINATION IS THE FAILURE THAT WOULD INVALIDATE EVERYTHING DOWNSTREAM

72 of the 1,202 candidates have been promoted into `gold_questions.jsonl`, which
is the n=99 eval set. Training on those would make every subsequent eval number
meaningless, and it would look like an improvement. The exclusion is asserted
here, in the builder, rather than done once by hand — `--dry-run` prints the
overlap it removed, and the run aborts if any excluded id survives into the
output.

The prompt shape is delegated to `common.build_rag_messages`, the same function
`eval.py` uses at inference, so train and inference cannot drift apart. That
mismatch is the project's #1 documented failure mode (Section 8.7).
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (  # noqa: E402
    GOLD_CANDIDATES_PATH,
    GOLD_PATH,
    REFUSAL_RE,
    REPO_ROOT,
    RULE_ID_RE,
    build_rag_messages,
    read_jsonl,
    write_jsonl_atomic,
)

# Imported from the auditor rather than redefined, so "what the builder
# excludes" and "what the audit looks for" cannot drift into disagreeing —
# a divergent copy of exactly this comparison is what let the five through.
from audit_sft import jaccard as _jaccard  # noqa: E402
from audit_sft import tokens as _tokens  # noqa: E402


def build_examples(records: list[dict]) -> list[dict]:
    """One training line per record, in the shape RAG inference uses.

    No retrieved context is attached. These questions carry their own card
    references and the verified answer cites its own rules, so injecting
    retrieval here would train the model to expect a context block that the
    answer does not actually depend on.
    """
    return [
        {"messages": build_rag_messages(r["question"])
                     + [{"role": "assistant", "content": r["answer"]}]}
        for r in records
    ]


def stratified_split(records: list[dict], valid_frac: float, seed: int):
    """Split within category, so a thin category cannot land entirely in one side."""
    by_cat: dict[str, list[dict]] = {}
    for r in records:
        by_cat.setdefault(r.get("category") or "uncategorized", []).append(r)
    rng = random.Random(seed)
    train, valid = [], []
    for cat in sorted(by_cat):
        rows = by_cat[cat][:]
        rng.shuffle(rows)
        n_valid = max(1, round(len(rows) * valid_frac)) if len(rows) > 1 else 0
        valid.extend(rows[:n_valid])
        train.extend(rows[n_valid:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, default=GOLD_CANDIDATES_PATH)
    ap.add_argument("--gold", type=Path, default=GOLD_PATH,
                    help="records here are EXCLUDED — this is the eval set")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/datasets/verified")
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--near-threshold", type=float, default=0.75,
                    help="question-token overlap with a gold record at which a "
                         "candidate is excluded as the same question reworded")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    candidates = read_jsonl(args.candidates)
    gold_records = read_jsonl(args.gold)
    gold_ids = {g["id"] for g in gold_records}

    # Three filters, because matching on `id` alone provably was not enough.
    #
    # `audit_sft.py` found five gold eval questions present verbatim in the
    # training set, and the id filter had passed all five:
    #
    #   3 were promoted into gold under a `qa-*` id while keeping their
    #     rulesguru_id (1156, 1796, 3518). `c["id"]` is "rg-1156" and
    #     `g["id"]` is "qa-amy-casts-assassin-s-trophy-...", so they never
    #     compared equal. Excluding on rulesguru_id catches these exactly.
    #
    #   2 are DIFFERENT RulesGuru entries that ask the same question — the
    #     upstream database contains duplicates (rg-308 and rg-777 are both the
    #     Serum Powder mulligan question), and one gold record predates the
    #     RulesGuru pull entirely and carries no rulesguru_id. Nothing exact
    #     can catch those, so question text is the third filter.
    #
    # The threshold is deliberately loose. Dropping a handful of extra training
    # records costs almost nothing against 1,130; keeping one contaminated
    # record invalidates every eval number a model trained here produces. The
    # asymmetry is the whole argument, and it points one way.
    gold_rg_ids = {str(g["rulesguru_id"]) for g in gold_records if g.get("rulesguru_id")}
    gold_qtokens = [_tokens(g.get("question") or "") for g in gold_records]

    kept, excluded = [], []
    for c in candidates:
        if c["id"] in gold_ids:
            excluded.append((c, "same id as a gold record"))
        elif c.get("rulesguru_id") and str(c["rulesguru_id"]) in gold_rg_ids:
            excluded.append((c, "same rulesguru_id as a gold record"))
        elif (ct := _tokens(c.get("question") or "")) and max(
                (_jaccard(ct, gt) for gt in gold_qtokens), default=0.0) >= args.near_threshold:
            excluded.append((c, f"question text >={args.near_threshold:.0%} overlap with a gold record"))
        else:
            kept.append(c)
    dropped_by = Counter(r for _, r in excluded)

    removed = len(candidates) - len(kept)
    usable = [c for c in kept if (c.get("question") or "").strip()
              and (c.get("answer") or "").strip()]

    # Assertion, not a comment. If this ever fires, every eval number produced
    # by a model trained on this set describes questions it was trained on.
    leaked = [c["id"] for c in usable
              if c["id"] in gold_ids
              or (c.get("rulesguru_id") and str(c["rulesguru_id"]) in gold_rg_ids)]
    if leaked:
        raise SystemExit(f"CONTAMINATION: {len(leaked)} eval ids survived the filter: {leaked[:5]}")

    refusals = [c for c in usable if REFUSAL_RE.search(c["answer"])]
    cited = sum(1 for c in usable if RULE_ID_RE.search(c["answer"]))

    print(f"candidates              : {len(candidates)}")
    print(f"  excluded (in gold set): {removed}"
          + (f"  [{', '.join(f'{k}={v}' for k, v in dropped_by.most_common())}]" if dropped_by else ""))
    print(f"  usable                : {len(usable)}")
    print(f"  refusal-shaped answers: {len(refusals)} ({len(refusals) / len(usable):.1%})")
    print(f"  citing a rule inline  : {cited} ({cited / len(usable):.0%})")
    print("  by category           : "
          + ", ".join(f"{c}={n}" for c, n in
                      Counter(r.get("category") or "?" for r in usable).most_common()))

    train, valid = stratified_split(usable, args.valid_frac, args.seed)
    print(f"\nsplit: train={len(train)} valid={len(valid)}")

    # The trap this repo has already fallen into once: a config comment claiming
    # ~1 epoch while the run did 0.45. Recompute rather than inherit.
    for batch, iters in ((4, 600), (4, 1000), (8, 600)):
        print(f"  batch={batch} iters={iters} -> "
              f"{iters * batch / max(1, len(train)):.2f} epochs")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        ex = build_examples(train[:1])[0]
        print("sample assistant target:")
        print("  " + ex["messages"][-1]["content"][:220])
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.out_dir / "train.jsonl", build_examples(train))
    write_jsonl_atomic(args.out_dir / "valid.jsonl", build_examples(valid))
    # Every exclusion, with WHICH filter caught it. Recording only the `id`
    # matches would leave the file describing 72 of the 90 — and an audit trail
    # that silently omits the cases the audit was written to find is worse than
    # none, because it reads as confirmation.
    write_jsonl_atomic(args.out_dir / "excluded_ids.jsonl",
                       [{"id": c["id"], "rulesguru_id": c.get("rulesguru_id"),
                         "reason": reason}
                        for c, reason in excluded])
    print(f"\nwrote {len(train)} train / {len(valid)} valid -> {args.out_dir}")
    print("excluded_ids.jsonl records what was held out, so the exclusion is auditable")


if __name__ == "__main__":
    main()
