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


def wiki_examples(path: Path) -> list[dict]:
    """SFT lines from the wiki gloss, each carrying its provenance in the answer.

    Off unless `--with-wiki` is passed. Two things make this different from every
    other input here, and both are stated in the text the model is trained on
    rather than only in a comment:

      * it is a GLOSS, not the rules, so an answer derived from it must not
        present itself as the Comprehensive Rules;
      * it is editable, so the revision is the only durable identifier.

    The pin is verified first. Training on a corpus that has drifted from the one
    a number was computed against is the contamination failure in a different
    coat: it looks like an improvement while making the run unreproducible.
    """
    from common import verify_wiki_pin
    verify_wiki_pin(path)
    out = []
    for c in read_jsonl(path):
        answer = (f"{c['text']}\n\n"
                  f"(Source: MTG Wiki, {c['title']}, revision {c['revision']} — "
                  f"community-edited reference, not the Comprehensive Rules.)")
        out.append({"messages": build_rag_messages(
            f"Explain: {c['title']}" + (f" ({c['heading']})" if c.get("heading") else ""))
            + [{"role": "assistant", "content": answer}]})
    return out


def gameplay_examples(path: Path) -> list[dict]:
    """SFT lines from reviewed board positions' reference lines.

    Off unless `--with-gameplay` is passed, the same explicit-opt-in shape as
    `--with-wiki`, for the same reason: this is a distribution the verified
    rules set carries none of (a board instead of a sentence, the gameplay
    action grammar instead of prose), so mixing it in is a labelled decision,
    not a default.

    Only positions carrying `reference_actions` qualify — the 100%-correct
    line, refused into the gold set unless the parser agrees it is legal
    (Section 21.85's rubric_server invariant) — never `answer`, which is a
    prose explanation for a human, not the action-grammar output the model is
    trained to produce. As of this writing that is 8 of 32 positions; the
    other 24 have a correct *explanation* but no line verified against
    `legal_actions`, and training on an unverified line would teach the
    action grammar from an example nobody checked.

    `common.build_position_messages(pos, closed=False)` is the SAME function
    `eval_positions.py` uses to build the `open` arm's prompt — sharing it is
    what keeps train and inference from drifting apart (Section 8.7's lesson,
    applied to the gameplay track this time).
    """
    from common import build_position_messages, read_jsonl as _read
    out = []
    for pos in _read(path):
        actions = pos.get("reference_actions")
        if not actions:
            continue
        target = "\n".join(actions)
        out.append({"messages": build_position_messages(pos, closed=False)
                     + [{"role": "assistant", "content": target}]})
    return out


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
    ap.add_argument("--with-wiki", action="store_true",
                    help="also train on the MTG Wiki gloss (data/processed/wiki_chunks.jsonl). "
                         "OFF by default and deliberately explicit: the corpus is "
                         "community-edited, and training bakes it into weights where it can "
                         "no longer be labelled unofficial, traced to a revision, or removed "
                         "if a page turns out to be wrong. Licence CC BY-NC-SA 2.5 — "
                         "noncommercial use only, and the adapter arguably inherits.")
    ap.add_argument("--with-gameplay", action="store_true",
                    help="also train on data/gold/positions.jsonl records carrying "
                         "reference_actions (PLAN_NEXT.md item 3's gameplay-mixture "
                         "experiment). OFF by default and deliberately explicit — a "
                         "separate, labelled experiment, not folded into the default set.")
    ap.add_argument("--positions", type=Path, default=REPO_ROOT / "data/gold/positions.jsonl")
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
    train_lines = build_examples(train)
    # Wiki lines go to TRAIN only, never to valid. Validation loss is read as a
    # signal about the verified gold set; mixing a second distribution into it
    # would move the curve for a reason unrelated to the thing being measured.
    if args.with_wiki:
        from common import WIKI_CHUNKS_PATH
        wiki = wiki_examples(WIKI_CHUNKS_PATH)
        train_lines += wiki
        print(f"  + {len(wiki)} wiki gloss lines (CC BY-NC-SA 2.5, unofficial) "
              f"-> train only")
    if args.with_gameplay:
        gameplay = gameplay_examples(args.positions)
        train_lines += gameplay
        print(f"  + {len(gameplay)} gameplay reference-line examples "
              f"(reviewed, legal_actions-verified) -> train only")
    write_jsonl_atomic(args.out_dir / "train.jsonl", train_lines)
    write_jsonl_atomic(args.out_dir / "valid.jsonl", build_examples(valid))
    # Every exclusion, with WHICH filter caught it. Recording only the `id`
    # matches would leave the file describing 72 of the 90 — and an audit trail
    # that silently omits the cases the audit was written to find is worse than
    # none, because it reads as confirmation.
    write_jsonl_atomic(args.out_dir / "excluded_ids.jsonl",
                       [{"id": c["id"], "rulesguru_id": c.get("rulesguru_id"),
                         "reason": reason}
                        for c, reason in excluded])
    # len(train_lines), not len(train): with --with-wiki the file holds more
    # lines than there are gold records, and reporting the record count for a
    # file that has more in it is how an epoch figure ends up describing a
    # dataset that does not exist (Section 21.13's duplicate count, inverted).
    print(f"\nwrote {len(train_lines)} train / {len(valid)} valid -> {args.out_dir}")
    print("excluded_ids.jsonl records what was held out, so the exclusion is auditable")


if __name__ == "__main__":
    main()
