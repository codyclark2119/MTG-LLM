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


def _record_context(rec: dict, card_index, ruling_index, embed_model,
                     lm_tokenizer, max_seq_length: int) -> str:
    """The same context shape `retrieve_hybrid.build_context` emits at eval
    time, but resolving cards from the record's own verified `cards` field
    (exact/normalized dictionary lookup) rather than `find_in_text` scanning
    the question prose for `[[brackets]]`.

    That distinction is the whole point (Section 21.136): `gold_candidates.
    jsonl` questions are plain prose with no bracket syntax anywhere, so
    `find_in_text` resolves zero cards on essentially all of them — which is
    exactly why `eval.py --with-cards` has never attached real card or
    ruling text to a `gold_questions.jsonl` run. The `cards` field was
    authored alongside the question, not scanned from it, so resolving
    THOSE names carries none of find_in_text's need for bracket markup.

    Truncated to fit `max_seq_length` ALONGSIDE the question and answer,
    never at their expense (Section 21.137). The first version of this
    function had no such budget: 960 of 1,001 built examples had a context
    long enough that `mlx_lm.lora`'s fixed-length truncation cut the
    ANSWER away entirely, training on empty or near-empty targets, and the
    resulting run's loss diverged to permanent NaN around iteration 260.
    Sections are dropped least-essential-first — Rules text (redundant with
    the plain rules-RAG shape v4 already covers), then Official rulings,
    then Cards referenced — until what remains plus the question and answer
    fits, so a training line's target is NEVER the thing sacrificed to fit
    a token budget.
    """
    from rag import retrieve as rag_retrieve

    names = rec.get("cards") or []
    resolved, seen = [], set()
    for n in names:
        card, _how = card_index.resolve(n)
        if card and card["name"] not in seen:
            seen.add(card["name"])
            resolved.append(card)
    rulings = ruling_index.find_for_cards(resolved) if ruling_index else []
    rules_hits = rag_retrieve(rec["question"], k=3, model_and_tokenizer=embed_model)

    cards_part = "Cards referenced:\n" + "\n\n".join(c["text"] for c in resolved) if resolved else None
    rulings_part = "Official rulings:\n" + "\n\n".join(r["text"] for r in rulings) if rulings else None
    rules_part = "Rules text:\n" + "\n\n".join(h["text"] for h in rules_hits)

    def _fits(context: str) -> bool:
        messages = (build_rag_messages(rec["question"], context, preformatted=True)
                    + [{"role": "assistant", "content": rec["answer"]}])
        return len(lm_tokenizer.apply_chat_template(messages, add_generation_prompt=False)) <= max_seq_length

    # Dropped least-essential-first until the question and answer, which are
    # never sacrificed, fit alongside whatever context remains.
    for combo in ([cards_part, rulings_part, rules_part],
                  [cards_part, rulings_part],
                  [cards_part],
                  []):
        combo = [p for p in combo if p]
        context = "\n\n".join(combo)
        if not combo or _fits(context):
            return context
    return ""


def build_examples_with_context(records: list[dict], max_seq_length: int = 2048) -> list[dict]:
    """Like `build_examples`, but each line carries REAL retrieved context —
    the CARDS_RAG_SYSTEM_PROMPT/RAG_SYSTEM_PROMPT shape `finetuned_rag` and
    `finetuned_rag_cards_rulings` are evaluated under.

    Off by default (`--with-retrieved-context`). Every adapter trained so
    far (v1-v5) trained on the bare question with NO retrieval context at
    all — `stamp_adapter.unseen_arms` has been warning about exactly this
    gap since Section 21.50, and it is why `finetuned_rag`'s own numbers
    have never been more than a guess at how the adapter behaves outside
    the one shape it was actually trained on. This is the retrofit
    half of the two-part rulings experiment the user chose (PLAN_NEXT.md
    item 3): train on the real shape first, author a new ruling-grounded
    corpus second, and compare.
    """
    from card_lookup import CardIndex
    from retrieve_hybrid import RulingIndex
    from rag import MODEL_ID as EMBED_MODEL_ID
    from mlx_embeddings import load as load_embed_model
    from mlx_lm import load as load_lm
    from eval import BASE_MODEL_ID

    print("loading card index, ruling index, embedding model, and tokenizer for retrieved context ...")
    card_index = CardIndex()
    ruling_index = RulingIndex()
    # Loaded once and passed to every _record_context call — rag.retrieve
    # reloads the embedding model from scratch on every call when this is
    # left as None, which is fine for a single lookup but would reload it
    # ~1,100 times here.
    embed_model = load_embed_model(EMBED_MODEL_ID)
    # Only the tokenizer is needed (to measure what mlx_lm.lora will actually
    # count), but mlx_lm.load returns model+tokenizer together; the model
    # itself is never used below and MLX does not materialize its weights
    # until they are read.
    _unused_model, lm_tokenizer = load_lm(BASE_MODEL_ID)

    out = []
    n_with_cards = n_with_rulings = n_trimmed = 0
    for i, r in enumerate(records, 1):
        context = _record_context(r, card_index, ruling_index, embed_model,
                                   lm_tokenizer, max_seq_length)
        if context.startswith("Cards referenced:"):
            n_with_cards += 1
        if "Official rulings:" in context:
            n_with_rulings += 1
        if "Rules text:" not in context:
            n_trimmed += 1
        out.append({"messages": build_rag_messages(r["question"], context, preformatted=True)
                     + [{"role": "assistant", "content": r["answer"]}]})
        if i % 200 == 0 or i == len(records):
            print(f"  context built: {i}/{len(records)}")
    print(f"  {n_with_cards}/{len(records)} examples carry real card context "
          f"({n_with_rulings}/{len(records)} also carry official rulings)")
    print(f"  {n_trimmed}/{len(records)} examples had their context trimmed "
          f"(Rules text dropped, or further) to keep the full answer inside "
          f"max_seq_length={max_seq_length}")
    return out


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
    ap.add_argument("--with-retrieved-context", action="store_true",
                    help="build every train/valid line with REAL retrieved card/ruling/"
                         "rules context (Section 21.136), instead of the bare question "
                         "every adapter through v5 trained on. Resolves cards from each "
                         "record's own verified `cards` field, not by scanning the "
                         "question for [[brackets]] (which gold_candidates.jsonl's prose "
                         "never contains). This changes the SHAPE of the whole default "
                         "set, not an additive mixture like --with-wiki/--with-gameplay — "
                         "point --out-dir at a new directory rather than overwriting "
                         "data/datasets/verified.")
    ap.add_argument("--max-seq-length", type=int, default=2048,
                    help="only used with --with-retrieved-context: the training "
                         "config's own max_seq_length, so context gets trimmed to "
                         "the SAME budget mlx_lm.lora will actually enforce, rather "
                         "than a mismatched guess (Section 21.137).")
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

    if args.with_retrieved_context:
        def builder(records):
            return build_examples_with_context(records, max_seq_length=args.max_seq_length)
    else:
        builder = build_examples

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        ex = builder(train[:1])[0]
        print("sample system prompt (first 80 chars):", ex["messages"][0]["content"][:80])
        print("sample assistant target:")
        print("  " + ex["messages"][-1]["content"][:220])
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_lines = builder(train)
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
    write_jsonl_atomic(args.out_dir / "valid.jsonl", builder(valid))
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
