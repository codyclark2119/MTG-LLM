"""Build a real-world eval set from community-vetted r/MTGRules Q&A.

Complements the synthetic held-out set from Section 7.4/9.1 with actual
questions Magic players asked and had answered, filtered to
well-upvoted (community-vetted) responses. Source dataset:
Javier-Jimenez99/reddit-mtgrules-qa on Hugging Face (CC-BY-SA-4.0),
itself scraped from Reddit's r/MTGRules.

This is NOT run through the same citation-verification pipeline as
Section 7's generated data with full confidence — Reddit answers were
posted across many years of Comprehensive Rules revisions, so a rule
number a post cites may have been renumbered or changed meaning since.
Every citation found in a response is checked against the CURRENT
pinned rules.jsonl and flagged, not silently trusted, if it doesn't
resolve. Each record also gets a best-effort RAG retrieval against the
current chunk index so there's a present-day grounding reference
alongside the original community answer.

Reddit's upvote score turns out to reward humor as much as accuracy —
spot-checking the top-scored candidates surfaced joke replies ("Isn't
Marty's cause to get back to the future?" at score 13) sitting right
next to genuine rulings. Response-vs-retrieved-chunk embedding
similarity was tried as a cheap filter but didn't separate the two
reliably (a real ruling scored 0.297, a joke scored 0.370 — no usable
threshold). So an actual LLM judge classifies every score/length-
filtered candidate as a genuine ruling attempt or not before anything
else happens to it.

Usage:
    python scripts/build_reddit_eval.py [--min-score 10] [--limit 200]
"""

import argparse
import json
from pathlib import Path

from common import CHUNKS_PATH, CR_VERSION, INDEX_PATH, REPO_ROOT, RULES_PATH, SYSTEM_PROMPT, load_rule_ids
from common import RULE_ID_RE as CROSS_REF_RE

MIN_PROMPT_LEN = 30
MIN_RESPONSE_LEN = 20


# Named for what it does, not for the role it plays. eval.py also had a
# JUDGE_SYSTEM_PROMPT that meant something entirely different (grade this
# answer, not classify this comment) — the same one-name-two-meanings trap
# Section 15.3 documented for CROSS_REF_RE.
RULING_FILTER_PROMPT = (
    "You classify Reddit r/MTGRules posts. Given a QUESTION and a REPLY, answer "
    "with exactly one word: YES if the reply is a genuine, substantive attempt to "
    "answer the Magic: The Gathering rules question, or NO if it is a joke, meme, "
    "non-answer, off-topic comment, or doesn't actually address the question. "
    "Answer with only YES or NO, nothing else."
)


def is_genuine_ruling(generate_fn, model, tokenizer, prompt_text: str, response_text: str) -> bool:
    user = f"QUESTION:\n{prompt_text}\n\nREPLY:\n{response_text}"
    messages = [
        {"role": "system", "content": RULING_FILTER_PROMPT},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    out = generate_fn(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
    return out.strip().upper().startswith("Y")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="Javier-Jimenez99/reddit-mtgrules-qa")
    parser.add_argument("--min-score", type=int, default=10)
    parser.add_argument("--pre-judge-limit", type=int, default=400, help="cap on candidates sent to the LLM judge")
    parser.add_argument("--limit", type=int, default=200, help="final eval-set size after judging")
    parser.add_argument("--judge-model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--rules", type=Path, default=RULES_PATH)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval/sets/reddit_questions.jsonl")
    parser.add_argument("--manifest-out", type=Path, default=REPO_ROOT / "eval/sets/reddit_questions.manifest.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-cards", action="store_true",
        help="keep only questions naming a card that resolves — builds the card-focused set "
             "needed to give the Section 13.5 comparison enough statistical power",
    )
    parser.add_argument(
        "--exclude-from", type=Path, nargs="*", default=[],
        help="jsonl files whose questions must be excluded (existing eval sets, training data)",
    )
    parser.add_argument("--skip-judge", action="store_true", help="skip LLM quality filtering (fast, for testing only)")
    parser.add_argument("--skip-retrieval", action="store_true", help="skip RAG grounding lookup (faster, for testing)")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train")

    candidates = [
        r for r in ds
        if r["score"] >= args.min_score
        and len(r["prompt"]) >= MIN_PROMPT_LEN
        and len(r["response"]) >= MIN_RESPONSE_LEN
    ]
    print(f"{len(candidates)}/{len(ds)} pass score>={args.min_score} + length filters")

    # Exclude anything already used for training or in an existing eval set,
    # so a "new" eval set can't quietly re-test questions the model was
    # trained on or that were already scored.
    excluded_questions: set[str] = set()
    for path in args.exclude_from:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if "messages" in rec:
                    for m in rec["messages"]:
                        if m["role"] == "user":
                            t = m["content"]
                            excluded_questions.add(t.split("\n\nQuestion: ", 1)[-1].strip())
                elif "question" in rec:
                    excluded_questions.add(rec["question"].strip())
    if excluded_questions:
        before = len(candidates)
        candidates = [r for r in candidates if r["prompt"].strip() not in excluded_questions]
        print(f"excluded {before - len(candidates)} already used in training/eval ({len(excluded_questions)} known)")

    if args.require_cards:
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).parent))
        from card_lookup import CardIndex

        print("loading card index to filter for resolvable card references ...")
        card_index = CardIndex()
        before = len(candidates)
        candidates = [r for r in candidates if card_index.find_in_text(r["prompt"])]
        print(f"{len(candidates)}/{before} name at least one card that resolves")

    # Highest-signal answers first, but cap so we don't just take the single
    # most-upvoted topic cluster repeated — sort by score, then take an even
    # spread rather than a strict top-N.
    candidates.sort(key=lambda r: -r["score"])
    if args.pre_judge_limit and len(candidates) > args.pre_judge_limit:
        stride = len(candidates) / args.pre_judge_limit
        candidates = [candidates[int(i * stride)] for i in range(args.pre_judge_limit)]

    if not args.skip_judge:
        from mlx_lm import generate as lm_generate
        from mlx_lm import load as load_lm

        print(f"loading {args.judge_model} for quality judging...")
        judge_model, judge_tokenizer = load_lm(args.judge_model)

        kept = []
        for i, r in enumerate(candidates, 1):
            if is_genuine_ruling(lm_generate, judge_model, judge_tokenizer, r["prompt"], r["response"]):
                kept.append(r)
            if i % 50 == 0 or i == len(candidates):
                print(f"  judged {i}/{len(candidates)}, {len(kept)} kept so far")
        print(f"judge kept {len(kept)}/{len(candidates)} as genuine ruling attempts")
        candidates = kept

    if args.limit and len(candidates) > args.limit:
        candidates.sort(key=lambda r: -r["score"])
        stride = len(candidates) / args.limit
        candidates = [candidates[int(i * stride)] for i in range(args.limit)]

    rule_ids = load_rule_ids(args.rules)

    embed_model = None
    if not args.skip_retrieval:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from rag import MODEL_ID, retrieve
        from mlx_embeddings import load as load_embedder

        print(f"loading {MODEL_ID} for retrieval grounding...")
        embed_model = load_embedder(MODEL_ID)

    records = []
    for r in candidates:
        cited = sorted(set(CROSS_REF_RE.findall(r["response"])))
        cited_valid = [c for c in cited if c in rule_ids]
        cited_stale = [c for c in cited if c not in rule_ids]

        retrieved_rule_ids = []
        if not args.skip_retrieval:
            try:
                hits = retrieve(
                    r["prompt"], k=3, chunks_path=args.chunks, index_path=args.index,
                    model_and_tokenizer=embed_model,
                )
                for h in hits:
                    retrieved_rule_ids.extend(h["rule_ids"])
            except Exception as e:
                print(f"  retrieval failed for a prompt, continuing without it: {e}")

        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["prompt"].strip()},
                    {"role": "assistant", "content": r["response"].strip()},
                ],
                "source": "reddit:r/MTGRules",
                "reddit_score": r["score"],
                "cited_rule_ids": cited_valid,
                "cited_rule_ids_stale": cited_stale,
                "retrieved_rule_ids": sorted(set(retrieved_rule_ids)),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stale_count = sum(1 for r in records if r["cited_rule_ids_stale"])
    args.manifest_out.write_text(
        f"# Reddit r/MTGRules eval set\n\n"
        f"- Source: {args.dataset} (Hugging Face, CC-BY-SA-4.0), itself scraped from "
        f"Reddit's r/MTGRules\n"
        f"- Filter: score >= {args.min_score}, prompt >= {MIN_PROMPT_LEN} chars, "
        f"response >= {MIN_RESPONSE_LEN} chars, {'LLM-filtered for genuine on-topic rulings (Reddit score alone rewards jokes as much as accuracy — see script docstring), ' if not args.skip_judge else ''}"
        f"evenly sampled down to {len(records)} records\n"
        f"- License: CC-BY-SA-4.0 — attribution required, derivatives must share-alike\n"
        f"- These are real community answers, not verified by an actual Magic rules judge — "
        f"correctness is not guaranteed even after LLM filtering. Explicit rule-number "
        f"citations in the original answers are checked against the currently-pinned CR "
        f"({CR_VERSION}) and flagged in `cited_rule_ids_stale` when they don't resolve — "
        f"Reddit posts span many CR revisions, so a cited number may have since been "
        f"renumbered or reused. {stale_count}/{len(records)} records have at least one "
        f"stale citation.\n"
        f"- `retrieved_rule_ids`: best-effort RAG grounding against the current chunk "
        f"index (Section 6), for cross-checking the community answer against current rules "
        f"text — not a guarantee of correctness either.\n",
        encoding="utf-8",
    )

    print(f"\n{len(records)} examples -> {args.out}")
    print(f"manifest -> {args.manifest_out}")
    print(f"{stale_count}/{len(records)} records have a citation that doesn't resolve against the current CR")

if __name__ == "__main__":
    main()
