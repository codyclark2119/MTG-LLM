"""Build the supervised fine-tuning dataset from chunked rules + glossary.

Implements README Section 7. Design notes on how each generation-workflow
step (7.2) maps to code:

  1. Seed with real questions -> assign_category() buckets every chunk by
     which of the 7.1 categories its section/rule-group naturally covers
     (e.g. section 4 = Zones -> "zone transition", rule group 704 ->
     "state-based actions"), then the model is prompted to write a
     question of that style from the chunk's own text. Categories the
     README calls out as the hardest ("Magic rules interactions are logic
     puzzles", Section 1.1) get a second, scenario-style question per
     chunk. Definition recall is generated deterministically straight
     from glossary.jsonl instead of via the LLM, since the glossary
     already *is* a curated, human-reviewed definition bank.
  2. Generate answers via RAG -> each chunk already carries its own
     directly-cross-referenced rules and glossary definitions (built in
     Section 5's reference-aware bundling), so it IS the retrieval result
     a query about its content would return. The model is instructed to
     answer strictly from that text and nothing else.
  3. Cite rule IDs -> enforced via prompt instruction; citation presence
     is checked afterward and gaps are flagged for review, not silently
     dropped.
  4. Human-review a sample -> this script cannot judge MTG-rules
     correctness. It flags low-confidence records (missing citations) to
     data/processed/sft_needs_review.jsonl for a human pass.
  5. Format as instruction/response JSONL -> build_messages().

Usage:
    python scripts/build_sft.py                  # full run
    python scripts/build_sft.py --limit 20        # smoke test on 20 chunks
    python scripts/build_sft.py --resume           # continue an interrupted run
    python scripts/build_sft.py --skip-generation  # rebuild splits from existing raw generations
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from common import CHUNKS_PATH, DATASETS_DIR, GLOSSARY_PATH, INDEX_PATH, REPO_ROOT, RULES_PATH, build_rag_messages, read_jsonl
from common import RULE_ID_RE as CROSS_REF_RE

REDIRECT_RE = re.compile(r"^See [^.]+\.$")

CATEGORY_STYLE = {
    "definition recall": 'a concise definition-recall question (e.g. "What is X?")',
    "turn-structure walkthrough": (
        "a question asking the player to walk through a sequence of turn/phase/step structure in order"
    ),
    "priority reasoning": "a question about who has priority, when, or what a player may do at a given point",
    "interaction puzzle": (
        'a two-player interaction scenario question in the style "Player A does X, '
        'Player B responds with Y — what happens?"'
    ),
    "state-based actions": "a question about when a specific state-based action is checked and what happens",
    "zone transition": "a question about where an object goes when it changes zones and what triggers that",
    "layer-system question": "a question about how continuous effects are applied in layers and in what order",
    "templating/keyword meaning": "a question asking what a specific keyword action or keyword ability means mechanically",
}
PUZZLE_STYLE = (
    'a two-player interaction scenario question in the style "Player A does X, then Player B '
    'does Y — what happens, and why?", testing precise sequencing'
)
# How many question variants to generate per chunk assigned to each
# category. The README singles these out as the hardest, most
# leverage-heavy categories ("Magic rules interactions are logic puzzles",
# Section 1.1), and several of them (priority, state-based actions, layers)
# only have a handful of source chunks, so they need a bigger multiplier
# than "interaction puzzle" to end up with a usable number of examples.
BOOST_COUNTS = {
    "interaction puzzle": 2,
    "priority reasoning": 2,
    "state-based actions": 4,
    "layer-system question": 4,
}


def assign_category(chunk: dict) -> str:
    section, group = chunk["section"], chunk["rule_group"]
    if group == "117":
        return "priority reasoning"
    if section == "4":
        return "zone transition"
    if section == "5":
        # Turn-structure text is often really about priority timing
        # ("once it begins, the active player gets priority") rather than
        # step sequencing — route by content instead of lumping it all
        # into one category.
        return "priority reasoning" if "priority" in chunk["text"].lower() else "turn-structure walkthrough"
    if group == "613":
        return "layer-system question"
    if group == "704":
        return "state-based actions"
    if group in ("701", "702"):
        return "templating/keyword meaning"
    if section == "6":
        return "interaction puzzle"
    return "definition recall"


load_jsonl = read_jsonl  # one definition, in common.py


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def parse_json_object(raw: str) -> dict | None:
    # Models sometimes add stray commentary or markdown fences around the
    # JSON; pull out the first {...} block rather than requiring an exact match.
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    q, a = obj.get("question", "").strip(), obj.get("answer", "").strip()
    if not q or not a:
        return None
    return {"question": q, "answer": a}


def generate_qa(model, tokenizer, chunk: dict, style: str, max_tokens: int):
    from mlx_lm import generate

    system = (
        "You write Magic: The Gathering rules-comprehension training data. "
        'Given ONLY the rules text provided, output a single JSON object with keys '
        f'"question" and "answer". The question should be {style}, answerable '
        "strictly from the provided text. The answer must cite specific rule "
        "numbers (formatted like 509.1a) that appear in the provided text, be a "
        "few sentences, and use no outside knowledge. Output ONLY the JSON "
        "object, no markdown fences, no commentary."
    )
    user = f"Rules text:\n{chunk['text']}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    result = parse_json_object(raw)
    if result is None:
        # One retry with a stricter nudge — cheap insurance against an
        # occasional stray preamble before the JSON.
        messages[0]["content"] += " Respond with valid JSON only."
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        result = parse_json_object(raw)
    return result


def run_generation(chunks: list[dict], model_id: str, raw_out: Path, max_tokens: int, limit: int | None) -> None:
    from mlx_lm import load

    done = {(r["chunk_id"], r["variant"]) for r in load_jsonl(raw_out)}

    tasks = []
    for chunk in chunks[:limit] if limit else chunks:
        category = assign_category(chunk)
        tasks.append((chunk, category, CATEGORY_STYLE[category], "base"))
        for extra in range(1, BOOST_COUNTS.get(category, 1)):
            tasks.append((chunk, category, PUZZLE_STYLE, f"puzzle-{extra}"))
    tasks = [t for t in tasks if (t[0]["chunk_id"], t[3]) not in done]

    if not tasks:
        print("nothing to generate (already complete, or --limit excludes everything)")
        return

    print(f"loading {model_id} ...")
    model, tokenizer = load(model_id)

    start = time.time()
    ok, failed = 0, 0
    for i, (chunk, category, style, variant) in enumerate(tasks, 1):
        qa = generate_qa(model, tokenizer, chunk, style, max_tokens)
        if qa is None:
            failed += 1
            print(f"  [{i}/{len(tasks)}] chunk {chunk['chunk_id']} ({variant}): failed to parse JSON, skipping", file=sys.stderr)
            continue
        append_jsonl(
            raw_out,
            {
                "chunk_id": chunk["chunk_id"],
                "variant": variant,
                "category": category,
                "question": qa["question"],
                "answer": qa["answer"],
                "supporting_rule_ids": chunk["rule_ids"] + chunk["cross_ref_rule_ids"],
                "source": f"chunk:{chunk['chunk_id']}",
            },
        )
        ok += 1
        if i % 25 == 0 or i == len(tasks):
            elapsed = time.time() - start
            rate = elapsed / i
            eta_min = rate * (len(tasks) - i) / 60
            print(f"  [{i}/{len(tasks)}] ok={ok} failed={failed} ({rate:.1f}s/ex, ~{eta_min:.0f} min left)")

    print(f"generation complete: {ok} examples written, {failed} failed -> {raw_out}")


def categorize_by_rule_ids(rule_ids: list[str], section_by_group: dict[str, str]) -> str:
    """Infer a 7.1 category from which rules a question's retrieval landed on.

    Reddit questions arrive uncategorized, and running an 800-call LLM
    classification pass to label them would cost more than it's worth when
    the retrieved rule IDs already say what the question is about. Reuses
    assign_category()'s section/group mapping so both sources land in the
    same taxonomy.
    """
    if not rule_ids:
        return "interaction puzzle"
    votes: dict[str, int] = {}
    for rid in rule_ids:
        group = rid.split(".")[0]
        section = section_by_group.get(group)
        pseudo_chunk = {"section": section, "rule_group": group, "text": ""}
        category = assign_category(pseudo_chunk)
        votes[category] = votes.get(category, 0) + 1
    return max(votes.items(), key=lambda kv: kv[1])[0]


def load_eval_questions(*paths: Path) -> set[str]:
    """Every question already committed to a held-out eval set.

    eval/sets/rules_questions.jsonl was carved out of this same generation pool by
    an earlier run, so re-splitting from scratch would silently leak held-out
    questions into training and invalidate every Section 9 comparison. These
    are excluded by exact question text before any split happens.
    """
    questions = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                for m in record["messages"]:
                    if m["role"] == "user":
                        # Strip the RAG wrapper if present so bare and grounded
                        # phrasings of the same question both match.
                        text = m["content"]
                        if "\n\nQuestion: " in text:
                            text = text.split("\n\nQuestion: ", 1)[1]
                        questions.add(text.strip())
    return questions


def generate_reddit_sourced(
    reddit_dataset: str, count: int, min_score: int, excluded: set[str],
    model_id: str, raw_out: Path, chunks_path: Path, index_path: Path,
    max_tokens: int, seed: int,
) -> None:
    """Answer REAL player questions, grounded in retrieved rules text.

    Section 9.5 flagged that the synthetic set's question phrasing is
    narrow — it's all generated from rules text, so it reads like rules
    text. Reddit supplies genuinely messy player phrasing (card names,
    partial context, mid-game confusion). Only the QUESTIONS are taken;
    answers are generated fresh against retrieved rules so citation quality
    and format stay consistent with the rest of the set, rather than
    inheriting Reddit's casual uncited prose.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from datasets import load_dataset
    from mlx_embeddings import load as load_embedder
    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm

    from rag import MODEL_ID as EMBED_MODEL_ID
    from rag import retrieve

    done = {r["question"] for r in load_jsonl(raw_out) if r.get("variant") == "reddit-sourced"}

    ds = load_dataset(reddit_dataset, split="train")
    candidates = [
        r for r in ds
        if r["score"] >= min_score
        and len(r["prompt"]) >= 40
        and r["prompt"].strip() not in excluded
        and r["prompt"].strip() not in done
    ]
    random.Random(seed).shuffle(candidates)
    candidates = candidates[:count]
    if not candidates:
        print("no new reddit questions to generate")
        return

    print(f"loading models for {len(candidates)} reddit-sourced generations ...")
    embed_model = load_embedder(EMBED_MODEL_ID)
    model, tokenizer = load_lm(model_id)

    start = time.time()
    ok = 0
    for i, r in enumerate(candidates, 1):
        question = r["prompt"].strip()
        try:
            hits = retrieve(question, k=3, chunks_path=chunks_path, index_path=index_path, model_and_tokenizer=embed_model)
        except Exception as e:
            print(f"  retrieval failed, skipping: {e}", file=sys.stderr)
            continue
        context = "\n\n".join(h["text"] for h in hits)
        supporting = sorted({rid for h in hits for rid in h["rule_ids"]})

        system = (
            "You are a Magic: The Gathering rules expert writing training data. "
            "Answer the player's question using ONLY the provided rules text. "
            "Cite specific rule numbers (formatted like 509.1a) that appear in that "
            "text. Be concise and precise. If the provided rules text does not "
            "contain enough information to answer, say so plainly instead of guessing."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Rules text:\n{context}\n\nQuestion: {question}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        answer = lm_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False).strip()
        if not answer:
            continue

        append_jsonl(
            raw_out,
            {
                "chunk_id": None,
                "variant": "reddit-sourced",
                "category": "interaction puzzle",
                "question": question,
                "answer": answer,
                "supporting_rule_ids": supporting,
                "grounding_context": context,
                "source": "reddit-question+rag-answer",
            },
        )
        ok += 1
        if i % 25 == 0 or i == len(candidates):
            rate = (time.time() - start) / i
            print(f"  [{i}/{len(candidates)}] ok={ok} ({rate:.1f}s/ex, ~{rate * (len(candidates) - i) / 60:.0f} min left)")

    print(f"reddit-sourced generation complete: {ok} examples -> {raw_out}")


def definition_examples(glossary: list[dict], limit: int, seed: int) -> list[dict]:
    candidates = [g for g in glossary if not REDIRECT_RE.match(g["definition"]) and len(g["definition"]) > 20]
    random.Random(seed).shuffle(candidates)
    examples = []
    for g in candidates[:limit]:
        examples.append(
            {
                "chunk_id": None,
                "variant": "definition",
                "category": "definition recall",
                "question": f'What does "{g["term"]}" mean in Magic: The Gathering?',
                "answer": g["definition"],
                "supporting_rule_ids": g["related_rules"],
                "source": f"glossary:{g['term']}",
            }
        )
    return examples


def has_citation(record: dict) -> bool:
    cited = set(CROSS_REF_RE.findall(record["answer"]))
    supporting = set(record["supporting_rule_ids"])
    return bool(cited & supporting)


def build_messages(record: dict) -> dict:
    return {
        "messages": build_rag_messages(record["question"])
        + [{"role": "assistant", "content": record["answer"]}]
    }


def truncate_context(context: str, budget_chars: int) -> str:
    """Trim grounding context to a char budget, on chunk/line boundaries.

    Sizing this matters more than it looks. Grounded examples put the
    context in the prompt and the answer at the END, so any example longer
    than the trainer's max_seq_length gets its answer truncated away —
    leaving no target tokens and producing a NaN loss (observed at
    iteration 20 of the first v2 attempt). Long sequences also blow up
    attention memory quadratically: uncapped k=3 contexts drove peak memory
    to 65GB on a 36GB machine, which swapped hard and took the machine down
    twice. The budget must leave room for the question and answer within
    max_seq_length.
    """
    if len(context) <= budget_chars:
        return context
    kept: list[str] = []
    used = 0
    for para in context.split("\n\n"):
        if used + len(para) > budget_chars:
            break
        kept.append(para)
        used += len(para) + 2
    if not kept:  # single oversized paragraph — hard cut on a line boundary
        lines = context[:budget_chars].split("\n")
        return "\n".join(lines[:-1]) if len(lines) > 1 else context[:budget_chars]
    return "\n\n".join(kept)


def build_grounded_messages(record: dict, context: str) -> dict:
    """Training example in the SAME shape RAG inference uses.

    The v1 dataset trained only on bare question -> answer, but the intended
    production architecture (Section 6.1) puts retrieved rules text in the
    prompt. That train/inference mismatch is the most likely cause of the
    Section 9.5 result where the fine-tuned model ignored correct retrieved
    context and answered from parametric memory instead: it had never once
    been shown an example where the answer was supposed to come from text in
    the prompt.

    Shape is delegated to common.build_rag_messages, which eval.py also uses,
    so the two cannot drift apart again.
    """
    return {
        "messages": build_rag_messages(record["question"], context)
        + [{"role": "assistant", "content": record["answer"]}]
    }


def stratified_split(records: list[dict], train_frac: float, valid_frac: float, seed: int):
    by_category: dict[str, list[dict]] = {}
    for r in records:
        by_category.setdefault(r["category"], []).append(r)

    train, valid, eval_ = [], [], []
    rng = random.Random(seed)
    for cat_records in by_category.values():
        rng.shuffle(cat_records)
        n = len(cat_records)
        n_train = round(n * train_frac)
        n_valid = round(n * valid_frac)
        train += cat_records[:n_train]
        valid += cat_records[n_train : n_train + n_valid]
        eval_ += cat_records[n_train + n_valid :]
    return train, valid, eval_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--glossary", type=Path, default=GLOSSARY_PATH)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--raw-out", type=Path, default=REPO_ROOT / "data/processed/sft_raw_generations.jsonl")
    # Chunk-derived "definition recall" is already ~40% of the generated
    # set (definitional/structural rules genuinely make up a large share
    # of the CR) — kept modest so glossary defs supplement rather than
    # dominate (Section 7.4: "don't over-index on definitions").
    parser.add_argument("--defs-limit", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--out-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--eval-out", type=Path, default=REPO_ROOT / "eval/sets/rules_questions.jsonl")
    parser.add_argument("--needs-review-out", type=Path, default=REPO_ROOT / "data/processed/sft_needs_review.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N chunks (smoke testing)")
    parser.add_argument("--skip-generation", action="store_true", help="rebuild splits from an existing raw-out file")
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--rules", type=Path, default=RULES_PATH)
    parser.add_argument("--reddit-dataset", default="Javier-Jimenez99/reddit-mtgrules-qa")
    parser.add_argument("--reddit-questions", type=int, default=0, help="generate N RAG-grounded answers to real reddit questions")
    parser.add_argument("--reddit-min-score", type=int, default=3)
    parser.add_argument(
        "--grounded", action="store_true",
        help="also emit a RAG-format copy of every example whose grounding context is known (fixes the "
             "train/inference mismatch identified in Section 9.5)",
    )
    parser.add_argument(
        "--held-out", type=Path, nargs="*",
        default=[Path("eval/sets/rules_questions.jsonl"), Path("eval/sets/reddit_questions.jsonl")],
        help="eval files whose questions must never enter train/valid",
    )
    parser.add_argument("--freeze-eval", action="store_true", help="keep the existing eval file rather than re-splitting one")
    # ~4000 chars ≈ 1000 tokens of context, leaving comfortable room for the
    # question and answer inside the trainer's 2048-token max_seq_length.
    # See truncate_context() for why exceeding it is not a soft failure.
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--max-example-chars", type=int, default=7000, help="drop grounded copies that still exceed this")
    args = parser.parse_args()

    chunks = load_jsonl(args.chunks)
    glossary = load_jsonl(args.glossary)
    chunk_text_by_id = {c["chunk_id"]: c["text"] for c in chunks}

    held_out_questions = load_eval_questions(*args.held_out)
    print(f"{len(held_out_questions)} held-out eval questions will be excluded from train/valid")

    if not args.skip_generation:
        run_generation(chunks, args.model, args.raw_out, args.max_tokens, args.limit)

    if args.reddit_questions:
        generate_reddit_sourced(
            args.reddit_dataset, args.reddit_questions, args.reddit_min_score,
            held_out_questions, args.model, args.raw_out, args.chunks, args.index,
            args.max_tokens, args.seed,
        )

    generated = load_jsonl(args.raw_out)

    # Reddit-sourced records were written with a placeholder category; assign
    # a real one from the rules their retrieval landed on so the stratified
    # split and the balance report both mean something.
    rules = load_jsonl(args.rules)
    section_by_group = {r["rule_id"].split(".")[0]: r["section"] for r in rules}
    for r in generated:
        if r.get("variant") == "reddit-sourced":
            r["category"] = categorize_by_rule_ids(r.get("supporting_rule_ids", []), section_by_group)

    defs = definition_examples(glossary, args.defs_limit, args.seed)
    all_records = generated + defs

    if not all_records:
        print("no examples to split — run generation first", file=sys.stderr)
        sys.exit(1)

    leaked = [r for r in all_records if r["question"].strip() in held_out_questions]
    all_records = [r for r in all_records if r["question"].strip() not in held_out_questions]
    if leaked:
        print(f"excluded {len(leaked)} records whose question appears in a held-out eval set")

    needs_review = [r for r in all_records if not has_citation(r)]
    args.needs_review_out.parent.mkdir(parents=True, exist_ok=True)
    with args.needs_review_out.open("w", encoding="utf-8") as f:
        for r in needs_review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.freeze_eval:
        # Section 9's numbers are only comparable across runs if the exam
        # doesn't change under them — keep the existing eval file as-is and
        # split the remainder into train/valid only.
        train_frac = args.train_frac / (args.train_frac + args.valid_frac)
        train, valid, eval_ = stratified_split(all_records, train_frac, 1 - train_frac, args.seed)
        eval_ = []
    else:
        train, valid, eval_ = stratified_split(all_records, args.train_frac, args.valid_frac, args.seed)

    oversized_dropped = 0

    def emit(records: list[dict], path: Path) -> int:
        nonlocal oversized_dropped
        written = 0
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(build_messages(r), ensure_ascii=False) + "\n")
                written += 1
                if args.grounded:
                    context = r.get("grounding_context") or chunk_text_by_id.get(r.get("chunk_id"))
                    if context:
                        grounded = build_grounded_messages(
                            r, truncate_context(context, args.max_context_chars)
                        )
                        if sum(len(m["content"]) for m in grounded["messages"]) > args.max_example_chars:
                            oversized_dropped += 1
                            continue
                        f.write(json.dumps(grounded, ensure_ascii=False) + "\n")
                        written += 1
        return written

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_train = emit(train, args.out_dir / "train.jsonl")
    n_valid = emit(valid, args.out_dir / "valid.jsonl")

    if eval_:
        args.eval_out.parent.mkdir(parents=True, exist_ok=True)
        with args.eval_out.open("w", encoding="utf-8") as f:
            for r in eval_:
                record = build_messages(r)
                record["category"] = r["category"]
                record["supporting_rule_ids"] = r["supporting_rule_ids"]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_category_counts: dict[str, int] = {}
    by_variant_counts: dict[str, int] = {}
    for r in all_records:
        by_category_counts[r["category"]] = by_category_counts.get(r["category"], 0) + 1
        by_variant_counts[r.get("variant", "?")] = by_variant_counts.get(r.get("variant", "?"), 0) + 1

    print(f"\n{len(all_records)} source examples ({len(generated)} generated + {len(defs)} definitions)")
    for cat, n in sorted(by_category_counts.items()):
        print(f"  {cat}: {n}")
    print("variants:", dict(sorted(by_variant_counts.items())))
    print(f"split: train={len(train)} valid={len(valid)} eval={len(eval_) if eval_ else '(frozen)'}")
    print(f"written training lines: train.jsonl={n_train} valid.jsonl={n_valid}" + (" (bare + grounded)" if args.grounded else ""))
    if oversized_dropped:
        print(f"dropped {oversized_dropped} grounded copies still over {args.max_example_chars} chars after truncation")
    print(f"citation check: {len(all_records) - len(needs_review)}/{len(all_records)} cite a supporting rule ID")
    if needs_review:
        print(f"{len(needs_review)} examples flagged for human review -> {args.needs_review_out}")
    print(
        "\nThis dataset was generated automatically and has not been verified "
        "for MTG-rules correctness — human-review a sample before trusting it "
        "(README Section 7.2 step 4), starting with the needs-review file."
    )

if __name__ == "__main__":
    main()
