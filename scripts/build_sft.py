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

CROSS_REF_RE = re.compile(r"\b\d{3}\.\d+[a-z]?\b")
REDIRECT_RE = re.compile(r"^See [^.]+\.$")

SYSTEM_PROMPT = (
    "You are a Magic: The Gathering rules expert. Answer precisely and "
    "cite comprehensive rule numbers."
)

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


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


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
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": record["answer"]},
        ]
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
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--glossary", type=Path, default=Path("data/processed/glossary.jsonl"))
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--raw-out", type=Path, default=Path("data/processed/sft_raw_generations.jsonl"))
    # Chunk-derived "definition recall" is already ~40% of the generated
    # set (definitional/structural rules genuinely make up a large share
    # of the CR) — kept modest so glossary defs supplement rather than
    # dominate (Section 7.4: "don't over-index on definitions").
    parser.add_argument("--defs-limit", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--out-dir", type=Path, default=Path("data/datasets"))
    parser.add_argument("--eval-out", type=Path, default=Path("eval/rules_questions.jsonl"))
    parser.add_argument("--needs-review-out", type=Path, default=Path("data/processed/sft_needs_review.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="only process the first N chunks (smoke testing)")
    parser.add_argument("--skip-generation", action="store_true", help="rebuild splits from an existing raw-out file")
    args = parser.parse_args()

    chunks = load_jsonl(args.chunks)
    glossary = load_jsonl(args.glossary)

    if not args.skip_generation:
        run_generation(chunks, args.model, args.raw_out, args.max_tokens, args.limit)

    generated = load_jsonl(args.raw_out)
    defs = definition_examples(glossary, args.defs_limit, args.seed)
    all_records = generated + defs

    if not all_records:
        print("no examples to split — run generation first", file=sys.stderr)
        sys.exit(1)

    needs_review = [r for r in all_records if not has_citation(r)]
    args.needs_review_out.parent.mkdir(parents=True, exist_ok=True)
    with args.needs_review_out.open("w", encoding="utf-8") as f:
        for r in needs_review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train, valid, eval_ = stratified_split(all_records, args.train_frac, args.valid_frac, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(build_messages(r), ensure_ascii=False) + "\n")
    with (args.out_dir / "valid.jsonl").open("w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(build_messages(r), ensure_ascii=False) + "\n")

    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    with args.eval_out.open("w", encoding="utf-8") as f:
        for r in eval_:
            record = build_messages(r)
            record["category"] = r["category"]
            record["supporting_rule_ids"] = r["supporting_rule_ids"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_category_counts: dict[str, int] = {}
    for r in all_records:
        by_category_counts[r["category"]] = by_category_counts.get(r["category"], 0) + 1

    print(f"\n{len(all_records)} total examples ({len(generated)} generated + {len(defs)} definitions)")
    for cat, n in sorted(by_category_counts.items()):
        print(f"  {cat}: {n}")
    print(f"split: train={len(train)} valid={len(valid)} eval={len(eval_)}")
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
