"""Evaluate rules comprehension across systems and eval sets (README Section 9).

"Do not judge by loss alone. Build a real rules exam." Runs every eval
question through four system configurations and scores each:

  - base            plain mlx-community/Qwen2.5-7B-Instruct-4bit, no context
  - base_rag        base model + RAG-retrieved rules context (Section 6)
  - finetuned       the Section 8 LoRA adapter alone, no context
  - finetuned_rag   the adapter + RAG-retrieved context (the intended
                    final architecture per Section 6.1: "fine-tuned
                    reasoner + rules retrieval")

Section 9.4 only asks for fine-tuned vs. base and vs. base+RAG, but the
Section 8.6 spot-checks found the adapter ignoring correct retrieved
context outright — that's specifically a finetuned_rag failure, so it
gets its own arm rather than being inferred from the other three.

Scoring (Section 9.3):
  - Automated: does each answer cite a rule ID that (a) actually exists
    in the pinned CR, and (b) overlaps the reference's supporting rule
    IDs where we have them.
  - Model-graded: the base model (not the adapter under test, and not
    involved in generating any candidate for a given question) scores
    every system's answer for a question against the reference in one
    call, 1-5.
  - A small consistency check reruns a subset of questions on the
    finetuned_rag arm (the one meant for production) and reports
    score agreement.
  - Human review is out of scope for this script — the report flags
    the lowest-scoring cases for a human pass, it doesn't replace one.

Usage:
    python scripts/eval.py [--synthetic-limit 70] [--reddit-limit 40]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag import MODEL_ID as EMBED_MODEL_ID
from rag import retrieve

CROSS_REF_RE = re.compile(r"\b\d{3}\.\d+[a-z]?\b")
BASE_MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
ADAPTER_PATH = "models/mtg-rules-adapter-best"

SYSTEM_PROMPT = (
    "You are a Magic: The Gathering rules expert. Answer precisely and "
    "cite comprehensive rule numbers."
)
RAG_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    " Use ONLY the provided rules text to answer; do not rely on outside knowledge."
)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert Magic: The Gathering rules judge grading AI-generated answers. "
    "You will be given a QUESTION, a REFERENCE ANSWER (assumed correct), and several "
    "CANDIDATE ANSWERS labeled by system name. Score each candidate's correctness "
    "against the reference on a 1-5 scale: 5 = fully correct and complete, "
    "4 = correct but missing a minor detail, 3 = partially correct, "
    "2 = mostly incorrect, 1 = wrong or fabricated (including a fabricated or "
    "contradicted rule citation). Output ONLY a JSON object mapping each system name "
    "to {\"score\": <1-5>, \"note\": \"<one short phrase>\"}. No other text."
)


def load_rule_ids(rules_path: Path) -> set[str]:
    ids = set()
    with rules_path.open(encoding="utf-8") as f:
        for line in f:
            ids.add(json.loads(line)["rule_id"])
    return ids


def load_questions(synthetic_path: Path, reddit_path: Path, synthetic_limit: int, reddit_limit: int) -> list[dict]:
    questions = []

    with synthetic_path.open(encoding="utf-8") as f:
        synthetic = [json.loads(l) for l in f]
    for r in synthetic[:synthetic_limit]:
        questions.append(
            {
                "source": "synthetic",
                "category": r.get("category"),
                "question": r["messages"][1]["content"],
                "reference": r["messages"][2]["content"],
                "supporting_rule_ids": r.get("supporting_rule_ids", []),
            }
        )

    with reddit_path.open(encoding="utf-8") as f:
        reddit = [json.loads(l) for l in f]
    reddit.sort(key=lambda r: -r["reddit_score"])
    if reddit_limit and len(reddit) > reddit_limit:
        stride = len(reddit) / reddit_limit
        reddit = [reddit[int(i * stride)] for i in range(reddit_limit)]
    for r in reddit:
        questions.append(
            {
                "source": "reddit",
                "category": None,
                "question": r["messages"][1]["content"],
                "reference": r["messages"][2]["content"],
                "supporting_rule_ids": r.get("cited_rule_ids", []) + r.get("retrieved_rule_ids", []),
            }
        )

    return questions


def build_prompt(tokenizer, question: str, context: str | None) -> str:
    if context:
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Rules text:\n{context}\n\nQuestion: {question}"},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def retrieve_context(question: str, embed_model, k: int = 3) -> str:
    hits = retrieve(question, k=k, model_and_tokenizer=embed_model)
    return "\n\n".join(h["text"] for h in hits)


def generate_all_answers(questions: list[dict], embed_model, max_tokens: int, adapter_path_under_test: str) -> dict[str, list[str]]:
    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm

    answers: dict[str, list[str]] = {}
    contexts = [retrieve_context(q["question"], embed_model) for q in questions]

    for arm_name, adapter_path in [("base", None), ("finetuned", adapter_path_under_test)]:
        print(f"loading model for arm(s) using adapter_path={adapter_path} ...")
        model, tokenizer = load_lm(BASE_MODEL_ID, adapter_path=adapter_path)

        for use_rag, out_key in [(False, arm_name), (True, f"{arm_name}_rag")]:
            print(f"generating arm: {out_key}")
            out = []
            for i, q in enumerate(questions, 1):
                ctx = contexts[i - 1] if use_rag else None
                prompt = build_prompt(tokenizer, q["question"], ctx)
                out.append(lm_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False))
                if i % 25 == 0 or i == len(questions):
                    print(f"  {out_key}: {i}/{len(questions)}")
            answers[out_key] = out

    return answers


def score_citations(answer: str, supporting_rule_ids: list[str], valid_rule_ids: set[str]) -> dict:
    cited = set(CROSS_REF_RE.findall(answer))
    return {
        "cited": sorted(cited),
        "has_fabricated": bool(cited - valid_rule_ids),
        "matches_reference": bool(cited & set(supporting_rule_ids)) if supporting_rule_ids else None,
    }


def judge_batch(lm_generate, judge_model, judge_tokenizer, question: str, reference: str, candidates: dict[str, str], max_tokens: int) -> dict:
    candidate_block = "\n\n".join(f"CANDIDATE ({name}):\n{text}" for name, text in candidates.items())
    user = f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\n{candidate_block}"
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    prompt = judge_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    raw = lm_generate(judge_model, judge_tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", type=Path, default=Path("eval/rules_questions.jsonl"))
    parser.add_argument("--reddit", type=Path, default=Path("eval/reddit_questions.jsonl"))
    parser.add_argument("--synthetic-limit", type=int, default=70)
    parser.add_argument("--reddit-limit", type=int, default=40)
    parser.add_argument("--rules", type=Path, default=Path("data/processed/rules.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--judge-max-tokens", type=int, default=500)
    parser.add_argument("--consistency-sample", type=int, default=15)
    parser.add_argument("--adapter-path", default=ADAPTER_PATH, help="adapter under test for the finetuned arms")
    parser.add_argument("--out", type=Path, default=Path("eval/eval_results.jsonl"))
    parser.add_argument("--report-out", type=Path, default=Path("eval/EVAL_REPORT.md"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from mlx_embeddings import load as load_embedder
    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm

    questions = load_questions(args.synthetic, args.reddit, args.synthetic_limit, args.reddit_limit)
    print(f"{len(questions)} eval questions ({args.synthetic_limit} synthetic + up to {args.reddit_limit} reddit)")

    valid_rule_ids = load_rule_ids(args.rules)
    print(f"loading {EMBED_MODEL_ID} for retrieval ...")
    embed_model = load_embedder(EMBED_MODEL_ID)

    answers = generate_all_answers(questions, embed_model, args.max_tokens, args.adapter_path)
    arm_names = list(answers.keys())

    # Consistency check: rerun a subset of finetuned_rag questions and see
    # how often the judge would even need to know — same generation config,
    # does the model give a stable answer.
    print(f"loading {BASE_MODEL_ID} (adapter) for consistency rerun ...")
    ft_model, ft_tokenizer = load_lm(BASE_MODEL_ID, adapter_path=args.adapter_path)
    consistency_idx = list(range(min(args.consistency_sample, len(questions))))
    consistency_reruns = []
    for i in consistency_idx:
        ctx = retrieve_context(questions[i]["question"], embed_model)
        prompt = build_prompt(ft_tokenizer, questions[i]["question"], ctx)
        consistency_reruns.append(lm_generate(ft_model, ft_tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False))
    del ft_model, ft_tokenizer

    print(f"loading {BASE_MODEL_ID} (no adapter) as judge ...")
    judge_model, judge_tokenizer = load_lm(BASE_MODEL_ID)

    results = []
    for i, q in enumerate(questions):
        candidates = {arm: answers[arm][i] for arm in arm_names}
        judged = judge_batch(lm_generate, judge_model, judge_tokenizer, q["question"], q["reference"], candidates, args.judge_max_tokens)

        per_arm = {}
        for arm in arm_names:
            citation = score_citations(candidates[arm], q["supporting_rule_ids"], valid_rule_ids)
            judge_result = judged.get(arm, {})
            per_arm[arm] = {
                "answer": candidates[arm],
                "citation": citation,
                "judge_score": judge_result.get("score"),
                "judge_note": judge_result.get("note"),
            }

        results.append(
            {
                "source": q["source"],
                "category": q["category"],
                "question": q["question"],
                "reference": q["reference"],
                "arms": per_arm,
                "consistency_rerun": consistency_reruns[i] if i in consistency_idx else None,
            }
        )
        if (i + 1) % 20 == 0 or i + 1 == len(questions):
            print(f"judged {i + 1}/{len(questions)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate.
    summary = {arm: {"scores": [], "fabricated": 0, "matches": 0, "match_total": 0} for arm in arm_names}
    for r in results:
        for arm, data in r["arms"].items():
            if data["judge_score"] is not None:
                summary[arm]["scores"].append(data["judge_score"])
            if data["citation"]["has_fabricated"]:
                summary[arm]["fabricated"] += 1
            if data["citation"]["matches_reference"] is not None:
                summary[arm]["match_total"] += 1
                if data["citation"]["matches_reference"]:
                    summary[arm]["matches"] += 1

    consistency_agree = sum(
        1 for i in consistency_idx
        if results[i]["arms"]["finetuned_rag"]["answer"].strip() == consistency_reruns[i].strip()
    )

    lines = ["# Section 9 Evaluation Report\n"]
    lines.append(f"{len(questions)} questions ({args.synthetic_limit} synthetic + {len(questions) - args.synthetic_limit} reddit)\n")
    lines.append("| Arm | Avg score (1-5) | N scored | Fabricated citation | Citation matches reference |")
    lines.append("| --- | --- | --- | --- | --- |")
    for arm in arm_names:
        s = summary[arm]
        avg = sum(s["scores"]) / len(s["scores"]) if s["scores"] else float("nan")
        match_rate = f"{s['matches']}/{s['match_total']}" if s["match_total"] else "n/a"
        lines.append(f"| {arm} | {avg:.2f} | {len(s['scores'])}/{len(questions)} | {s['fabricated']}/{len(questions)} | {match_rate} |")
    lines.append("")
    lines.append(f"Consistency (finetuned_rag, {len(consistency_idx)} questions rerun): {consistency_agree}/{len(consistency_idx)} identical on rerun.\n")

    if summary.get("finetuned_rag") and summary.get("base_rag"):
        ft_scores = summary["finetuned_rag"]["scores"]
        base_scores = summary["base_rag"]["scores"]
        ft_avg = sum(ft_scores) / len(ft_scores) if ft_scores else 0
        base_avg = sum(base_scores) / len(base_scores) if base_scores else 0
        verdict = (
            "Fine-tuning (with RAG) beats RAG-only on this exam."
            if ft_avg > base_avg
            else "Fine-tuning (with RAG) does NOT beat RAG-only on this exam — per Section 9.4, fix the SFT data, not the hyperparameters."
        )
        lines.append(f"**Section 9.4 verdict:** finetuned_rag avg {ft_avg:.2f} vs base_rag avg {base_avg:.2f}. {verdict}\n")

    lowest = sorted(results, key=lambda r: min((a["judge_score"] or 5) for a in r["arms"].values()))[:10]
    lines.append("## Lowest-scoring cases (for human review)\n")
    for r in lowest:
        worst_arm = min(r["arms"], key=lambda a: (r["arms"][a]["judge_score"] or 5))
        lines.append(f"- [{r['source']}] \"{r['question'][:100]}\" — worst: {worst_arm} (score {r['arms'][worst_arm]['judge_score']}, {r['arms'][worst_arm]['judge_note']})")

    args.report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nfull results -> {args.out}")
    print(f"report -> {args.report_out}")


if __name__ == "__main__":
    main()
