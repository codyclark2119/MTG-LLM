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
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: F401  (SYSTEM_PROMPT re-exported for callers)
    CARDS_RAG_SYSTEM_PROMPT,
    REPO_ROOT,
    RAG_SYSTEM_PROMPT,
    RULES_PATH,
    SYSTEM_PROMPT,
    load_rule_ids,
)
from common import RULE_ID_RE as CROSS_REF_RE
from rag import MODEL_ID as EMBED_MODEL_ID
from rag import retrieve

# DEFAULTS, not constants. Both of these are overridable per run (--base-model,
# --adapter-path) and both are recorded in the results file, because a stale
# default here has already produced a wrong run once: ADAPTER_PATH pointed at
# the v1 adapter long after v2 superseded it, so a bare `python scripts/eval.py`
# silently evaluated the old one.
#
# Nothing downstream may read these directly — take the model from args and
# pass it down. That is what keeps swapping in a new checkpoint (or a larger
# base model, which fits at 36GB for inference) a flag change rather than an
# edit.
BASE_MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
ADAPTER_PATH = "models/mtg-rules-adapter-v2-best"

# The v1 judge prompt and its judge_batch() were removed in the Section 17
# review: Section 9.7 replaced them with the length-neutral, anonymized v2
# judge and RE-SCORED both earlier runs under it (rules_v1_rejudged.md,
# rules_v2_rejudged.md), so nothing referenced them any more. The v1
# results they produced are preserved in eval/; the code is in git history.

# v2 judge. Section 9.6 measured a length bias in the v1 prompt (removed
# above, see the note): score
# correlated with answer length at r = +0.21, and the four arms ranked by
# score in exactly the order they ranked by verbosity. A near-verbatim
# correct one-line answer was scored 4 "missing detail" while a longer
# restatement of the same fact scored 5 — penalizing the fine-tuned model
# for the concision its training data taught it.
#
# Two changes: correctness and citation validity are scored separately so a
# right-but-terse answer can't be docked for thoroughness it was never asked
# for, and length-neutrality is stated as an explicit rule rather than left
# implicit. Candidates are also anonymized behind randomized A/B/C/D labels
# in judge_batch_anonymized so the judge can't favor a system by name.
JUDGE_SYSTEM_PROMPT_V2 = (
    "You are an expert Magic: The Gathering rules judge grading answers.\n\n"
    "You get a QUESTION, a REFERENCE ANSWER (treat as correct), and several "
    "CANDIDATE answers labeled A, B, C, D. Score each candidate on two "
    "independent 1-5 scales:\n\n"
    "correctness — does it state the same ruling as the reference?\n"
    "  5 = states the same ruling, no contradictions\n"
    "  4 = same ruling, one small imprecision\n"
    "  3 = partially right, or right but omits something the question asked for\n"
    "  2 = mostly wrong\n"
    "  1 = wrong, or contradicts the reference\n\n"
    "citation — are the comprehensive-rule numbers it cites real and relevant?\n"
    "  5 = cites a correct, relevant rule number\n"
    "  3 = cites nothing at all\n"
    "  1 = cites a rule number that is fabricated, wrong, or contradicts its own claim\n\n"
    "CRITICAL SCORING RULES:\n"
    "- Judge ONLY factual accuracy. Length, verbosity, tone, and formatting are "
    "IRRELEVANT.\n"
    "- A short answer that states the correct ruling is FULLY correct. Do NOT "
    "deduct for brevity, for omitting background, or for 'missing detail' when "
    "the ruling itself is right and complete.\n"
    "- A long answer is not better for being long. Extra correct detail earns "
    "nothing; extra INCORRECT detail must be penalized.\n"
    "- If a candidate says the reference's rules text does not answer the "
    "question, and that is true, score correctness 4-5 rather than penalizing "
    "it for declining to guess.\n\n"
    "Output ONLY a JSON object mapping each label to "
    '{"correctness": <1-5>, "citation": <1-5>, "note": "<short phrase>"}. '
    "No other text."
)


# v3 judge. Sections 9.6-9.9 kept treating judge disagreement as a prompt
# wording problem, but the disagreement is structural: asking for a holistic
# 1-5 "how close is this to my one phrasing?" has no objective answer, so two
# judges landed at r = +0.43 and effects below ~0.5 became unmeasurable.
#
# This prompt does not ask for a score at all. It asks which enumerated
# claims a candidate asserted and which known misconceptions it fell into —
# extraction questions with checkable answers — and the score is computed
# from the counts here in Python. Two judges can still disagree about whether
# a claim was asserted, but they can no longer disagree about the arithmetic.
#
# Requires a rubric, so it only applies to questions carrying `key_points`
# (see data/gold/SCHEMA.md). Everything else falls back to V2.
JUDGE_SYSTEM_PROMPT_V3 = (
    "You are an expert Magic: The Gathering rules judge.\n\n"
    "You get a QUESTION, a numbered list of KEY POINTS (facts a correct "
    "answer must state), an optional numbered list of COMMON ERRORS (false "
    "claims a correct answer must avoid), and several CANDIDATE answers "
    "labeled A, B, C, D.\n\n"
    "For each candidate, report:\n"
    "  points_hit  — the numbers of the KEY POINTS the candidate actually "
    "asserts. Count a point as hit if the candidate states it in ANY wording, "
    "including paraphrase or implication. Do not require matching vocabulary.\n"
    "  errors_made — the numbers of the COMMON ERRORS the candidate asserts. "
    "Only list an error the candidate actually commits.\n"
    "  citation    — 1-5 on whether cited comprehensive-rule numbers are real "
    "and relevant: 5 = correct relevant rule cited, 3 = cites nothing, "
    "1 = fabricated, wrong, or self-contradicting citation.\n\n"
    "CRITICAL:\n"
    "- Length, verbosity, tone, and formatting are IRRELEVANT. A one-sentence "
    "answer that states every key point hits every key point.\n"
    "- Extra correct information neither adds nor removes points.\n"
    "- Do NOT award a point the candidate never makes, and do NOT withhold a "
    "point that is stated in different words than the rubric uses.\n\n"
    "Output ONLY a JSON object mapping each label to "
    '{"points_hit": [<numbers>], "errors_made": [<numbers>], '
    '"citation": <1-5>, "note": "<short phrase>"}. No other text.'
)


def rubric_correctness(points_hit, errors_made, n_points: int, n_errors: int) -> dict:
    """Turn rubric extraction into a 1-5 correctness score, in Python.

    Mapped onto 1-5 so results stay comparable with the V2-judged runs in
    Sections 9.5-9.9 rather than starting a fresh, incomparable scale.

    Asserting a documented misconception halves credit instead of zeroing
    it: an answer can state the right ruling and still tack on a wrong
    reason, and that is meaningfully better than an answer that gets the
    ruling wrong, but clearly worse than a clean one.
    """
    hit = {i for i in points_hit if isinstance(i, int) and 1 <= i <= n_points}
    err = {i for i in errors_made if isinstance(i, int) and 1 <= i <= n_errors}
    fraction = len(hit) / n_points if n_points else 0.0
    if err:
        fraction *= 0.5
    return {
        "correctness": round(1 + 4 * fraction, 2),
        "points_hit": sorted(hit),
        "points_total": n_points,
        "errors_made": sorted(err),
    }


def judge_batch_rubric(
    lm_generate, judge_model, judge_tokenizer, question: str,
    key_points: list[str], common_errors: list[str],
    candidates: dict[str, str], max_tokens: int, rng: random.Random,
) -> dict:
    """Score against an enumerated rubric behind randomized A/B/C/D labels."""
    arms = list(candidates)
    rng.shuffle(arms)
    label_to_arm = dict(zip((chr(ord("A") + i) for i in range(len(arms))), arms))

    points_block = "\n".join(f"{i}. {p}" for i, p in enumerate(key_points, 1))
    user = f"QUESTION:\n{question}\n\nKEY POINTS:\n{points_block}\n"
    if common_errors:
        errors_block = "\n".join(f"{i}. {e}" for i, e in enumerate(common_errors, 1))
        user += f"\nCOMMON ERRORS:\n{errors_block}\n"
    user += "\n" + "\n\n".join(f"CANDIDATE {label}:\n{candidates[arm]}" for label, arm in label_to_arm.items())

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT_V3},
        {"role": "user", "content": user},
    ]
    prompt = judge_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    raw = lm_generate(judge_model, judge_tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        scored = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}

    out = {}
    for label, arm in label_to_arm.items():
        entry = scored.get(label)
        if not isinstance(entry, dict):
            continue
        computed = rubric_correctness(
            entry.get("points_hit") or [], entry.get("errors_made") or [],
            len(key_points), len(common_errors),
        )
        citation = entry.get("citation")
        out[arm] = {
            **computed,
            "citation": citation if isinstance(citation, (int, float)) else 3,
            "note": entry.get("note", ""),
            "scored_by": "rubric",
        }
    return out


def load_questions(synthetic_path: Path, reddit_path: Path, synthetic_limit: int, reddit_limit: int) -> list[dict]:
    questions = []

    with synthetic_path.open(encoding="utf-8") as f:
        synthetic = [json.loads(l) for l in f if l.strip()]
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
        reddit = [json.loads(l) for l in f if l.strip()]
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


def stratified_sample(rows: list[dict], limit: int, key: str = "category") -> list[dict]:
    """Take `limit` rows spread as evenly as possible across `key`.

    The RulesGuru candidates are wildly unbalanced — 265 priority-reasoning
    against 62 turn-structure — so a flat stride under-samples exactly the
    category the corpus was pulled to fix. Round-robin across categories
    instead, striding within each so the pick stays a cross-section rather
    than the first few of each group. Small categories exhaust and drop out;
    their budget spills to the rest.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get(key) or "uncategorized", []).append(r)

    # Stride within each group, deterministically, largest budget first.
    ordered = {
        name: [members[int(i * len(members) / min(len(members), limit))]
               for i in range(min(len(members), limit))]
        for name, members in sorted(groups.items())
    }

    picked: list[dict] = []
    round_idx = 0
    while len(picked) < limit and any(round_idx < len(v) for v in ordered.values()):
        for name in sorted(ordered):
            if round_idx < len(ordered[name]) and len(picked) < limit:
                picked.append(ordered[name][round_idx])
        round_idx += 1
    return picked


def load_gold_questions(path: Path, limit: int | None = None, stratify: bool = True) -> list[dict]:
    """Load rubric-bearing eval rows (gold set or RulesGuru candidates).

    These carry `key_points`, which routes them to the V3 rubric judge.
    Never truncated: the files are ordered by id, which correlates with
    topic, so the first N is not a cross-section.
    """
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if limit and len(rows) > limit:
        if stratify:
            rows = stratified_sample(rows, limit)
        else:
            stride = len(rows) / limit
            rows = [rows[int(i * stride)] for i in range(limit)]

    questions = []
    for r in rows:
        questions.append(
            {
                "source": r.get("source", "gold"),
                "category": r.get("category"),
                "question": r["messages"][1]["content"],
                "reference": r["messages"][2]["content"],
                "supporting_rule_ids": r.get("supporting_rule_ids", []),
                "key_points": r.get("key_points", []),
                "common_errors": r.get("common_errors", []),
                "gold_id": r.get("gold_id"),
                "difficulty": r.get("difficulty"),
            }
        )
    return questions


def score_one_question(lm_generate, judge_model, judge_tokenizer, q: dict,
                       candidates: dict[str, str], max_tokens: int, rng: random.Random) -> dict:
    """Route to the rubric judge when a rubric exists, else the V2 judge."""
    if q.get("key_points"):
        return judge_batch_rubric(
            lm_generate, judge_model, judge_tokenizer, q["question"],
            q["key_points"], q.get("common_errors") or [], candidates, max_tokens, rng,
        )
    return judge_batch_anonymized(
        lm_generate, judge_model, judge_tokenizer, q["question"], q["reference"],
        candidates, max_tokens, rng,
    )


def build_prompt(tokenizer, question: str, context: str | None, preformatted: bool = False) -> str:
    # Shape comes from common.build_rag_messages so that what is evaluated is
    # what build_sft.py trained on — see Section 8.7.
    messages = build_rag_messages(question, context, preformatted)
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def retrieve_context(question: str, embed_model, k: int = 3) -> str:
    hits = retrieve(question, k=k, model_and_tokenizer=embed_model)
    return "\n\n".join(h["text"] for h in hits)


def generate_all_answers(
    questions: list[dict], embed_model, max_tokens: int, adapter_path_under_test: str,
    with_cards: bool = False, base_model_id: str = BASE_MODEL_ID,
) -> dict[str, list[str]]:
    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm

    answers: dict[str, list[str]] = {}
    contexts = [retrieve_context(q["question"], embed_model) for q in questions]

    card_contexts = None
    if with_cards:
        # Card-augmented arms: rules retrieval unchanged, plus the cards the
        # question actually names, resolved by lookup rather than embedding
        # (see scripts/retrieve_hybrid.py for why they aren't merged).
        from card_lookup import CardIndex
        from retrieve_hybrid import build_context

        print("loading card index for card-augmented arms ...")
        card_index = CardIndex()
        card_contexts = [
            build_context(q["question"], card_index, embed_model=embed_model)["context"]
            for q in questions
        ]
        named = sum(1 for c in card_contexts if c.startswith("Cards referenced:"))
        print(f"  {named}/{len(questions)} questions had at least one card resolved")

    for arm_name, adapter_path in [("base", None), ("finetuned", adapter_path_under_test)]:
        print(f"loading {base_model_id} for arm(s) using adapter_path={adapter_path} ...")
        model, tokenizer = load_lm(base_model_id, adapter_path=adapter_path)

        # (contexts, arm name, whether that context is already section-labeled)
        variants = [(contexts, f"{arm_name}_rag", False), (None, arm_name, False)]
        if with_cards:
            variants.append((card_contexts, f"{arm_name}_rag_cards", True))

        for ctxs, out_key, preformatted in variants:
            print(f"generating arm: {out_key}")
            out = []
            for i, q in enumerate(questions, 1):
                ctx = ctxs[i - 1] if ctxs is not None else None
                prompt = build_prompt(tokenizer, q["question"], ctx, preformatted)
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


def judge_batch_anonymized(
    lm_generate, judge_model, judge_tokenizer, question: str, reference: str,
    candidates: dict[str, str], max_tokens: int, rng: random.Random,
) -> dict:
    """Score candidates behind randomized A/B/C/D labels.

    The v1 judge saw real system names ("base", "finetuned_rag"), which
    leaves it free to reward a name rather than an answer, and always in
    the same order. Shuffling per question removes both the name signal and
    any fixed position effect; scores are mapped back afterward.
    """
    arms = list(candidates)
    rng.shuffle(arms)
    labels = [chr(ord("A") + i) for i in range(len(arms))]
    label_to_arm = dict(zip(labels, arms))

    block = "\n\n".join(f"CANDIDATE {label}:\n{candidates[arm]}" for label, arm in label_to_arm.items())
    user = f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\n{block}"
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT_V2},
        {"role": "user", "content": user},
    ]
    prompt = judge_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    raw = lm_generate(judge_model, judge_tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        scored = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}

    out = {}
    for label, arm in label_to_arm.items():
        entry = scored.get(label)
        if isinstance(entry, dict):
            out[arm] = entry
    return out


def rescore(args) -> None:
    """Re-judge stored answers with the recalibrated judge.

    Generation is by far the expensive half of this script, and the answers
    being scored don't change when only the judge changes — so re-scoring
    reads the previous results file instead of regenerating 440 answers.
    """
    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm

    with args.rescore_from.open(encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]
    print(f"re-scoring {len(results)} questions from {args.rescore_from}")

    print(f"loading {args.judge_model} as judge ...")
    judge_model, judge_tokenizer = load_lm(args.judge_model)
    rng = random.Random(args.seed)

    for i, r in enumerate(results, 1):
        candidates = {arm: data["answer"] for arm, data in r["arms"].items()}
        # Stored results carry the rubric when the question had one, so a
        # rescore keeps rubric-judging rubric questions rather than silently
        # dropping back to prose comparison.
        judged = score_one_question(
            lm_generate, judge_model, judge_tokenizer, r, candidates, args.judge_max_tokens, rng,
        )
        for arm, data in r["arms"].items():
            entry = judged.get(arm, {})
            data["correctness"] = entry.get("correctness")
            data["citation_score"] = entry.get("citation")
            data["judge_note_v2"] = entry.get("note")
            if entry.get("scored_by") == "rubric":
                data["scored_by"] = "rubric"
                data["points_hit"] = entry.get("points_hit")
                data["points_total"] = entry.get("points_total")
                data["errors_made"] = entry.get("errors_made")
        if i % 20 == 0 or i == len(results):
            print(f"  re-scored {i}/{len(results)}")

    with args.out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    arms = list(results[0]["arms"])
    lines = ["# Section 9 Evaluation Report (recalibrated judge)\n"]
    lines.append(
        f"{len(results)} questions, re-scored from `{args.rescore_from.name}` with the v2 judge: "
        "correctness and citation scored separately, length/style explicitly excluded, "
        "candidates anonymized behind randomized A/B/C/D labels.\n"
    )
    # Name the judge in the body. The earlier reports recorded it only in the
    # filename (cards_n100_judge2.md), which puts the single
    # most important variable of a two-judge study outside the document.
    lines.append(f"- judge: `{args.judge_model}`\n")
    lines.append("| Arm | Correctness (1-5) | Citation (1-5) | Avg answer chars |")
    lines.append("| --- | --- | --- | --- |")
    for arm in arms:
        cs = [r["arms"][arm]["correctness"] for r in results if r["arms"][arm]["correctness"] is not None]
        qs = [r["arms"][arm]["citation_score"] for r in results if r["arms"][arm]["citation_score"] is not None]
        ln = [len(r["arms"][arm]["answer"]) for r in results]
        c = sum(cs) / len(cs) if cs else float("nan")
        q = sum(qs) / len(qs) if qs else float("nan")
        lines.append(f"| {arm} | {c:.2f} (n={len(cs)}) | {q:.2f} | {sum(ln) / len(ln):.0f} |")

    # The bias this recalibration targets: does score still track length?
    pairs = [
        (len(r["arms"][a]["answer"]), r["arms"][a]["correctness"])
        for r in results for a in arms if r["arms"][a]["correctness"] is not None
    ]
    if len(pairs) > 2:
        n = len(pairs)
        mx = sum(p[0] for p in pairs) / n
        my = sum(p[1] for p in pairs) / n
        cov = sum((x - mx) * (y - my) for x, y in pairs) / n
        sx = (sum((x - mx) ** 2 for x, _ in pairs) / n) ** 0.5
        sy = (sum((y - my) ** 2 for _, y in pairs) / n) ** 0.5
        r_len = cov / (sx * sy) if sx and sy else float("nan")
        lines.append(f"\nCorrelation(answer length, correctness): **r = {r_len:+.3f}** "
                     f"(v1 judge measured r = +0.21 against its single blended score).\n")

    args.report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nresults -> {args.out}\nreport -> {args.report_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", type=Path, default=REPO_ROOT / "eval/sets/rules_questions.jsonl")
    parser.add_argument("--reddit", type=Path, default=REPO_ROOT / "eval/sets/reddit_questions.jsonl")
    parser.add_argument("--synthetic-limit", type=int, default=70)
    parser.add_argument("--reddit-limit", type=int, default=40)
    parser.add_argument("--gold", type=Path, nargs="*", default=[],
                        help="rubric-bearing eval files (eval/sets/gold_questions_eval.jsonl, "
                             "eval/sets/rulesguru_candidates.jsonl); these route to the V3 rubric judge")
    parser.add_argument("--gold-limit", type=int, default=None,
                        help="sample this many rows from each --gold file")
    parser.add_argument("--no-gold-stratify", action="store_true",
                        help="sample --gold-limit by flat stride instead of balancing across categories")
    parser.add_argument("--gold-only", action="store_true",
                        help="evaluate only the --gold files, skipping the synthetic and reddit sets")
    parser.add_argument("--rules", type=Path, default=RULES_PATH)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--judge-max-tokens", type=int, default=500)
    parser.add_argument("--consistency-sample", type=int, default=15)
    parser.add_argument("--adapter-path", default=ADAPTER_PATH, help="adapter under test for the finetuned arms")
    parser.add_argument("--base-model", default=BASE_MODEL_ID,
                        help="base model for every arm and for the consistency rerun. A larger "
                             "4-bit model fits at 36GB for inference even though training one "
                             "does not, so this is the cheap capability lever.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval/runs/latest.jsonl",
                        help="archived runs use a descriptive stem (see eval/README.md); "
                             "the default is deliberately neutral so a bare run cannot "
                             "overwrite a published experiment")
    parser.add_argument("--report-out", type=Path, default=REPO_ROOT / "eval/reports/latest.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-cards", action="store_true",
                        help="add {base,finetuned}_rag_cards arms using card-name lookup + rules retrieval")
    parser.add_argument("--judge-model", default=None,
                        help="model used as judge. Defaults to --base-model — which is ALSO the "
                             "'base' arm under test, so an independent judge is needed to rule out "
                             "self-preference bias (Section 9.9).")
    parser.add_argument("--rescore-from", type=Path, default=None,
                        help="re-judge stored answers from a previous results file instead of regenerating")
    args = parser.parse_args()

    # --judge-model defaults to whatever base model is under test rather than to
    # a hardcoded id, so pointing --base-model at something else doesn't leave
    # the judge silently behind on the old model.
    if args.judge_model is None:
        args.judge_model = args.base_model

    if args.rescore_from:
        rescore(args)
        return

    from mlx_embeddings import load as load_embedder
    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm

    if args.gold_only and not args.gold:
        raise SystemExit("--gold-only needs at least one --gold file")

    questions = []
    if not args.gold_only:
        questions = load_questions(args.synthetic, args.reddit, args.synthetic_limit, args.reddit_limit)
        print(f"{len(questions)} prose-judged questions "
              f"({args.synthetic_limit} synthetic + up to {args.reddit_limit} reddit)")
    for path in args.gold:
        gold = load_gold_questions(path, args.gold_limit, stratify=not args.no_gold_stratify)
        questions.extend(gold)
        by_cat = Counter(q["category"] for q in gold)
        print(f"{len(gold)} rubric-judged questions from {path}")
        print("    " + ", ".join(f"{c}: {n}" for c, n in sorted(by_cat.items())))
    if not questions:
        raise SystemExit("no eval questions loaded")

    n_rubric = sum(1 for q in questions if q.get("key_points"))
    print(f"{len(questions)} eval questions total ({n_rubric} with a rubric -> V3 judge)")

    valid_rule_ids = load_rule_ids(args.rules)
    print(f"loading {EMBED_MODEL_ID} for retrieval ...")
    embed_model = load_embedder(EMBED_MODEL_ID)

    answers = generate_all_answers(questions, embed_model, args.max_tokens, args.adapter_path,
                                   args.with_cards, base_model_id=args.base_model)
    arm_names = list(answers.keys())

    # Consistency check: rerun a subset of finetuned_rag questions and see
    # how often the judge would even need to know — same generation config,
    # does the model give a stable answer.
    print(f"loading {args.base_model} (adapter) for consistency rerun ...")
    ft_model, ft_tokenizer = load_lm(args.base_model, adapter_path=args.adapter_path)
    consistency_idx = list(range(min(args.consistency_sample, len(questions))))
    consistency_reruns = []
    for i in consistency_idx:
        ctx = retrieve_context(questions[i]["question"], embed_model)
        prompt = build_prompt(ft_tokenizer, questions[i]["question"], ctx)
        consistency_reruns.append(lm_generate(ft_model, ft_tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False))
    del ft_model, ft_tokenizer

    # This previously loaded BASE_MODEL_ID and ignored --judge-model entirely,
    # so `eval.py --judge-model <other>` produced a run judged by the base model
    # but labelled as if it had used the other one. Only the --rescore-from path
    # (which reads args.judge_model at its own load site) was correct — and
    # every published two-judge result went through rescore, so Section 9.9 and
    # the Llama reports are unaffected. Fixed here so the flag means what it
    # says on the generate-and-judge path too.
    print(f"loading {args.judge_model} (no adapter) as judge ...")
    judge_model, judge_tokenizer = load_lm(args.judge_model)

    judge_rng = random.Random(args.seed)
    results = []
    for i, q in enumerate(questions):
        candidates = {arm: answers[arm][i] for arm in arm_names}
        judged = score_one_question(
            lm_generate, judge_model, judge_tokenizer, q, candidates, args.judge_max_tokens, judge_rng,
        )

        per_arm = {}
        for arm in arm_names:
            citation = score_citations(candidates[arm], q["supporting_rule_ids"], valid_rule_ids)
            judge_result = judged.get(arm, {})
            per_arm[arm] = {
                "answer": candidates[arm],
                "citation": citation,
                "correctness": judge_result.get("correctness"),
                "citation_score": judge_result.get("citation"),
                "judge_score": judge_result.get("correctness"),
                "judge_note": judge_result.get("note"),
            }
            if judge_result.get("scored_by") == "rubric":
                per_arm[arm].update(
                    scored_by="rubric",
                    points_hit=judge_result.get("points_hit"),
                    points_total=judge_result.get("points_total"),
                    errors_made=judge_result.get("errors_made"),
                )

        results.append(
            {
                "source": q["source"],
                "category": q["category"],
                "question": q["question"],
                "reference": q["reference"],
                # Carried so --rescore-from can re-run the rubric judge; without
                # it a rescore would quietly downgrade these to prose scoring.
                "key_points": q.get("key_points", []),
                "common_errors": q.get("common_errors", []),
                "gold_id": q.get("gold_id"),
                "difficulty": q.get("difficulty"),
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
    source_counts = Counter(q["source"].split(":")[0] for q in questions)
    composition = ", ".join(f"{n} {s}" for s, n in source_counts.most_common())
    lines.append(f"{len(questions)} questions ({composition})")
    # Provenance in the report itself. Previously the only record of which judge
    # scored a run was the filename someone chose for it, which is how
    # cards_n100_judge2.md ended up carrying its most important
    # variable in its name.
    lines.append(
        f"\n- base model: `{args.base_model}`\n"
        f"- adapter under test: `{args.adapter_path}`\n"
        f"- judge: `{args.judge_model}`"
        + ("  (**same model as the `base` arm** — self-preference bias is not "
           "ruled out; re-judge with --rescore-from and an independent judge, Section 9.9)"
           if args.judge_model == args.base_model else "")
    )
    if n_rubric:
        lines.append(
            f"\n{n_rubric} scored against enumerated rubrics (V3 judge); "
            f"{len(questions) - n_rubric} scored against a prose reference (V2 judge). "
            "Rubric correctness is computed from key points hit, not assigned holistically.\n"
        )
    else:
        lines.append("")
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
