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
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import BASE_MODEL_ID as _BASE_MODEL_ID
from common import (  # noqa: F401  (SYSTEM_PROMPT re-exported for callers)
    CARDS_RAG_SYSTEM_PROMPT,
    GOLD_CANDIDATES_PATH,
    GOLD_PATH,
    REPO_ROOT,
    RAG_SYSTEM_PROMPT,
    RULES_PATH,
    SYSTEM_PROMPT,
    build_rag_messages,
    is_hand_authored,
    load_rule_ids,
    pearson_r,
    read_jsonl,
)
from common import RULE_ID_RE as CROSS_REF_RE
from common import AUTO_K_RULES, K_RULES_NO_CARDS, k_rules_arg
from stamp_adapter import check as prompt_stamp_check
from stamp_adapter import unseen_arms
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
# Re-exported so `from eval import BASE_MODEL_ID` keeps working for the
# three modules that already do it; the definition moved to common.py
# (Section 21.146) so a server can read it without importing this file.
BASE_MODEL_ID = _BASE_MODEL_ID

# v2-best, deliberately, after run 3 — NOT a stale default (Section 18.3).
#
# Run 3 trained a full epoch against v2's 0.45 and was evaluated under both
# judges: `finetuned_rag` moved +0.04 (Qwen) and +0.02 (Llama), while the two
# control arms — byte-identical answers that cannot have changed — moved up to
# 0.23 on judge variance alone. There is no measured basis to prefer
# ckpt1322, so nothing was promoted and `models/mtg-rules-adapter-v3-best`
# deliberately does not exist.
#
# Note also that v2-best IS iteration 600 of the v3 run: the two checkpoints
# are bit-identical, 224 tensors, max absolute difference 0.0. So this default
# does not point at an older experiment — it points at the earlier of two
# indistinguishable checkpoints from the same curve, and the cheaper one to
# reproduce.
#
# Evaluate ckpt1322 with `--adapter-path models/mtg-rules-adapter-v3-ckpt1322`;
# the choice is recorded in the results file either way.
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


# v4 judge. Kappa on the blunder call between two judges is +0.24 (Section
# 20.3) -- weak, 31 of 88 calls disputed -- and every other number in the
# project is read through the judge, so that caps everything.
#
# Section 21.3 found the mechanism on the worst position. The candidate answered
# "CAST Shock TARGET Grizzly Bears / PASS", committing NONE of the three
# enumerated errors. One judge correctly reported none. The other reported ALL
# THREE. It was not reading errors off the answer; it was signalling that the
# answer was bad.
#
# So V4 asks for the receipt. Every claimed point_hit and errors_made must come
# with a QUOTE from the candidate, and the quote is then checked in Python
# against the candidate's actual text. A claim whose quote does not appear is
# DROPPED, and the drop count is reported -- which turns "did the judge invent
# this?" from a question about the judge into a string containment test.
#
# Same philosophy as V3: ask for something checkable, then do the arithmetic in
# Python. V3 removed the judge's discretion over the SCORE; V4 removes its
# discretion over whether the evidence exists.
#
# The quote does NOT have to lexically match the rubric line. V3's
# paraphrase-tolerance is deliberate and preserved -- a point can be stated in
# any wording, and the quote is only required to be real text from the answer.
# Requiring lexical overlap would trade one bias for another.
# V5: V3, plus a quote requirement on ERRORS ONLY (Section 21.37).
#
# The positive controls measured the judge charging the REFERENCE ANSWER with a
# common_error on 40% of questions, and 21.37 found the mechanism: an error whose
# polarity OPPOSES the answer's verdict fires 30% of the time against 7% for one
# that agrees with it. The judge matches the error's content and misses the
# negation — "it is a legal target" reads as asserted by an answer that says "it
# is NOT a legal target".
#
# A quote is the direct remedy, because the negation lives in the text: there is
# no verbatim span in "it's not a legal target for Flashback" that asserts the
# opposite. `verify_quoted_claims` already discards a claim whose quote is not in
# the candidate, and it is tested.
#
# Errors ONLY, unlike V4. V4 demanded a quote for every key point too and cost
# 60-85 points of coverage by blowing the output budget (Section 21.14). Errors
# fire on roughly a third of candidates and there are 2-3 of them against 3-4
# key points, so this asks for a small fraction of V4's output.
JUDGE_SYSTEM_PROMPT_V5 = (
    JUDGE_SYSTEM_PROMPT_V3
    .replace(
        "  errors_made — the numbers of the COMMON ERRORS the candidate asserts. "
        "Only list an error the candidate actually commits.",
        "  errors_made — for each COMMON ERROR the candidate asserts, its number AND "
        "a short VERBATIM quote from that candidate showing where it is asserted. "
        "An answer that states the OPPOSITE of an error has not committed it — "
        "check the polarity of what the candidate actually says before listing it. "
        "If you cannot quote the candidate asserting the error, do not list it.")
    .replace(
        '"errors_made": [<numbers>]',
        '"errors_made": [{"n": <number>, "quote": "<verbatim>"}]')
)


JUDGE_SYSTEM_PROMPT_V4 = (
    "You are an expert Magic: The Gathering rules judge.\n\n"
    "You get a QUESTION, a numbered list of KEY POINTS (facts a correct "
    "answer must state), an optional numbered list of COMMON ERRORS (false "
    "claims a correct answer must avoid), and several CANDIDATE answers "
    "labeled A, B, C, D.\n\n"
    "For each candidate, report:\n"
    "  points_hit  — for each KEY POINT the candidate asserts, its number AND "
    "a short VERBATIM quote from that candidate showing where. Count a point "
    "as hit if the candidate states it in ANY wording, including paraphrase or "
    "implication; the quote does not need to match the rubric's words, it only "
    "needs to be the part of the answer that carries the claim.\n"
    "  errors_made — for each COMMON ERROR the candidate actually commits, its "
    "number AND a short VERBATIM quote from that candidate showing where.\n"
    "  citation    — 1-5 on whether cited comprehensive-rule numbers are real "
    "and relevant: 5 = correct relevant rule cited, 3 = cites nothing, "
    "1 = fabricated, wrong, or self-contradicting citation.\n\n"
    "CRITICAL:\n"
    "- Every quote must be copied EXACTLY from that candidate's text. Do not "
    "paraphrase the quote, do not quote the rubric, and do not quote a "
    "different candidate. A claim you cannot quote will be discarded.\n"
    "- If the candidate does not commit any listed error, return an EMPTY "
    "errors_made list. Do not list errors to signal that an answer is poor.\n"
    "- Length, verbosity, tone, and formatting are IRRELEVANT. A one-sentence "
    "answer that states every key point hits every key point.\n"
    "- Extra correct information neither adds nor removes points.\n\n"
    "Output ONLY a JSON object mapping each label to "
    '{"points_hit": [{"n": <number>, "quote": "<verbatim>"}], '
    '"errors_made": [{"n": <number>, "quote": "<verbatim>"}], '
    '"citation": <1-5>, "note": "<short phrase>"}. No other text.'
)


def _normalize_for_quote(s: str) -> str:
    """Collapse whitespace and case so a quote survives reformatting."""
    return " ".join((s or "").split()).lower()


def _as_claim_list(v) -> list:
    """A judge's `points_hit`/`errors_made` field, coerced to a list.

    Valid JSON of the wrong SHAPE — `"points_hit": 3` where `[3]` was asked
    for — parses cleanly and then raises TypeError inside the set
    comprehension below, killing a multi-hour run at whatever question the
    judge happened to fumble. Neither `json.loads` guard catches it: the JSON
    was fine, and only the shape was not.

    A bare scalar is wrapped rather than discarded. Discarding would score that
    answer 1.0 and write the number to the results file as though it were
    measured, which is worse than the crash it replaces — the crash at least
    announces itself.
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]


def _claim_index(i, n_max: int) -> bool:
    """A usable 1-based rubric index.

    `bool` is a subclass of `int`, so a judge answering `"points_hit": true`
    would otherwise be read as claiming point 1 — a fabricated claim, awarded
    silently, from a response that named no point at all.
    """
    return isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= n_max


def verify_quoted_claims(claims, answer: str, n_max: int) -> tuple[list[int], int]:
    """Keep only claims whose quote really appears in the answer.

    Returns (kept_numbers, n_dropped). Accepts the V3 shape (bare integers) too,
    so a mixed or partially-malformed response degrades to V3 behaviour rather
    than to nothing -- an unquoted claim is kept, because V3 never asked for a
    quote and dropping it would silently penalize a judge that answered the
    older question.
    """
    kept: list[int] = []
    dropped = 0
    hay = _normalize_for_quote(answer)
    for c in _as_claim_list(claims):
        if isinstance(c, int) and not isinstance(c, bool):
            if 1 <= c <= n_max:
                kept.append(c)
            continue
        if not isinstance(c, dict):
            continue
        n = c.get("n")
        if not _claim_index(n, n_max):
            continue
        quote = c.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            kept.append(n)
            continue
        # Short quotes match too easily to be evidence of anything.
        needle = _normalize_for_quote(quote)
        if len(needle) >= 12 and needle not in hay:
            dropped += 1
            continue
        kept.append(n)
    return kept, dropped


# Correctness scoring version, recorded in every row it produces.
#
# "points_only" (current): correctness = 1 + 4 * (points_hit / n_points).
# "halved_v3"  (through Section 21.27): the same, halved whenever `errors_made`
#              was non-empty.
#
# Named rather than implied, because the two produce different numbers from the
# SAME judge output, and a run file that does not say which one it used cannot
# be compared to anything.
SCORING = "points_only"


def rubric_correctness(points_hit, errors_made, n_points: int, n_errors: int,
                       halve_on_error: bool = False) -> dict:
    """Turn rubric extraction into a 1-5 correctness score, in Python.

    Mapped onto 1-5 so results stay comparable with the V2-judged runs in
    Sections 9.5-9.9 rather than starting a fresh, incomparable scale.

    THE HALVING IS OFF (Section 21.28)
    ---------------------------------
    Through Section 21.27 an asserted misconception halved credit, on the
    reasoning that an answer can state the right ruling and tack on a wrong
    reason — better than getting the ruling wrong, worse than a clean answer.
    That reasoning is sound and the mechanism it depended on is not.

    The positive controls measured the judge inventing a `common_error` against
    the REFERENCE ANSWER — which definitionally cannot commit one — on 40% of
    questions. The halving turned that into a real cost:

        oracle mean when the judge invents no error : 4.86  (n=59)
        oracle mean when it invents one             : 2.39  (n=39)

    2.47 points off the correct answer, on 40% of the set, for errors it did not
    make. Removing the term widens the oracle/wrong separation from 2.77 to
    3.22, so the error half was subtracting resolution from a scale that works
    without it.

    `errors_made` is still extracted, still returned, and still what blunder
    rate is defined on. What changed is that a field with a measured 40%
    false-positive rate no longer moves the headline score.

    `halve_on_error=True` reproduces every number published through Section
    21.27 from the same stored judge output, which is what makes the change
    auditable rather than a break in the record.
    """
    hit = {i for i in _as_claim_list(points_hit) if _claim_index(i, n_points)}
    err = {i for i in _as_claim_list(errors_made) if _claim_index(i, n_errors)}
    fraction = len(hit) / n_points if n_points else 0.0
    if err and halve_on_error:
        fraction *= 0.5
    return {
        "correctness": round(1 + 4 * fraction, 2),
        "points_hit": sorted(hit),
        "points_total": n_points,
        "errors_made": sorted(err),
        "scoring": "halved_v3" if halve_on_error else SCORING,
    }


# Everything a rubric grading carries BESIDES the score itself. Both writers —
# main() on a fresh run and rescore() on a stored one — assembled their per-arm
# dict field by field, so a value computed in judge_batch_rubric reached the
# stored run only if someone had remembered to list it in both places. Nobody
# had: `scoring` reached neither (while CLAUDE.md said every row carried it),
# and `all_errors_fired` / `error_contradiction` reached neither, which left
# rescore()'s Section 21.26 report block gated on a key that could not exist.
# `quote_drops` was listed in rescore() only, having been caught once already —
# the same bug, fixed in one copy. One definition now, so the next field added
# to rubric_correctness is carried by both without a second edit.
RUBRIC_DIAGNOSTICS = ("scoring", "quote_drops", "all_errors_fired",
                      "error_contradiction")


def carry_diagnostics(dest: dict, entry: dict) -> dict:
    """Copy the non-score fields of a rubric grading onto a stored arm record."""
    for k in RUBRIC_DIAGNOSTICS:
        if entry.get(k) is not None:
            dest[k] = entry[k]
    return dest


# The two judge system prompts above say "labeled A, B, C, D" in prose, while
# the labels themselves are generated as chr(ord("A") + i) for however many
# arms there are. At four arms those agree. At five they do not: the judge
# would be shown CANDIDATE A through E and told in the same breath that there
# are four, which is an invitation to silently drop the last one.
#
# Rewritten ONLY when the count is not four, so every published run -- all of
# which used exactly four arms -- reproduces the prompt byte for byte and stays
# comparable. Found while adding a fifth position arm, before it could score
# anything.
_FOUR_LABEL_PHRASE = "labeled A, B, C, D"


def judge_prompt_for(base: str, labels: list[str]) -> str:
    """The judge system prompt, with its label list matching the real one."""
    if len(labels) == 4:
        return base
    if len(labels) == 1:
        named = f"labeled {labels[0]}"
    else:
        named = "labeled " + ", ".join(labels[:-1]) + f" and {labels[-1]}"
    return base.replace(_FOUR_LABEL_PHRASE, named)


def apply_judge_template(judge_tokenizer, messages: list[dict]) -> str:
    """Render a JUDGE prompt, folding `system` into the first user turn when the
    model's chat template refuses a system role.

    `gemma-2-27b-it` raises `TemplateError: System role not supported` and dies
    before grading anything — a whole model family unusable as a judge over a
    formatting convention. All three judge prompts open with a system message.

    **Judge paths only, deliberately.** `build_prompt()` renders the prompt for
    the model UNDER TEST from `common.build_rag_messages`, and an adapter is
    only valid for the format it was trained on (Section 8.7) — silently
    reshaping that would invalidate the adapter and present as a capability
    result. Evaluating a Gemma-family model as a subject needs a deliberate
    decision and a re-stamp, not a fallback. So this is not called there.

    Judges are stateless graders with no adapter, so the same reshaping is free.
    It does change the judge prompt for such a model, which is a reason to
    compare its numbers only against other runs of itself.
    """
    try:
        return judge_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except Exception as exc:                      # jinja2.TemplateError, and kin
        if "system" not in str(exc).lower():
            raise
        folded, carried = [], ""
        for m in messages:
            if m["role"] == "system":
                carried += m["content"].rstrip() + "\n\n"
            elif m["role"] == "user" and carried:
                folded.append({"role": "user", "content": carried + m["content"]})
                carried = ""
            else:
                folded.append(m)
        if carried:                               # system with no user turn after it
            folded.append({"role": "user", "content": carried.rstrip()})
        return judge_tokenizer.apply_chat_template(folded, add_generation_prompt=True)


def judge_batch_rubric(
    lm_generate, judge_model, judge_tokenizer, question: str,
    key_points: list[str], common_errors: list[str],
    candidates: dict[str, str], max_tokens: int, rng: random.Random,
    judge_version: str = "v3",
) -> dict:
    """Score against an enumerated rubric behind randomized A/B/C/D labels.

    `judge_version` selects the prompt. "v3" is the default and is unchanged, so
    every published number reproduces. "v4" additionally requires a verbatim
    quote behind each claimed point or error and discards claims whose quote is
    not in the candidate's text (Section 21.7).
    """
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
        {"role": "system", "content": judge_prompt_for(
            {"v4": JUDGE_SYSTEM_PROMPT_V4,
             "v5": JUDGE_SYSTEM_PROMPT_V5}.get(judge_version, JUDGE_SYSTEM_PROMPT_V3),
            list(label_to_arm))},
        {"role": "user", "content": user},
    ]
    prompt = apply_judge_template(judge_tokenizer, messages)
    raw = lm_generate(judge_model, judge_tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        scored = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}

    # A single candidate is often returned UNWRAPPED (Section 21.39). Asked to
    # grade one answer "labeled A", the judge emits
    #   {"points_hit": [...], "errors_made": [...], "citation": 3}
    # rather than {"A": {...}} — which is reasonable, and which `scored.get("A")`
    # silently reads as "the judge said nothing about A".
    #
    # Only when there is exactly one label, and only when the object looks like
    # an entry rather than a label map. With several arms an unwrapped object
    # cannot be attributed and must still fail.
    if (len(label_to_arm) == 1 and not any(k in scored for k in label_to_arm)
            and any(k in scored for k in ("points_hit", "errors_made", "citation"))):
        scored = {next(iter(label_to_arm)): scored}

    out = {}
    for label, arm in label_to_arm.items():
        entry = scored.get(label)
        if not isinstance(entry, dict):
            continue
        raw_points = entry.get("points_hit") or []
        raw_errors = entry.get("errors_made") or []
        drops = 0
        if judge_version == "v5":
            # Errors only. V5 leaves points_hit in the V3 shape on purpose —
            # that half of the instrument works (the judge credits the reference
            # answer with 90% of its own key points), and V4 showed that asking
            # for quotes on everything costs most of the coverage.
            raw_errors, drops = verify_quoted_claims(
                raw_errors, candidates[arm], len(common_errors))
        elif judge_version == "v4":
            # The receipt check. A claim the judge cannot quote from THIS
            # candidate is discarded, and the count is carried so fabrication
            # is a reported number rather than an impression.
            answer_text = candidates[arm]
            raw_points, d1 = verify_quoted_claims(raw_points, answer_text, len(key_points))
            raw_errors, d2 = verify_quoted_claims(raw_errors, answer_text, len(common_errors))
            drops = d1 + d2
        computed = rubric_correctness(
            raw_points, raw_errors, len(key_points), len(common_errors),
        )
        computed["quote_drops"] = drops
        # The "every error at once" signature (Section 21.26).
        #
        # A rubric's common_errors are alternative wrong answers, so committing
        # all of them is usually not a thing an answer can do. Firing them all
        # is the judge using the error list as a "this answer is bad" flag —
        # and blunder rate is defined on `errors_made` being non-empty, so it
        # lands directly on the gate metric.
        #
        # Reported, never corrected. Sometimes an answer really is wrong on
        # every axis, and silently dropping errors would change a published
        # metric on a heuristic. The contradiction flag is the sharper one:
        # stating half the key points while committing every listed
        # misconception is not a judgement, it is two claims that cannot both
        # hold.
        n_err = len(common_errors)
        fired_all = n_err >= 3 and len(computed["errors_made"]) == n_err
        computed["all_errors_fired"] = fired_all
        computed["error_contradiction"] = bool(
            fired_all and key_points
            and len(computed["points_hit"]) >= len(key_points) / 2)
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
    """Load rubric-bearing eval rows from flat or chat-format JSONL.

    The canonical gold set stores `question`/`answer` fields, while derived
    eval sets store the same content in chat `messages`. Both carry
    `key_points`, which routes them to the V3 rubric judge.
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
        messages = r.get("messages") or []
        if messages:
            user = next((m["content"] for m in messages if m.get("role") == "user"), "")
            reference = next((m["content"] for m in reversed(messages)
                              if m.get("role") == "assistant"), "")
        else:
            user = r.get("question", "")
            reference = r.get("answer", "")
        questions.append(
            {
                "source": r.get("source", "gold"),
                "category": r.get("category"),
                "question": user,
                "reference": reference,
                "supporting_rule_ids": r.get("supporting_rule_ids", []),
                "key_points": r.get("key_points", []),
                "common_errors": r.get("common_errors", []),
                "gold_id": r.get("gold_id") or r.get("id"),
                "difficulty": r.get("difficulty"),
                # Carried so card-augmented arms can resolve cards by exact
                # lookup instead of scanning the question for `[[brackets]]`
                # that neither real gold corpus contains (Section 21.136).
                # Dropping it here is what would make that fix a silent no-op.
                "cards": r.get("cards") or [],
            }
        )
    return questions


def score_one_question(lm_generate, judge_model, judge_tokenizer, q: dict,
                       candidates: dict[str, str], max_tokens: int, rng: random.Random,
                       judge_version: str = "v3") -> dict:
    """Route to the rubric judge when a rubric exists, else the V2 judge."""
    if q.get("key_points"):
        return judge_batch_rubric(
            lm_generate, judge_model, judge_tokenizer, q["question"],
            q["key_points"], q.get("common_errors") or [], candidates, max_tokens, rng,
            judge_version=judge_version,
        )
    return judge_batch_anonymized(
        lm_generate, judge_model, judge_tokenizer, q["question"], q["reference"],
        candidates, max_tokens, rng,
    )


def build_prompt(tokenizer, question: str, context: str | None, preformatted: bool = False,
                 exemplars: list[tuple[str, str]] | None = None) -> str:
    # Shape comes from common.build_rag_messages so that what is evaluated is
    # what build_sft.py trained on — see Section 8.7.
    messages = build_rag_messages(question, context, preformatted, exemplars)
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def load_exemplars(path: Path, n: int, eval_questions: list[dict]) -> list[tuple[str, str]]:
    """Few-shot demonstrations, drawn from a corpus disjoint from the eval set.

    Contamination discipline is the same here as for training data, for the
    same reason: an exemplar that IS an eval question makes the model's answer
    a lookup rather than a demonstration of reasoning, and it would read as a
    capability gain. `build_sft_verified.py` learned this the expensive way —
    matching on `id` alone let 18 eval questions through a filter that
    asserted it had excluded them (Section 21.13) — so this checks the
    QUESTION TEXT, which is the thing that would actually leak, rather than
    trusting that two files with different names hold different questions.

    Picked deterministically (first N after sorting by id) rather than
    randomly: a few-shot result that moves when the seed moves is a result
    about the seed, and this project already has one unseeded-sampling
    finding it did not want (Section 18.2's validation curve).
    """
    from audit_sft import jaccard as _jaccard
    from audit_sft import tokens as _tokens

    pool = [r for r in read_jsonl(path)
            if (r.get("question") or "").strip() and (r.get("answer") or "").strip()]
    eval_toks = [_tokens(q.get("question") or "") for q in eval_questions]

    picked: list[tuple[str, str]] = []
    skipped = 0
    for rec in sorted(pool, key=lambda r: str(r.get("id", ""))):
        qt = _tokens(rec["question"])
        if qt and max((_jaccard(qt, et) for et in eval_toks), default=0.0) >= 0.5:
            skipped += 1
            continue
        picked.append((rec["question"], rec["answer"]))
        if len(picked) >= n:
            break
    if len(picked) < n:
        raise SystemExit(
            f"only {len(picked)} uncontaminated exemplars available in {path} "
            f"(asked for {n}); {skipped} were too close to an eval question")
    print(f"  {len(picked)} few-shot exemplars from {path.name} "
          f"({skipped} skipped as too close to an eval question)")
    return picked


def retrieve_context(question: str, embed_model, k: int = 3) -> str:
    hits = retrieve(question, k=k, model_and_tokenizer=embed_model)
    return "\n\n".join(h["text"] for h in hits)


def generate_all_answers(
    questions: list[dict], embed_model, max_tokens: int, adapter_path_under_test: str,
    with_cards: bool = False, with_rulings: bool = False,
    base_model_id: str = BASE_MODEL_ID, base_only: bool = False,
    truncation_out: dict[str, int] | None = None,
    exemplars: list[tuple[str, str]] | None = None, k_rules: int | str = 3,
    no_plain_rag: bool = False, routing_out: dict[str, int] | None = None,
) -> dict[str, list[str]]:
    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm

    answers: dict[str, list[str]] = {}
    # The rules-only `_rag` arm is NOT routed, and cannot be: routing keys off
    # whether a card resolved, and this arm resolves none by construction. It
    # is the card-free branch of the policy, so under AUTO it takes that
    # branch's k -- routing it on its own (empty) card list would give the same
    # answer through a confusing path, and passing "auto" straight through
    # would make its context depend on a lookup it never performs.
    plain_k = K_RULES_NO_CARDS if k_rules == AUTO_K_RULES else k_rules
    contexts = [retrieve_context(q["question"], embed_model, k=plain_k) for q in questions]

    card_contexts = None
    if with_cards:
        # Card-augmented arms: rules retrieval unchanged, plus the cards the
        # question actually names, resolved by lookup rather than embedding
        # (see scripts/retrieve_hybrid.py for why they aren't merged).
        from card_lookup import CardIndex
        from retrieve_hybrid import RulingIndex, build_context

        print("loading card index for card-augmented arms ...")
        card_index = CardIndex()
        ruling_index = RulingIndex() if with_rulings else None
        # Prefer the record's own hand-verified `cards` field over scanning the
        # question for `[[brackets]]` it almost certainly does not contain
        # (Section 21.136: find_in_text resolves 0/99 on the real gold set).
        # Falls back to text-scanning when a record carries no `cards`, so a
        # corpus that DOES use bracket syntax keeps working unchanged.
        built = [
            build_context(q["question"], card_index, embed_model=embed_model,
                          ruling_index=ruling_index, k_rules=k_rules,
                          card_names=q.get("cards") or None)
            for q in questions
        ]
        card_contexts = [b["context"] for b in built]
        # Under AUTO the header's "k: 3" line becomes a lie by omission -- k is
        # no longer one number for the run, and a report that names the POLICY
        # without the SPLIT cannot be read against a fixed-k run at all. This
        # is the same discipline as recording an overridable default in the
        # output; routing just makes the default per-question.
        if routing_out is not None and k_rules == AUTO_K_RULES:
            for b in built:
                routing_out[f"k{b['k_rules_used']}"] = \
                    routing_out.get(f"k{b['k_rules_used']}", 0) + 1
        named = sum(1 for c in card_contexts if c.startswith("Cards referenced:"))
        from_field = sum(1 for q in questions if q.get("cards"))
        print(f"  {named}/{len(questions)} questions had at least one card resolved"
              f"  ({from_field} via their own `cards` field, "
              f"{len(questions) - from_field} via [[bracket]] scanning)")
        if with_rulings:
            with_official = sum(1 for c in card_contexts if "Official rulings:" in c)
            print(f"  {with_official}/{len(questions)} questions had official rulings")

    # An adapter is only valid for the prompt format it saw. Checked HERE,
    # before generating hundreds of answers, because the failure it guards
    # against (Section 8.7) presents as the model having got worse — so the
    # cost of not checking is a full run plus the wrong conclusion drawn from
    # it. A missing stamp is a warning, not an error: adapters trained before
    # `stamp_adapter.py` existed have nothing to compare against.
    if adapter_path_under_test:
        ok, msg = prompt_stamp_check(Path(adapter_path_under_test))
        if not ok:
            raise SystemExit(msg)
        print(f"  {msg}")
        # The fingerprint asks whether the prompts were EDITED since training.
        # This asks whether the adapter ever saw the one an arm is about to use,
        # which a matching fingerprint cannot tell you (Section 8.7, 21.50).
        # Must list the arms this run will ACTUALLY generate. Warning about
        # `finetuned_rag` under --no-plain-rag would name an arm that never runs.
        planned = ["finetuned"] if no_plain_rag else ["finetuned", "finetuned_rag"]
        if with_cards:
            planned.append("finetuned_rag_cards_rulings" if with_rulings
                           else "finetuned_rag_cards")
        for arm in unseen_arms(Path(adapter_path_under_test), planned):
            print(f"  WARNING: arm `{arm}` uses a system prompt this adapter's "
                  f"training set contains ZERO times.")

    # `base_only` skips the adapter arms entirely. Two reasons, both real:
    # an adapter trained on one base model cannot load onto a different-sized
    # one at all (a 7B LoRA has the wrong dimensions for a 32B), and a
    # base-model capability comparison does not want to pay for three arms
    # it will not read. Note this CHANGES THE ARM COUNT, so a base-only run
    # is comparable only to another base-only run -- Section 21.5 measured a
    # byte-identical arm moving 23 points when an arm was added, because
    # `judge_batch_rubric` grades every candidate for a question in one
    # batched call.
    arm_specs = [("base", None)]
    if not base_only:
        arm_specs.append(("finetuned", adapter_path_under_test))

    for arm_name, adapter_path in arm_specs:
        print(f"loading {base_model_id} for arm(s) using adapter_path={adapter_path} ...")
        model, tokenizer = load_lm(base_model_id, adapter_path=adapter_path)

        # (contexts, arm name, whether that context is already section-labeled)
        #
        # `no_plain_rag` drops the rules-only variant. It exists because at
        # `k_rules=0` that arm receives an EMPTY context, falls back to the
        # no-context prompt, and becomes byte-identical to the bare `base`
        # arm -- measured at 53/53 questions, against 0/53 at k=3 (Section
        # 21.143). `judge_batch_rubric` grades every candidate for a question
        # in ONE batched call, so that hands the judge a duplicate and makes
        # a k=0 run non-comparable to a k=3 one even though both have three
        # arms. Dropping it lets both configurations run as the same two
        # DISTINCT arms. Section 21.5 is about arm COUNT; this is the same
        # failure reached through composition instead.
        variants = [] if no_plain_rag else [(contexts, f"{arm_name}_rag", False)]
        variants.append((None, arm_name, False))
        if with_cards:
            suffix = "_rag_cards_rulings" if with_rulings else "_rag_cards"
            variants.append((card_contexts, f"{arm_name}{suffix}", True))

        for ctxs, out_key, preformatted in variants:
            print(f"generating arm: {out_key}")
            out = []
            for i, q in enumerate(questions, 1):
                ctx = ctxs[i - 1] if ctxs is not None else None
                prompt = build_prompt(tokenizer, q["question"], ctx, preformatted, exemplars)
                out.append(lm_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False))
                if i % 25 == 0 or i == len(questions):
                    print(f"  {out_key}: {i}/{len(questions)}")
            answers[out_key] = out

            # How many answers ran to the token ceiling instead of finishing.
            # This is measured, printed and stored because it is invisible
            # otherwise and it is CORRELATED WITH THE TREATMENT: a verbose arm
            # gets cut off mid-reasoning while a terse one never does, so the
            # rubric metric (points_hit / n_points) silently penalises the
            # verbose arm for a setting rather than for its answer. Found at
            # the default max_tokens=300, where the 7B base arms hit the cap on
            # 26-40 of 53 questions (median answer exactly 300 tokens) while
            # the v6 adapter's terse arms hit it on 2-4. See Section 21.138.
            capped = sum(1 for a in out if len(tokenizer.encode(a)) >= max_tokens)
            if truncation_out is not None:
                truncation_out[out_key] = capped
            if capped:
                pct = capped / max(1, len(out))
                flag = "  <-- HIGH, scores for this arm are truncation-limited" if pct >= 0.25 else ""
                print(f"  {out_key}: {capped}/{len(out)} answers hit the "
                      f"{max_tokens}-token ceiling ({pct:.0%}){flag}")

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
        {"role": "system", "content": judge_prompt_for(
            JUDGE_SYSTEM_PROMPT_V2, list(label_to_arm))},
        {"role": "user", "content": user},
    ]
    prompt = apply_judge_template(judge_tokenizer, messages)
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


def rubric_provenance(gold_path: Path, candidates_path: Path) -> dict[str, str]:
    """{question id -> 'hand-authored (judge:XX)' | 'machine-drafted'}.

    Resolved at REPORT time by joining on `gold_id`, not threaded through the
    eval rows. Two reasons. Runs already written can be segmented — including
    the Section 14.6 pilot — and a rubric rewritten after a run was scored is
    reflected the next time the comparison is run, instead of being frozen
    into the row as whatever it was on the day.
    """
    prov: dict[str, str] = {}
    for path in (candidates_path, gold_path):  # gold wins on overlap
        for rec in read_jsonl(path):
            src = str(rec.get("rubric_source") or "")
            prov[rec["id"]] = src if is_hand_authored(rec) else "machine-drafted"
    return prov


def _author_of(rubric_source: str) -> str:
    """'hand-authored (Cody Clark); decomposed...' -> 'Cody Clark'."""
    m = re.search(r"\(([^)]+)\)", rubric_source or "")
    return m.group(1) if m else (rubric_source or "unattributed")


def compare_judges(path_a: Path, path_b: Path, report_out: Path,
                   gold_path: Path, candidates_path: Path) -> None:
    """Inter-judge agreement on identical stored answers, split by rubric source.

    Section 14.6 established that rubric craft is what drives judge agreement
    (r +0.30 -> +0.62 hand vs machine, on the same questions and answers). That
    was a one-off analysis; this makes it a standing readout, because it is
    also the acceptance test for contributed rubrics.

    It does NOT support "find the bad rubrics and rewrite them". Section 16.12
    read six of eight disputes sitting on two of eight items as evidence that
    disagreement localizes; at n=8 that was noise. Measured over both full sets
    (Section 21.20), disagreement is diffuse — Gini 0.45-0.48 across rubric
    items, the worst 10% of items carrying only ~23% of disputes, and 91 of 99
    rules records and 19 of 22 positions carrying at least one. The table below
    is a prompt to READ a few disputed items and understand them, not a repair
    list: there is no handful of rubrics whose rewriting would move kappa.

    Both files must be the SAME answers judged twice (`--rescore-from`).
    Comparing two independent generations measures two things at once and
    settles neither.
    """
    def load(p: Path) -> dict:
        return {r["gold_id"]: r for r in read_jsonl(p, missing_ok=False) if r.get("gold_id")}

    a, b = load(path_a), load(path_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit(f"{path_a} and {path_b} share no gold_id — are both from --rescore-from?")
    prov = rubric_provenance(gold_path, candidates_path)
    arms = [x for x in a[shared[0]]["arms"] if x in b[shared[0]]["arms"]]

    # group -> list of (score_a, score_b, points_hit_a, points_hit_b, id, arm)
    #
    # Segmented by AUTHOR, not just hand-vs-machine. This used to collapse every
    # hand-authored rubric into one bucket, which made the per-author breakdown
    # CLAUDE.md advertises impossible: attribution rides on each submission
    # precisely so one file can hold several people's work, and the readout
    # threw that away. Every record in the set is hand-authored now, so the
    # binary split has nothing left to contrast.
    groups: dict[str, list] = {}
    for qid in shared:
        src = prov.get(qid, "machine-drafted")
        label = "machine-drafted" if src == "machine-drafted" else _author_of(src)
        for arm in arms:
            xa, xb = a[qid]["arms"][arm], b[qid]["arms"][arm]
            ca, cb = xa.get("correctness"), xb.get("correctness")
            if ca is None or cb is None:
                continue
            row = (ca, cb, xa.get("points_hit"), xb.get("points_hit"), qid, arm)
            groups.setdefault(label, []).append(row)
            groups.setdefault("ALL", []).append(row)

    def stats(rows: list) -> dict:
        pairs = [(r[0], r[1]) for r in rows]
        n = len(pairs)
        if not n:
            return {}
        exact = sum(1 for x, y in pairs if x == y) / n
        mean_gap = sum(abs(x - y) for x, y in pairs) / n
        ph = [(r[2], r[3]) for r in rows if r[2] is not None and r[3] is not None]
        same_ph = (sum(1 for x, y in ph if sorted(x) == sorted(y)) / len(ph)) if ph else float("nan")
        return {"n": n, "r": pearson_r(pairs), "exact": exact,
                "gap": mean_gap, "same_points": same_ph,
                "n_questions": len({r[4] for r in rows})}

    authors = sorted(g for g in groups if g not in ("ALL", "machine-drafted"))
    order = ["ALL"] + authors + (["machine-drafted"] if "machine-drafted" in groups else [])
    # Name the judges, and verify they differ. Identifying them by filename is
    # the record Section 21.40 found insufficient, in the one report where the
    # judge identity is the entire subject (Section 21.54).
    rows_a, rows_b = list(a.values()), list(b.values())
    lines = [f"# Inter-judge agreement: `{path_a.name}` vs `{path_b.name}`", "",
             "- " + judges_of(rows_a, path_a.stem),
             "- " + judges_of(rows_b, path_b.stem), ""]
    warn = assert_two_judges(rows_a, rows_b)
    if warn:
        lines += [warn]
    lines += [f"{len(shared)} questions x {len(arms)} arms, identical stored answers.",
             "Segmented by who wrote the rubric — the variable Section 14.6 found "
             "dominates agreement.", "",
             "| Rubric source | Questions | Pairs | Pearson r | Exact | Mean gap | Same points_hit |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for g in order:
        s = stats(groups[g])
        name = f"**{g}**" if g == "ALL" else g
        lines.append(f"| {name} | {s['n_questions']} | {s['n']} | {s['r']:+.2f} | "
                     f"{s['exact']:.0%} | {s['gap']:.2f} | {s['same_points']:.0%} |")

    both = len([g for g in groups if g != "ALL"]) > 1
    lines += ["", "Reference: Section 14.6 measured r **+0.30** machine-drafted vs "
              "**+0.62** hand-authored on 20 questions.", ""]
    if not both:
        only = order[-1]
        lines.append(f"> Only **{only}** rubrics are present, so there is no contrast to read "
                     f"here — the row is a baseline for when another source arrives.")
    elif min(len({r[4] for r in groups[g]}) for g in authors) < 25:
        lines.append("> At least one segment is under 25 questions. Section 14.6 needed 20 to "
                     "separate hand from machine, an effect far larger than the difference "
                     "between two careful authors — read a split here as a prompt to look at "
                     "specific records, not as a measured difference.")
    lines += ["", "## Questions the judges disagree on most", "",
              "Worth reading to understand *how* the judges differ — but not a repair "
              "list. Section 21.20 measured disagreement as diffuse rather than "
              "localized (Gini 0.45–0.48 across rubric items; 91 of 99 records carry at "
              "least one dispute), so rewriting the rows below would not move kappa. "
              "The lever is the judge, not this table.", "",
              "| Gold id | Arm | Judge A | Judge B | Gap | Rubric |",
              "| --- | --- | --- | --- | --- | --- |"]
    worst = sorted(groups.get("ALL", []), key=lambda r: -abs(r[0] - r[1]))[:12]
    for ca, cb, _, _, qid, arm in worst:
        if ca == cb:
            break
        lines.append(f"| `{qid}` | {arm} | {ca:.1f} | {cb:.1f} | {abs(ca - cb):.1f} | "
                     f"{prov.get(qid, 'machine-drafted') if prov.get(qid) == 'machine-drafted' else _author_of(prov.get(qid, ''))} |")

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for g in order:
        s = stats(groups[g])
        print(f"  {g:16s} n={s['n']:>4}  r={s['r']:+.2f}  exact={s['exact']:.0%}  gap={s['gap']:.2f}")
    print(f"\n-> {report_out}")


def judges_of(rows: list[dict], label: str) -> str:
    """The judge model named on a set of stored rows, or a stated absence.

    Section 21.40's lesson was "a rate is a statement about a judge", and every
    row has carried `judge_model` since. The one report where the judge IS the
    subject — inter-judge agreement — still identified its two inputs by
    *filename*, which is the exact record 21.40 found insufficient
    (`cards_n100_judge2.md` carrying its most important variable in its name).

    Returns a display string. A file with no `judge_model` says so rather than
    going blank: archived runs predate the field, and "unrecorded" is a fact
    about the evidence where a silent omission looks like nothing was wrong.
    """
    seen = {r.get("judge_model") for r in rows}
    named = sorted(m for m in seen if m)
    if not named:
        return f"{label}: judge unrecorded (archived before Section 21.40)"
    if len(named) > 1:
        return f"{label}: MIXED judges in one file — " + ", ".join(named)
    return f"{label}: `{named[0]}`"


def assert_two_judges(rows_a: list[dict], rows_b: list[dict]) -> str | None:
    """Warn when an 'inter-judge' comparison did not actually vary the judge.

    Comparing a file with itself, or two files judged by the same model,
    produces perfect agreement and a report headed "Inter-judge agreement".
    Nothing caught that: the arm and id overlap checks both pass, and kappa
    +1.00 reads as a *result*. The whole point of --rescore-from is that the
    judge differs, so the case worth guarding is the one where it does not.

    Returns a warning line, or None when the two judges genuinely differ.
    Never fatal — a run whose judges are unrecorded is still worth comparing,
    it just cannot claim to have varied the judge.
    """
    ja = {r.get("judge_model") for r in rows_a if r.get("judge_model")}
    jb = {r.get("judge_model") for r in rows_b if r.get("judge_model")}
    if not ja or not jb:
        return ("> **One or both files do not record their judge**, so this cannot verify "
                "that the judge actually varied. Identify them by provenance before "
                "reading the agreement number (Section 21.40).\n")
    if ja == jb:
        return (f"> **Both files were judged by the same model** (`{sorted(ja)[0]}`). This is "
                "not an inter-judge comparison — agreement here measures determinism, not "
                "agreement, and the judge is deterministic. Re-run one side with "
                "`--rescore-from` and a different `--judge-model`.\n")
    return None


def coverage_lines(results: list[dict]) -> list[str]:
    """How much of the set the judge actually graded, as report lines.

    The per-arm "(n=)" beside each mean carries this, but as a parenthetical,
    which reads as a footnote rather than as the headline it is: a V4 run scored
    14 of 99 and the report presented four confident-looking averages over the
    14 (Section 21.14). A judge's parse failures are not random — they track
    rubric size — so the graded subset is SELECTED, and a mean over it is a mean
    over the short rubrics.

    Lives here as ONE definition because it did not: the guard was written into
    `rescore()` and the first-pass writer never got it, so the report for a fresh
    run — the common case, and the one a V4 experiment actually produces — had no
    coverage warning at all. Same defect as the judge stamp in 21.51 and the
    diagnostics copy list before `carry_diagnostics`: a hardening applied to one
    of two writers is a hardening that fires on half the runs.
    """
    n_unjudged = sum(1 for r in results for d in r["arms"].values()
                     if d.get("correctness") is None)
    n_total = sum(len(r["arms"]) for r in results)
    if not n_unjudged:
        return []
    frac = n_unjudged / n_total
    out = [f"> **{n_unjudged} of {n_total} arm-answers ({frac:.0%}) went unjudged** — the "
           "judge returned JSON that could not be parsed, usually by running out of "
           "output tokens. They are excluded rather than counted as wrong.\n"]
    if frac >= 0.2:
        out.append(
            "> The failures are not spread evenly: longer rubrics need longer judge "
            "output, so what remains is a subset selected by rubric size rather than "
            "by anything about the answers. **Read nothing into the means below** "
            "until the run is repeated with a larger `--judge-max-tokens`.\n")
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
            judge_version=args.judge_prompt,
        )
        # The judge travels with the rate (Section 21.40) — and this is the ONE
        # path where the judge is guaranteed to differ from the file it read,
        # since re-judging with a different model is the entire purpose of
        # --rescore-from. It was also the one path that did not update these,
        # so a run rescored by Mistral kept "Qwen2.5-32B" on every row while the
        # derived report header correctly said Mistral. An absent field reads as
        # unknown; a wrong one reads as a fact, and --compare reads the file.
        r["judge_model"] = args.judge_model
        r["judge_prompt"] = args.judge_prompt
        r["rescored_from"] = args.rescore_from.name
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
                # Includes the V4 signal (how often the judge claims something
                # it cannot quote), which was the one field this copy already
                # carried. See RUBRIC_DIAGNOSTICS.
                carry_diagnostics(data, entry)
        if i % 20 == 0 or i == len(results):
            print(f"  re-scored {i}/{len(results)}")

    with args.out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    arms = list(results[0]["arms"])
    # Which judge prompt ran is DERIVED, never assumed. This header used to say
    # "with the v2 judge" unconditionally, while score_one_question routes to
    # the V3 rubric judge for any record carrying key_points — so once the gold
    # set became fully rubric-backed, every rescore report described a prompt
    # that had not run. A reader comparing it against the V3 first-pass report
    # would conclude the two-judge study varied model AND prompt, and discount
    # a result that is actually clean. Same trap as the stale ADAPTER_PATH:
    # the default is fine, printing it without checking is not.
    n_rubric = sum(1 for r in results for d in r["arms"].values()
                   if d.get("scored_by") == "rubric")
    n_prose = sum(len(r["arms"]) for r in results) - n_rubric
    # ...and WHICH rubric prompt, which the derivation above still could not
    # see: `scored_by` is "rubric" for V3 and V4 alike, so a `--judge-prompt v4`
    # run produced a report headed "V3 rubric judge" that went on to claim the
    # prompt was identical to the first pass. The same trap as the comment
    # above, one level down. `args.judge_prompt` is known; use it.
    vlabel = "V4" if args.judge_prompt == "v4" else "V3"
    if n_prose == 0:
        how = (f"the **{vlabel} rubric judge**: the judge reports which enumerated key points "
               "and which common errors each answer made, and the score is computed in "
               "Python from those counts"
               + (", and every claim must be backed by a verbatim quote from the "
                  "candidate or it is discarded" if vlabel == "V4" else ""))
    elif n_rubric == 0:
        how = ("the **V2 prose judge**: correctness and citation scored separately, "
               "length/style explicitly excluded, candidates anonymized behind "
               "randomized A/B/C/D labels")
    else:
        how = (f"a mix of both judge prompts — {n_rubric} arm-answers against enumerated "
               f"rubrics (V3) and {n_prose} against a prose reference (V2)")
    lines = ["# Section 9 Evaluation Report (recalibrated judge)\n"]
    lines.append(
        f"{len(results)} questions, re-scored from `{args.rescore_from.name}` with {how}.\n"
    )
    # This sentence is the whole justification for reading the rescore as a
    # two-judge study, so it may only appear when it is TRUE. A V4 rescore
    # varies the prompt as well as the model, which is the one thing Section 9.9
    # says not to do — and the report used to print the reassurance anyway.
    if n_prose == 0 and vlabel == "V3":
        lines.append(
            "The judge PROMPT is identical to the first pass; only the judge MODEL differs. "
            "That is what Section 9.9 requires — vary the judge and nothing else.\n")
    elif n_prose == 0:
        lines.append(
            "> **This rescore changed the judge PROMPT (V3 -> V4), not only the judge "
            "model.** Section 9.9 asks for one variable at a time, so this is not a "
            "two-judge agreement measurement and must not be read against a V3 run as "
            "though it were.\n")
    # Name the judge in the body. The earlier reports recorded it only in the
    # filename (cards_n100_judge2.md), which puts the single
    # most important variable of a two-judge study outside the document.
    lines.append(f"- judge: `{args.judge_model}`\n")
    # Which scoring rule produced these numbers. Section 21.28 changed it, and
    # the same judge output yields different scores under each rule — so a
    # report that does not say which one it used cannot be compared to anything.
    scorings = {d.get("scoring") for r in results for d in r["arms"].values() if d.get("scoring")}
    if scorings:
        lines.append(f"- scoring: `{'`, `'.join(sorted(scorings))}`"
                     + (" — correctness is `points_hit / n_points`; `errors_made` is "
                        "reported but does not move the score (Section 21.28)\n"
                        if scorings == {"points_only"} else "\n"))

    # V4 only: how many claims the judge made and could not back with a quote
    # from the candidate. This is the number the V4 prompt exists to produce —
    # fabrication measured rather than inferred.
    drops = [d.get("quote_drops") for r in results for d in r["arms"].values()
             if d.get("quote_drops") is not None]
    if drops:
        n_claims_dropped = sum(drops)
        n_answers_affected = sum(1 for d in drops if d)
        lines.append(
            f"- **unverifiable claims discarded: {n_claims_dropped}** across "
            f"{n_answers_affected}/{len(drops)} arm-answers. Each was a key point or "
            "common error the judge asserted and then could not quote from the "
            "candidate it was grading (V4, Section 21.7).\n"
        )
    lines += coverage_lines(results)
    # The all-errors-fired rate, which is judge-specific and large: measured
    # 50% of Qwen's blunder calls against 16% of Llama's on this same set
    # (Section 21.26).
    fired = sum(1 for r in results for d in r["arms"].values() if d.get("all_errors_fired"))
    contra = sum(1 for r in results for d in r["arms"].values() if d.get("error_contradiction"))
    blunders = sum(1 for r in results for d in r["arms"].values() if d.get("errors_made"))
    if blunders and any("all_errors_fired" in d for r in results for d in r["arms"].values()):
        lines.append(
            f"- **every listed error fired at once: {fired}/{blunders} blunder calls "
            f"({fired / blunders:.0%})**, of which {contra} also credit the answer with half "
            "the key points or more — two claims that cannot both hold. `common_errors` are "
            "alternative wrong answers; committing all of them is usually not something an "
            "answer can do, and blunder rate is defined on this field (Section 21.26).\n")
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
    parser.add_argument("--no-plain-rag", action="store_true",
                        help="drop the rules-only `_rag` arm. Required for an honest\n"
                             "--k-rules 0 comparison: with k=0 that arm gets an empty\n"
                             "context and becomes byte-identical to the bare base arm\n"
                             "(53/53 questions, vs 0/53 at k=3), which hands the batched\n"
                             "judge a duplicate candidate. CHANGES THE ARM COUNT -- see\n"
                             "Section 21.5 and 21.143.")
    parser.add_argument("--k-rules", type=k_rules_arg, default=3, metavar="K",
                        help="how many CR rules chunks to retrieve per question, or `auto` to\n"
                             "choose PER QUESTION: 0 when the question resolved a card, 3 when\n"
                             "it did not. The two halves were measured separately and disagree\n"
                             "in sign — +0.25 for k=0 with cards (21.144), +0.65 for k=3 without\n"
                             "them (21.155) — so no single k is right for both (Section 21.156).\n"
                             "`auto` changes the CARD arm only; the rules-only `_rag` arm resolves\n"
                             "no cards and always takes the card-free k.")
    parser.add_argument("--few-shot", type=int, default=0, metavar="N",
                        help="prepend N worked (question, answer) examples as completed "
                             "prior turns before the real question — in-context learning, "
                             "never tried in this project before (Section 21.140). Drawn "
                             "from --few-shot-source, which must be a corpus DISJOINT from "
                             "the eval set; overlap is checked on question text, not just "
                             "id. Leaves the system prompts and therefore "
                             "prompt_fingerprint() untouched, so it cannot invalidate a "
                             "stored adapter.")
    parser.add_argument("--few-shot-source", type=Path, default=GOLD_CANDIDATES_PATH,
                        help="corpus to draw --few-shot exemplars from (default: the "
                             "RulesGuru candidate pool, which is the training-data source "
                             "and is not itself an eval set)")
    parser.add_argument("--base-only", action="store_true",
                        help="skip the finetuned arms entirely and evaluate the base model "
                             "alone. Required when --base-model differs from the one the "
                             "adapter was trained on (a 7B LoRA cannot load onto a 32B). "
                             "CHANGES THE ARM COUNT, so a --base-only run is comparable only "
                             "to another --base-only run (Section 21.5).")
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
    parser.add_argument("--compare", type=Path, nargs=2, default=None,
                        metavar=("A.jsonl", "B.jsonl"),
                        help="report inter-judge agreement between two runs of the SAME "
                             "answers, segmented by who wrote each rubric. This is the "
                             "acceptance test for contributed rubrics (Section 14.6).")
    parser.add_argument("--gold-set", type=Path, default=GOLD_PATH,
                        help="--compare joins on gold_id against this to find who wrote each rubric")
    parser.add_argument("--candidates", type=Path, default=GOLD_CANDIDATES_PATH)
    parser.add_argument("--with-cards", action="store_true",
                        help="add {base,finetuned}_rag_cards arms using card-name lookup + rules retrieval")
    parser.add_argument("--with-rulings", action="store_true",
                        help="with --with-cards, add official WotC rulings to the card context")
    parser.add_argument("--judge-model", default=None,
                        help="model used as judge. Defaults to --base-model — which is ALSO the "
                             "'base' arm under test, so an independent judge is needed to rule out "
                             "self-preference bias (Section 9.9).")
    parser.add_argument("--judge-prompt", choices=("v3", "v4", "v5"), default="v3",
                        help="v4 requires a verbatim quote behind every claimed point "
                             "or error and discards claims it cannot verify (Section 21.7). "
                             "v3 is the default so published numbers reproduce.")
    parser.add_argument("--rescore-from", type=Path, default=None,
                        help="re-judge stored answers from a previous results file instead of regenerating")
    args = parser.parse_args()

    if args.with_rulings and not args.with_cards:
        raise SystemExit("--with-rulings requires --with-cards")

    # `--candidates` and `--gold-set` are read ONLY by compare_judges. Passing
    # either to a generation run is accepted by argparse and then does nothing,
    # and the failure is silent and expensive: `--candidates
    # data/gold/card_ruling_candidates.jsonl` looks exactly like "evaluate that
    # benchmark", so the run proceeds on the DEFAULT 110 prose questions and
    # produces a plausible report against the wrong question set an hour later.
    # The benchmark flags are `--gold-only --gold <path>`. Same family as the
    # `--judge-model` that was accepted and ignored on one code path.
    if not args.compare:
        for flag, value, default in (("--candidates", args.candidates, GOLD_CANDIDATES_PATH),
                                     ("--gold-set", args.gold_set, GOLD_PATH)):
            if value != default:
                raise SystemExit(
                    f"{flag} is only read by --compare and would be silently ignored here.\n"
                    f"To evaluate a question set, use:  --gold-only --gold {value}")

    # --judge-model defaults to whatever base model is under test rather than to
    # a hardcoded id, so pointing --base-model at something else doesn't leave
    # the judge silently behind on the old model.
    if args.judge_model is None:
        args.judge_model = args.base_model

    if args.compare:
        compare_judges(args.compare[0], args.compare[1], args.report_out,
                       args.gold_set, args.candidates)
        return

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

    exemplars = None
    if args.few_shot:
        exemplars = load_exemplars(args.few_shot_source, args.few_shot, questions)

    truncation: dict[str, int] = {}
    routing: dict[str, int] = {}
    answers = generate_all_answers(questions, embed_model, args.max_tokens, args.adapter_path,
                                   args.with_cards, args.with_rulings,
                                   base_model_id=args.base_model, base_only=args.base_only,
                                   truncation_out=truncation, exemplars=exemplars,
                                   k_rules=args.k_rules, routing_out=routing,
                                   no_plain_rag=args.no_plain_rag)
    arm_names = list(answers.keys())

    # Consistency check: rerun a subset of finetuned_rag questions and see
    # how often the judge would even need to know — same generation config,
    # does the model give a stable answer. Under --base-only there is no
    # adapter to load (and loading one trained on a different base model
    # would fail outright), so the rerun uses the plain base model and
    # measures the same thing for the arm that actually exists: `base_rag`.
    #
    # The rerun must reproduce the arm it NAMES, or it reports stability for a
    # configuration nothing was generated under. Two things can move that arm
    # out from under it: `--no-plain-rag` deletes `*_rag` entirely (so the arm
    # becomes the bare one, generated with NO context), and `--few-shot`
    # changes every arm's prompt (so the rerun must carry the same exemplars).
    # Both are checked here rather than assumed — a rerun that silently used a
    # different prompt would report a stability number for a prompt shape that
    # never ran, which is Section 21.61's shape: a flag that changes generation
    # and a consumer of that output nobody audited.
    consistency_base = "base" if args.base_only else "finetuned"
    consistency_arm = consistency_base if args.no_plain_rag else f"{consistency_base}_rag"
    print(f"loading {args.base_model} "
          f"({'no adapter' if args.base_only else 'adapter'}) for consistency rerun ...")
    ft_model, ft_tokenizer = load_lm(
        args.base_model, adapter_path=None if args.base_only else args.adapter_path)
    consistency_idx = list(range(min(args.consistency_sample, len(questions))))
    consistency_reruns = []
    for i in consistency_idx:
        # Same k the `_rag` arm was generated under, including under AUTO --
        # this rerun exists to measure whether that arm is STABLE, so a
        # different context makes it measure a configuration nothing ran.
        rerun_k = K_RULES_NO_CARDS if args.k_rules == AUTO_K_RULES else args.k_rules
        ctx = (None if args.no_plain_rag
               else retrieve_context(questions[i]["question"], embed_model, k=rerun_k))
        prompt = build_prompt(ft_tokenizer, questions[i]["question"], ctx,
                              exemplars=exemplars)
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
                carry_diagnostics(per_arm[arm], judge_result)

        results.append(
            {
                "source": q["source"],
                "category": q["category"],
                # Which judge produced these numbers, on every row. Section 21.40
                # spent four sections diagnosing a 40% false-error rate as a
                # property of `errors_made` when it was a property of the 7B, and
                # not one stored run recorded which model it had been judged by —
                # the filename was the only record. A rate is a statement about
                # a judge, so the judge travels with the rate.
                "judge_model": args.judge_model,
                "judge_prompt": args.judge_prompt,
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
    summary = {arm: {"scores": [], "fabricated": 0, "matches": 0, "match_total": 0,
                     "cited_any": 0} for arm in arm_names}
    for r in results:
        for arm, data in r["arms"].items():
            if data["judge_score"] is not None:
                summary[arm]["scores"].append(data["judge_score"])
            if data["citation"]["has_fabricated"]:
                summary[arm]["fabricated"] += 1
            if data["citation"]["cited"]:
                summary[arm]["cited_any"] += 1
            if data["citation"]["matches_reference"] is not None:
                summary[arm]["match_total"] += 1
                if data["citation"]["matches_reference"]:
                    summary[arm]["matches"] += 1

    consistency_agree = sum(
        1 for i in consistency_idx
        if results[i]["arms"][consistency_arm]["answer"].strip() == consistency_reruns[i].strip()
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
        + (f"- adapter under test: `{args.adapter_path}`\n"
           if not args.base_only else
           # Counted, not asserted. This said "3 arms, not 6" unconditionally,
           # which is right only for --base-only --with-cards; the 21.144 runs
           # that decided the router's k=0 branch have TWO arms and their
           # headers claimed three. Arm count is exactly what 21.5 makes
           # load-bearing for comparability, so a header stating the wrong one
           # invites the bad comparison the sentence exists to prevent.
           f"- adapter: **none (`--base-only`)** — {len(arm_names)} arm"
           f"{'s' if len(arm_names) != 1 else ''}"
           f" ({', '.join(f'`{a}`' for a in arm_names)}), not 6. Comparable only to a "
           "run with the SAME arms (Section 21.5: arm count changes scores)\n")
        + f"- max tokens: `{args.max_tokens}`\n"
        + f"- k (rules chunks retrieved): `{args.k_rules}`\n"
        + (f"  - **routed per question** (Section 21.156): "
           + ", ".join(f"`k={k[1:]}` on {n} question{'s' if n != 1 else ''}"
                       for k, n in sorted(routing.items()))
           + ". The rules-only `_rag` arm is not routed — it resolves no cards, "
             f"so it ran at `k={K_RULES_NO_CARDS}` throughout.\n"
           if routing else "")
        + ("- **plain `_rag` arm dropped** (`--no-plain-rag`), so this run's arm "
           "count differs from a default run and the two are not comparable "
           "(Section 21.5)\n" if args.no_plain_rag else "")
        + (f"- few-shot: **{args.few_shot} exemplars** from "
           f"`{args.few_shot_source.name}`\n" if args.few_shot else "")
        + f"- judge: `{args.judge_model}`"
        + ("  (**same model as the `base` arm** — self-preference bias is not "
           "ruled out; re-judge with --rescore-from and an independent judge, Section 9.9)"
           if args.judge_model == args.base_model else "")
    )
    # Truncation, beside the scores rather than only in the run log. An arm
    # that ran out of tokens did not finish its reasoning, so its rubric
    # coverage is capped by a setting rather than by its ability — and the
    # rate is correlated with how verbose an arm is, which is exactly the
    # thing being compared (Section 21.138).
    if truncation and any(truncation.values()):
        n_q = len(questions)
        worst = max(truncation.values()) / max(1, n_q)
        lines.append(
            "\n**Answers stopped by the token ceiling** (not by finishing): "
            + ", ".join(f"`{a}` {c}/{n_q}" for a, c in truncation.items() if c)
            + (f"\n\n> At {worst:.0%} on the worst arm, scores below are "
               "truncation-limited, not ability-limited — re-run with a higher "
               "`--max-tokens` before comparing arms of different verbosity."
               if worst >= 0.25 else "")
        )
    # Beside the table, not only in the run log. The generation-time warning
    # prints at the start of a run and the number it qualifies arrives at the
    # end of one; the number is what gets quoted (Section 21.50).
    _unseen = (unseen_arms(Path(args.adapter_path), list(arm_names))
               if args.adapter_path else [])
    if _unseen:
        lines.append(
            "\n- **" + ", ".join(f"`{a}`" for a in _unseen) + "**: this adapter's "
            "training set contains its system prompt **zero** times. The prompt "
            "fingerprint matches — nothing was edited — but the weights never saw "
            "this shape, which is Section 8.7's mechanism reached without a prompt "
            "change. Read the row as a statement about the training *shape*, not "
            "about the training *data*."
        )
    if n_rubric:
        lines.append(
            f"\n{n_rubric} scored against enumerated rubrics (V3 judge); "
            f"{len(questions) - n_rubric} scored against a prose reference (V2 judge). "
            "Rubric correctness is computed from key points hit, not assigned holistically.\n"
        )
    else:
        lines.append("")
    # The same coverage guard the rescore path has carried since 21.14. It must
    # sit ABOVE the table, because the thing it qualifies is every mean in it.
    lines += coverage_lines(results)
    # Grounding is reported BESIDE the score, never folded into it. Section 19.1
    # measured `base` scoring highest under Qwen while fabricating a rule id on
    # 35 of 99 questions: the V3 judge scores which enumerated claims an answer
    # made, and inventing a citation is not one of them, so fabrication is very
    # nearly free under the score alone. Blending the two would re-create
    # exactly the confounded single number V3 exists to take apart, so instead
    # the column sits next to it and the check below refuses to let the score
    # be read on its own.
    #
    # This costs no judge call and has no judge noise: a rule id either resolves
    # against the pinned CR or it does not.
    # "Grounded" used to sit here as `cited_any - fabricated`: the answers that
    # cited a rule and invented none. It reads as a quality measure and is not
    # one -- it RISES whenever an arm cites more, whatever the accuracy. Section
    # 21.158 caught it inverting on the comparison it mattered for: dropping the
    # CR section took this arm from 32 citing answers to 41 and from 3
    # fabrications to 7, and "grounded" went UP, 29 to 34. The rate carries the
    # denominator, so it cannot do that.
    lines.append("| Arm | Avg score (1-5) | N scored | Cited a rule | Fabricated citation | "
                 "Fabrication rate | Citation matches reference |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for arm in arm_names:
        s = summary[arm]
        avg = sum(s["scores"]) / len(s["scores"]) if s["scores"] else float("nan")
        match_rate = f"{s['matches']}/{s['match_total']}" if s["match_total"] else "n/a"
        # Of the answers that cited anything -- an arm that stays silent cannot
        # fabricate, so `fabricated/n` alone rewards not citing at all.
        fab_rate = (f"{s['fabricated'] / s['cited_any']:.0%}" if s["cited_any"]
                    else "n/a (cited nothing)")
        lines.append(f"| {arm} | {avg:.2f} | {len(s['scores'])}/{len(questions)} | "
                     f"{s['cited_any']}/{len(questions)} | {s['fabricated']}/{len(questions)} | "
                     f"{fab_rate} | {match_rate} |")
    lines.append("")

    # The within-run trip-wire below compares ARMS. It structurally cannot see
    # the comparison 21.158 got wrong, which was across two RUNS: k=0 scored
    # +0.25 and doubled fabrication, both printed, in two reports both read.
    # A k=0 run is the one configuration where the model is asked to cite the
    # CR with no CR text in front of it, so it says so on its own face.
    if args.k_rules == 0 or (args.k_rules == AUTO_K_RULES):
        lines.append(
            "> **No CR text was retrieved for "
            + ("the card arm in this run" if args.k_rules == AUTO_K_RULES
               else "this run (`--k-rules 0`)")
            + ", and the prompt still asks for rule citations.** Measured on the "
            "32B card benchmark, removing that section made the model cite the CR "
            "*more* -- 32 answers to 41 -- from memory, and fabricate on 3 against "
            "7 (Section 21.158). Read the fabrication RATE against a `k=3` run "
            "before reading the score column; the score alone moved +0.25 in the "
            "opposite direction.\n")

    # The Section 19.1 misreading, detected rather than left to the reader: if
    # the top-scoring arm is not also the least-fabricating one, say so in the
    # report instead of hoping whoever reads the table remembers.
    scored_arms = [a for a in arm_names if summary[a]["scores"]]
    if scored_arms:
        top = max(scored_arms, key=lambda a: sum(summary[a]["scores"]) / len(summary[a]["scores"]))
        cleanest = min(scored_arms, key=lambda a: summary[a]["fabricated"])
        if top != cleanest and summary[top]["fabricated"] > summary[cleanest]["fabricated"]:
            lines.append(
                f"> **The highest-scoring arm is not the least-fabricating one.** `{top}` scores "
                f"best while fabricating {summary[top]['fabricated']}/{len(questions)} "
                f"citations, against `{cleanest}`'s {summary[cleanest]['fabricated']}/{len(questions)}. "
                "The rubric judge scores which enumerated claims an answer made; a "
                "fabricated rule id is not one of them, so it costs almost nothing here. "
                "Do not read the score column without this one (Section 19.1).\n"
            )
    lines.append(f"Consistency ({consistency_arm}, {len(consistency_idx)} questions rerun): {consistency_agree}/{len(consistency_idx)} identical on rerun.\n")

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
