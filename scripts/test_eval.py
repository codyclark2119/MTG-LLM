"""Self-checking tests for the scoring arithmetic in eval.py.

    python scripts/test_eval.py

WHY

`test_actions.py` covers the action parser at 84 assertions. The code that
turns a judge's JSON into the numbers this project publishes had none — and it
is the more consequential half. A parser bug shows up as a bad legality rate,
which looks wrong. A bug in `rubric_correctness` shows up as a plausible score,
which does not.

Everything here is pure arithmetic and string work: no model, no GPU, no
network. Same style as test_actions.py — plain asserts behind a runner, because
there is no pytest in the env.

The cases that matter are the ones a REAL judge response produces: an index out
of range, a claim repeated, a quote that does not appear, a field of the wrong
type. Each of those has to degrade to a number that is still meaningful, or
fail loudly. Silently scoring is the one thing it must not do.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import pearson_r  # noqa: E402
from eval import (  # noqa: E402
    _FOUR_LABEL_PHRASE,
    JUDGE_SYSTEM_PROMPT_V2,
    JUDGE_SYSTEM_PROMPT_V3,
    JUDGE_SYSTEM_PROMPT_V4,
    RUBRIC_DIAGNOSTICS,
    _author_of,
    carry_diagnostics,
    judge_batch_rubric,
    judge_prompt_for,
    rubric_correctness,
    score_citations,
    stratified_sample,
    verify_quoted_claims,
)

CHECKS_RUN = 0


def check(label: str, got, want) -> bool:
    global CHECKS_RUN
    CHECKS_RUN += 1
    if got == want or (got != got and want != want):  # NaN == NaN by hand
        return True
    print(f"  FAIL {label}\n       got  {got!r}\n       want {want!r}")
    return False


def near(label: str, got, want, tol=1e-9) -> bool:
    global CHECKS_RUN
    CHECKS_RUN += 1
    if abs(got - want) <= tol:
        return True
    print(f"  FAIL {label}\n       got  {got!r}\n       want {want!r} (+-{tol})")
    return False


def test_rubric_correctness() -> int:
    """The 1-5 score every rubric-judged number in this project comes from."""
    failed = 0
    score = lambda ph, em, np_, ne: rubric_correctness(ph, em, np_, ne)["correctness"]

    # --- the scale endpoints ------------------------------------------------
    failed += not check("all points, no errors -> 5", score([1, 2, 3, 4], [], 4, 3), 5.0)
    failed += not check("no points, no errors -> 1", score([], [], 4, 3), 1.0)
    failed += not check("half the points -> 3", score([1, 2], [], 4, 3), 3.0)
    failed += not check("one of four -> 2", score([1], [], 4, 3), 2.0)

    # --- errors no longer move the score (Section 21.28) --------------------
    # The halving was removed after the positive controls measured the judge
    # inventing an error against the REFERENCE ANSWER on 40% of questions,
    # costing it 2.47 points it definitionally could not have lost.
    failed += not check("all points + an error -> still 5", score([1, 2, 3, 4], [1], 4, 3), 5.0)
    failed += not check("no points + an error -> 1", score([], [1], 4, 3), 1.0)
    failed += not check("errors do not move the score",
                        score([1, 2], [1, 2, 3], 4, 3), score([1, 2], [], 4, 3))

    # ...but errors are still EXTRACTED and reported, because blunder rate is
    # defined on them. Dropping the field would have been a different and much
    # worse change than dropping its effect on the score.
    r = rubric_correctness([1, 2, 3, 4], [1, 3], 4, 3)
    failed += not check("errors_made still reported", r["errors_made"], [1, 3])
    failed += not check("scoring version recorded", r["scoring"], "points_only")

    # --- the old rule stays reachable, and reproduces the published numbers -
    # Every figure through Section 21.27 was computed this way. Without this
    # path the record could not be re-derived, and the change would be a break
    # rather than a revision.
    old = lambda ph, em: rubric_correctness(ph, em, 4, 3, halve_on_error=True)["correctness"]
    failed += not check("halved: all points + an error -> 3", old([1, 2, 3, 4], [1]), 3.0)
    failed += not check("halved: two errors halve once, not twice", old([1, 2, 3, 4], [1, 2]), 3.0)
    failed += not check("halved: no error is unchanged", old([1, 2, 3, 4], []), 5.0)
    failed += not check("halved: records its own version",
                        rubric_correctness([1], [], 4, 3, halve_on_error=True)["scoring"],
                        "halved_v3")

    # --- a judge that miscounts must not move the score ---------------------
    # These are not hypotheticals: judge output has claimed point 7 of 4, and
    # has listed the same point twice in one response.
    failed += not check("point index above n_points is dropped",
                        score([1, 2, 7], [], 4, 3), 3.0)
    failed += not check("duplicate points cannot exceed the scale",
                        score([2, 2, 2, 2, 2], [], 4, 3), 2.0)
    failed += not check("point index 0 is dropped", score([0, 1], [], 4, 3), 2.0)
    failed += not check("negative point index is dropped", score([-1, 1], [], 4, 3), 2.0)
    failed += not check("non-integer claims are ignored",
                        score(["1", None, 1.0, 2], [], 4, 3), 2.0)

    # An out-of-range ERROR is the dangerous direction: blunder rate is defined
    # on errors_made being non-empty, so a hallucinated error index that
    # survived would both halve the score AND count as a blunder.
    failed += not check("error index above n_errors does not halve",
                        score([1, 2, 3, 4], [9], 4, 3), 5.0)
    r = rubric_correctness([1, 2], [9], 4, 3)
    failed += not check("...and does not register as a blunder", r["errors_made"], [])

    # --- the returned record ------------------------------------------------
    r = rubric_correctness([3, 1, 1], [2], 4, 3)
    failed += not check("points_hit is sorted and deduped", r["points_hit"], [1, 3])
    failed += not check("errors_made is sorted", r["errors_made"], [2])
    failed += not check("points_total is the rubric size", r["points_total"], 4)

    # --- valid JSON of the wrong SHAPE --------------------------------------
    # `"points_hit": 3` for `[3]`. Both json.loads guards pass it through, and
    # it used to raise TypeError inside the set comprehension — an uncaught
    # crash, with no handler anywhere between here and main(), that would kill
    # a multi-hour run at whatever question the judge fumbled.
    failed += not check("a bare scalar is read as the claim it names",
                        score(3, [], 4, 3), 2.0)
    failed += not check("...on the errors field too",
                        rubric_correctness([1, 2, 3, 4], 1, 4, 3,
                                           halve_on_error=True)["correctness"], 3.0)
    failed += not check("None claims -> no points", score(None, None, 4, 3), 1.0)
    failed += not check("a tuple is a list", score((1, 2), [], 4, 3), 3.0)

    # bool is a subclass of int, so `"points_hit": true` would otherwise read
    # as "hit point 1" — a fabricated claim awarded silently, from a response
    # that named no point at all.
    failed += not check("True is not point 1", score(True, [], 4, 3), 1.0)
    failed += not check("True in a list is not point 1", score([True, 2], [], 4, 3), 2.0)
    failed += not check("True is not error 1 either", score([1, 2, 3, 4], [True], 4, 3), 5.0)

    # --- a rubric with no key points ---------------------------------------
    # Current behaviour, asserted so a change is deliberate: n_points == 0
    # scores 1.0 for every arm. validate_gold.py requires key_points, so this
    # should be unreachable from the gold set -- but `--gold` takes any path.
    failed += not check("empty rubric scores 1.0, not a crash", score([], [], 0, 0), 1.0)
    return failed


def test_verify_quoted_claims() -> int:
    """V4's receipt check: a claim the judge cannot quote is not a claim."""
    failed = 0
    answer = ("Trample assigns only lethal damage to the blocker, and the rest "
              "carries through to the defending player.")

    # --- V3 compatibility: bare integers are kept ---------------------------
    # A V3 response must not be silently zeroed by a V4 code path; it never
    # asked for a quote.
    failed += not check("bare ints kept", verify_quoted_claims([1, 3], answer, 4), ([1, 3], 0))
    failed += not check("bare int out of range dropped, not counted",
                        verify_quoted_claims([1, 9], answer, 4), ([1], 0))

    # --- the real V4 shape --------------------------------------------------
    good = [{"n": 1, "quote": "assigns only lethal damage to the blocker"}]
    failed += not check("quote present -> kept", verify_quoted_claims(good, answer, 4), ([1], 0))

    bad = [{"n": 2, "quote": "deathtouch makes any damage lethal"}]
    failed += not check("quote absent -> dropped and counted",
                        verify_quoted_claims(bad, answer, 4), ([], 1))

    mixed = good + bad
    failed += not check("mixed response keeps the honest claim",
                        verify_quoted_claims(mixed, answer, 4), ([1], 1))

    # --- normalization: a quote must survive reformatting -------------------
    reflowed = [{"n": 1, "quote": "ASSIGNS ONLY LETHAL\n  DAMAGE to the blocker"}]
    failed += not check("case and whitespace normalized",
                        verify_quoted_claims(reflowed, answer, 4), ([1], 0))

    # --- a short quote is not evidence --------------------------------------
    # Under 12 chars matches too easily to mean anything, so it is kept rather
    # than checked. Asserted in both directions: a short quote that is NOT in
    # the answer must also be kept, or the threshold is doing nothing.
    failed += not check("short quote kept when present",
                        verify_quoted_claims([{"n": 1, "quote": "trample"}], answer, 4), ([1], 0))
    failed += not check("short quote kept when absent (below the threshold)",
                        verify_quoted_claims([{"n": 1, "quote": "zzzz"}], answer, 4), ([1], 0))

    # --- degradation, not collapse ------------------------------------------
    failed += not check("missing quote key degrades to V3",
                        verify_quoted_claims([{"n": 1}], answer, 4), ([1], 0))
    failed += not check("empty quote degrades to V3",
                        verify_quoted_claims([{"n": 1, "quote": "   "}], answer, 4), ([1], 0))
    failed += not check("non-int n dropped",
                        verify_quoted_claims([{"n": "1", "quote": "x" * 20}], answer, 4), ([], 0))
    failed += not check("out-of-range n dropped before the quote is checked",
                        verify_quoted_claims([{"n": 9, "quote": "not in the answer at all"}],
                                             answer, 4), ([], 0))
    failed += not check("a bare string claim is ignored",
                        verify_quoted_claims(["point 1"], answer, 4), ([], 0))
    failed += not check("None claims -> empty", verify_quoted_claims(None, answer, 4), ([], 0))
    failed += not check("a bare scalar is wrapped, not a crash",
                        verify_quoted_claims(2, answer, 4), ([2], 0))
    failed += not check("True is not claim 1", verify_quoted_claims(True, answer, 4), ([], 0))
    failed += not check("True as an n is not claim 1",
                        verify_quoted_claims([{"n": True, "quote": "x" * 20}], answer, 4), ([], 0))
    failed += not check("empty answer drops every long quote",
                        verify_quoted_claims(good, "", 4), ([], 1))
    return failed


def test_judge_prompt_for() -> int:
    """The label list the judge is told about must match the one it is shown."""
    failed = 0
    base = JUDGE_SYSTEM_PROMPT_V3

    # THE assertion in this file. `judge_prompt_for` works by str.replace on a
    # literal phrase; if that phrase ever stops appearing in a prompt, the
    # replace becomes a silent no-op and a five-arm run is told there are four.
    # No test would notice, because the function still returns a valid prompt.
    # Exactly once, too: a second occurrence would be rewritten in one place
    # and left stale in the other, which is worse than not rewriting at all.
    for name, prompt in (("V2", JUDGE_SYSTEM_PROMPT_V2),
                         ("V3", JUDGE_SYSTEM_PROMPT_V3),
                         ("V4", JUDGE_SYSTEM_PROMPT_V4)):
        failed += not check(f"{name} contains the phrase being rewritten, exactly once",
                            prompt.count(_FOUR_LABEL_PHRASE), 1)

    # Four arms must reproduce every published run byte for byte.
    four = judge_prompt_for(base, ["A", "B", "C", "D"])
    failed += not check("four arms -> unchanged", four, base)
    failed += not check("four arms -> the same object, not a copy", four is base, True)

    five = judge_prompt_for(base, ["A", "B", "C", "D", "E"])
    failed += not check("five arms -> five named", "labeled A, B, C, D and E" in five, True)

    # The direction with teeth is FEWER arms, not more. Told about a candidate
    # it was never shown, a judge has an entry to fill and nothing to fill it
    # from -- and `judge_batch_rubric` reads scores back by label, so an
    # invented "D" would be dropped silently rather than raising.
    # (as a whole word — "CANDIDATE" and "Do NOT" both contain a D)
    three = judge_prompt_for(base, list("ABC"))
    failed += not check("three arms", "labeled A, B and C" in three, True)
    failed += not check("...and no fourth label is mentioned",
                        re.findall(r"\bD\b", three), [])
    failed += not check("two arms", "labeled A and B" in judge_prompt_for(base, list("AB")), True)
    failed += not check("one arm", "labeled A" in judge_prompt_for(base, ["A"]), True)
    # A single arm must not read as a list of one with a stray "and".
    failed += not check("one arm has no conjunction",
                        " and " in judge_prompt_for(base, ["A"]).split("labeled A")[1][:6], False)

    # Only the label phrase changes; the rest of the prompt is untouched.
    failed += not check("prompt length changes only by the phrase",
                        len(five) - len(base),
                        len("labeled A, B, C, D and E") - len(_FOUR_LABEL_PHRASE))
    return failed


def test_score_citations() -> int:
    """Grounding: which rule ids an answer cites, and whether they exist."""
    failed = 0
    valid = {"509.1a", "702.19b", "104.3a"}

    r = score_citations("Trample (702.19b) applies here, see also 509.1a.", ["702.19b"], valid)
    failed += not check("cited ids found in prose", r["cited"], ["509.1a", "702.19b"])
    failed += not check("no fabrication", r["has_fabricated"], False)
    failed += not check("overlaps the reference", r["matches_reference"], True)

    r = score_citations("By rule 999.9z the creature dies.", ["702.19b"], valid)
    failed += not check("a rule id that does not exist is fabricated",
                        r["has_fabricated"], True)
    failed += not check("...and does not match the reference", r["matches_reference"], False)

    # matches_reference is None, not False, when there is nothing to match
    # against. False would read as "cited the wrong rules" in the report.
    r = score_citations("Trample (702.19b) applies.", [], valid)
    failed += not check("no reference ids -> None, not False", r["matches_reference"], None)

    r = score_citations("See 509.1a. As 509.1a says, 509.1a is the rule.", [], valid)
    failed += not check("repeated citations dedupe", r["cited"], ["509.1a"])

    r = score_citations("No citations here at all.", ["702.19b"], valid)
    failed += not check("no citations -> nothing cited", r["cited"], [])
    failed += not check("no citations -> nothing fabricated", r["has_fabricated"], False)
    failed += not check("no citations -> does not match", r["matches_reference"], False)

    # The regex finds ids in PROSE, so it must not be fooled by neighbouring
    # digits. "1509.1a" is not rule 509.1a, and a version string is not a rule.
    r = score_citations("Version 2.19 of the doc, item 1509.1a.", [], valid)
    failed += not check("neighbouring digits do not produce a rule id", r["cited"], [])
    return failed


def test_stratified_sample() -> int:
    """The sampler that decides which questions an eval actually asks."""
    failed = 0
    rows = ([{"category": "a", "i": i} for i in range(10)]
            + [{"category": "b", "i": i} for i in range(10)]
            + [{"category": "c", "i": i} for i in range(2)])

    picked = stratified_sample(rows, 9)
    failed += not check("returns exactly `limit`", len(picked), 9)
    cats = sorted(p["category"] for p in picked)
    failed += not check("spreads across categories", cats.count("c"), 2)
    failed += not check("...and the deep ones absorb the rest",
                        cats.count("a") + cats.count("b"), 7)

    # Determinism. Comparing two runs requires the same questions in both, and
    # nothing here seeds an RNG -- so it must be a pure function of the input.
    failed += not check("deterministic", stratified_sample(rows, 9), picked)

    failed += not check("limit above n returns everything available",
                        len(stratified_sample(rows, 100)), len(rows))
    failed += not check("no duplicates", len({id(p) for p in picked}), len(picked))

    # A missing category key must not crash or silently merge groups.
    mixed = rows + [{"i": 99}]
    failed += not check("missing category is its own bucket",
                        any(p.get("i") == 99 and "category" not in p
                            for p in stratified_sample(mixed, 100)), True)
    return failed


def test_pearson_r() -> int:
    """The headline statistic: inter-judge agreement (Section 14.6)."""
    failed = 0
    failed += not near("perfect agreement", pearson_r([(1, 1), (2, 2), (3, 3), (4, 4)]), 1.0)
    failed += not near("perfect inversion", pearson_r([(1, 4), (2, 3), (3, 2), (4, 1)]), -1.0)

    # A judge that gives everything the same score has zero variance, so r is
    # undefined -- it must be NaN, never 0.0. Reported as 0.0 it would read as
    # "the judges are uncorrelated", which is a much weaker claim than "one of
    # them said nothing".
    nan = pearson_r([(3, 1), (3, 2), (3, 3)])
    failed += not check("constant series -> NaN", nan != nan, True)
    under = pearson_r([(1, 1), (2, 2)])
    failed += not check("under three pairs -> NaN", under != under, True)

    r = pearson_r([(5, 5), (4, 3), (3, 4), (2, 2), (1, 1)])
    failed += not check("a realistic pair correlates but is not 1.0", 0.8 < r < 1.0, True)
    return failed


def test_author_of() -> int:
    """Attribution rides on the submission, and `--compare` splits on it."""
    failed = 0
    failed += not check("name in parens",
                        _author_of("hand-authored (Cody Clark); decomposed from RulesGuru"),
                        "Cody Clark")
    failed += not check("judge token form", _author_of("hand-authored (judge:CC)"), "judge:CC")
    failed += not check("no parens -> the string itself",
                        _author_of("machine-drafted"), "machine-drafted")
    failed += not check("empty -> unattributed", _author_of(""), "unattributed")
    failed += not check("None -> unattributed", _author_of(None), "unattributed")
    return failed


class _StubTokenizer:
    """`apply_chat_template` is all judge_batch_rubric needs from a tokenizer."""

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return "\n".join(m["content"] for m in messages)


def _judge_returning(payload: str):
    """A fake lm_generate that answers with `payload`, recording the prompt."""
    seen = {}

    def gen(model, tokenizer, prompt, max_tokens, verbose=False):
        seen["prompt"] = prompt
        return payload

    return gen, seen


def test_judge_batch_rubric() -> int:
    """The whole judge path, with the model stubbed out.

    The label mapping is the piece worth covering: candidates are shuffled
    behind A/B/C/D and mapped back afterward, so an off-by-one here would
    attribute every arm's score to a different arm — a result that looks
    entirely normal and is exactly wrong.
    """
    failed = 0
    kp = ["Trample carries damage past the blocker", "Doom Blade can target it"]
    ce = ["Thinks a chump block stops all the damage"]
    cands = {"base": "A", "base_rag": "B", "finetuned": "C", "finetuned_rag": "D"}

    # Every arm answers with its own name, and the judge awards point 1 to
    # whichever label carries "base_rag". If the mapping is right, base_rag is
    # the only arm scoring above 1.0 whatever the shuffle does.
    import json as _json
    import random as _random

    for seed in range(6):
        rng = _random.Random(seed)
        # Find which label base_rag lands on by replaying the same shuffle.
        arms = list(cands)
        _random.Random(seed).shuffle(arms)
        label_of = {arm: chr(ord("A") + i) for i, arm in enumerate(arms)}
        payload = _json.dumps({lbl: {"points_hit": [1, 2] if arm == "base_rag" else [],
                                     "errors_made": [], "citation": 5, "note": ""}
                               for arm, lbl in label_of.items()})
        gen, _ = _judge_returning(payload)
        out = judge_batch_rubric(gen, None, _StubTokenizer(), "q?", kp, ce,
                                 dict(cands), 256, rng)
        failed += not check(f"seed {seed}: the scored arm is the intended one",
                            [a for a in out if out[a]["correctness"] > 1.0], ["base_rag"])

    # --- a single candidate returned UNWRAPPED (Section 21.39) ---------------
    # Asked to grade one answer "labeled A", the judge emits the entry directly
    # instead of {"A": {...}}. That is reasonable, and `scored.get("A")` read it
    # as "the judge said nothing" — which showed up as a 33% coverage loss that
    # looked like a cost of the V5 prompt and was an artifact of the harness.
    gen, _ = _judge_returning(_json.dumps(
        {"points_hit": [1, 2], "errors_made": [], "citation": 5, "note": ""}))
    out = judge_batch_rubric(gen, None, _StubTokenizer(), "q?", kp, ce,
                             {"only": "x"}, 256, __import__("random").Random(0))
    failed += not check("unwrapped single candidate is read", list(out), ["only"])
    failed += not check("...with its points", out["only"]["points_hit"], [1, 2])

    # Only for ONE arm. With several, an unwrapped object cannot be attributed
    # to any of them and must still fail rather than be given to an arbitrary arm.
    gen, _ = _judge_returning(_json.dumps(
        {"points_hit": [1, 2], "errors_made": [], "citation": 5, "note": ""}))
    out = judge_batch_rubric(gen, None, _StubTokenizer(), "q?", kp, ce,
                             dict(cands), 256, __import__("random").Random(0))
    failed += not check("unwrapped multi-arm output is still rejected", out, {})

    # A properly wrapped single candidate must not be double-wrapped.
    gen, _ = _judge_returning(_json.dumps(
        {"A": {"points_hit": [1], "errors_made": [], "citation": 5, "note": ""}}))
    out = judge_batch_rubric(gen, None, _StubTokenizer(), "q?", kp, ce,
                             {"only": "x"}, 256, __import__("random").Random(0))
    failed += not check("wrapped single candidate still works", out["only"]["points_hit"], [1])

    # --- the shape bug, through the real path -------------------------------
    # This is what would have killed a run: valid JSON, wrong shape.
    gen, _ = _judge_returning('{"A": {"points_hit": 1, "errors_made": null, "citation": 5}}')
    out = judge_batch_rubric(gen, None, _StubTokenizer(), "q?", kp, ce,
                             {"only": "x"}, 256, __import__("random").Random(0))
    failed += not check("a scalar points_hit scores instead of crashing",
                        out["only"]["correctness"], 3.0)

    # --- judge output that is not usable at all -----------------------------
    for label, payload in (("no JSON", "I cannot judge this."),
                           ("malformed JSON", '{"A": {"points_hit": [1,}')):
        gen, _ = _judge_returning(payload)
        out = judge_batch_rubric(gen, None, _StubTokenizer(), "q?", kp, ce,
                                 dict(cands), 256, __import__("random").Random(0))
        failed += not check(f"{label} -> empty, not a partial score", out, {})

    # An arm the judge simply omitted must be absent, never a default score.
    # Reported as 1.0 it would read as "this arm answered badly".
    gen, _ = _judge_returning('{"A": {"points_hit": [1, 2], "errors_made": [], "citation": 5}}')
    out = judge_batch_rubric(gen, None, _StubTokenizer(), "q?", kp, ce,
                             dict(cands), 256, __import__("random").Random(0))
    failed += not check("omitted arms are absent, not zero-scored", len(out), 1)

    # --- the "every error at once" signature (Section 21.26) ----------------
    # Found by reading answers: on pos-removal-timing-0001 all three arms
    # played the correct line and the judge returned errors [1,2,3,4]. Measured
    # across runs it is 50% of the Qwen judge's blunder calls against 16% of
    # Llama's, and blunder rate is defined on this field.
    kp4, ce4 = ["a", "b", "c", "d"], ["e1", "e2", "e3", "e4"]
    for label, entry, want_fired, want_contra in (
            ("correct line + all errors", {"points_hit": [1, 2, 3, 4], "errors_made": [1, 2, 3, 4]}, True, True),
            ("no points + all errors", {"points_hit": [], "errors_made": [1, 2, 3, 4]}, True, False),
            ("half points + all errors", {"points_hit": [1, 2], "errors_made": [1, 2, 3, 4]}, True, True),
            ("one error only", {"points_hit": [1, 2], "errors_made": [2]}, False, False),
            ("no errors", {"points_hit": [1, 2], "errors_made": []}, False, False)):
        gen, _ = _judge_returning(_json.dumps({"A": {**entry, "citation": 3, "note": ""}}))
        got = judge_batch_rubric(gen, None, _StubTokenizer(), "q?", kp4, ce4,
                                 {"only": "x"}, 256, __import__("random").Random(0))["only"]
        failed += not check(f"{label}: fired_all", got["all_errors_fired"], want_fired)
        failed += not check(f"{label}: contradiction", got["error_contradiction"], want_contra)

    # Under three listed errors the signature means nothing — firing both of
    # two is ordinary. The threshold must not fire there.
    gen, _ = _judge_returning(_json.dumps({"A": {"points_hit": [1], "errors_made": [1, 2],
                                                 "citation": 3, "note": ""}}))
    got = judge_batch_rubric(gen, None, _StubTokenizer(), "q?", ["a", "b"], ["e1", "e2"],
                             {"only": "x"}, 256, __import__("random").Random(0))["only"]
    failed += not check("two errors is not the signature", got["all_errors_fired"], False)

    # --- V4 drops an unquotable claim, and says how many --------------------
    payload = _json.dumps({"A": {
        "points_hit": [{"n": 1, "quote": "carries damage past the blocker"},
                       {"n": 2, "quote": "a sentence that is nowhere in the answer"}],
        "errors_made": [], "citation": 5, "note": ""}})
    gen, _ = _judge_returning(payload)
    out = judge_batch_rubric(gen, None, _StubTokenizer(), "q?", kp, ce,
                             {"only": "Trample carries damage past the blocker."},
                             256, __import__("random").Random(0), judge_version="v4")
    failed += not check("V4 keeps the quotable claim", out["only"]["points_hit"], [1])
    failed += not check("V4 counts the drop", out["only"]["quote_drops"], 1)

    # V3 on the same response must keep both — it never asked for a quote.
    gen, _ = _judge_returning(payload)
    out = judge_batch_rubric(gen, None, _StubTokenizer(), "q?", kp, ce,
                             {"only": "Trample carries damage past the blocker."},
                             256, __import__("random").Random(0), judge_version="v3")
    failed += not check("V3 ignores quotes entirely", out["only"]["points_hit"], [])
    return failed


def test_behaviour_opening() -> int:
    """common_errors must be assertable claims, not player behaviours (21.35).

    The judge is asked which of these the candidate ASSERTED. `key_points` are
    claims, so that question is answerable and the judge credits the reference
    answer with 90% of its own points. `common_errors` were 89% behaviour
    descriptions, and the same question becomes "did this text describe a player
    doing that?" — vaguer, and the leading explanation for a 40% false-positive
    rate against an answer that cannot commit an error at all.
    """
    from common import looks_like_behaviour as f  # noqa: E402
    failed = 0
    for text, want in (
            # the old form: capitalised third-person verb, what a player DOES
            ("Adds Centaur Courser to the block, spending a 3/3", True),
            ("Holds Doom Blade for a better target", True),
            ("Treats reach as letting the Asp attack in the air", True),
            ("Keeps the second Forest for Giant Growth", True),
            # claims a wrong answer could actually contain
            ("A chump block stops all the trample damage", False),
            ("Trample damage is fully absorbed by any blocker", False),
            ("Adding Centaur Courser to the block is worth the 3 life", False),
            ("Player A controls both permanents", False),
            ("Because the trigger is put on the stack first, it resolves first", False),
            # excluded openers: legitimate claim starts that look third-person
            ("Has haste, so it can attack immediately", False),
            ("Is a legal target because it is nonblack", False),
            ("Triggers resolve in the order they are announced", False),
            ("This hand should be mulliganed", False)):
        failed += not check(f"{'warns' if want else 'silent'}: {text[:44]}",
                            f(text), want)
    return failed


def test_error_assertions() -> int:
    """The negative control's construction (Section 21.40).

    The whole defensibility of this control is that the planted sentence is the
    rubric line VERBATIM — the moment a paraphrase creeps in, a miss stops being
    attributable to the judge. So the checks are: the claim survives untouched,
    behaviour-form entries are refused rather than reshaped, and the rotation
    actually rotates.
    """
    from calibrate_judge import (ASSERTION_TAIL, build_error_assertions,
                                 question_text, reference_answer, separation)
    failed = 0

    # --- the pair is mandatory (Section 21.43) -----------------------------
    # Three published conclusions came from one side of this control. A
    # one-sided call must raise, not return a partial result.
    one = [{"fired": [1], "n_fired": 1, "hit": True}]
    for label, args in (("no clean half", ([], one)), ("no planted half", (one, [])),
                        ("neither half", ([], []))):
        try:
            separation(*args)
            failed += not check(f"separation raises: {label}", "returned", "raised")
        except ValueError:
            failed += not check(f"separation raises: {label}", "raised", "raised")

    # Llama's real shape: perfect on the planted half, fires at 58% of clean
    # answers. Ranked on the planted column alone it is the BEST of four judges.
    clean = [{"fired": [1] if i < 14 else [], "n_fired": 1 if i < 14 else 0}
             for i in range(24)]
    planted = [{"fired": [1], "n_fired": 1, "hit": True} for _ in range(24)]
    s = separation(clean, planted)
    failed += not check("p(fire|clean)", round(s["p_fire_clean"], 4), round(14 / 24, 4))
    failed += not check("p(fire|error)", s["p_fire_error"], 1.0)
    failed += not check("separation subtracts", round(s["separation"], 4), round(10 / 24, 4))
    failed += not check("hit rate carried", s["hit_planted"], 1.0)
    # A judge that fires at EVERYTHING scores a perfect 100% on the planted
    # half and separates nothing. This is the case one column cannot see.
    s0 = separation(planted, planted)
    failed += not check("fires-at-everything separates zero", s0["separation"], 0.0)

    # --- padding a clean answer to answer length (Section 21.41) -----------
    from calibrate_judge import pad_clean
    rt = {"104.1": "A game ends immediately when a player wins.",
          "601.2": "To cast a spell, a player follows the steps below."}
    rec = {"rule_citations": ["104.1", "601.2"]}
    padded = pad_clean(rec, "Yes.", rt)
    failed += not check("padding lengthens", len(padded) > len("Yes."), True)
    failed += not check("original answer survives", padded.startswith("Yes."), True)
    for rid in ("104.1", "601.2"):
        failed += not check(f"CR text {rid} appended verbatim", rt[rid] in padded, True)
    # A record with no resolvable citation must come back UNCHANGED — counting
    # it as padded would mix two lengths in one rate, which is the confound
    # 21.41 was written about.
    failed += not check("unresolvable citation leaves it alone",
                        pad_clean({"rule_citations": ["999.9"]}, "Yes.", rt), "Yes.")
    failed += not check("no citations leaves it alone",
                        pad_clean({}, "Yes.", rt), "Yes.")
    # Falls back to supporting_rule_ids, which is what the eval file carries.
    failed += not check("supporting_rule_ids also works",
                        rt["104.1"] in pad_clean({"supporting_rule_ids": ["104.1"]}, "Y.", rt),
                        True)

    # --- the reference answer, three record shapes -------------------------
    failed += not check("position/gold use `answer`",
                        reference_answer({"answer": "Block."}), "Block.")
    failed += not check("eval file uses the assistant turn",
                        reference_answer({"messages": [{"role": "user", "content": "Q"},
                                                       {"role": "assistant", "content": "A."}]}),
                        "A.")
    # "" rather than a guess: run_judge_report refuses to run on it, because
    # grading a blank as the clean answer would score perfect specificity.
    failed += not check("no reference yields empty", reference_answer({"messages": []}), "")

    claim = "Centaur Courser should be added to the block alongside Sedge Scorpion"
    recs = [{"id": f"p{i}", "battlefield": ["Mountain"], "key_points": ["k"],
             "common_errors": [claim, "Second claim about blocking", "Third claim here"]}
            for i in range(3)]
    cases, skipped_e, skipped_r = build_error_assertions(recs)
    failed += not check("all three usable", len(cases), 3)
    failed += not check("nothing skipped", (skipped_e, skipped_r), (0, 0))
    failed += not check("claim appears verbatim", claim in cases[0]["answer"], True)
    failed += not check("answer is claim + tail", cases[0]["answer"], claim + ASSERTION_TAIL)
    # Rotation: record i plants error i % len, so three records must not all
    # plant error 1 — otherwise the result describes first entries only.
    failed += not check("rotates the planted error", [c["n"] for c in cases], [1, 2, 3])

    # A behaviour-shaped entry cannot be asserted, so it must be dropped, never
    # rewritten — a rewrite is the prose that made this control undefensible.
    beh = {"id": "b", "battlefield": [], "key_points": ["k"],
           "common_errors": ["Adds Centaur Courser to the block, spending a 3/3"]}
    cases, skipped_e, skipped_r = build_error_assertions([beh])
    failed += not check("behaviour entry yields no case", cases, [])
    failed += not check("counted as a dropped record", skipped_r, 1)
    failed += not check("counted as a skipped entry", skipped_e, 1)

    # Mixed: keep the claim, drop the behaviour, still usable.
    mixed = {"id": "m", "battlefield": [], "key_points": ["k"],
             "common_errors": ["Adds Centaur Courser to the block", claim]}
    cases, skipped_e, skipped_r = build_error_assertions([mixed])
    failed += not check("mixed record is usable", len(cases), 1)
    failed += not check("picks the claim, not the behaviour", cases[0]["n"], 2)
    failed += not check("mixed: entry skipped but record kept", (skipped_e, skipped_r), (1, 0))

    # A position renders its board; a rules question uses its text. Getting this
    # backwards would judge every position against an empty question. Checked
    # against a REAL position rather than a stub — a hand-built fixture missing
    # a key renders differently from the records this actually runs on.
    from common import POSITIONS_PATH, read_jsonl, render_position
    pos = read_jsonl(POSITIONS_PATH)[0]
    failed += not check("position renders its board",
                        question_text(pos), render_position(pos))
    failed += not check("rendered board is non-empty", bool(question_text(pos).strip()), True)
    failed += not check("rules question uses its text",
                        question_text({"question": "Does it resolve?"}), "Does it resolve?")
    failed += not check("falls back to the user message",
                        question_text({"messages": [{"role": "system", "content": "s"},
                                                    {"role": "user", "content": "Q?"}]}), "Q?")
    return failed


def test_carry_diagnostics() -> int:
    """Everything rubric_correctness computes must reach the stored run.

    Both writers used to list the fields by hand, so a value could be computed,
    consumed by a report, and never stored — which is what happened to
    `all_errors_fired`: rescore()'s Section 21.26 block is gated on the key
    being present, and no writer ever wrote it. The regression this guards is
    someone adding a field to rubric_correctness and not to the copy list.
    """
    failed = 0
    computed = rubric_correctness([1, 2], [], 2, 3)
    for k in ("scoring",):
        failed += not check(f"rubric_correctness emits {k}", k in computed, True)
    # Anything rubric_correctness produces that is not a score must be listed,
    # or it silently stops reaching disk the moment it is added.
    scores = {"correctness", "points_hit", "points_total", "errors_made"}
    failed += not check("every non-score field of a grading is carried",
                        sorted(set(computed) - scores - set(RUBRIC_DIAGNOSTICS)), [])

    entry = {"correctness": 5.0, "scoring": "points_only", "quote_drops": 2,
             "all_errors_fired": True, "error_contradiction": False}
    dest: dict = {}
    carry_diagnostics(dest, entry)
    failed += not check("carries scoring", dest.get("scoring"), "points_only")
    failed += not check("carries quote_drops", dest.get("quote_drops"), 2)
    failed += not check("carries all_errors_fired", dest.get("all_errors_fired"), True)
    # False is a real value, not an absence — `if entry.get(k)` would drop it,
    # and error_contradiction is False on most gradings.
    failed += not check("carries error_contradiction=False",
                        dest.get("error_contradiction"), False)
    failed += not check("does not invent the score", "correctness" in dest, False)
    # A prose (V2) grading has none of these; it must not gain empty keys.
    prose: dict = {}
    carry_diagnostics(prose, {"correctness": 3.0, "note": "ok"})
    failed += not check("prose grading carries nothing", prose, {})
    return failed


def test_unseen_arms() -> int:
    """An arm whose system prompt the training set never contained (21.50).

    The fingerprint check answers "were the prompts edited since training".
    This answers the question it cannot see — "did the weights ever meet this
    shape" — which is Section 8.7's actual mechanism, reachable with every
    fingerprint matching. It is tested here because a silent [] is the failure
    mode: the warning simply never prints and the run reads as a data result.
    """
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    from common import PROMPT_STAMP_FILE, REPO_ROOT
    from stamp_adapter import unseen_arms

    failed = 0
    arms = ["base", "base_rag", "finetuned", "finetuned_rag"]

    def stamped(counts) -> _Path:
        d = _Path(tempfile.mkdtemp())
        body = {"prompt_fingerprint": "x"}
        if counts is not None:
            body["dataset_prompt_counts"] = counts
        (d / PROMPT_STAMP_FILE).write_text(_json.dumps(body), encoding="utf-8")
        return d

    # The real case: data/datasets/verified is plain-prompt only.
    failed += not check("rag arm flagged when the set has no RAG examples",
                        unseen_arms(stamped({"SYSTEM_PROMPT": 1112}), arms),
                        ["finetuned_rag"])
    # Both shapes present (the v2 dataset) — nothing to say.
    failed += not check("both shapes present flags nothing",
                        unseen_arms(stamped({"SYSTEM_PROMPT": 1679,
                                             "RAG_SYSTEM_PROMPT": 1292}), arms), [])
    # A zero count is an absence, not a presence. `in counts` would pass it.
    failed += not check("a count of 0 is an absence",
                        unseen_arms(stamped({"SYSTEM_PROMPT": 10,
                                             "RAG_SYSTEM_PROMPT": 0}), arms),
                        ["finetuned_rag"])
    # base arms never load the adapter, so they can never be flagged — a false
    # positive here would fire on every run and train the reader to ignore it.
    failed += not check("base arms are never flagged",
                        unseen_arms(stamped({"RAG_SYSTEM_PROMPT": 5}), arms),
                        ["finetuned"])
    # A stamp written without --dataset records the prompts as they are now,
    # which is not evidence about training. Silence beats a guess.
    failed += not check("no dataset counts means no claim",
                        unseen_arms(stamped(None), arms), [])
    failed += not check("no stamp at all means no claim",
                        unseen_arms(_Path(tempfile.mkdtemp()), arms), [])
    # And the adapter this was written for.
    failed += not check("v4 flags finetuned_rag on disk",
                        unseen_arms(REPO_ROOT / "models/mtg-rules-adapter-v4", arms),
                        ["finetuned_rag"])
    return failed


def test_rescore_stamps_judge() -> int:
    """--rescore-from must relabel the rows it rewrites (Section 21.51).

    Checked by AST rather than by running it: `rescore` loads a judge model, so
    executing it needs a GPU and this suite may not have one. What is verified
    is the property that broke — which row-level keys the function assigns —
    and that survives reading the source.

    The regression is specific and recurs: re-judging with a DIFFERENT model is
    the entire purpose of --rescore-from, so it is the one path where the stored
    `judge_model` is guaranteed wrong if nobody updates it. The report header is
    derived from args and was right; the data file kept the old judge's name,
    and the data file is what --compare and any later rescore read.
    """
    import ast as _ast

    failed = 0
    from common import REPO_ROOT
    src = REPO_ROOT / "scripts" / "eval.py"   # never cwd-relative; this suite runs from anywhere
    tree = _ast.parse(src.read_text(encoding="utf-8"))
    fn = next((n for n in _ast.walk(tree)
               if isinstance(n, _ast.FunctionDef) and n.name == "rescore"), None)
    failed += not check("rescore() exists", fn is not None, True)
    if fn is None:
        return failed

    assigned = set()
    for node in _ast.walk(fn):
        if isinstance(node, _ast.Assign):
            for t in node.targets:
                if isinstance(t, _ast.Subscript) and _ast.unparse(t.value) == "r":
                    assigned.add(_ast.unparse(t.slice).strip("'\""))

    for key in ("judge_model", "judge_prompt"):
        failed += not check(f"rescore stamps {key} on the row", key in assigned, True)
    # Provenance: which file these answers came from, since a rescore's own
    # filename is the only other record of it.
    failed += not check("rescore records what it rescored from",
                        "rescored_from" in assigned, True)
    return failed


def test_coverage_lines() -> int:
    """The unjudged-coverage guard, and that BOTH writers emit it (21.53).

    Section 21.14's V4 run scored 14 of 99 and printed four confident averages
    over the 14. The guard written for that lived in `rescore()` only, so a
    fresh run — the common case — had no coverage warning at all. The structural
    half of this test is the point: the arithmetic was never broken, the second
    caller was missing.
    """
    import ast as _ast

    from common import REPO_ROOT
    from eval import coverage_lines

    failed = 0
    def rows(n_arms: int, n_rows: int, n_ungraded: int):
        out = []
        left = n_ungraded
        for _ in range(n_rows):
            arms = {}
            for a in range(n_arms):
                arms[f"arm{a}"] = {"correctness": None if left > 0 else 3.0}
                if left > 0:
                    left -= 1
            out.append({"arms": arms})
        return out

    failed += not check("full coverage says nothing", coverage_lines(rows(4, 10, 0)), [])
    # Below the 20% bar: report the count, but do not tell the reader to stop.
    one = coverage_lines(rows(4, 10, 4))          # 4/40 = 10%
    failed += not check("10% unjudged reports the count", len(one), 1)
    failed += not check("10% unjudged does not withhold",
                        any("Read nothing into" in x for x in one), False)
    # At and above it, the means are over a subset selected by rubric size.
    two = coverage_lines(rows(4, 10, 8))          # 8/40 = 20%
    failed += not check("20% unjudged withholds the means",
                        any("Read nothing into" in x for x in two), True)
    failed += not check("the count is reported as a fraction of arm-answers",
                        "8 of 40" in two[0], True)
    # The real case: 14 of 99 questions graded, four arms.
    v4 = coverage_lines(rows(4, 99, (99 - 14) * 4))
    failed += not check("a 14/99 run withholds",
                        any("Read nothing into" in x for x in v4), True)

    # Structural: both report writers must call it. The arithmetic above passed
    # for a year while only one of them did.
    tree = _ast.parse((REPO_ROOT / "scripts" / "eval.py").read_text(encoding="utf-8"))
    for name in ("rescore", "main"):
        fn = next((n for n in _ast.walk(tree)
                   if isinstance(n, _ast.FunctionDef) and n.name == name), None)
        calls = {_ast.unparse(n.func) for n in _ast.walk(fn) if isinstance(n, _ast.Call)}
        failed += not check(f"{name}() emits the coverage guard",
                            "coverage_lines" in calls, True)
    return failed


def test_judge_identity() -> int:
    """An inter-judge report must name its judges and check they differ (21.54).

    `compare_judges` identified its two inputs by FILENAME — the record Section
    21.40 found insufficient — in the one report whose entire subject is which
    judge said what. And nothing stopped comparing a file with itself: the id
    and arm overlap checks both pass, kappa comes out +1.00, and the header
    still says "Inter-judge agreement".
    """
    from eval import assert_two_judges, judges_of

    failed = 0
    A = [{"judge_model": "vendor/A"}] * 3
    B = [{"judge_model": "vendor/B"}] * 3

    failed += not check("names the judge", judges_of(A, "runA"), "runA: `vendor/A`")
    # An unrecorded judge is a fact about the evidence, not a blank.
    failed += not check("absence is stated, not omitted",
                        "unrecorded" in judges_of([{}], "runA"), True)
    # A file whose rows disagree is a harness bug, and averaging across it would
    # report a rate for a judge that never graded all of it.
    failed += not check("mixed judges in one file are flagged",
                        "MIXED" in judges_of([{"judge_model": "A"}, {"judge_model": "B"}], "r"),
                        True)

    failed += not check("two different judges pass silently",
                        assert_two_judges(A, B), None)
    same = assert_two_judges(A, list(A))
    failed += not check("the same judge twice is flagged", same is not None, True)
    # It must say WHY, since kappa +1.00 otherwise reads as a result.
    failed += not check("says what it actually measured",
                        "determinism" in (same or ""), True)
    failed += not check("an unrecorded side cannot claim the judge varied",
                        assert_two_judges([{}], B) is not None, True)
    # Symmetric: the missing side may be either one.
    failed += not check("unrecorded on the other side too",
                        assert_two_judges(A, [{}]) is not None, True)
    return failed


def test_adjudication_scoring() -> int:
    """`unsure` must be excluded and every exclusion reported (21.55).

    The form asks "Genuinely ambiguous — I could argue it either way", stores it
    on the submission, and `score_run` read `errors_present` and nothing else.
    So a reviewer following the instruction to use the box changed nothing, and
    their coin flip entered the precision number as ground truth. Human
    adjudication exists to be the tiebreaker; one that silently includes coin
    flips is not one.
    """
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    sys.path.insert(0, str(Path(__file__).parent))
    from adjudicate import score_run

    failed = 0
    run = _Path(tempfile.mkdtemp()) / "run.jsonl"
    run.write_text("\n".join(_json.dumps(r) for r in [
        {"id": "p1", "arms": {"a": {"errors_made": [1]}}},
        {"id": "p2", "arms": {"a": {"errors_made": [2]}}},
        {"id": "p3", "arms": {"a": {"errors_made": []}}},
        # Ungraded by this judge: must not be scored as "fired nothing".
        {"id": "p4", "arms": {"a": {"errors_made": None}}},
    ]), encoding="utf-8")

    def v(key, errs, **kw):
        return {"key": key, "errors_present": errs, **kw}

    firm = [v("p1::a", [1]), v("p2::a", [2])]
    s = score_run(run, firm, {})
    failed += not check("two firm agreeing verdicts: precision 1.0", s["precision"], 1.0)
    failed += not check("n counts them", s["n"], 2)
    failed += not check("nothing excluded", s["excluded_unsure"], 0)

    # The same two, plus one ambiguous verdict that would drag precision down.
    with_unsure = firm + [v("p3::a", [3], unsure=True)]
    s2 = score_run(run, with_unsure, {})
    failed += not check("an ambiguous verdict does not enter precision",
                        s2["precision"], 1.0)
    failed += not check("it is excluded from n", s2["n"], 2)
    failed += not check("and it is reported, not dropped silently",
                        s2["excluded_unsure"], 1)

    # An answer this judge never graded cannot count against it either way.
    s3 = score_run(run, firm + [v("p4::a", [1])], {})
    failed += not check("ungraded answers do not enter n", s3["n"], 2)
    failed += not check("ungraded answers are reported", s3["unmatched"], 1)

    # not_covered is COUNTED (the human agreed no listed error occurred) but
    # must be reported, since it means the metric is narrower than "was good".
    s4 = score_run(run, [v("p3::a", [], not_covered=True)], {})
    failed += not check("not_covered still scores", s4["n"], 1)
    failed += not check("not_covered is reported", s4["not_covered"], 1)

    # A verdict is keyed record_id::arm, and that key is stable while the text
    # behind it is not — regenerate the arms and the same key names a different
    # answer (Section 21.62). Without the digest, 22 human verdicts would have
    # scored silently against text their author never saw.
    from adjudicate import answer_sha
    graded = {"id": "p5", "arms": {"a": {"errors_made": [1], "answer": "CAST Shock\nPASS"}}}
    run2 = _Path(tempfile.mkdtemp()) / "r2.jsonl"
    run2.write_text(_json.dumps(graded), encoding="utf-8")
    same = v("p5::a", [1], answer_sha=answer_sha("CAST Shock\nPASS"))
    s5 = score_run(run2, [same], {})
    failed += not check("a matching digest scores normally", s5["n"], 1)
    failed += not check("and is not counted stale", s5["stale"], 0)

    other = v("p5::a", [1], answer_sha=answer_sha("PHASE upkeep\nCAST Shock\nPASS"))
    s6 = score_run(run2, [other], {})
    failed += not check("a verdict on different text does not score", s6["n"], 0)
    failed += not check("...and is reported as stale", s6["stale"], 1)
    # Verdicts predating the digest must still score — refusing them would
    # discard human work to enforce a field that did not exist when it was done.
    s7 = score_run(run2, [v("p5::a", [1])], {})
    failed += not check("a verdict with no digest still scores", s7["n"], 1)

    # A charge the reviewer's form never displayed cannot be a false positive.
    # The form showed `common_errors` while the judge was given
    # `common_errors + PROTOCOL_ERRORS`, so every protocol charge fell into
    # `judge - human` by construction (Section 21.75). The trigger rate tracks
    # the condition under test — 0% of charges on a pre-protocol run, 68% on a
    # protocol run — so it reversed the comparison it corrupted: 2.2% vs 2.8%
    # unfixed, 5.6% vs 2.8% fixed.
    run3 = _Path(tempfile.mkdtemp()) / "r3.jsonl"
    run3.write_text(_json.dumps(
        {"id": "p6", "arms": {"a": {"errors_made": [1, 9]}}}), encoding="utf-8")
    # The reviewer saw 4 entries and ticked entry 1. Entry 9 is a protocol
    # charge they were never offered.
    s8 = score_run(run3, [v("p6::a", [1], n_shown=4)], {})
    failed += not check("a charge above n_shown is not a false positive",
                        s8["precision"], 1.0)
    failed += not check("...and is counted, not dropped silently",
                        s8["charges_not_shown"], 1)

    # Same run, same verdict, but the reviewer WAS shown all 11 entries and did
    # not tick 9. Now it is a real false positive.
    s9 = score_run(run3, [v("p6::a", [1], n_shown=11)], {})
    failed += not check("a charge within n_shown IS a false positive",
                        s9["precision"], 0.5)
    failed += not check("nothing restricted", s9["charges_not_shown"], 0)

    # A verdict predating form_version 3 carries no n_shown; the fallback is the
    # rubric's own strategy count, which is exactly what that form displayed.
    s10 = score_run(run3, [v("p6::a", [1])],
                    {"p6": {"common_errors": ["a", "b", "c", "d"]}})
    failed += not check("legacy verdicts fall back to the strategy count",
                        s10["charges_not_shown"], 1)
    failed += not check("...and score as if restricted", s10["precision"], 1.0)

    # Two things go stale under a verdict and the digest checks only one. The
    # rubric grew from 4 entries to 11 while the answers did not change, so the
    # done-set would have marked all 16 covered tasks done and the reviewer
    # would have skipped exactly the ones needing redoing (Section 21.75).
    from common import verdict_is_current
    cur = {"answer_sha": "abc", "n_shown": 11}
    failed += not check("current text and current rubric: done",
                        verdict_is_current(cur, "abc", 11), True)
    failed += not check("same text, SHORTER rubric: not done",
                        verdict_is_current({"answer_sha": "abc", "n_shown": 4}, "abc", 11),
                        False)
    failed += not check("no n_shown (pre-v3) is not done",
                        verdict_is_current({"answer_sha": "abc"}, "abc", 11), False)
    failed += not check("different text is not done",
                        verdict_is_current(cur, "xyz", 11), False)

    # Both writers of this predicate must exist. No test over inputs and
    # outputs can see a missing caller, and this repo has shipped that bug five
    # times (21.74).
    _sd = Path(__file__).parent
    for who in ("adjudicate.py", "rubric_server.py"):
        failed += not check(f"{who} uses the shared staleness predicate",
                            "verdict_is_current" in (_sd / who).read_text(), True)

    # The note changed job at form_version 4: v3 asked for it only when no box
    # applied, v4 asks for the reasoning on every faulted verdict. So an absent
    # note means different things under the two and they must never be pooled
    # (Section 21.77). And the field must be READ by something — it was
    # collected, stored and read by nothing, which is 21.55's shape exactly.
    from adjudicate import notes_report
    mixed = [
        {"key": "p1::a", "errors_present": [1], "note": "", "form_version": 3},
        {"key": "p2::a", "errors_present": [1], "note": "why", "form_version": 4},
        {"key": "p3::a", "errors_present": [], "not_covered": True,
         "note": "no entry for a second land drop", "form_version": 4},
    ]
    out = "\n".join(notes_report(mixed, {}))
    failed += not check("faulted verdicts are counted", "2/3 verdicts carry a note" in out, True)
    failed += not check("pre-v4 verdicts are called out as not comparable",
                        "pre-v4" in out, True)
    failed += not check("not_covered notes are surfaced separately",
                        "no entry for a second land drop" in out, True)

    # A version bump must not discard human work — only a rubric that GREW does.
    # The reviewer's checkbox verdict is still a correct verdict when the note's
    # job changes underneath it.
    failed += not check("a v3 verdict survives the v4 form",
                        verdict_is_current({"answer_sha": "a", "n_shown": 11,
                                            "form_version": 3}, "a", 11), True)

    # Kappa is UNDEFINED when either rater has no variance, and it returns nan,
    # which in a table reads as a missing number rather than "this sample cannot
    # answer the question". The first v4 batch was 9/9 blundered under both
    # raters — 100% agreement, zero information (Section 21.78).
    run4 = _Path(tempfile.mkdtemp()) / "r4.jsonl"
    run4.write_text("\n".join(_json.dumps(r) for r in [
        {"id": "q1", "arms": {"a": {"errors_made": [1], "answer": "x"}}},
        {"id": "q2", "arms": {"a": {"errors_made": [1], "answer": "y"}}},
    ]), encoding="utf-8")
    allbad = [v("q1::a", [1], n_shown=4), v("q2::a", [1], n_shown=4)]
    s11 = score_run(run4, allbad, {})
    failed += not check("a one-sided sample is flagged degenerate",
                        s11["blunder_degenerate"], True)
    # BOTH raters must vary. An earlier version of this case varied only the
    # human and still fired, correctly: the run called both answers blundered.
    run5 = _Path(tempfile.mkdtemp()) / "r5.jsonl"
    run5.write_text("\n".join(_json.dumps(r) for r in [
        {"id": "q1", "arms": {"a": {"errors_made": [1], "answer": "x"}}},
        {"id": "q2", "arms": {"a": {"errors_made": [], "answer": "y"}}},
    ]), encoding="utf-8")
    mixed_calls = [v("q1::a", [1], n_shown=4), v("q2::a", [], n_shown=4)]
    s12 = score_run(run5, mixed_calls, {})
    failed += not check("a sample with variance is not flagged",
                        s12["blunder_degenerate"], False)

    # Pooling form versions can manufacture the variance that makes kappa
    # computable, so which versions were scored is reported.
    failed += not check("form versions scored are recorded",
                        score_run(run4, [v("q1::a", [1], n_shown=4, form_version=2),
                                         v("q2::a", [], n_shown=4, form_version=4)],
                                  {})["form_versions"], [2, 4])

    # Two verdicts on the SAME text by the same author, one offered more entries
    # than the other: the reviewer graded that answer once, so it contributes
    # once, and the one that saw more rubric wins (Section 21.78).
    sha = answer_sha("x")
    both = [v("q1::a", [1], n_shown=4, answer_sha=sha),
            v("q1::a", [1], n_shown=11, answer_sha=sha)]
    s13 = score_run(run4, both, {})
    failed += not check("the same answer contributes one verdict", s13["n"], 1)
    failed += not check("...and the superseded one is reported", s13["superseded"], 1)
    failed += not check("the verdict that saw more rubric wins",
                        s13["charges_not_shown"], 0)

    # But two verdicts on DIFFERENT text must both survive — they are about
    # different answers (21.62), and the stale filter decides which applies.
    s14 = score_run(run4, [v("q1::a", [1], n_shown=11, answer_sha=sha),
                           v("q1::a", [1], n_shown=11, answer_sha=answer_sha("other"))], {})
    failed += not check("different text is not collapsed as superseded",
                        s14["superseded"], 0)

    # The ingest dedup, which is where the same assumption destroyed work rather
    # than mis-scoring it: six of six regrades dropped as "already on file"
    # because the identity was (key, author) (Section 21.65).
    def dedup(on_file, incoming):
        exact = {(x.get("key"), x.get("author"), x.get("answer_sha")) for x in on_file}
        loose = {(x.get("key"), x.get("author")) for x in on_file}
        def seen(x):
            if x.get("answer_sha"):
                return (x.get("key"), x.get("author"), x["answer_sha"]) in exact
            return (x.get("key"), x.get("author")) in loose
        return [x for x in incoming if not seen(x)]

    old_v = {"key": "p::a", "author": "me", "answer_sha": "aaaaaaaaaaaa"}
    regrade = {"key": "p::a", "author": "me", "answer_sha": "bbbbbbbbbbbb"}
    failed += not check("a regrade against different text is kept",
                        len(dedup([old_v], [regrade])), 1)
    failed += not check("the identical verdict is not duplicated",
                        len(dedup([old_v], [dict(old_v)])), 0)
    # A legacy row must match on (key, author), or it re-appends on every ingest
    # forever: the copy on file was backfilled with a digest it does not carry.
    failed += not check("a legacy row with no digest is recognised",
                        len(dedup([old_v], [{"key": "p::a", "author": "me"}])), 0)
    failed += not check("a different author is always new",
                        len(dedup([old_v], [dict(regrade, author="you")])), 1)
    return failed


def test_cohens_kappa() -> int:
    """Chance-corrected agreement, one definition (21.57).

    Raw agreement flatters a skewed call, which is why
    `eval_positions.compare_judges` reports kappa first and says so in its
    docstring. `adjudicate.score_run` — the judge-versus-HUMAN comparison,
    where the human's blunder calls are skewed and chance correction matters
    MORE — reported raw agreement with no correction at all.
    """
    from common import cohens_kappa

    failed = 0
    failed += not check("perfect agreement is 1.0",
                        cohens_kappa([(True, True), (False, False)] * 5), 1.0)
    failed += not check("independent calls are ~0",
                        round(cohens_kappa([(True, False), (False, True),
                                            (True, True), (False, False)]), 2), 0.0)
    # The case raw agreement gets wrong: both sides say True 90% of the time and
    # agree 82% — which is chance, not agreement.
    skewed = [(True, True)] * 8 + [(True, False), (False, True)]
    failed += not check("a skewed call agrees 80% by raw count",
                        round(sum(1 for x, y in skewed if x == y) / len(skewed), 2), 0.8)
    failed += not check("...and kappa sees through it",
                        cohens_kappa(skewed) < 0.15, True)
    # Undefined rather than 1.0 when one side never varies: with pe == 1 the
    # formula divides by zero, and returning 1.0 there would report perfect
    # agreement for a judge that said the same thing every time.
    failed += not check("a constant call is NaN, not 1.0",
                        cohens_kappa([(True, True)] * 10) != cohens_kappa([(True, True)] * 10),
                        True)
    failed += not check("empty is NaN", cohens_kappa([]) != cohens_kappa([]), True)
    return failed


def test_mana_problems() -> int:
    """A legal_action the position cannot pay for (21.59).

    `timing_problems` asks whether a play is legal *when*; nothing asked whether
    it was payable *at all*. A board offering `CAST Doom Blade` with `{G}{G}`
    available passed every existing check — the card resolves, the action
    parses, the step allows an instant.

    Both directions are asserted. A check that only ever passes hides every real
    result, and this one currently finds zero problems in the gold set.
    """
    sys.path.insert(0, str(Path(__file__).parent / "gameplay"))
    from positions import mana_problems, parse_mana

    failed = 0
    # parse_mana refuses to guess rather than reporting a cost of zero, because
    # "not modelled" rendered as "free" fails in the direction that reads as a pass.
    failed += not check("generic plus coloured", parse_mana("{1}{B}"), ({"B": 1}, 1))
    failed += not check("pure generic", parse_mana("{2}"), ({}, 2))
    for unsupported in ("{X}{R}", "{W/U}{W}", "{B/P}", ""):
        failed += not check(f"refuses to model {unsupported!r}",
                            parse_mana(unsupported), None)

    class _Idx:
        def __init__(self, cost): self.cost = cost
        def resolve(self, name): return {"mana_cost": self.cost, "name": name}, "exact"

    pos = {"mana_available": "{G}{G}", "legal_actions": ["CAST Doom Blade TARGET Bear"]}
    failed += not check("an unaffordable colour is caught",
                        len(mana_problems(pos, _Idx("{1}{B}"))), 1)
    failed += not check("an affordable spell is not flagged",
                        mana_problems(pos, _Idx("{1}{G}")), [])
    # Right colour, too little total mana.
    failed += not check("too little total mana is caught",
                        len(mana_problems(pos, _Idx("{4}{G}"))), 1)
    # Silences that must stay silent rather than becoming false positives.
    failed += not check("no card index means skipped, not failed",
                        mana_problems(pos, None), [])
    failed += not check("a position with no stated pool is not checked",
                        mana_problems({"legal_actions": pos["legal_actions"]}, _Idx("{9}")), [])
    failed += not check("an unmodellable cost is skipped, not called free",
                        mana_problems(pos, _Idx("{X}{B}{B}{B}")), [])
    return failed


def test_tap_problems() -> int:
    """Declared taps checked against the board, never the judge (21.60)."""
    sys.path.insert(0, str(Path(__file__).parent / "gameplay"))
    from actions import parse_output
    from positions import mana_abilities, tap_problems

    failed = 0
    forest = {"text": "Forest\nBasic Land — Forest\n({T}: Add {G}.)"}
    temple = {"text": "Temple of Silence\nLand\nThis land enters tapped.\n{T}: Add {W} or {B}."}
    tomb = {"text": "Ancient Tomb\nLand\n{T}: Add {C}{C}. This land deals 2 damage to you."}
    bear = {"text": "Grizzly Bears\nCreature — Bear"}

    failed += not check("a basic's ability", mana_abilities(forest), ["{G}"])
    failed += not check("both halves of a dual", mana_abilities(temple), ["{W}", "{B}"])
    failed += not check("a multi-symbol ability", mana_abilities(tomb), ["{C}{C}"])
    # None, not [] — "cannot parse an ability" must not become "produces
    # nothing", which would flag every correct tap of that land.
    failed += not check("no mana ability reads as unknown", mana_abilities(bear), None)

    class _Idx:
        def __init__(self, card): self.card = card
        def resolve(self, name): return self.card, "exact"

    board = {"battlefield": [{"controller": "you", "card": "Forest", "tapped": False},
                             {"controller": "you", "card": "Forest", "tapped": False},
                             {"controller": "opp", "card": "Mountain", "tapped": False}]}
    def probs(ans, idx=None):
        return tap_problems(board, parse_output(ans).actions, idx or _Idx(forest))

    failed += not check("a correct tap is clean", probs("TAP Forest FOR {G}\nPASS"), [])
    failed += not check("two taps of two copies is clean",
                        probs("TAP Forest FOR {G}\nTAP Forest FOR {G}\nPASS"), [])
    failed += not check("a third tap exceeds what is untapped",
                        len(probs("TAP Forest FOR {G}\n" * 3 + "PASS")), 1)
    failed += not check("mana the land cannot add is caught",
                        len(probs("TAP Forest FOR {B}\nPASS")), 1)
    # Your opponent's lands are not yours to tap.
    failed += not check("tapping the opponent's permanent is caught",
                        len(probs("TAP Mountain FOR {R}\nPASS")), 1)
    failed += not check("tapping something absent is caught",
                        len(probs("TAP Island FOR {U}\nPASS")), 1)
    # A dual produces either half, and neither is wrong.
    board2 = {"battlefield": [{"controller": "you", "card": "Temple of Silence", "tapped": False}]}
    for colour in ("{W}", "{B}"):
        failed += not check(f"a dual may add {colour}",
                            tap_problems(board2, parse_output(f"TAP Temple of Silence FOR {colour}").actions,
                                         _Idx(temple)), [])
    # Silence, not a pass: no card index means unchecked.
    failed += not check("no card index means skipped",
                        tap_problems(board, parse_output("TAP Forest FOR {B}").actions, None), [])
    # An already-tapped land is not available.
    tappedboard = {"battlefield": [{"controller": "you", "card": "Forest", "tapped": True}]}
    failed += not check("an already-tapped land cannot be tapped",
                        len(tap_problems(tappedboard, parse_output("TAP Forest FOR {G}").actions,
                                         _Idx(forest))), 1)
    return failed


def test_phase_declaration() -> int:
    """`PHASE <step>` checked against the board, and kept out of the play count (21.61)."""
    sys.path.insert(0, str(Path(__file__).parent / "gameplay"))
    from actions import legality, parse_output
    from positions import phase_problems

    failed = 0
    board = {"phase": "opponent's declare attackers"}

    def probs(ans):
        return phase_problems(board, parse_output(ans).actions)

    failed += not check("naming the step agrees", probs("PHASE declare attackers\nPASS"), [])
    failed += not check("the fuller phrasing also agrees",
                        probs("PHASE opponent's declare attackers\nPASS"), [])
    failed += not check("a different step is caught",
                        len(probs("PHASE declare blockers\nPASS")), 1)
    failed += not check("a wildly wrong step is caught",
                        len(probs("PHASE upkeep\nPASS")), 1)
    # Silence is "not stated", never "agreed".
    failed += not check("no declaration reports nothing", probs("CAST Shock\nPASS"), [])
    # Prose that opens with the word is not a declaration — there is no suffix
    # for the whole-word test to notice, so a closed vocabulary does the work.
    failed += not check("'Phase two of my plan' is prose",
                        probs("Phase two of my plan is to attack\nPASS"), [])

    # A declaration is not a play. Verbosity must not change legality, or
    # requiring it would collapse Gate 1 and read as "verbosity makes the model
    # play worse" — the shape this repo has been bitten by twice.
    legal = ["CAST Ambush Viper", "PASS"]
    terse = legality(parse_output("CAST Ambush Viper\nPASS"), legal)
    verbose = legality(parse_output(
        "PHASE declare attackers\nTAP Forest FOR {G}\nTAP Forest FOR {G}\n"
        "CAST Ambush Viper\nPASS"), legal)
    failed += not check("terse answer is legal", terse["all_legal"], True)
    failed += not check("the same answer told verbosely is equally legal",
                        verbose["all_legal"], True)
    failed += not check("and counts the same number of plays",
                        verbose["n_actions"], terse["n_actions"])
    # A wrong play is still caught through the declarations.
    wrong = legality(parse_output("PHASE declare attackers\nCAST Doom Blade\nPASS"), legal)
    failed += not check("a wrong play is still illegal", wrong["all_legal"], False)
    # And verbosity cannot be used to escape the do-nothing detector (21.58).
    failed += not check("a verbose answer that does nothing is still only_pass",
                        parse_output("PHASE upkeep\nTAP Forest FOR {G}\nPASS").only_pass, True)
    return failed


def test_payment_and_battlefield_casts() -> int:
    """Taps that do not pay for the casts, and casting what is already in play (21.66).

    Both came from reviewer notes on real verbose answers, and neither is
    visible to `legal_actions`: an over-tapped payment names only legal taps,
    and a spell already on the battlefield is absent from the list for a reason
    the list cannot state.
    """
    sys.path.insert(0, str(Path(__file__).parent / "gameplay"))
    from actions import parse_output
    from positions import battlefield_cast_problems, payment_problems

    failed = 0
    board = {"battlefield": [{"controller": "you", "card": "Plains", "tapped": False},
                             {"controller": "you", "card": "Plains", "tapped": False},
                             {"controller": "you", "card": "Mountain", "tapped": False},
                             {"controller": "you", "card": "Serra Angel", "tapped": False}],
             "players": {"you": {"hand": ["Lightning Strike"]}}}

    class _Idx:                       # Lightning Strike is {1}{R}
        def resolve(self, name): return {"mana_cost": "{1}{R}", "name": name}, "exact"

    def pay(ans):
        return payment_problems(board, parse_output(ans).actions, _Idx())

    strike = "CAST Lightning Strike TARGET Wall of Omens"
    failed += not check("an exact payment is clean",
                        pay(f"TAP Plains FOR {{W}}\nTAP Mountain FOR {{R}}\n{strike}"), [])
    # The reviewer's case: three lands for a two-mana spell.
    failed += not check("over-tapping is caught",
                        len(pay(f"TAP Plains FOR {{W}}\nTAP Plains FOR {{W}}\n"
                                f"TAP Mountain FOR {{R}}\n{strike}")), 1)
    failed += not check("under-tapping is caught",
                        len(pay(f"TAP Mountain FOR {{R}}\n{strike}")), 1)
    # Enough mana, wrong colours: two Plains cannot pay {1}{R}.
    failed += not check("a colour the pool lacks is caught",
                        any("{R}" in x for x in
                            pay(f"TAP Plains FOR {{W}}\nTAP Plains FOR {{W}}\n{strike}")), True)
    # Silence where nothing was declared — a terse answer is not over-tapping,
    # and firing here would report a finding on every run predating the grammar.
    failed += not check("no taps declared reports nothing", pay(strike), [])
    failed += not check("taps with no cast reports nothing",
                        pay("TAP Plains FOR {W}\nPASS"), [])
    failed += not check("no card index reports nothing",
                        payment_problems(board, parse_output(
                            f"TAP Plains FOR {{W}}\n{strike}").actions, None), [])

    # Casting something already on the battlefield. Needs no card data.
    failed += not check("casting a permanent already in play is caught",
                        len(battlefield_cast_problems(board,
                            parse_output("CAST Serra Angel\nPASS").actions)), 1)
    failed += not check("casting from hand is clean",
                        battlefield_cast_problems(board, parse_output(strike).actions), [])
    # A card in BOTH is a second copy and is castable — the only unambiguous
    # case is on the battlefield and NOT in hand.
    both = {**board, "players": {"you": {"hand": ["Serra Angel"]}}}
    failed += not check("a second copy in hand is castable",
                        battlefield_cast_problems(both,
                            parse_output("CAST Serra Angel\nPASS").actions), [])
    # The opponent's permanents are not yours and say nothing about your hand.
    opp = {"battlefield": [{"controller": "opp", "card": "Serra Angel", "tapped": False}],
           "players": {"you": {"hand": []}}}
    failed += not check("the opponent's permanent is not flagged",
                        battlefield_cast_problems(opp,
                            parse_output("CAST Serra Angel\nPASS").actions), [])
    return failed


def test_protocol_findings() -> int:
    """The seven protocol classes, decided by a parser (21.70).

    These are the rubric entries a machine can confirm, which is what makes
    per-error precision computable without a human or a second judge. The two
    payment classes are asserted separately because a reviewer split them: not
    tapping enough makes the spell uncastable, while tapping too much is a legal
    play that should still be discouraged.
    """
    sys.path.insert(0, str(Path(__file__).parent / "gameplay"))
    from actions import parse_output
    from common import PROTOCOL_ERRORS, PROTOCOL_INVALIDATING
    from positions import protocol_findings

    failed = 0
    failed += not check("seven classes are defined", len(PROTOCOL_ERRORS), 7)
    failed += not check("over-tapping is the only non-invalidating one",
                        sorted(set(range(1, 8)) - set(PROTOCOL_INVALIDATING)), [6])

    board = {"phase": "precombat main",
             "legal_actions": ["CAST Shock TARGET Bear", "PASS"],
             "battlefield": [{"controller": "you", "card": "Mountain", "tapped": False},
                             {"controller": "you", "card": "Mountain", "tapped": False},
                             {"controller": "you", "card": "Serra Angel", "tapped": False}],
             "players": {"you": {"hand": ["Shock"]}}}

    # {1}{R}, so one Mountain under-pays and three over-pay. A one-mana spell
    # cannot express under-tapping here: two identical CAST lines collapse to
    # one action (REPEAT_IS_MEANINGFUL covers PLAY and TAP, not CAST), so the
    # single tap paid for it exactly and the first version of this test asserted
    # a shortfall that did not exist.
    class _Idx:
        def resolve(self, name): return {"mana_cost": "{1}{R}", "name": name}, "exact"

    def f(ans, idx=None):
        return protocol_findings(board, parse_output(ans), idx)

    failed += not check("only PASS is class 1", f("PASS")[1], True)
    failed += not check("a loop is class 2", f("CAST Shock TARGET Bear\n" * 6)[2], True)
    failed += not check("an unavailable play is class 3",
                        f("CAST Doom Blade TARGET Bear\nPASS")[3], True)
    failed += not check("the right phase is not class 4",
                        f("PHASE precombat main\nCAST Shock TARGET Bear\nPASS")[4], False)
    failed += not check("a wrong phase is class 4",
                        f("PHASE declare blockers\nCAST Shock TARGET Bear\nPASS")[4], True)
    # The split the reviewer asked for.
    under = f("TAP Mountain FOR {R}\nCAST Shock TARGET Bear\nPASS", _Idx())
    over = f("TAP Mountain FOR {R}\nTAP Mountain FOR {R}\n"
             "TAP Mountain FOR {R}\nCAST Shock TARGET Bear\nPASS", _Idx())
    exact = f("TAP Mountain FOR {R}\nTAP Mountain FOR {R}\n"
              "CAST Shock TARGET Bear\nPASS", _Idx())
    failed += not check("under-tapping is class 5, not 6", (under[5], under[6]), (True, False))
    failed += not check("over-tapping is class 6, not 5", (over[5], over[6]), (False, True))
    failed += not check("exact payment is neither", (exact[5], exact[6]), (False, False))
    failed += not check("casting what is in play is class 7",
                        f("CAST Serra Angel\nPASS")[7], True)
    # Undecidable must stay None: counting a check that could not run as "the
    # judge was wrong" is the one-sided-control mistake (21.43).
    failed += not check("no card index leaves payment undecided",
                        (f("TAP Mountain FOR {R}\nCAST Shock TARGET Bear")[5],
                         f("TAP Mountain FOR {R}\nCAST Shock TARGET Bear")[6]), (None, None))
    failed += not check("a position with no phase leaves class 4 undecided",
                        protocol_findings({"legal_actions": ["PASS"]},
                                          parse_output("PHASE upkeep\nPASS"))[4], None)
    return failed


def test_turn_scenarios() -> int:
    """Multi-step scenarios: teacher-forced, and self-checking (21.71)."""
    import copy

    sys.path.insert(0, str(Path(__file__).parent / "gameplay"))
    from turns import expand_steps, validate_scenario

    failed = 0
    base = {"id": "t1", "turn": 3, "phase": "precombat main", "active_player": "you",
            "priority": "you", "battlefield": [],
            "players": {"you": {"life": 20, "hand": ["Shock"], "library_count": 30},
                        "opp": {"life": 6, "hand_count": 1, "library_count": 30}},
            "steps": [
                {"phase": "precombat main", "legal_actions": ["CAST Shock TARGET Bear"],
                 "reference_actions": ["CAST Shock TARGET Bear"],
                 "key_points": ["Shock the Bear"]},
                {"phase": "declare attackers", "legal_actions": ["ATTACK Courser"],
                 "reference_actions": ["ATTACK Courser"], "key_points": ["Attack"],
                 "players": {"you": {"life": 20, "hand": [], "library_count": 30},
                             "opp": {"life": 6, "hand_count": 1, "library_count": 30}}}]}

    steps = expand_steps(base)
    failed += not check("one position per step", len(steps), 2)
    failed += not check("each step is separately addressable",
                        [x["id"] for x in steps], ["t1::step1", "t1::step2"])
    # A step is a complete position, so every existing check works unchanged.
    failed += not check("a step carries its own phase", steps[1]["phase"], "declare attackers")
    failed += not check("a step carries its own legal_actions",
                        steps[1]["legal_actions"], ["ATTACK Courser"])
    failed += not check("unstated fields inherit from the scenario", steps[1]["turn"], 3)
    # The sequence is carried as knowledge, not derived by a rules engine.
    failed += not check("step 1 is told nothing was done yet",
                        any("Already done" in k for k in steps[0].get("known_information") or []),
                        False)
    failed += not check("step 2 is told what step 1 did",
                        any("CAST Shock TARGET Bear" in k
                            for k in steps[1].get("known_information") or []), True)
    failed += not check("a clean scenario validates", validate_scenario(base), [])

    # One step is a position, not a scenario — the distinction is the point.
    one = dict(base, steps=base["steps"][:1])
    failed += not check("a single-step scenario is refused",
                        any("at least 2 steps" in p for p in validate_scenario(one)), True)
    # A step with no reference_actions leaves the next step blind to the line.
    noref = copy.deepcopy(base)
    del noref["steps"][0]["reference_actions"]
    failed += not check("a middle step with no reference_actions is caught",
                        any("reference_actions" in p for p in validate_scenario(noref)), True)
    # The trap this validator exists for: a cast card still in hand later reads
    # as the model ignoring a play when it is the scenario lying to it.
    stale = copy.deepcopy(base)
    del stale["steps"][1]["players"]
    failed += not check("a spent card still in hand is caught",
                        any("still in hand" in p for p in validate_scenario(stale)), True)
    return failed


def test_half_answer() -> int:
    """The `partial` control in the calibration harness (Section 21.33).

    It produces the mid-scale row the migration trip-wires are read against, and
    the first version overshot by a whole sentence — 79% of the text, and on 14
    of 99 gold answers it returned the reference verbatim, so the calibration was
    comparing the oracle against itself and calling one of them "partial".
    """
    failed = 0
    from calibrate_judge import half_answer  # noqa: E402

    four = "One. Two. Three. Four."
    got = half_answer(four)
    failed += not check("four sentences -> about half", got, "One. Two.")

    # THE invariant: never the whole answer, however few sentences there are.
    two = "Yes. Because the trigger goes on the stack first."
    failed += not check("two sentences -> not the whole thing",
                        half_answer(two) != two.strip(), True)
    failed += not check("two sentences -> the first", half_answer(two), "Yes.")

    # ...and never empty, so the judge always has something to grade.
    # (sentence starts are uppercase — `_SENT_RE` requires it, so a lowercase
    # continuation is deliberately NOT a boundary and the text stays one part)
    for label, text in (("two", two), ("four", four),
                        ("three", "Alpha. Beta. Gamma."),
                        ("uneven", "Yes. " + "X" * 200 + ". Short.")):
        failed += not check(f"{label}: non-empty", bool(half_answer(text).strip()), True)
        failed += not check(f"{label}: is a proper prefix",
                            text.strip().startswith(half_answer(text)[:20]), True)

    # A single sentence cannot be halved at a boundary; returning it whole is
    # the documented fallback, not an overshoot.
    one = "Colorless is not a color."
    failed += not check("single sentence returns itself", half_answer(one), one)
    failed += not check("empty input", half_answer(""), "")
    failed += not check("None input", half_answer(None), "")

    # The overshoot itself: the sentence that crosses halfway must be DROPPED,
    # not included. "Yes." + a long sentence: including the long one would be
    # ~100% of the text.
    lopsided = "Yes. " + "Y" * 300 + "."
    failed += not check("the crossing sentence is dropped",
                        len(half_answer(lopsided)) < len(lopsided) / 2, True)

    # A lowercase continuation is not a sentence boundary, so there is nothing
    # to cut and the whole string comes back. Asserted so the regex's contract
    # is pinned rather than rediscovered — the first draft of this test used a
    # lowercase filler and failed for that reason, not for a real defect.
    no_boundary = "Yes. " + "y" * 300 + "."
    failed += not check("no uppercase start -> no boundary -> unchanged",
                        half_answer(no_boundary), no_boundary)
    return failed


def main() -> None:
    failed = 0
    for name, fn in (("rubric_correctness", test_rubric_correctness),
                     ("verify_quoted_claims", test_verify_quoted_claims),
                     ("judge_prompt_for", test_judge_prompt_for),
                     ("score_citations", test_score_citations),
                     ("stratified_sample", test_stratified_sample),
                     ("pearson_r", test_pearson_r),
                     ("_author_of", test_author_of),
                     ("judge_batch_rubric", test_judge_batch_rubric),
                     ("carry_diagnostics", test_carry_diagnostics),
                     ("error_assertions", test_error_assertions),
                     ("half_answer", test_half_answer),
                     ("unseen_arms", test_unseen_arms),
                     ("rescore_stamps_judge", test_rescore_stamps_judge),
                     ("coverage_lines", test_coverage_lines),
                     ("judge_identity", test_judge_identity),
                     ("adjudication_scoring", test_adjudication_scoring),
                     ("cohens_kappa", test_cohens_kappa),
                     ("mana_problems", test_mana_problems),
                     ("tap_problems", test_tap_problems),
                     ("phase_declaration", test_phase_declaration),
                     ("payment_and_battlefield", test_payment_and_battlefield_casts),
                     ("protocol_findings", test_protocol_findings),
                     ("turn_scenarios", test_turn_scenarios),
                     ("behaviour_opening", test_behaviour_opening)):
        print(f"{name} ...")
        failed += fn()

    if failed:
        print(f"\n{failed} check(s) FAILED")
        raise SystemExit(1)
    print(f"\nall checks passed ({CHECKS_RUN} assertions)")


if __name__ == "__main__":
    main()
