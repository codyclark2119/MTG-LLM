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
                     ("behaviour_opening", test_behaviour_opening)):
        print(f"{name} ...")
        failed += fn()

    if failed:
        print(f"\n{failed} check(s) FAILED")
        raise SystemExit(1)
    print(f"\nall checks passed ({CHECKS_RUN} assertions)")


if __name__ == "__main__":
    main()
