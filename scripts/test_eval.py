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
    _author_of,
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
                     ("half_answer", test_half_answer)):
        print(f"{name} ...")
        failed += fn()

    if failed:
        print(f"\n{failed} check(s) FAILED")
        raise SystemExit(1)
    print(f"\nall checks passed ({CHECKS_RUN} assertions)")


if __name__ == "__main__":
    main()
