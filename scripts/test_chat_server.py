"""The chat surface's request handling and rating durability.

Phase 2's ship gate says a rating must be submitted end-to-end and confirmed
to persist before any link is handed out, because "a form can look wired and
save nothing" — which is exactly what happened to the adjudication form's
ambiguity checkbox: collected, stored, and read by nothing (Section 21.55).

The model is never loaded here. A fake engine stands in, so these run in
milliseconds with no GPU, matching every other test in this repo.
"""

import json
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chat_server
import chat_auth
from triage_ratings import effective_k
from chat_server import (append_rating, build_app, read_ratings,
                         validate_ask, validate_rating)

FAILED = 0


def check(label: str, got, want) -> None:
    global FAILED
    if got != want:
        print(f"  FAIL {label}\n    got  {got!r}\n    want {want!r}")
        FAILED += 1


class FakeEngine:
    def __init__(self):
        self.calls = []

    def answer(self, question: str) -> dict:
        self.calls.append(question)
        return {"answer": f"answer to {question}", "context": "Cards referenced:\nX",
                "cards": ["Grizzly Bears"], "rulings": ["Grizzly Bears"],
                "rules_chunks": ["c1"], "k_rules_used": 0, "elapsed_s": 0.1}

    def config(self) -> dict:
        # `auto` is the shipped default (Section 21.156), and it is the POLICY:
        # the k a given answer actually got is `k_rules_used`, per question.
        return {"base_model": "fake", "adapter_path": None, "k_rules": "auto",
                "max_tokens": 800}


def test_validation() -> None:
    check("empty question rejected", bool(validate_ask({"question": "  "})), True)
    check("missing question rejected", bool(validate_ask({})), True)
    check("normal question accepted", validate_ask({"question": "why?"}), [])
    check("over-long question rejected",
          bool(validate_ask({"question": "x" * 3000})), True)

    seen = {"abc"}
    check("rating for an unissued id rejected",
          bool(validate_rating({"answer_id": "nope", "rating": "up"}, seen)), True)
    check("bad rating value rejected",
          bool(validate_rating({"answer_id": "abc", "rating": "meh"}, seen)), True)
    check("valid rating accepted",
          validate_rating({"answer_id": "abc", "rating": "up"}, seen), [])
    check("over-long note rejected",
          bool(validate_rating({"answer_id": "abc", "rating": "up",
                                "note": "x" * 3000}, seen)), True)


def test_roundtrip() -> None:
    """Ask, then rate, then read the file back off disk.

    The read-back is the point. Section 21.55's failure was a field that every
    layer accepted and the last layer never read, so asserting on the response
    alone would reproduce exactly the bug this file exists to prevent.
    """
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory() as tmp:
        ratings = Path(tmp) / "ratings.jsonl"
        client = TestClient(build_app(FakeEngine(), ratings))

        check("health reports config", client.get("/api/health").json()["k_rules"], "auto")

        r = client.post("/api/ask", json={"question": "does trample work?"})
        check("ask returns 200", r.status_code, 200)
        body = r.json()
        check("ask returns an answer_id", bool(body.get("answer_id")), True)
        check("ask returns resolved cards", body["cards"], ["Grizzly Bears"])
        # The answer text must NOT have to be echoed back by the client to rate it.
        check("ask does not leak the raw context to the client",
              "context" in body, False)

        bad = client.post("/api/rate", json={"answer_id": "forged", "rating": "up"})
        check("forged answer_id refused", bad.status_code, 400)
        check("nothing written for a refused rating", read_ratings(ratings), [])

        ok = client.post("/api/rate", json={"answer_id": body["answer_id"],
                                            "rating": "down", "note": "wrong rule"})
        check("rating accepted", ok.status_code, 200)

        rows = read_ratings(ratings)
        check("exactly one row persisted", len(rows), 1)
        row = rows[0]
        check("rating persisted", row["rating"], "down")
        check("note persisted", row["note"], "wrong rule")
        check("question persisted", row["question"], "does trample work?")
        check("answer persisted", row["answer"], "answer to does trample work?")
        # The whole reason this field exists (plan Phase 2 step 3): without it
        # a retrieval miss and a reasoning miss are indistinguishable later.
        check("the retrieved context actually used persisted",
              row["context"], "Cards referenced:\nX")
        check("config stamped on the row", row["config"]["k_rules"], "auto")
        # The policy alone is not the treatment. A row stamped `k_rules: "auto"`
        # and nothing else says which switch was on, not which branch this
        # answer took — and the branches were chosen on two different
        # benchmarks pointing in opposite directions (Section 21.156). Asserted
        # on the PERSISTED row, because that is what triage reads months later.
        check("the k this answer actually got rides on the row",
              row["config"]["k_rules_used"], 0)
        check("triage groups on it", effective_k(row), 0)
        check("timestamp recorded", bool(row.get("rated_at")), True)


def test_login_gate() -> None:
    """The gate must cover the PAGE, not only the API.

    Serving the app shell to an unauthenticated visitor and relying on the
    endpoints to refuse leaks the surface's shape and wording, and invites the
    next endpoint to be added without a check.
    """
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory() as tmp:
        auth = chat_auth.Auth("s3cret", secret_key="k" * 64)
        client = TestClient(build_app(FakeEngine(), Path(tmp) / "r.jsonl", auth))

        check("locked out of the API", client.post("/api/ask", json={"question": "q"}).status_code, 401)
        check("locked out of the PAGE too",
              "Enter the access password" in client.get("/").text, True)
        check("health degrades to liveness only",
              client.get("/api/health").json(), {"ok": True})

        check("wrong password refused",
              client.post("/api/login", json={"password": "nope"}).status_code, 401)
        check("still locked out after a failed login",
              client.post("/api/ask", json={"question": "q"}).status_code, 401)

        ok = client.post("/api/login", json={"password": "s3cret"})
        check("correct password accepted", ok.status_code, 200)
        check("session cookie issued", chat_auth.COOKIE_NAME in ok.cookies, True)
        # TestClient keeps the cookie, so these exercise the real session path.
        check("API reachable after login",
              client.post("/api/ask", json={"question": "q"}).status_code, 200)
        check("page served after login", "MTG Rules Assistant" in client.get("/").text, True)
        check("health full after login", "k_rules" in client.get("/api/health").json(), True)

        client.post("/api/logout")
        check("logout ends the session",
              client.post("/api/ask", json={"question": "q"}).status_code, 401)


def test_session_cookie_cannot_be_forged() -> None:
    """A cookie is `expiry.HMAC`; neither half may be attacker-controlled."""
    auth = chat_auth.Auth("pw", secret_key="k" * 64)
    good = auth.issue()
    check("a freshly issued session is valid", auth.valid_session(good), True)
    expiry, _, sig = good.partition(".")

    check("garbage refused", auth.valid_session("nonsense"), False)
    check("empty refused", auth.valid_session(""), False)
    check("no signature refused", auth.valid_session(f"{expiry}."), False)
    check("tampered signature refused", auth.valid_session(f"{expiry}.{'0' * len(sig)}"), False)
    # The attack the HMAC exists to stop: keep a real signature, extend the life.
    check("expiry cannot be extended with a stolen signature",
          auth.valid_session(f"{int(expiry) + 999999}.{sig}"), False)
    check("an expired session is refused",
          auth.valid_session(f"{int(time.time()) - 10}.{auth._sign(int(time.time()) - 10)}"),
          False)
    # A different key must not accept another server's cookie.
    other = chat_auth.Auth("pw", secret_key="j" * 64)
    check("a cookie from another key is refused", other.valid_session(good), False)


def test_lockout_and_public_bind() -> None:
    auth = chat_auth.Auth("pw", secret_key="k" * 64)
    for _ in range(8):
        auth.check_password("wrong", ip="1.2.3.4")
    check("brute force locks out", auth.locked_out("1.2.3.4"), True)
    check("...and the right password is refused while locked",
          auth.check_password("pw", ip="1.2.3.4"), False)
    check("a different client is unaffected", auth.locked_out("5.6.7.8"), False)

    # Fail closed: a public bind with no password must be refused outright.
    off = chat_auth.Auth(None)
    check("no password + public bind refused",
          bool(chat_auth.require_password_for_public(off, "0.0.0.0")), True)
    check("no password + loopback allowed",
          chat_auth.require_password_for_public(off, "127.0.0.1"), [])
    check("password + public bind allowed",
          chat_auth.require_password_for_public(auth, "0.0.0.0"), [])


def test_trailing_blank_line() -> None:
    """Five jsonl readers in this repo crashed on one; this one must not."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "r.jsonl"
        p.write_text(json.dumps({"a": 1}) + "\n\n")
        check("trailing blank line tolerated", read_ratings(p), [{"a": 1}])


def test_no_model_import_at_module_level() -> None:
    """A public server must not pay for mlx to answer /api/health.

    Also the reason `chat_server` can be imported by test_webui and by this
    file with no GPU present at all.
    """
    src = Path(__file__).parent.joinpath("chat_server.py").read_text()
    head = src.split("class Engine")[0]
    for banned in ("import mlx", "from mlx"):
        check(f"module level does not {banned}", banned in head, False)


def test_prose_card_resolution() -> None:
    """Unbracketed card names must resolve, and rules words must NOT.

    Bracket-only resolution meant a live user asking "Does Lightning Bolt kill
    a Grizzly Bears?" got ZERO card text and fell onto the rules-only arm —
    the worst of the three (21.154). Prose scanning fixes that, and is only
    safe under two measured rules, both asserted here.
    """
    from card_lookup import CardIndex

    ci = CardIndex()
    names = lambda q, **kw: [c["name"] for c in ci.find_in_text(q, **kw)]

    # Default is unchanged: bracket-only, because every stored eval number was
    # produced under it.
    check("default stays bracket-only",
          names("Does Lightning Bolt kill a Grizzly Bears?"), [])
    check("brackets still work", names("[[Shock]] it"), ["Shock"])

    check("prose resolves multi-word names",
          sorted(names("Does Lightning Bolt kill a Grizzly Bears?", scan_prose=True)),
          ["Grizzly Bears", "Lightning Bolt"])

    # Rule 1: >=2 words. Eighteen ordinary rules words are also card names, so
    # a single-word match would inject a card into a pure rules question.
    for word in ("How does lifelink work?", "Can I exile it?",
                 "Does regeneration still apply?", "What about shock lands?"):
        check(f"no card invented for: {word}", names(word, scan_prose=True), [])

    # The stoplist: phrases that are card names AND ordinary rules language.
    check("'the end' is stoplisted",
          names("What happens at the end of the turn?", scan_prose=True), [])
    check("'the command zone' is stoplisted",
          names("Put it into the command zone", scan_prose=True), [])

    # Rule 2: longest match wins and its span is masked, so a shorter card name
    # cannot also match inside it and spend a slot on the wrong card.
    check("a longer name shadows the shorter one it contains",
          names("I cast Genesis Wave for five", scan_prose=True), ["Genesis Wave"])


def test_triage_classification() -> None:
    """Triage must separate the three failure halves, including the subtle one.

    `eval.score_citations` reports fabrication by EXISTENCE, which cannot see
    the case a real user question produced: an answer citing 702.7a
    "(Deathtouch)" and 702.13a "(Trample)" — both real ids, and both actually
    First Strike and Intimidate. A citation pointing at a real rule that says
    something else survives a spot check, so it needs its own category.
    """
    from common import load_rule_ids
    from triage_ratings import classify

    valid = load_rule_ids()
    ctx = "702.2c. Any nonzero amount of combat damage assigned to a creature ..."

    cases = [
        ("everything cited was in context",
         {"answer": "By 702.2c it is lethal.", "context": ctx}, "grounded"),
        ("invented a rule id",
         {"answer": "See 999.9z.", "context": ctx}, "invented_citation"),
        ("cited a real rule never shown to it",
         {"answer": "See 702.19b.", "context": ctx}, "ungrounded_citation"),
        ("nothing retrieved at all",
         {"answer": "See 702.7a.", "context": ""}, "no_context"),
        ("cited nothing",
         {"answer": "It just dies.", "context": ctx}, "no_citation"),
    ]
    for label, row, want in cases:
        check(f"triage: {label}", classify(row, valid)["observation"], want)

    # The observation is neutral; the RATING decides what it means. This was a
    # real bug: `grounded` was named `reasoning_miss`, so the second rating ever
    # collected — an answer the user LIKED — was reported as a reasoning
    # failure. One name, two meanings (CROSS_REF_RE, JUDGE_SYSTEM_PROMPT, PASS).
    liked = dict(row_grounded := {"answer": "By 702.2c it is lethal.", "context": ctx},
                 rating="up")
    disliked = dict(row_grounded, rating="down")
    check("same observation under both ratings",
          classify(liked, valid)["observation"], classify(disliked, valid)["observation"])
    check("an up-rated grounded answer is not called a failure",
          "reasoned wrong" in classify(liked, valid)["action"], False)
    check("a down-rated grounded answer IS a reasoning failure",
          "reasoned wrong" in classify(disliked, valid)["action"], True)
    check("an up-rated INVENTED citation still warns",
          "WARNING" in classify(dict(row := {"answer": "See 999.9z.", "context": ctx},
                                     rating="up"), valid)["action"], True)

    # The evidence must survive even when a coarser category wins, or the
    # diagnosis is a label with nothing behind it.
    d = classify({"answer": "See 702.7a and 702.13a.", "context": ""}, valid)
    check("triage keeps ungrounded ids under no_context",
          d["ungrounded"], ["702.13a", "702.7a"])
    # Every observation must map to an action under BOTH ratings, or triage
    # sorts an answer into a bucket it then cannot direct.
    from triage_ratings import OBSERVATIONS, RATING_ACTIONS
    for obs in OBSERVATIONS:
        for rating in ("up", "down"):
            check(f"({obs}, {rating}) has an action", (obs, rating) in RATING_ACTIONS, True)


def main() -> None:
    test_prose_card_resolution()
    test_triage_classification()
    test_validation()
    test_roundtrip()
    test_login_gate()
    test_session_cookie_cannot_be_forged()
    test_lockout_and_public_bind()
    test_trailing_blank_line()
    test_no_model_import_at_module_level()
    if FAILED:
        print(f"\n{FAILED} check(s) failed")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
