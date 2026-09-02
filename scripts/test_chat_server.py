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
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chat_server
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
                "rules_chunks": ["c1"], "elapsed_s": 0.1}

    def config(self) -> dict:
        return {"base_model": "fake", "adapter_path": None, "k_rules": 3,
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
        client = TestClient(build_app(FakeEngine(), ratings, token=None))

        check("health reports config", client.get("/api/health").json()["k_rules"], 3)

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
        check("config stamped on the row", row["config"]["k_rules"], 3)
        check("timestamp recorded", bool(row.get("rated_at")), True)


def test_token_gate() -> None:
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory() as tmp:
        client = TestClient(build_app(FakeEngine(), Path(tmp) / "r.jsonl", token="s3cret"))
        check("no token refused", client.post("/api/ask", json={"question": "q"}).status_code, 401)
        check("wrong token refused",
              client.post("/api/ask", json={"question": "q"},
                          headers={"x-token": "nope"}).status_code, 401)
        check("right token accepted",
              client.post("/api/ask", json={"question": "q"},
                          headers={"x-token": "s3cret"}).status_code, 200)


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
        ("had the rules, reasoned wrong",
         {"answer": "By 702.2c it is lethal.", "context": ctx}, "reasoning_miss"),
        ("invented a rule id",
         {"answer": "See 999.9z.", "context": ctx}, "nonexistent_citation"),
        ("cited a real rule never shown to it",
         {"answer": "See 702.19b.", "context": ctx}, "unretrieved_citation"),
        ("nothing retrieved at all",
         {"answer": "See 702.7a.", "context": ""}, "no_context"),
        ("cited nothing",
         {"answer": "It just dies.", "context": ctx}, "no_citation"),
    ]
    for label, row, want in cases:
        check(f"triage: {label}", classify(row, valid)["category"], want)

    # The evidence must survive even when a coarser category wins, or the
    # diagnosis is a label with nothing behind it.
    d = classify({"answer": "See 702.7a and 702.13a.", "context": ""}, valid)
    check("triage keeps unretrieved ids under no_context",
          d["unretrieved"], ["702.13a", "702.7a"])
    # Every category must map to an action, or triage sorts without directing.
    from triage_ratings import CATEGORY_ACTION
    for _, row, want in cases:
        check(f"category {want} has an action", want in CATEGORY_ACTION, True)


def main() -> None:
    test_triage_classification()
    test_validation()
    test_roundtrip()
    test_token_gate()
    test_trailing_blank_line()
    test_no_model_import_at_module_level()
    if FAILED:
        print(f"\n{FAILED} check(s) failed")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
