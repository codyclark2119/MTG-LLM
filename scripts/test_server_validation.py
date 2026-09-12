"""Validation and AUTH at the public rubric server boundary.

This is the only server in this repo meant to be reachable from the internet
(`webui.py` is LAN-only and contains a subprocess runner; `chat_server.py`
holds a 21.6 GB model), so its boundary is the one that has to be right rather
than merely present. Two halves live here: what a submitted body may contain,
and who may reach the endpoints at all.

No model, no network — `TestClient` drives the ASGI app directly.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rubric_server  # noqa: E402
from rubric_server import (MAX_AUTHOR_LENGTH, MAX_RUBRIC_ITEMS,  # noqa: E402
                           MAX_RUBRIC_ITEM_LENGTH, validate_adjudication_input,
                           validate_position_input, validate_rubric_input)

FAILED = 0


def check(label: str, got, want) -> None:
    global FAILED
    if got != want:
        print(f"  FAIL {label}\n    got  {got!r}\n    want {want!r}")
        FAILED += 1


TOKEN = "contributor-token-abc"
EXPORT_TOKEN = "export-token-xyz"

TASKS = [{"id": "q1", "question": "does trample work?", "answer": "yes",
          "category": "combat", "key_points": [], "common_errors": []}]


def client_for(tmp: Path, token=TOKEN, export_token=EXPORT_TOKEN, **kw):
    from fastapi.testclient import TestClient
    app = rubric_server.build_app(TASKS, tmp / "submissions.jsonl", token,
                                  kind="rubric", export_token=export_token, **kw)
    return TestClient(app, follow_redirects=False)


def test_auth_boundary() -> None:
    """Bad token, query bootstrap, clean redirect, cookie flags, healthz."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        c = client_for(tmp)

        # --- health is unauthenticated, and must stay that way -------------
        # Gating it leaves every fly machine marked unhealthy and the app down.
        h = c.get("/healthz")
        check("healthz needs no token", h.status_code, 200)
        check("healthz reports liveness", h.json()["ok"], True)

        # --- a bad token is refused ----------------------------------------
        check("no token at all is refused", c.get("/rubric").status_code, 401)
        check("a wrong query token is refused",
              c.get("/rubric?t=wrong").status_code, 401)
        check("a wrong cookie is refused",
              c.get("/rubric", cookies={"mlrt": "wrong"}).status_code, 401)
        check("a wrong header is refused",
              c.get("/rubric", headers={"x-token": "wrong"}).status_code, 401)
        # A prefix of the real token must not pass — the reason the comparison
        # is compare_digest and not `!=`.
        check("a prefix of the real token is refused",
              c.get("/rubric?t=" + TOKEN[:-1]).status_code, 401)
        check("a non-ASCII token is a refusal, not a 500",
              c.get("/rubric?t=%C3%A9%C3%A9").status_code, 401)
        # The surrogate case cannot be put in a URL at all, so it is asserted
        # on the comparison itself: `str.encode` raises on a lone surrogate,
        # and an auth check that raises is a 500, not a refusal.
        check("a lone surrogate is a mismatch, not an exception",
              rubric_server._secret_matches("\udcff", TOKEN), False)
        check("a non-str supplied value is a mismatch",
              rubric_server._secret_matches(None, TOKEN), False)
        check("no expected secret is never a match",
              rubric_server._secret_matches("anything", None), False)
        check("the right token matches",
              rubric_server._secret_matches(TOKEN, TOKEN), True)

        # --- the query token is exchanged for a cookie, then removed --------
        r = c.get("/rubric?t=" + TOKEN)
        check("a query token redirects rather than serving", r.status_code, 303)
        location = r.headers["location"]
        check("the redirect drops the token from the URL", "t=" in location, False)
        check("...and lands on the same path", location.endswith("/rubric"), True)

        raw = r.headers["set-cookie"]
        check("the cookie is set on the redirect", "mlrt=" in raw, True)
        check("the cookie is HttpOnly", "HttpOnly" in raw, True)
        check("the cookie is SameSite=Lax", "SameSite=lax" in raw.replace("Lax", "lax"), True)
        check("the cookie has an explicit path", "Path=/" in raw, True)
        check("the cookie has a max-age", "Max-Age=" in raw, True)
        # Plain HTTP in the test: Secure would set a cookie the browser then
        # never sends back.
        check("no Secure flag over plain HTTP", "Secure" in raw, False)

        # --- and the cookie alone then works -------------------------------
        follow = c.get("/rubric")
        check("the cookie from the redirect authenticates", follow.status_code, 200)
        check("the contributor gets the real form",
              "<!doctype html>" in follow.text.lower(), True)

        # --- HTTPS deployments get Secure ----------------------------------
        c2 = client_for(tmp)
        r2 = c2.get("/rubric?t=" + TOKEN, headers={"x-forwarded-proto": "https"})
        check("a forwarded HTTPS request sets Secure",
              "Secure" in r2.headers["set-cookie"], True)

        # The unit, directly, at both ends.
        flags = rubric_server.cookie_flags()
        check("cookie_flags is HttpOnly", flags["httponly"], True)
        check("cookie_flags is lax", flags["samesite"], "lax")
        check("cookie_flags has an explicit path", flags["path"], "/")
        check("cookie_flags is not Secure on plain HTTP", flags["secure"], False)
        check("force_secure overrides",
              rubric_server.cookie_flags(force_secure=True)["secure"], True)


def test_secrets_are_compared_in_constant_time() -> None:
    """Asserted on the SOURCE, because timing is not observable from a test.

    A `!=` on a shared secret behaves identically to `compare_digest` in every
    functional test — it returns the same answers — and differs only in how
    long it takes to return the wrong one, which leaks the length of the
    matching prefix. So no black-box assertion can catch a regression here.
    `test_chat_server` reads the source for the same kind of property (the
    module-level mlx import), and this is that tool used for this property.
    """
    src = Path(__file__).parent.joinpath("rubric_server.py").read_text()

    check("the comparison helper exists",
          "def _secret_matches(" in src, True)
    body = src.split("def _secret_matches(", 1)[1].split("\ndef ", 1)[0]
    check("_secret_matches uses hmac.compare_digest",
          "hmac.compare_digest" in body, True)
    check("...and does not fall back to ==", " == expected" in body, False)

    middleware = src.split("async def auth(", 1)[1].split("\n    @app.", 1)[0]
    check("the middleware compares through the helper",
          "_secret_matches(supplied, token)" in middleware, True)
    for direct in ("supplied != token", "supplied == token",
                   'request.query_params.get("t") == token'):
        check(f"the middleware does not use {direct!r}", direct in middleware, False)

    export = src.split("def api_export(", 1)[1].split("\n    @app.", 1)[0]
    check("the export gate also uses the helper",
          "_secret_matches(" in export, True)


def test_export_needs_the_stronger_credential() -> None:
    """The contributor token must not unlock everyone else's submissions."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        subs = tmp / "submissions.jsonl"
        subs.write_text(json.dumps({"id": "q1", "author": "someone else"}) + "\n")
        c = client_for(tmp)
        c.get("/rubric?t=" + TOKEN)          # become an authenticated contributor

        denied = c.get("/api/export")
        check("a contributor cannot export", denied.status_code, 403)
        check("...and is told which credential is missing",
              "RUBRIC_EXPORT_TOKEN" in denied.text, True)
        check("the contributor token is explicitly not enough",
              "x-export-token" in denied.text, True)
        check("no submission content leaked in the refusal",
              "someone else" in denied.text, False)

        check("the contributor token sent AS the export token is refused",
              c.get("/api/export",
                    headers={"x-export-token": TOKEN}).status_code, 403)

        allowed = c.get("/api/export", headers={"x-export-token": EXPORT_TOKEN})
        check("the export credential works", allowed.status_code, 200)
        check("...and returns the log", "someone else" in allowed.text, True)

        # Fail closed: with a contributor token in force and NO export token
        # configured, export is refused rather than falling back to the weaker
        # secret.
        closed = client_for(tmp, export_token=None)
        closed.get("/rubric?t=" + TOKEN)
        check("unset export token fails closed",
              closed.get("/api/export").status_code, 403)

        # A loopback server with no auth at all is unaffected: `main` refuses a
        # public bind without a token, so this configuration is local-only.
        local = client_for(tmp, token=None, export_token=None)
        check("a tokenless loopback server can still export",
              local.get("/api/export").status_code, 200)


def test_public_submissions_are_bounded() -> None:
    """Every public endpoint that appends to a file must bound what it accepts."""
    clean = {"author": "Reviewer", "key_points": "one\ntwo",
             "legal_actions": "CAST Shock", "question": "q"}
    check("a normal position passes", validate_position_input(clean), [])

    check("a missing author is refused",
          any("attribution" in p for p in validate_position_input(
              {**clean, "author": ""})), True)
    check("an over-long author is refused",
          any(str(MAX_AUTHOR_LENGTH) in p for p in validate_position_input(
              {**clean, "author": "x" * (MAX_AUTHOR_LENGTH + 1)})), True)
    check("too many key points are refused",
          any(str(MAX_RUBRIC_ITEMS) in p for p in validate_position_input(
              {**clean, "key_points": "\n".join(["k"] * (MAX_RUBRIC_ITEMS + 1))})), True)
    check("an over-long key point is refused",
          any(str(MAX_RUBRIC_ITEM_LENGTH) in p for p in validate_position_input(
              {**clean, "key_points": "x" * (MAX_RUBRIC_ITEM_LENGTH + 1)})), True)
    check("an over-long board is refused",
          bool(validate_position_input(
              {**clean, "board": "x" * (MAX_RUBRIC_ITEM_LENGTH * MAX_RUBRIC_ITEMS + 1)})),
          True)
    check("a non-text board is refused",
          bool(validate_position_input({**clean, "board": {"nope": 1}})), True)
    check("a list of key points is accepted too",
          validate_position_input({**clean, "key_points": ["one", "two"]}), [])

    # And the endpoint actually applies it, rather than validating in a
    # function nothing calls (Section 21.55's shape).
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        c = client_for(tmp)
        c.get("/rubric?t=" + TOKEN)
        huge = {"author": "x" * (MAX_AUTHOR_LENGTH + 1), "key_points": "a\nb",
                "legal_actions": "CAST Shock"}
        r = c.post("/api/position", json=huge)
        check("/api/position enforces the bound", r.status_code, 400)
        check("nothing was appended for a refused position",
              (tmp / "submissions.jsonl").exists(), False)

        # The body cap, before any parsing.
        big = c.post("/api/position", content=b"x" * (rubric_server.MAX_BODY_BYTES + 1),
                     headers={"content-type": "application/json"})
        check("an oversized body is refused with 413", big.status_code, 413)


def main() -> None:
    rubric = {"id": "q1", "author": "Reviewer", "key_points": ["one", "two"],
              "common_errors": []}
    assert validate_rubric_input(rubric, {"q1"}) == []
    assert any("author is required" in p for p in validate_rubric_input(
        {**rubric, "author": ""}, {"q1"}))
    assert any("at least 2 key points" in p for p in validate_rubric_input(
        {**rubric, "key_points": ["one"]}, {"q1"}))
    assert any("must be a list" in p for p in validate_rubric_input(
        {**rubric, "common_errors": "none"}, {"q1"}))

    task = {"key": "q1::arm", "common_errors": ["error one", "error two"]}
    clean = {"key": task["key"], "author": "Reviewer", "errors_present": [],
             "unsure": False, "not_covered": False, "note": ""}
    assert validate_adjudication_input(clean, task) == []
    assert any("author is required" in p for p in validate_adjudication_input(
        {**clean, "author": ""}, task))
    assert any("must contain integers" in p for p in validate_adjudication_input(
        {**clean, "errors_present": ["1"]}, task))
    assert any("a note is required" in p for p in validate_adjudication_input(
        {**clean, "errors_present": [1]}, task))
    assert any("a note is required" in p for p in validate_adjudication_input(
        {**clean, "not_covered": True}, task))
    assert any("must be boolean" in p for p in validate_adjudication_input(
        {**clean, "unsure": "false"}, task))

    test_public_submissions_are_bounded()
    test_auth_boundary()
    test_secrets_are_compared_in_constant_time()
    test_export_needs_the_stronger_credential()
    if FAILED:
        print(f"\n{FAILED} check(s) failed")
        raise SystemExit(1)
    print("server input validation and auth boundary passed")


if __name__ == "__main__":
    main()
