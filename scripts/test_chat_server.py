"""The chat surface's request handling and rating durability.

Phase 2's ship gate says a rating must be submitted end-to-end and confirmed
to persist before any link is handed out, because "a form can look wired and
save nothing" — which is exactly what happened to the adjudication form's
ambiguity checkbox: collected, stored, and read by nothing (Section 21.55).

The model is never loaded here. A fake engine stands in, so these run in
milliseconds with no GPU, matching every other test in this repo.
"""

import asyncio
import json
import sys
import threading
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
        # The fake runs ROUTED, and so does the shipped profile since the 32B
        # (see SERVING_PROFILE). This test predates that and its real job is to
        # prove `k_rules_used` survives to the stored row, which is only
        # observable when the policy and the treatment differ — `auto` with a
        # per-answer k of 0 is exactly that case, whatever the default is.
        return {"base_model": "fake", "adapter_path": None, "k_rules": "auto",
                "max_tokens": 800, "keyword_rules": True}


class BlockingEngine(FakeEngine):
    """A fake whose inference can be held open, and which counts overlap.

    `max_in_flight` is the assertion that matters for serialization: MLX shares
    one Metal device and the embedder is one shared mutable object, so two
    answers running at once is a correctness problem, not a throughput win.
    """

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.in_flight = 0
        self.max_in_flight = 0
        self._count_lock = threading.Lock()

    def answer(self, question: str) -> dict:
        with self._count_lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.entered.set()
        try:
            # Bounded so a regression FAILS instead of hanging the suite.
            if not self.release.wait(5):
                raise AssertionError("inference was never released")
            return super().answer(question)
        finally:
            with self._count_lock:
                self.in_flight -= 1


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
        # 21.161 turned this on for everyone; a rating is only interpretable
        # against the retrieval config that produced the answer.
        check("keyword-rule injection stamped on the row",
              row["config"]["keyword_rules"], True)
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


def test_ask_does_not_block_the_event_loop() -> None:
    """The regression: /api/ask used to freeze every other request for ~30s.

    `engine.answer` is retrieval plus MLX generation — 28.4s on the served 32B
    (21.163) — and it was called directly from an `async def`, so it ran ON the
    event loop. For the whole of that time the server answered nothing at all:
    not the login page, not the health check, not a rating POST for an answer
    that had already come back. The page's elapsed counter made it look alive
    while it was not serving.

    Proof, not inspection: inference is held open, and a lightweight endpoint
    must COMPLETE before it is released. If the answer runs on the loop that
    request cannot be served, so this fails rather than merely looking right.
    No model is loaded — the fake blocks on an Event.
    """
    import httpx

    engine = BlockingEngine()
    with tempfile.TemporaryDirectory() as tmp:
        app = build_app(engine, Path(tmp) / "r.jsonl")

        async def scenario():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://test") as c:
                ask = asyncio.create_task(c.post("/api/ask", json={"question": "q"}))
                # Waited off the loop, so this await cannot itself be what
                # keeps the loop turning.
                await asyncio.to_thread(engine.entered.wait, 5)
                check("inference actually started", engine.entered.is_set(), True)
                check("the ask is still in flight", ask.done(), False)

                # THE ASSERTION. A served response while inference is held is
                # only possible if the loop is free.
                health = await asyncio.wait_for(c.get("/api/health"), timeout=5)
                check("health completes while inference is held",
                      health.status_code, 200)
                check("...and the ask has still not returned", ask.done(), False)

                engine.release.set()
                done = await asyncio.wait_for(ask, timeout=5)
                check("the ask completes once inference is released",
                      done.status_code, 200)
                check("the held answer is the one that comes back",
                      done.json()["answer"], "answer to q")

        asyncio.run(scenario())


def test_answers_stay_single_flight() -> None:
    """Off the event loop, but still ONE generation at a time.

    Offloading to a thread is only half of it: the naive fix runs every ask in
    the threadpool concurrently, which is worse than the bug it replaces. Two
    MLX generations share one Metal device and `build_context` shares one
    embedder object, so overlap is a correctness problem before it is a
    latency one, and 21.163's memory ceiling (17.6 GB resident) does not admit
    a second copy at all.

    The lock is INSIDE `Engine.answer`, which the fake does not use, so what is
    asserted here is the server's limiter and nothing else.
    """
    import httpx

    engine = BlockingEngine()
    with tempfile.TemporaryDirectory() as tmp:
        app = build_app(engine, Path(tmp) / "r.jsonl")

        async def scenario():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://test") as c:
                first = asyncio.create_task(c.post("/api/ask", json={"question": "a"}))
                await asyncio.to_thread(engine.entered.wait, 5)
                second = asyncio.create_task(c.post("/api/ask", json={"question": "b"}))
                # Give the second every chance to slip into the engine.
                await asyncio.sleep(0.25)
                check("the second ask did NOT enter inference", engine.in_flight, 1)
                check("the second ask is queued, not running", second.done(), False)

                engine.release.set()
                r1 = await asyncio.wait_for(first, timeout=5)
                r2 = await asyncio.wait_for(second, timeout=5)
                check("first ask succeeded", r1.status_code, 200)
                check("queued ask succeeded too", r2.status_code, 200)
                check("the two answers are distinct",
                      r1.json()["answer_id"] != r2.json()["answer_id"], True)
                # The whole point: they never overlapped.
                check("never more than one generation at a time",
                      engine.max_in_flight, 1)

        asyncio.run(scenario())


def test_forwarded_ip_is_not_trusted_by_default() -> None:
    """A spoofed X-Forwarded-For must not hand out a fresh throttling bucket.

    The old `client_ip` returned the first X-Forwarded-For value whenever the
    header was present, on the reasoning that it "only" buckets rate limits.
    But the throttle is the only thing in front of a shared password, so an
    identity the client picks is an identity the client rotates: a LAN client
    could send a new value per request and never be locked out, which is not a
    slower brute force — it is no brute-force limit at all.
    """
    from fastapi.testclient import TestClient

    peer = "testclient"  # the peer address Starlette's TestClient reports

    def app_for(trust_proxy: bool, auth):
        return build_app(FakeEngine(), Path(tempfile.gettempdir()) / "unused.jsonl",
                         auth, trust_proxy=trust_proxy)

    # --- direct / LAN: the header is ignored outright -----------------------
    auth = chat_auth.Auth("pw", secret_key="k" * 64)
    client = TestClient(app_for(False, auth))
    for i in range(chat_auth._MAX_ATTEMPTS):
        client.post("/api/login", json={"password": "wrong"},
                    headers={"x-forwarded-for": f"10.0.0.{i}"})
    check("a direct client is locked out despite rotating the header",
          client.post("/api/login", json={"password": "wrong"},
                      headers={"x-forwarded-for": "10.9.9.9"}).status_code, 429)
    check("...and the CORRECT password is refused while locked",
          client.post("/api/login", json={"password": "pw"},
                      headers={"x-forwarded-for": "10.9.9.9"}).status_code, 429)
    check("the lockout was bucketed on the real peer",
          auth.locked_out(peer), True)
    check("exactly one identity was tracked, not eight",
          auth.tracked_identities(), 1)

    # --- trusted-proxy mode: deliberate, and bucketed per forwarded client --
    proxied = chat_auth.Auth("pw", secret_key="k" * 64)
    pclient = TestClient(app_for(True, proxied))
    # TestClient's peer is not loopback, so even in trust_proxy mode the header
    # must be ignored: the peer is not the proxy.
    for i in range(chat_auth._MAX_ATTEMPTS):
        pclient.post("/api/login", json={"password": "wrong"},
                     headers={"x-forwarded-for": f"10.0.0.{i}"})
    check("trust_proxy does not trust a NON-trusted peer",
          proxied.locked_out(peer), True)

    # The unit the server actually calls, exercised at both ends of the policy.
    hdrs = {"x-forwarded-for": "9.9.9.9, 203.0.113.7"}
    check("direct peer wins over any header",
          chat_auth.client_ip("192.168.1.50", hdrs), "192.168.1.50")
    check("trusted loopback peer yields the RIGHTMOST forwarded entry",
          chat_auth.client_ip("127.0.0.1", hdrs, trust_proxy=True), "203.0.113.7")
    check("a trusted peer prefers CF-Connecting-IP",
          chat_auth.client_ip("127.0.0.1",
                              {"cf-connecting-ip": "203.0.113.9", **hdrs},
                              trust_proxy=True), "203.0.113.9")
    check("trust_proxy with a LAN peer still uses the peer",
          chat_auth.client_ip("192.168.1.50", hdrs, trust_proxy=True), "192.168.1.50")
    check("no headers at all degrades to the peer",
          chat_auth.client_ip("192.168.1.50", {}), "192.168.1.50")
    check("no peer at all is a placeholder, not a crash",
          chat_auth.client_ip(None, {}), "-")


def test_attempt_table_is_bounded() -> None:
    """Rotating identities must not grow the table, and must not flush a lockout.

    Two separate failures. The table was unbounded, so anything that could
    reach /api/login could grow it forever — and it expired an entry only when
    the SAME address came back, which is exactly the address a rotating
    attacker never sends again. Bounding it naively then creates a second bug:
    if eviction is indiscriminate, filling the table EVICTS a real lockout, so
    the bound becomes the way to defeat the throttle.
    """
    auth = chat_auth.Auth("pw", secret_key="k" * 64)
    for _ in range(chat_auth._MAX_ATTEMPTS):
        auth.check_password("wrong", ip="1.2.3.4")
    check("victim is locked out", auth.locked_out("1.2.3.4"), True)

    # A FIXED count, not one derived from the bound. Deriving it would make
    # this test's runtime a function of the constant under test, so raising the
    # bound would hang the suite instead of failing it — which is how a test
    # stops being a gate. The bound itself is asserted separately, below.
    rotations = 8192
    for i in range(rotations):
        auth.check_password("wrong", ip=f"10.{i // 65536}.{i // 256 % 256}.{i % 256}")

    check("the bound is small enough for `rotations` above to exceed it",
          chat_auth._MAX_TRACKED_IPS < rotations, True)
    check(f"{rotations} identities do not grow the table past its bound",
          auth.tracked_identities() <= chat_auth._MAX_TRACKED_IPS, True)
    check("the real lockout SURVIVED the flood", auth.locked_out("1.2.3.4"), True)

    # Global expiry: a window that has run out is collected even though that
    # address never comes back. Asserted by moving the entries into the past.
    stale = chat_auth.Auth("pw", secret_key="k" * 64)
    stale.check_password("wrong", ip="5.5.5.5")
    check("one entry recorded", stale.tracked_identities(), 1)
    stale._attempts["5.5.5.5"] = (1, time.time() - chat_auth._LOCKOUT_S - 1)
    # Any other address's traffic must be enough to collect it.
    stale.check_password("wrong", ip="6.6.6.6")
    check("a stale entry is expired by SOMEONE ELSE's request, not only its own",
          "5.5.5.5" in stale._attempts, False)


def test_serving_profile_is_coherent() -> None:
    """The shipped model and the retrieval settings that depend on it must agree.

    THIS IS THE DRIFT GATE. 21.163 changed one literal — the serving model, 7B
    to 32B — and every setting that had been chosen FOR THE 7B stayed behind
    it, because the model lived in `common.py` and the retrieval defaults lived
    in `chat_server.py`'s argparse and in prose in three more files. Nothing
    could fail, because nothing compared them.

    So the k for each model is asserted here as a table WITH its citation, and
    the profile must match it. Changing the served model now fails this test
    until the k is revisited, which is the review 21.163 did not get.
    """
    from chat_server import build_parser
    from common import (AUTO_K_RULES, BASE_MODEL_ID, CHAT_MODEL_ID,
                        CHAT_SERVING_PROFILE, CHAT_SERVING_PROFILE_FINGERPRINT,
                        K_RULES_NO_CARDS)

    profile = CHAT_SERVING_PROFILE

    # 1. The profile is the one the repo thinks it is. Any behaviourally
    #    relevant field moving without this pin moving is the "identifier
    #    survives while its meaning changes" bug (21.13, 21.62, 21.65, 21.78).
    check("the serving profile matches its pinned fingerprint",
          profile.fingerprint(), CHAT_SERVING_PROFILE_FINGERPRINT)
    check("the profile serves the chat model constant",
          profile.model_id, CHAT_MODEL_ID)
    check("the serving model is NOT the measurement baseline",
          profile.model_id != BASE_MODEL_ID, True)

    # 2. The measured k for each model. A serving model absent from this table
    #    has no measured retrieval policy, which is the condition that must
    #    fail rather than default to whatever was there before.
    measured_k = {
        # 21.144 +0.25 on the k=0 branch (15/6/32, p=0.078), reproduced in
        # 21.158's own table; the k=3 branch is 21.155/21.157.
        "mlx-community/Qwen2.5-32B-Instruct-4bit": AUTO_K_RULES,
        # 21.158: on the 7B the k=0 branch is -0.02 and takes fabricated
        # citations 0/53 -> 5/53, so the router's card branch is not used.
        "mlx-community/Qwen2.5-7B-Instruct-4bit": K_RULES_NO_CARDS,
    }
    check("the served model has a measured k policy",
          profile.model_id in measured_k, True)
    check("k_rules is the setting measured ON THE MODEL THAT SHIPS",
          profile.k_rules, measured_k.get(profile.model_id))

    # 3. chat_server derives, it does not restate. Read off a REAL parse, not
    #    by reading the source: `--help` exiting 0 has hidden a NameError here.
    args = build_parser().parse_args([])
    check("--base-model default comes from the profile",
          args.base_model, profile.model_id)
    check("--k-rules default comes from the profile", args.k_rules, profile.k_rules)
    check("--keyword-rules default comes from the profile",
          args.keyword_rules, profile.keyword_rules)
    check("--max-tokens default comes from the profile",
          args.max_tokens, profile.max_tokens)
    check("--adapter-path default comes from the profile",
          args.adapter_path, profile.adapter_path)
    # The 21.139 result that keeps the adapters off the serving path.
    check("no adapter is served", profile.adapter_path, None)

    # 4. No serving-path help string may still claim the 7B is what runs.
    #    A prose copy of a config value is the thing this profile replaced.
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    parser = build_parser()
    with redirect_stdout(buf):
        parser.print_help()
    helptext = buf.getvalue()
    check("the 32B is named as what this serves", "32B" in helptext, True)
    for stale in ("the 7B this serves", "the 7B this service actually serves",
                  "on the 7B this service"):
        check(f"help does not claim {stale!r}", stale in helptext, False)


def test_ratings_carry_the_serving_profile() -> None:
    """A rating is only interpretable against the configuration that made it.

    `k_rules_used` was already stamped (21.156). What was missing is WHICH
    SHIPPED CONFIGURATION that k came from: with the policy changing from flat
    k=3 to `auto` between profiles, two rows can carry the same `k_rules_used`
    and belong to different serving regimes. The profile id plus its
    fingerprint is what keeps them separable later.
    """
    from fastapi.testclient import TestClient
    from common import CHAT_SERVING_PROFILE

    with tempfile.TemporaryDirectory() as tmp:
        ratings = Path(tmp) / "ratings.jsonl"
        engine = FakeEngine()
        # The fake's config() is the bare dict a pre-profile engine returned;
        # the real Engine merges profile.stamp() into it. Assert on the real
        # merge rather than on the fake, or this tests the fake.
        stamped = dict(engine.config(), **CHAT_SERVING_PROFILE.stamp())
        engine.config = lambda: stamped

        client = TestClient(build_app(engine, ratings))
        aid = client.post("/api/ask", json={"question": "q"}).json()["answer_id"]
        client.post("/api/rate", json={"answer_id": aid, "rating": "up"})

        rows = read_ratings(ratings)
        check("one row persisted", len(rows), 1)
        cfg = rows[0]["config"]
        check("the profile id rides on the stored rating",
              cfg["serving_profile"], CHAT_SERVING_PROFILE.profile_id)
        check("...and so does its fingerprint",
              cfg["serving_profile_fingerprint"], CHAT_SERVING_PROFILE.fingerprint())
        check("the per-answer k is still there too", cfg["k_rules_used"], 0)


def test_rating_is_idempotent() -> None:
    """One generated answer is ONE feedback observation.

    The same issued `answer_id` could be rated repeatedly and each request
    appended another durable row. A double-clicked thumb, a retried POST, a
    page reload that resubmits — any of them wrote a second line naming the
    same answer, and nothing downstream could tell it apart from a second
    user agreeing: every count off `ratings.jsonl` weighted that answer twice.

    Asserted on the FILE, not the response. Section 21.55's failure was a layer
    that accepted a field nothing read; checking the JSON here would repeat it.
    """
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory() as tmp:
        ratings = Path(tmp) / "ratings.jsonl"
        client = TestClient(build_app(FakeEngine(), ratings))
        aid = client.post("/api/ask", json={"question": "q"}).json()["answer_id"]

        first = client.post("/api/rate", json={"answer_id": aid, "rating": "up",
                                               "note": "good"})
        check("the first rating is accepted", first.status_code, 200)
        check("...and says it was recorded", first.json().get("recorded"), True)
        check("one row on disk", len(read_ratings(ratings)), 1)

        # THE REGRESSION: byte-identical retry.
        again = client.post("/api/rate", json={"answer_id": aid, "rating": "up",
                                               "note": "good"})
        check("an identical retry is accepted, not an error", again.status_code, 200)
        check("...and reports that nothing was appended",
              again.json().get("recorded"), False)
        check("...and is marked a duplicate", again.json().get("duplicate"), True)
        check("STILL one row on disk", len(read_ratings(ratings)), 1)

        # Ten more for good measure — a retry loop must not be able to inflate it.
        for _ in range(10):
            client.post("/api/rate", json={"answer_id": aid, "rating": "up",
                                           "note": "good"})
        check("ten further retries append nothing", len(read_ratings(ratings)), 1)

        # A DIFFERENT verdict for the same answer is a disagreement, not a
        # second observation. It must not be silently stored as either.
        flip = client.post("/api/rate", json={"answer_id": aid, "rating": "down",
                                              "note": "good"})
        check("a conflicting rating is refused with 409", flip.status_code, 409)
        check("the refusal names the rating on file", flip.json().get("rating"), "up")
        check("nothing was appended for the conflict", len(read_ratings(ratings)), 1)
        # A changed NOTE on the same verdict is also a conflict: the note is
        # part of the observation, and silently keeping the first would discard
        # what the user actually wrote.
        renote = client.post("/api/rate", json={"answer_id": aid, "rating": "up",
                                                "note": "actually wrong"})
        check("a changed note on the same verdict conflicts", renote.status_code, 409)
        check("still exactly one row", len(read_ratings(ratings)), 1)

        # The stored row is the FIRST one, unmodified.
        row = read_ratings(ratings)[0]
        check("the persisted rating is the first one", row["rating"], "up")
        check("the persisted note is the first one", row["note"], "good")

        # A second answer is a second observation and must still be ratable.
        aid2 = client.post("/api/ask", json={"question": "q2"}).json()["answer_id"]
        check("a different answer can still be rated",
              client.post("/api/rate", json={"answer_id": aid2,
                                             "rating": "down"}).status_code, 200)
        check("now two rows", len(read_ratings(ratings)), 2)


def test_concurrent_duplicate_ratings_write_one_row() -> None:
    """Simultaneous identical retries must not both decide they are the first.

    A check-then-append with the state read outside the lock leaves exactly the
    window two racing retries need. `/api/rate` now persists inside one lock,
    so this is the test that the window is closed rather than merely narrow.
    """
    import httpx

    with tempfile.TemporaryDirectory() as tmp:
        ratings = Path(tmp) / "ratings.jsonl"
        app = build_app(FakeEngine(), ratings)

        async def scenario():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://test") as c:
                aid = (await c.post("/api/ask",
                                    json={"question": "q"})).json()["answer_id"]
                payload = {"answer_id": aid, "rating": "up", "note": "n"}
                results = await asyncio.gather(
                    *[c.post("/api/rate", json=dict(payload)) for _ in range(12)])

                codes = sorted(r.status_code for r in results)
                check("every concurrent retry got 200", codes, [200] * 12)
                recorded = [r.json().get("recorded") for r in results]
                check("exactly ONE of them claims to have recorded",
                      recorded.count(True), 1)
                check("exactly one row reached the file",
                      len(read_ratings(ratings)), 1)

        asyncio.run(scenario())


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
    test_serving_profile_is_coherent()
    test_ratings_carry_the_serving_profile()
    test_rating_is_idempotent()
    test_concurrent_duplicate_ratings_write_one_row()
    test_forwarded_ip_is_not_trusted_by_default()
    test_attempt_table_is_bounded()
    test_ask_does_not_block_the_event_loop()
    test_answers_stay_single_flight()
    test_trailing_blank_line()
    test_no_model_import_at_module_level()
    if FAILED:
        print(f"\n{FAILED} check(s) failed")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
