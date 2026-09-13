"""Rules chatbot with rating capture — Phase 2's first external surface.

A THIRD server, deliberately separate from the two that exist:

  * `rubric_server.py` is the deployable half and is deliberately model-free
    (it imports no mlx and serves only pre-generated tasks). Adding a model
    to it would put 4.4 GB of weights and the whole retrieval stack into an
    image whose entire virtue is that it contains four files.
  * `webui.py` is LAN-only and contains a real subprocess runner, which is
    remote code execution the moment it is public.

Neither should become a public chat surface, so this is its own program with
its own dependencies and its own threat model. It reaches the model, the card
index, the ruling index and the rules index, and it writes exactly one file:
the ratings log.

Configuration is not a free choice — it is what Phase 1 measured:

  * **The 32B, not the 7B** (Section 21.163, reversing 21.145). The latency
    gate had chosen the 7B on time to first token — 14.8s of blank screen
    against 3.0s — and it chose it against the 32B's +0.50, which had been
    measured only on the 53-question XMage-mined benchmark. Run at last on the
    99-question gold set, the 32B is worth **+0.30** (p = 0.085), but **+1.13
    on turn-structure walkthroughs**, the category the 7B was worst at by a
    wide margin, with complete misses falling 44/98 to 34/98. Reversed on an
    explicit quality-over-speed decision. The costs are unchanged and real:
    ~30.8s per answer, ~14.8s before the first token, 21.6 GB resident. The
    surface answers with a running elapsed counter because a silent 30s reads
    as a hang, and generation is serialized, so a second questioner waits out
    the first.
  * **Cards + official rulings attached** (Section 21.139). Best arm at both
    model sizes, and the arm that does NOT fabricate citations: 3/53 against
    17/53 for the no-retrieval arm.
  * **The model is loaded ONCE at startup.** `infer.py` reloads per
    invocation, which is right for a CLI and fatal for a chat surface.

`--keyword-rules` defaults to **on** (Section 21.161). A resolved card's own
chunk names the rules for its keywords, so trample arrives with 702.19 and its
subrules regardless of how the question was phrased. It is here because card
text *displaces* CR text: the arm this serves fabricates 5/99 rule citations
against a rules-only arm's 1/99, and injection takes it back to 1. Correctness
is a null and nothing measured got worse — but n=4 at p=0.125, so this is a
judgement call on a bounded downside, not a settled result.

`--k-rules` defaults to **`3`**. The provisional `auto` default was tested
on the full 99-question gold set after Section 21.165 required replication.
It did not replicate: k=0 lost 14-20 with 65 ties (mean delta -0.131, p=0.392)
and fabricated citations rose from 4/99 at k=3 to 21/99 at k=0. The
precommitted rule therefore restores flat k=3 for the shipped 32B profile.
`--k-rules auto` remains available as an explicit experimental override, and
`k_rules_used` remains recorded per answer.

The whole configuration lives in ONE object, `common.CHAT_SERVING_PROFILE`, and
every default below is read from it. It used to be three literals in three files
plus prose in a fourth, which is how 21.163 moved the model and left the 7B's
retrieval settings behind it.

**Ratings store the retrieved context that was actually used.** Without it a
low rating cannot be diagnosed: a retrieval miss and a reasoning miss look
identical in a rating alone, and that distinction is the whole subject of
Phase 1. Appended one JSON object per line, fsynced, mirroring
`rubric_server.append_submission`.

Usage:
    python scripts/chat_server.py                      # loopback only
    python scripts/chat_server.py --lan                # LAN, prints a token
    python scripts/chat_server.py --k-rules auto       # experimental per-question router
"""

import argparse
import secrets
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chat_auth
# Re-exported: both servers must serve the SAME pages and apply the SAME
# validation, so neither defines its own copy (see chat_common's docstring).
from chat_common import (INDEX_HTML, LOGIN_HTML, MAX_QUESTION_CHARS,  # noqa: F401
                         append_rating, read_ratings, validate_ask,
                         validate_rating)
from common import (CHAT_SERVING_PROFILE, REPO_ROOT, build_rag_messages,
                    k_rules_arg)

RATINGS_PATH = REPO_ROOT / "data" / "chat" / "ratings.jsonl"
MAX_QUESTION_CHARS = 2000
_WRITE_LOCK = threading.Lock()


class Engine:
    """Model and indexes, loaded once and held.

    Generation is serialized behind a lock. MLX shares one Metal device and
    two concurrent generations do not make either faster — they make both
    slower and the memory ceiling closer. A queue is honest; interleaved
    generation would be a latency bug that only appears under the load a
    public surface is exactly where you meet.

    That queue costs more since 21.163: the 32B is 21.6 GB resident and takes
    ~30.8s an answer, so a second questioner waits ~60s and a third ~90s. Two
    of these models do not fit beside each other on a 64 GB machine, so the
    lock is a hard serialization, not a tuning choice.
    """

    def __init__(self, base_model_id: str, k_rules: int | str, max_tokens: int,
                 adapter_path: str | None = None, keyword_rules: bool = False,
                 profile=CHAT_SERVING_PROFILE):
        self.base_model_id = base_model_id
        self.k_rules = k_rules
        self.keyword_rules = keyword_rules
        self.max_tokens = max_tokens
        self.adapter_path = adapter_path
        # The profile this process was STARTED from. Kept separately from the
        # four values above because a flag may override any of them: the
        # profile names the shipped configuration and the values name what this
        # run actually did, and a row where they disagree must read as an
        # override rather than as the profile having meant something else.
        self.profile = profile
        self._lock = threading.Lock()

        from mlx_embeddings import load as load_embedder
        from mlx_lm import load as load_lm

        from card_lookup import CardIndex
        from rag import MODEL_ID as EMBED_MODEL_ID
        from retrieve_hybrid import KeywordRuleIndex, RulingIndex

        print(f"loading embedder {EMBED_MODEL_ID} ...", flush=True)
        self.embed_model = load_embedder(EMBED_MODEL_ID)
        print("loading card index and rulings ...", flush=True)
        self.card_index = CardIndex()
        self.ruling_index = RulingIndex()
        # Opt-in. On the gold set the card arm fabricates 5/99 against the
        # rules-only arm's 1/99 — card text displaces CR text (21.160) — and
        # injection took that 5 back to 1 (21.159). Null on correctness, no
        # measured cost, but n=4 at p=0.125, so it is not switched on for
        # anyone by default.
        self.keyword_rule_index = KeywordRuleIndex() if keyword_rules else None
        print(f"loading {base_model_id} ...", flush=True)
        self.model, self.tokenizer = load_lm(base_model_id, adapter_path=adapter_path)
        print("ready", flush=True)

    def build_context(self, question: str) -> dict:
        from retrieve_hybrid import build_context
        # scan_prose=True: real users do not know the [[bracket]] convention,
        # and without it a card question resolves NO card text and falls onto
        # the rules-only arm — the worst of the three (21.154).
        return build_context(question, self.card_index, embed_model=self.embed_model,
                             ruling_index=self.ruling_index, k_rules=self.k_rules,
                             keyword_rule_index=self.keyword_rule_index,
                             scan_prose=True)

    def answer(self, question: str) -> dict:
        from mlx_lm import generate as lm_generate

        t0 = time.perf_counter()
        # The lock covers RETRIEVAL AS WELL AS GENERATION. It used to wrap
        # `lm_generate` alone, which left `build_context` running concurrently
        # on `self.embed_model` -- one mlx_embeddings model, one Metal device,
        # shared mutable state, reached from two threads. Generation was the
        # visibly expensive half, so the lock was put where the cost was rather
        # than where the sharing was. Nothing had exercised it, because until
        # `/api/ask` was taken off the event loop the route serialized itself
        # by blocking the whole server.
        #
        # Single-flight across the pair is also what the latency story already
        # assumed: 21.163 describes a second questioner waiting out the first,
        # and interleaving the retrieval halves would not make either answer
        # arrive sooner.
        with self._lock:
            result = self.build_context(question)
            context = result["context"]
            messages = build_rag_messages(question, context, preformatted=True)
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True)
            text = lm_generate(self.model, self.tokenizer, prompt=prompt,
                               max_tokens=self.max_tokens, verbose=False)
        return {
            "answer": text,
            "context": context,
            "cards": result["card_names"],
            "rulings": result["ruling_card_names"],
            "rules_chunks": result["rules_chunk_ids"],
            # Under `--k-rules auto` the config is no longer a run-level
            # constant, so the value that actually applied to THIS answer has
            # to travel with it — see config() below.
            "k_rules_used": result["k_rules_used"],
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    def config(self) -> dict:
        """Recorded on every answer AND every rating.

        A rating is only interpretable against the configuration that produced
        the answer. This project has an identifier-survives-while-meaning-
        changes bug in four separate sections (21.13, 21.62, 21.65, 21.78);
        stamping the config onto the row is the cheap version of not repeating
        it a fifth time.

        Under `--k-rules auto` this returns the POLICY, not the treatment: it
        is one string for the process while the k actually applied varies per
        question (Section 21.156). `/api/ask` therefore stamps `k_rules_used`
        alongside it, and `triage_ratings` reads that. A rating stamped only
        `k_rules: "auto"` names which switch was on, not which branch it took —
        the same distinction that made `_cards` runs unreadable in 21.136.
        """
        return {
            "base_model": self.base_model_id,
            "adapter_path": self.adapter_path,
            "k_rules": self.k_rules,
            "max_tokens": self.max_tokens,
            "keyword_rules": bool(self.keyword_rule_index),
            # Which shipped configuration this is, and whether that name still
            # means what it meant. See common.ServingProfile.fingerprint.
            **self.profile.stamp(),
        }




def build_app(engine, ratings_path: Path, auth=None, secure_cookies: bool = False,
              trust_proxy: bool = False):
    import anyio

    from fastapi import FastAPI, Request, Response
    from fastapi.responses import HTMLResponse, JSONResponse

    auth = auth if auth is not None else chat_auth.Auth(None)

    app = FastAPI()
    # One in-flight answer, repo-wide. See `/api/ask` for why this is a
    # CapacityLimiter and not a lock. Built here rather than at module scope so
    # two apps in one test process do not share a queue; anyio binds it to
    # whichever event loop first uses it.
    ask_limiter = anyio.CapacityLimiter(1)
    # answer_id -> the full record, so a rating names something the SERVER
    # produced rather than anything a client cares to post. Bounded, because
    # an unbounded dict on a long-running public process is a memory leak
    # with a queue in front of it.
    issued: dict[str, dict] = {}
    # A deque, not a list: eviction is `popleft`, which is O(1). `list.pop(0)`
    # is O(n) and shifts 2000 entries on every answer past the cap.
    order: deque[str] = deque()
    # answer_id -> (rating, note) already durably written. ONE GENERATED ANSWER
    # IS ONE FEEDBACK OBSERVATION: without this, a double-clicked button or a
    # retried POST appended a second row naming the same answer, and every
    # count computed off `ratings.jsonl` silently weighted that answer twice.
    # Section 21.55's lesson is that a form can look wired and save nothing;
    # this is the same failure in the other direction -- saving more than the
    # user said.
    rated: dict[str, tuple[str, str]] = {}
    MAX_ISSUED = 2000
    # `issued`/`order`/`rated` are read and written from Starlette's worker
    # THREADS (the sync routes) as well as from the event loop, and since
    # /api/ask went to a thread two answers can be remembered concurrently.
    # A dict is not safe to read-modify-write across threads.
    state_lock = threading.Lock()

    def remember(answer_id: str, row: dict) -> None:
        with state_lock:
            issued[answer_id] = row
            order.append(answer_id)
            while len(order) > MAX_ISSUED:
                evicted = order.popleft()
                issued.pop(evicted, None)
                # Dropped together: a rating for an answer we no longer hold
                # cannot be persisted anyway, and leaving the id here would
                # leak the one structure the eviction exists to bound.
                rated.pop(evicted, None)

    def persist_rating(answer_id: str, rating: str, note: str) -> tuple[str, tuple | None]:
        """Write one rating at most once. Runs in a worker thread.

        Returns (verdict, prior) where verdict is "created", "duplicate",
        "conflict" or "gone".

        The dedupe check and the append are under ONE lock on purpose. Checking
        first and appending after would leave exactly the window that two
        simultaneous retries need to both decide they are the first -- and the
        symptom would be a duplicate row in a file this project treats as a
        measurement, discovered months later with no way to tell which of the
        two the user meant.
        """
        fresh = (rating, note)
        with state_lock:
            prior = rated.get(answer_id)
            if prior is not None:
                return ("duplicate" if prior == fresh else "conflict"), prior
            row = issued.get(answer_id)
            if row is None:      # evicted between validation and here
                return "gone", None
            row = dict(row)
            row["rating"] = rating
            row["note"] = note
            row["rated_at"] = datetime.now(timezone.utc).isoformat()
            append_rating(ratings_path, row)
            rated[answer_id] = fresh
            return "created", fresh

    def authorized(request: Request) -> bool:
        return auth.valid_session(request.cookies.get(chat_auth.COOKIE_NAME))

    def client_ip(request: Request) -> str:
        # Used ONLY to bucket rate-limit counters, never to authorize -- but
        # "only rate limiting" was the reasoning that let this trust the first
        # X-Forwarded-For value whenever one was present. It is wrong: the
        # throttle is the only thing standing in front of a shared password,
        # and an identity the client chooses is an identity the client can
        # rotate, so a LAN or same-host attacker got an unlimited number of
        # fresh buckets and could never be locked out. `chat_auth.client_ip`
        # decides when a header may be believed; see its docstring.
        return chat_auth.client_ip(
            request.client.host if request.client else None,
            request.headers, trust_proxy=trust_proxy)

    @app.get("/login", response_class=HTMLResponse)
    def login_page() -> str:
        return LOGIN_HTML

    @app.post("/api/login")
    async def login(request: Request):
        body = await request.json()
        ip = client_ip(request)
        if auth.locked_out(ip):
            return JSONResponse(
                {"errors": ["too many attempts; try again in a few minutes"]},
                status_code=429)
        if not auth.check_password(body.get("password", ""), ip):
            return JSONResponse({"errors": ["incorrect password"]}, status_code=401)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(chat_auth.COOKIE_NAME, auth.issue(),
                        **chat_auth.cookie_kwargs(secure_cookies))
        return resp

    @app.post("/api/logout")
    def logout(response: Response):
        response.delete_cookie(chat_auth.COOKIE_NAME, path="/")
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        # The page itself is gated, not just the API. Serving the app shell to
        # an unauthenticated visitor and relying on the endpoints to refuse is
        # how a "locked" surface ends up leaking its shape and its wording.
        if not authorized(request):
            return HTMLResponse(LOGIN_HTML)
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/health")
    def health(request: Request):
        # Unauthenticated callers get liveness only. Config and rating
        # counts are operational detail and are not owed to the internet.
        if not authorized(request):
            return {"ok": True}
        return {"ok": True, **engine.config(), "ratings": len(read_ratings(ratings_path))}

    @app.post("/api/ask")
    async def ask(request: Request):
        if not authorized(request):
            return JSONResponse({"errors": ["unauthorized"]}, status_code=401)
        body = await request.json()
        errs = validate_ask(body)
        if errs:
            return JSONResponse({"errors": errs}, status_code=400)
        question = body["question"].strip()
        # OFF THE EVENT LOOP. `engine.answer` is ~30s of retrieval and MLX
        # generation (21.163), and calling it directly from an `async def`
        # froze the whole server for that whole time: the login page, the
        # health check and the rating POST all sat behind one questioner,
        # because a coroutine that blocks blocks the only thread running every
        # other coroutine. The elapsed counter 21.163 added made this worse to
        # diagnose, not better -- the page looked alive while the server was
        # not answering anything.
        #
        # `limiter=ask_limiter` is what keeps the one-at-a-time behaviour that
        # 21.163 describes and the memory ceiling requires. It is a
        # CapacityLimiter rather than a second threading.Lock on purpose:
        # anyio acquires it BEFORE it takes a worker thread, so a queued
        # questioner parks as a cheap awaiting task. Holding a lock inside the
        # thread would instead park a whole worker per waiter, and Starlette's
        # threadpool is the same pool the sync routes below run in -- queued
        # asks would have starved exactly the lightweight endpoints this
        # change exists to keep responsive.
        result = await anyio.to_thread.run_sync(engine.answer, question,
                                                limiter=ask_limiter)
        answer_id = secrets.token_hex(8)
        remember(answer_id, {
            "answer_id": answer_id,
            "question": question,
            "answer": result["answer"],
            # The retrieved context ACTUALLY USED. Without this a low rating
            # cannot be told apart from a retrieval miss, which is the one
            # distinction Phase 1 was built around.
            "context": result["context"],
            "cards": result["cards"],
            "rulings": result["rulings"],
            "rules_chunks": result["rules_chunks"],
            "elapsed_s": result["elapsed_s"],
            "config": {**engine.config(), "k_rules_used": result.get("k_rules_used")},
        })
        return {
            "answer_id": answer_id,
            "answer": result["answer"],
            "cards": result["cards"],
            "rulings": result["rulings"],
            "elapsed_s": result["elapsed_s"],
        }

    @app.post("/api/rate")
    async def rate(request: Request):
        if not authorized(request):
            return JSONResponse({"errors": ["unauthorized"]}, status_code=401)
        body = await request.json()
        with state_lock:
            seen = set(issued)
        errs = validate_rating(body, seen)
        if errs:
            return JSONResponse({"errors": errs}, status_code=400)
        # Off the loop: `append_rating` fsyncs, and a blocking disk write on
        # the event loop is the same bug as /api/ask above, only smaller.
        verdict, prior = await anyio.to_thread.run_sync(
            persist_rating, body["answer_id"], body["rating"],
            body.get("note", ""))
        if verdict == "created":
            return {"ok": True, "recorded": True}
        if verdict == "duplicate":
            # An identical retry is the SAME observation, not a second one.
            # 200 so a double-clicked button reads as success to the user, and
            # `recorded: False` so a caller can tell nothing was appended.
            return {"ok": True, "recorded": False, "duplicate": True}
        if verdict == "conflict":
            # Two DIFFERENT ratings for one answer is a real disagreement and
            # the server must not silently pick one, nor store both as if they
            # were independent evidence. 409 says which one is on file.
            return JSONResponse(
                {"errors": [f"this answer is already rated {prior[0]!r}; "
                            f"one answer is one rating"],
                 "rating": prior[0]}, status_code=409)
        return JSONResponse(
            {"errors": ["that answer is too old to rate"]}, status_code=400)

    return app






def build_parser() -> argparse.ArgumentParser:
    """Separate from `main` so a test can read the DEFAULTS off it.

    `--help` exiting 0 proves almost nothing (CLAUDE.md; it has hidden a
    NameError here before), and the drift this file is guarding against lives
    exactly in the defaults. `test_chat_server` parses an empty argv and
    compares the result to `CHAT_SERVING_PROFILE` field by field, so a literal
    reintroduced below fails a test instead of quietly serving the wrong
    retrieval settings for the model in force.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--lan", action="store_true",
                    help="bind 0.0.0.0. Requires CHAT_PASSWORD to be set — a "
                         "non-loopback bind with no password is refused")
    ap.add_argument("--secure-cookies", action="store_true",
                    help="mark the session cookie Secure; set this whenever the "
                         "service is reached over HTTPS")
    ap.add_argument("--trust-proxy", action="store_true",
                    help="believe CF-Connecting-IP / X-Forwarded-For for rate-limit "
                         "bucketing, but ONLY on requests that arrive from loopback "
                         "(%s). Set this when a cloudflared tunnel fronts a loopback "
                         "bind and every request would otherwise share one bucket. "
                         "Off by default: without a proxy in front, the header is "
                         "client input and trusting it lets one client rotate "
                         "throttling identities forever."
                         % ", ".join(sorted(chat_auth.TRUSTED_PEERS)))
    # Every default below comes from CHAT_SERVING_PROFILE. Do not reintroduce a
    # literal here: the model and the retrieval settings that depend on it drifted
    # apart precisely because they were separately-maintained constants.
    ap.add_argument("--base-model", default=CHAT_SERVING_PROFILE.model_id,
                    help="default is the 32B (Section 21.163). 21.145's latency gate had "
                         "chosen the 7B on time-to-first-token; that was reversed on an "
                         "explicit quality-over-speed call once the 32B was finally measured "
                         "on the MAIN benchmark: +0.30 overall (p=0.085) but +1.13 on "
                         "turn-structure walkthroughs, the 7B's worst category, and complete "
                         "misses down 44/98 to 34/98. Costs ~4x answer time, ~5x time to "
                         "first token and ~5x memory. Pass the 7B id for the fast path — but "
                         "note --k-rules below was chosen for the 32B.")
    ap.add_argument("--adapter-path", default=CHAT_SERVING_PROFILE.adapter_path,
                    help="not recommended: six fine-tunes all scored BELOW the "
                         "base model on the card/ruling benchmark (21.139)")
    ap.add_argument("--k-rules", type=k_rules_arg, default=CHAT_SERVING_PROFILE.k_rules,
                    help="CR chunks per question. Default `auto`: the per-question router "
                         "(21.156) — k=0 when card text already resolved, k=3 when it did "
                         "not. The k=3 branch is +0.65 card-free under two judges (21.155, "
                         "21.157). The k=0 branch is +0.25 on the 32B this serves (15/6/32, "
                         "p=0.078; 21.144, reproduced in 21.158) and was NOT used while the "
                         "gold replication: k=0 lost 14-20 with 65 ties and raised fabricated "
                         "citations from 4/99 to 21/99. Flat k=3 is the shipped setting; "
                         "pass `auto` only to reproduce the experimental router. "
                         "Recorded on every answer, per question, as k_rules_used.")
    ap.add_argument("--keyword-rules", action=argparse.BooleanOptionalAction,
                    default=CHAT_SERVING_PROFILE.keyword_rules,
                    help="ON by default. Inject the CR text for rules a resolved card's own "
                         "keywords name. Card text displaces CR text, and the card arm this "
                         "serves fabricates 5/99 against the rules-only arm's 1/99 (21.160); "
                         "injection takes that back to 1 (21.159) with a null on correctness "
                         "and no measured cost anywhere. The fabrication result is n=4 at "
                         "p=0.125 — enabled as a deliberate decision on suggestive evidence "
                         "(21.161), not because it is established. `--no-keyword-rules` to "
                         "disable. Recorded on every answer.")
    ap.add_argument("--max-tokens", type=int, default=CHAT_SERVING_PROFILE.max_tokens)
    ap.add_argument("--ratings", type=Path, default=RATINGS_PATH)
    return ap


def main() -> None:
    args = build_parser().parse_args()

    import uvicorn

    host = "0.0.0.0" if args.lan else args.host
    auth = chat_auth.from_env()

    # Checked BEFORE the model loads. Discovering an auth misconfiguration
    # after a 40-second model load is how a server gets started with the check
    # skipped "just this once".
    problems = chat_auth.require_password_for_public(auth, host)
    if problems:
        for p in problems:
            print(f"\n{p}\n")
        raise SystemExit(2)
    if auth.enabled and auth.ephemeral_key:
        print("  note: CHAT_SECRET_KEY is unset, so sessions are signed with a "
              "random key\n        and everyone is signed out when this process "
              "restarts.")
    if not auth.enabled:
        print("  note: no CHAT_PASSWORD set — loopback only, no sign-in required.")

    engine = Engine(args.base_model, args.k_rules, args.max_tokens, args.adapter_path,
                    keyword_rules=args.keyword_rules, profile=CHAT_SERVING_PROFILE)
    app = build_app(engine, args.ratings, auth, secure_cookies=args.secure_cookies,
                    trust_proxy=args.trust_proxy)

    where = socket.gethostbyname(socket.gethostname()) if args.lan else host
    print(f"\n  http://{where}:{args.port}/\n")
    uvicorn.run(app, host=host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
