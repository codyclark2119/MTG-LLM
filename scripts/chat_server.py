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

`--k-rules` defaults to **3**, for both halves of live traffic. On card-free
questions retrieval is worth +0.65 under two judges (21.155, 21.157) and takes
fabricated rule citations from 4/20 to 0/20. On card questions the +0.25 that
once argued for dropping it (21.144) was measured on the **32B**; re-measured on
the 7B this service actually serves, k=0 buys −0.02 and takes fabricated
citations from **0/53 to 5/53** (Section 21.158). A confidently-cited wrong
answer is the worst failure mode for a rules bot, so the default is the grounded
setting.

`auto` still routes per question (Section 21.156) and remains the better setting
on a 32B. Every answer records `k_rules_used`, not just the policy, so ratings
stay diagnosable whichever is in force.

**Ratings store the retrieved context that was actually used.** Without it a
low rating cannot be diagnosed: a retrieval miss and a reasoning miss look
identical in a rating alone, and that distinction is the whole subject of
Phase 1. Appended one JSON object per line, fsynced, mirroring
`rubric_server.append_submission`.

Usage:
    python scripts/chat_server.py                      # loopback only
    python scripts/chat_server.py --lan                # LAN, prints a token
    python scripts/chat_server.py --k-rules 3          # pin it, ignoring the router
"""

import argparse
import secrets
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import chat_auth
# Re-exported: both servers must serve the SAME pages and apply the SAME
# validation, so neither defines its own copy (see chat_common's docstring).
from chat_common import (INDEX_HTML, LOGIN_HTML, MAX_QUESTION_CHARS,  # noqa: F401
                         append_rating, read_ratings, validate_ask,
                         validate_rating)
from common import (CHAT_MODEL_ID, K_RULES_NO_CARDS, REPO_ROOT,
                    build_rag_messages, k_rules_arg)

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
                 adapter_path: str | None = None, keyword_rules: bool = False):
        self.base_model_id = base_model_id
        self.k_rules = k_rules
        self.keyword_rules = keyword_rules
        self.max_tokens = max_tokens
        self.adapter_path = adapter_path
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
        result = self.build_context(question)
        context = result["context"]
        messages = build_rag_messages(question, context, preformatted=True)
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        with self._lock:
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
        }




def build_app(engine, ratings_path: Path, auth=None, secure_cookies: bool = False):
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import HTMLResponse, JSONResponse

    auth = auth if auth is not None else chat_auth.Auth(None)

    app = FastAPI()
    # answer_id -> the full record, so a rating names something the SERVER
    # produced rather than anything a client cares to post. Bounded, because
    # an unbounded dict on a long-running public process is a memory leak
    # with a queue in front of it.
    issued: dict[str, dict] = {}
    order: list[str] = []
    MAX_ISSUED = 2000

    def remember(answer_id: str, row: dict) -> None:
        issued[answer_id] = row
        order.append(answer_id)
        while len(order) > MAX_ISSUED:
            issued.pop(order.pop(0), None)

    def authorized(request: Request) -> bool:
        return auth.valid_session(request.cookies.get(chat_auth.COOKIE_NAME))

    def client_ip(request: Request) -> str:
        # X-Forwarded-For is set by fly's proxy and is spoofable when nothing
        # strips it, so it is used ONLY to bucket rate-limit counters, never
        # to authorize. Worst case a forged header gets someone their own
        # bucket, which costs a slower brute force and nothing else.
        fwd = request.headers.get("x-forwarded-for", "")
        return fwd.split(",")[0].strip() or (request.client.host if request.client else "-")

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
        result = engine.answer(question)
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
        errs = validate_rating(body, set(issued))
        if errs:
            return JSONResponse({"errors": errs}, status_code=400)
        row = dict(issued[body["answer_id"]])
        row["rating"] = body["rating"]
        row["note"] = body.get("note", "")
        row["rated_at"] = datetime.now(timezone.utc).isoformat()
        append_rating(ratings_path, row)
        return {"ok": True}

    return app






def main() -> None:
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
    ap.add_argument("--base-model", default=CHAT_MODEL_ID,
                    help="default is the 32B (Section 21.163). 21.145's latency gate had "
                         "chosen the 7B on time-to-first-token; that was reversed on an "
                         "explicit quality-over-speed call once the 32B was finally measured "
                         "on the MAIN benchmark: +0.30 overall (p=0.085) but +1.13 on "
                         "turn-structure walkthroughs, the 7B's worst category, and complete "
                         "misses down 44/98 to 34/98. Costs ~4x answer time, ~5x time to "
                         "first token and ~5x memory. Pass the 7B id for the fast path.")
    ap.add_argument("--adapter-path", default=None,
                    help="not recommended: six fine-tunes all scored BELOW the "
                         "base model on the card/ruling benchmark (21.139)")
    ap.add_argument("--k-rules", type=k_rules_arg, default=K_RULES_NO_CARDS,
                    help="CR chunks per question. Default 3 for BOTH halves of live "
                         "traffic: +0.65 on card-free questions (21.155, two judges) "
                         "and, on the 7B this serves, k=0 buys -0.02 on card questions "
                         "while taking fabricated citations from 0/53 to 5/53 (Section "
                         "21.158). `auto` routes per question and is a 32B "
                         "configuration — it is not the default here because the +0.25 "
                         "behind it was measured on a model this service does not run. "
                         "Recorded on every answer, per question.")
    ap.add_argument("--keyword-rules", action=argparse.BooleanOptionalAction, default=True,
                    help="ON by default. Inject the CR text for rules a resolved card's own "
                         "keywords name. Card text displaces CR text, and the card arm this "
                         "serves fabricates 5/99 against the rules-only arm's 1/99 (21.160); "
                         "injection takes that back to 1 (21.159) with a null on correctness "
                         "and no measured cost anywhere. The fabrication result is n=4 at "
                         "p=0.125 — enabled as a deliberate decision on suggestive evidence "
                         "(21.161), not because it is established. `--no-keyword-rules` to "
                         "disable. Recorded on every answer.")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--ratings", type=Path, default=RATINGS_PATH)
    args = ap.parse_args()

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
                    keyword_rules=args.keyword_rules)
    app = build_app(engine, args.ratings, auth, secure_cookies=args.secure_cookies)

    where = socket.gethostbyname(socket.gethostname()) if args.lan else host
    print(f"\n  http://{where}:{args.port}/\n")
    uvicorn.run(app, host=host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
