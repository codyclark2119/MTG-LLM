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

  * **The 7B, not the 32B** (Section 21.145). The 32B is worth +0.50
    correctness and costs 4.0x the wall-clock and 4.9x the time to first
    token — 14.8s of blank screen against 3.0s. The gate's pre-committed
    rule put 30.8s in the "ship the 7B" band.
  * **Cards + official rulings attached** (Section 21.139). Best arm at both
    model sizes, and the arm that does NOT fabricate citations: 3/53 against
    17/53 for the no-retrieval arm.
  * **The model is loaded ONCE at startup.** `infer.py` reloads per
    invocation, which is right for a CLI and fatal for a chat surface.

`--k-rules` defaults to 3, the established configuration. Section 21.144
measured k=0 (no CR text at all) at +0.25 with p = 0.078 — the strongest
retrieval lead this project has, and short of significance. Every answer
records the k it was generated under, so real user ratings accumulate into
the A/B test the benchmark could not settle on its own.

**Ratings store the retrieved context that was actually used.** Without it a
low rating cannot be diagnosed: a retrieval miss and a reasoning miss look
identical in a rating alone, and that distinction is the whole subject of
Phase 1. Appended one JSON object per line, fsynced, mirroring
`rubric_server.append_submission`.

Usage:
    python scripts/chat_server.py                      # loopback only
    python scripts/chat_server.py --lan                # LAN, prints a token
    python scripts/chat_server.py --k-rules 0          # serve the 21.144 lead
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
from common import BASE_MODEL_ID, REPO_ROOT, build_rag_messages

RATINGS_PATH = REPO_ROOT / "data" / "chat" / "ratings.jsonl"
MAX_QUESTION_CHARS = 2000
_WRITE_LOCK = threading.Lock()


class Engine:
    """Model and indexes, loaded once and held.

    Generation is serialized behind a lock. MLX shares one Metal device and
    two concurrent generations on a 4.4 GB model do not make either faster —
    they make both slower and the memory ceiling closer. A queue is honest;
    interleaved generation would be a latency bug that only appears under the
    load a public surface is exactly where you meet.
    """

    def __init__(self, base_model_id: str, k_rules: int, max_tokens: int,
                 adapter_path: str | None = None):
        self.base_model_id = base_model_id
        self.k_rules = k_rules
        self.max_tokens = max_tokens
        self.adapter_path = adapter_path
        self._lock = threading.Lock()

        from mlx_embeddings import load as load_embedder
        from mlx_lm import load as load_lm

        from card_lookup import CardIndex
        from rag import MODEL_ID as EMBED_MODEL_ID
        from retrieve_hybrid import RulingIndex

        print(f"loading embedder {EMBED_MODEL_ID} ...", flush=True)
        self.embed_model = load_embedder(EMBED_MODEL_ID)
        print("loading card index and rulings ...", flush=True)
        self.card_index = CardIndex()
        self.ruling_index = RulingIndex()
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
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    def config(self) -> dict:
        """Recorded on every answer AND every rating.

        A rating is only interpretable against the configuration that produced
        the answer. This project has an identifier-survives-while-meaning-
        changes bug in four separate sections (21.13, 21.62, 21.65, 21.78);
        stamping the config onto the row is the cheap version of not repeating
        it a fifth time.
        """
        return {
            "base_model": self.base_model_id,
            "adapter_path": self.adapter_path,
            "k_rules": self.k_rules,
            "max_tokens": self.max_tokens,
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
            "config": engine.config(),
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
    ap.add_argument("--base-model", default=BASE_MODEL_ID,
                    help="default is the 7B, which is what Section 21.145's "
                         "latency gate selected — the 32B is 4x slower to a "
                         "complete answer and 4.9x to the first token")
    ap.add_argument("--adapter-path", default=None,
                    help="not recommended: six fine-tunes all scored BELOW the "
                         "base model on the card/ruling benchmark (21.139)")
    ap.add_argument("--k-rules", type=int, default=3,
                    help="3 is the established config; 0 serves Section 21.144's "
                         "lead (+0.25, p=0.078). Recorded on every answer.")
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

    engine = Engine(args.base_model, args.k_rules, args.max_tokens, args.adapter_path)
    app = build_app(engine, args.ratings, auth, secure_cookies=args.secure_cookies)

    where = socket.gethostbyname(socket.gethostname()) if args.lan else host
    print(f"\n  http://{where}:{args.port}/\n")
    uvicorn.run(app, host=host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
