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
import json
import os
import secrets
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BASE_MODEL_ID, REPO_ROOT, build_rag_messages

RATINGS_PATH = REPO_ROOT / "data" / "chat" / "ratings.jsonl"
MAX_QUESTION_CHARS = 2000
_WRITE_LOCK = threading.Lock()


def append_rating(path: Path, row: dict) -> None:
    """Append-only, fsynced, one JSON object per line.

    Same discipline as `rubric_server.append_submission` and for the same
    reason: this is the only durable record of what a user thought, it is
    written by a long-running server, and a crash mid-write must not be able
    to leave a half-line that makes the whole file unreadable.
    """
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def read_ratings(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:  # a trailing blank line is not a parse error (five readers learned this)
            rows.append(json.loads(line))
    return rows


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
        return build_context(question, self.card_index, embed_model=self.embed_model,
                             ruling_index=self.ruling_index, k_rules=self.k_rules)

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


def validate_ask(body: dict) -> list[str]:
    errs = []
    q = body.get("question")
    if not isinstance(q, str) or not q.strip():
        errs.append("question must be a non-empty string")
    elif len(q) > MAX_QUESTION_CHARS:
        errs.append(f"question exceeds {MAX_QUESTION_CHARS} characters")
    return errs


def validate_rating(body: dict, seen_ids: set[str]) -> list[str]:
    """A rating must name an answer this server actually produced.

    Accepting a client-supplied question/answer pair would let anyone write
    arbitrary rows into the training-adjacent record, which is the same class
    of mistake as trusting a client-supplied command line (see webui.py's
    allowlist). The server holds the answer; the client sends only an id.
    """
    errs = []
    aid = body.get("answer_id")
    if not isinstance(aid, str) or not aid:
        errs.append("answer_id must be a non-empty string")
    elif aid not in seen_ids:
        errs.append("answer_id is not one this server issued")
    rating = body.get("rating")
    if rating not in ("up", "down"):
        errs.append("rating must be 'up' or 'down'")
    note = body.get("note", "")
    if not isinstance(note, str) or len(note) > 2000:
        errs.append("note must be a string of at most 2000 characters")
    return errs


def build_app(engine, ratings_path: Path, token: str | None):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse

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
        if token is None:
            return True
        return request.headers.get("x-token") == token or \
            request.query_params.get("token") == token

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/health")
    def health() -> dict:
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


INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>MTG Rules Assistant</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  margin:0;background:#f6f7f9;color:#1a1a1a}
.wrap{max-width:760px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:19px;margin:0 0 4px}
.sub{color:#666;font-size:13px;margin-bottom:20px}
textarea{width:100%;box-sizing:border-box;padding:10px;font:inherit;
  border:1px solid #ccc;border-radius:8px;min-height:76px;background:#fff}
button{font:inherit;padding:8px 16px;border-radius:8px;border:1px solid #bbb;
  background:#fff;cursor:pointer}
button.primary{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
button:disabled{opacity:.5;cursor:default}
.answer{background:#fff;border:1px solid #e2e2e2;border-radius:10px;
  padding:16px;margin-top:18px;white-space:pre-wrap}
.meta{color:#777;font-size:12px;margin-top:10px}
.rate{margin-top:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.note{flex:1;min-width:200px;padding:6px 8px;font:inherit;border:1px solid #ccc;
  border-radius:6px}
.thanks{color:#137333;font-size:13px}
.err{color:#b00020;font-size:13px;margin-top:8px}
.disclaim{margin-top:24px;font-size:12px;color:#888;border-top:1px solid #e2e2e2;
  padding-top:12px}
</style></head><body><div class="wrap">
<h1>MTG Rules Assistant</h1>
<div class="sub">Ask a Magic: The Gathering rules question. Rate the answer &mdash;
that is what makes it better.</div>
<textarea id="q" placeholder="e.g. If I block with a creature that has first strike and it dies, does it still deal damage?"></textarea>
<div style="margin-top:10px"><button class="primary" id="askBtn">Ask</button></div>
<div id="out"></div>
<div class="disclaim">Answers are generated by a local model and can be wrong,
including when they cite a rule number. Check the Comprehensive Rules before
relying on anything here for a real game.</div>
</div>
<script>
function el(id){return document.getElementById(id);}
function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function tokenParam(){
  var m=location.search.match(/[?&]token=([^&]+)/);
  return m?m[1]:null;
}
function api(path,body){
  var t=tokenParam();
  var headers={'Content-Type':'application/json'};
  if(t)headers['x-token']=t;
  return fetch(path,{method:'POST',headers:headers,body:JSON.stringify(body)})
    .then(function(r){return r.json().then(function(j){
      if(!r.ok)throw new Error((j.errors||['request failed']).join('; '));
      return j;});});
}
function renderAnswer(res){
  var out=el('out');
  var bits=[];
  if(res.cards&&res.cards.length)bits.push('cards: '+esc(res.cards.join(', ')));
  if(res.rulings&&res.rulings.length)bits.push('rulings: '+esc(res.rulings.join(', ')));
  bits.push(res.elapsed_s+'s');
  out.innerHTML='<div class="answer">'+esc(res.answer)+'</div>'+
    '<div class="meta">'+bits.join(' &middot; ')+'</div>'+
    '<div class="rate" id="rate">'+
      '<button id="up">Helpful</button>'+
      '<button id="down">Not helpful</button>'+
      '<input class="note" id="note" placeholder="What was wrong? (optional)">'+
    '</div><div id="rerr"></div>';
  bindRate(res.answer_id);
}
function bindRate(answerId){
  ['up','down'].forEach(function(kind){
    el(kind).onclick=function(){
      api('/api/rate',{answer_id:answerId,rating:kind,note:el('note').value})
        .then(function(){
          el('rate').innerHTML='<span class="thanks">Thanks &mdash; recorded.</span>';
        })
        .catch(function(e){
          el('rerr').innerHTML='<div class="err">'+esc(e.message)+'</div>';
        });
    };
  });
}
function ask(){
  var q=el('q').value.trim();
  if(!q)return;
  el('askBtn').disabled=true;
  el('out').innerHTML='<div class="meta">thinking&hellip;</div>';
  api('/api/ask',{question:q})
    .then(function(res){renderAnswer(res);})
    .catch(function(e){el('out').innerHTML='<div class="err">'+esc(e.message)+'</div>';})
    .then(function(){el('askBtn').disabled=false;});
}
el('askBtn').onclick=ask;
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--lan", action="store_true",
                    help="bind 0.0.0.0 and require a token (printed at startup)")
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
    token = secrets.token_urlsafe(16) if args.lan else None

    engine = Engine(args.base_model, args.k_rules, args.max_tokens, args.adapter_path)
    app = build_app(engine, args.ratings, token)

    if token:
        ip = socket.gethostbyname(socket.gethostname())
        print(f"\n  LAN:  http://{ip}:{args.port}/?token={token}\n")
    else:
        print(f"\n  http://{host}:{args.port}/\n")
    uvicorn.run(app, host=host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
