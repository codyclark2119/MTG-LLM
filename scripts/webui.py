"""Local web console for the gold set: label, author, and run the pipeline.

Three views over one server:

  /#/label    one candidate at a time, with the CR text of every cited rule
              and live card resolution beside the answer
  /#/new      a form for records that come from a judge rather than from the
              RulesGuru corpus — the categories RulesGuru cannot supply
              (definition recall has zero candidates there) have to enter
              this way
  /#/scripts  run pipeline scripts and watch their output

WHY THE RUNNER IS AN ALLOWLIST
------------------------------
This server binds to the LAN on request, so it is reachable by anything on
the network. A generic "run a command" endpoint would therefore be remote
code execution on the laptop for anyone who can reach the port. Instead the
client can only name an action id from ACTIONS below and supply values for
that action's declared arguments; the command line is assembled here and
never accepted from the client. Arguments are typed, and free-text values
are passed as separate argv entries rather than through a shell.

Off loopback, a token is also required. It is generated at startup and
printed with the URL; the phone picks it up once from the query string and
keeps it in a cookie. `--no-auth` opts out for a trusted network.

Jobs run one at a time. Several of these scripts write the same files —
two concurrent writers to gold_questions.jsonl would interleave — so
serializing is a correctness property, not just tidiness.

Usage:
    python scripts/webui.py                 # 127.0.0.1:8765, no token
    python scripts/webui.py --lan           # all interfaces + token, phone-reachable
    python scripts/webui.py --lan --no-auth
"""

import argparse
import json
import secrets
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "gameplay"))
from common import ACTION_GRAMMAR, GOLD_PATH, POSITIONS_PATH, RULES_PATH, render_position
from label_store import CANDIDATES_PATH, CATEGORIES, DIFFICULTIES, Store
from positions import (
    POSITION_CATEGORIES,
    append_position,
    lint_common_errors,
    load_positions,
    next_position_id,
    position_from_form,
    validate_position,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every runnable action. `cmd` is a script path relative to the repo root;
# `args` declare what the UI may vary. Nothing outside this table can run.
ACTIONS = [
    {
        "id": "validate_gold", "group": "Gold set", "label": "Validate gold set",
        "desc": "Check ids, categories, citations against the pinned CR, and card names. "
                "Writes eval rows when asked.",
        "cmd": "scripts/validate_gold.py", "eta": "seconds", "writes": True,
        "args": [{"name": "--to-eval", "type": "flag", "default": True, "label": "also write eval rows"}],
    },
    {
        "id": "validate_positions", "group": "Gameplay", "label": "Validate positions",
        "desc": "Check every board position: card names against Oracle, legal actions against "
                "the action grammar, and that each carries the common_errors that blunder rate "
                "is measured from.",
        "cmd": "scripts/gameplay/positions.py", "eta": "~20s", "writes": False,
        "args": [{"name": "--no-cards", "type": "flag", "default": False,
                  "label": "skip Oracle name check (faster)"}],
    },
    {
        "id": "test_actions", "group": "Gameplay", "label": "Test the action grammar",
        "desc": "Parser assertions, including the traps: verbs must be whole words, BLOCK "
                "operands must not invert, and bracket meta-syntax must not break a match.",
        "cmd": "scripts/gameplay/test_actions.py", "eta": "instant", "writes": False, "args": [],
    },
    {
        "id": "eval_positions", "group": "Gameplay", "label": "Score positions (gates)",
        "desc": "Blunder rate, legality and the three gates across four arms, scored by TWO "
                "judges — Gates 2 and 3 both reversed between judges on identical answers "
                "(Section 16.12), so a one-judge verdict is a statement about the judge.",
        "cmd": "scripts/gameplay/eval_positions.py", "eta": "~20 min", "writes": True,
        "args": [
            {"name": "--positions", "type": "text",
             "default": "data/gold/positions.jsonl", "label": "position file"},
            {"name": "--limit", "type": "int", "default": 0, "label": "max positions (0 = all)"},
            {"name": "--no-retrieval", "type": "flag", "default": False,
             "label": "skip the card/rules arms"},
            {"name": "--second-judge", "type": "text",
             "default": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
             "label": "second judge (blank = one judge only)"},
        ],
    },
    {
        "id": "rulesguru_to_gold", "group": "Gold set", "label": "Rebuild gold candidates",
        "desc": "Re-derive gold_candidates.jsonl from the frozen RulesGuru snapshot.",
        "cmd": "scripts/rulesguru_to_gold.py", "eta": "~30s", "writes": True, "args": [],
    },
    {
        "id": "fetch_rulesguru", "group": "Gold set", "label": "Fetch more RulesGuru questions",
        "desc": "Additive only — existing questions are never rewritten, because the API "
                "re-randomizes card names on every request.",
        "cmd": "scripts/fetch_rulesguru.py", "eta": "~1 min", "writes": True,
        "args": [
            {"name": "--max", "type": "int", "default": 100, "label": "new questions"},
            {"name": "--count", "type": "int", "default": 100, "label": "per request"},
        ],
    },
    {
        "id": "author_emit", "group": "Gold set", "label": "Emit authoring worksheet",
        "desc": "Offline alternative to the labeling view — thinnest categories first.",
        "cmd": "scripts/author_rubrics.py", "eta": "seconds", "writes": True,
        "args": [
            {"name": "--emit", "type": "int", "default": 20, "label": "questions"},
            {"name": "--category", "type": "choice", "choices": [""] + CATEGORIES,
             "default": "", "label": "category (blank = auto)"},
        ],
    },
    {
        "id": "ingest_pastes", "group": "Gold set", "label": "Ingest pasted Q&A",
        "desc": "Parse data/gold/pastes/*.txt into gold records. Existing ids are skipped.",
        "cmd": "scripts/ingest_qa_pastes.py", "eta": "seconds", "writes": True,
        "args": [
            {"name": "--from", "type": "fixed", "value": ["data/gold/pastes/*.txt"], "glob": True},
            {"name": "--draft-rubric", "type": "flag", "default": True, "label": "draft rubrics"},
        ],
    },
    {
        "id": "rag_query", "group": "Retrieval", "label": "Query the rules index",
        "desc": "Top-k rules chunks for a question. Read-only.",
        "cmd": "scripts/rag.py", "eta": "seconds", "writes": False,
        "args": [
            {"name": None, "type": "fixed", "value": ["query"]},
            {"name": None, "type": "text", "default": "when are state-based actions checked?",
             "label": "question", "positional": True},
            {"name": "--k", "type": "int", "default": 3, "label": "chunks"},
        ],
    },
    {
        "id": "hybrid", "group": "Retrieval", "label": "Routed card + rules retrieval",
        "desc": "Card names by dictionary lookup, rules by embedding. Use [[card name]].",
        "cmd": "scripts/retrieve_hybrid.py", "eta": "seconds", "writes": False,
        "args": [
            {"name": None, "type": "text", "default": "Does [[Chatterfang]] double token creation?",
             "label": "question", "positional": True},
            {"name": "--k", "type": "int", "default": 3, "label": "rules chunks"},
        ],
    },
    {
        "id": "rag_index", "group": "Retrieval", "label": "Rebuild the vector index",
        "desc": "Re-embed all rules chunks. Required after chunk.py changes anything.",
        "cmd": "scripts/rag.py", "eta": "~1 min", "writes": True,
        "args": [{"name": None, "type": "fixed", "value": ["index"]}],
    },
    {
        "id": "ingest", "group": "Corpora", "label": "Re-parse the Comprehensive Rules",
        "desc": "Rebuild rules.jsonl and glossary.jsonl from the pinned CR text.",
        "cmd": "scripts/ingest.py", "eta": "seconds", "writes": True, "args": [],
    },
    {
        "id": "chunk", "group": "Corpora", "label": "Rebuild retrieval chunks",
        "desc": "Rebuild chunks.jsonl. The vector index must be rebuilt afterwards.",
        "cmd": "scripts/chunk.py", "eta": "seconds", "writes": True, "args": [],
    },
    {
        "id": "chunk_cards", "group": "Corpora", "label": "Rebuild card chunks",
        "desc": "From the full Oracle pool. Refuses to shrink the corpus by half without --force.",
        "cmd": "scripts/chunk_cards.py", "eta": "~1 min", "writes": True, "args": [],
    },
    {
        "id": "ingest_rulings", "group": "Corpora", "label": "Rebuild WotC rulings",
        "desc": "Join official rulings to cards and to the rules they cite.",
        "cmd": "scripts/ingest_rulings.py", "eta": "~1 min", "writes": True, "args": [],
    },
]
ACTIONS_BY_ID = {a["id"]: a for a in ACTIONS}


class Runner:
    """Runs one allowlisted action at a time, buffering its output."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.order: list[str] = []
        self.active: str | None = None

    def build_cmd(self, action: dict, values: dict) -> list[str]:
        cmd = [sys.executable, action["cmd"]]
        for spec in action["args"]:
            name, kind = spec.get("name"), spec["type"]
            if kind == "fixed":
                if spec.get("glob"):
                    # Expand here rather than handing a pattern to a shell.
                    import glob as _g
                    hits = sorted(_g.glob(str(REPO_ROOT / spec["value"][0])))
                    if name:
                        cmd.append(name)
                    cmd += hits or spec["value"]
                else:
                    if name:
                        cmd.append(name)
                    cmd += list(spec["value"])
                continue

            raw = values.get(spec.get("key") or name or spec.get("label"))
            if kind == "flag":
                if raw if raw is not None else spec.get("default"):
                    cmd.append(name)
            elif kind == "int":
                v = int(raw) if str(raw or "").strip() else spec.get("default")
                cmd += ([name] if name else []) + [str(int(v))]
            elif kind in ("text", "choice"):
                v = raw if raw not in (None, "") else spec.get("default", "")
                if v == "":
                    continue
                if kind == "choice" and v not in spec.get("choices", []):
                    raise ValueError(f"{v!r} is not an allowed value")
                cmd += ([name] if name else []) + [str(v)]
        return cmd

    def start(self, action_id: str, values: dict) -> dict:
        action = ACTIONS_BY_ID.get(action_id)
        if action is None:
            return {"ok": False, "error": f"unknown action {action_id!r}"}
        with self.lock:
            if self.active and self.jobs[self.active]["status"] == "running":
                return {"ok": False, "error": "Another job is still running. "
                                              "These scripts share output files, so they run one at a time."}
            try:
                cmd = self.build_cmd(action, values)
            except (ValueError, TypeError) as e:
                return {"ok": False, "error": str(e)}

            jid = f"{action_id}-{int(time.time() * 1000)}"
            job = {
                "id": jid, "action": action_id, "label": action["label"],
                "cmd": " ".join(Path(c).name if i == 1 else c for i, c in enumerate(cmd)),
                "status": "running", "returncode": None, "lines": [],
                "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ended": None,
            }
            self.jobs[jid] = job
            self.order.append(jid)
            self.active = jid

        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
        )
        job["_proc"] = proc
        threading.Thread(target=self._pump, args=(job, proc), daemon=True).start()
        return {"ok": True, "job": jid}

    def _pump(self, job: dict, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:
                # Progress bars emit \r; keep only the last segment so the log
                # doesn't fill with partial redraws.
                job["lines"].append(line.rstrip("\n").split("\r")[-1])
                if len(job["lines"]) > 4000:
                    del job["lines"][:1000]
        finally:
            proc.wait()
            job["returncode"] = proc.returncode
            job["status"] = "done" if proc.returncode == 0 else (
                "cancelled" if job.get("_cancelled") else "failed")
            job["ended"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            job.pop("_proc", None)
            with self.lock:
                if self.active == job["id"]:
                    self.active = None

    def cancel(self, jid: str) -> dict:
        job = self.jobs.get(jid)
        if not job or job["status"] != "running":
            return {"ok": False, "error": "not running"}
        job["_cancelled"] = True
        proc = job.get("_proc")
        if proc:
            proc.terminate()
        return {"ok": True}

    def view(self, jid: str, since: int = 0) -> dict | None:
        job = self.jobs.get(jid)
        if job is None:
            return None
        return {k: v for k, v in job.items() if not k.startswith("_")} | {
            "lines": job["lines"][since:], "total_lines": len(job["lines"]),
        }

    def history(self, n: int = 12) -> list[dict]:
        return [{"id": j, "label": self.jobs[j]["label"], "status": self.jobs[j]["status"],
                 "started": self.jobs[j]["started"], "returncode": self.jobs[j]["returncode"]}
                for j in reversed(self.order[-n:])]


def lan_ip() -> str:
    """Best-guess LAN address. The UDP socket is never actually sent on."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def build_app(store: Store, runner: Runner, author: str, token: str | None):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    app = FastAPI(title="magic-llm console")

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if token:
            supplied = (request.query_params.get("t")
                        or request.cookies.get("mlt")
                        or request.headers.get("x-token"))
            if supplied != token:
                return PlainTextResponse("Token required. Use the URL printed by the server.", 401)
            response = await call_next(request)
            if request.query_params.get("t") == token:
                response.set_cookie("mlt", token, max_age=86400 * 7, samesite="lax")
            return response
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    # -- labeling ---------------------------------------------------------
    @app.get("/api/queue")
    def queue():
        items, fixes = store.queue()
        return {"items": items, "fix_count": fixes, "stats": store.stats()}

    @app.get("/api/item/{rid}")
    def item(rid: str):
        d = store.detail(rid)
        return d if d else JSONResponse({"error": "not found"}, status_code=404)

    @app.post("/api/item/{rid}")
    async def save(rid: str, payload: dict):
        return store.save(rid, payload, author)

    @app.post("/api/reject/{rid}")
    async def reject(rid: str, payload: dict):
        return store.reject(rid, payload.get("reason", ""))

    @app.get("/api/check")
    def check(rules: str = "", cards: str = ""):
        rule_ids = [s for s in rules.replace(",", " ").split() if s]
        card_names = [s.strip() for s in cards.split("|") if s.strip()]
        return {"rules": store.check_rules(rule_ids), "cards": store.check_cards(card_names)}

    # -- new record -------------------------------------------------------
    @app.post("/api/new")
    async def create(payload: dict):
        return store.create(payload, author)

    @app.get("/api/meta")
    def meta():
        return {"categories": CATEGORIES, "difficulties": DIFFICULTIES,
                "actions": [{k: v for k, v in a.items()} for a in ACTIONS],
                "position_categories": POSITION_CATEGORIES,
                "action_grammar": ACTION_GRAMMAR}

    # -- positions --------------------------------------------------------
    # A board position is a gold record whose "question" is a board, so it
    # reuses the rubric fields and the V3 judge unchanged. It lives in its own
    # file: gold_questions.jsonl is mid-measurement (the n~100 two-judge
    # comparison) and mixing position categories into it would change the
    # composition that measurement depends on.
    @app.post("/api/position/preview")
    async def position_preview(payload: dict):
        pos = position_from_form(payload)
        pos["id"] = pos["id"] or "pos-preview"
        problems = validate_position(pos, store.card_index)
        # Warnings, not problems: a flagged rubric still saves. Section 16.12
        # measured this defect widening the two judges' blunder-rate gap to 28
        # points, but the lint is deliberately conservative and catches only
        # verbatim restatement, so it advises rather than blocks.
        return {"rendered": render_position(pos), "problems": problems,
                "warnings": lint_common_errors(pos),
                "n_battlefield": len(pos["battlefield"])}

    @app.post("/api/position")
    async def position_create(payload: dict):
        pos = position_from_form(payload)
        existing = load_positions(POSITIONS_PATH)
        pos["id"] = pos["id"] or next_position_id(existing, pos["category"])
        if any(p.get("id") == pos["id"] for p in existing):
            return {"ok": False, "problems": [f"id {pos['id']} already exists"]}
        pos["rubric_source"] = f"hand-authored ({author})" if author else "hand-authored"
        problems = validate_position(pos, store.card_index)
        if problems:
            return {"ok": False, "problems": problems}
        append_position(pos, POSITIONS_PATH)
        return {"ok": True, "id": pos["id"], "total": len(existing) + 1,
                "warnings": lint_common_errors(pos)}

    @app.get("/api/positions")
    def positions_list():
        rows = load_positions(POSITIONS_PATH)
        by_cat = {c: sum(1 for r in rows if r.get("category") == c)
                  for c in POSITION_CATEGORIES}
        return {"total": len(rows), "by_category": by_cat,
                "recent": [{"id": r["id"], "category": r.get("category"),
                            "difficulty": r.get("difficulty")} for r in rows[-8:]][::-1]}

    # -- scripts ----------------------------------------------------------
    @app.post("/api/run")
    async def run(payload: dict):
        return runner.start(payload.get("action", ""), payload.get("values") or {})

    @app.get("/api/job/{jid}")
    def job(jid: str, since: int = 0):
        v = runner.view(jid, since)
        return v if v else JSONResponse({"error": "not found"}, status_code=404)

    @app.post("/api/job/{jid}/cancel")
    def cancel(jid: str):
        return runner.cancel(jid)

    @app.get("/api/jobs")
    def jobs():
        return {"jobs": runner.history(), "active": runner.active}

    return app


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>magic-llm console</title>
<style>
:root{--paper:#F5F6F8;--panel:#fff;--ink:#131820;--soft:#59636F;--faint:#8A939E;
 --accent:#1C5A8C;--accent-bg:#EAF1F7;--ok:#1E7A4B;--bad:#94382C;--rule:#D9DEE4;--rule-soft:#E8EBEF;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#0F1319;--panel:#161B22;
 --ink:#E3E7EC;--soft:#9BA5B1;--faint:#6B7683;--accent:#79B0DC;--accent-bg:#16242F;--ok:#68C08D;
 --bad:#D59286;--rule:#2A313A;--rule-soft:#1F252D}}
:root[data-theme="dark"]{--paper:#0F1319;--panel:#161B22;--ink:#E3E7EC;--soft:#9BA5B1;--faint:#6B7683;
 --accent:#79B0DC;--accent-bg:#16242F;--ok:#68C08D;--bad:#D59286;--rule:#2A313A;--rule-soft:#1F252D}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.55}
button,input,select,textarea{font:inherit;color:inherit}
header{position:sticky;top:0;z-index:10;background:var(--panel);border-bottom:1px solid var(--rule);
 padding:.55rem .9rem;display:flex;gap:.8rem;align-items:center;flex-wrap:wrap}
.brand{font-weight:650;letter-spacing:-.01em;white-space:nowrap}
nav{display:flex;gap:.25rem}
nav a{text-decoration:none;color:var(--soft);padding:.28rem .6rem;border:1px solid transparent;border-radius:2px;font-size:.86rem}
nav a.on{color:var(--accent);border-color:var(--rule);background:var(--accent-bg)}
.chips{display:flex;gap:.3rem;flex-wrap:wrap;margin-left:auto}
.chip{font-size:.68rem;padding:.14rem .42rem;border:1px solid var(--rule);border-radius:2px;
 color:var(--soft);font-variant-numeric:tabular-nums;white-space:nowrap}
.chip b{color:var(--ink)}.chip.low{border-color:var(--bad);color:var(--bad)}
.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:1px;background:var(--rule);min-height:calc(100vh - 50px)}
@media(max-width:60rem){.split{grid-template-columns:1fr}}
.col{background:var(--paper);padding:1rem 1.15rem 3rem;min-width:0}
.page{max-width:52rem;margin:0 auto;padding:1.2rem 1.15rem 4rem}
h2{font-size:.7rem;text-transform:uppercase;letter-spacing:.11em;color:var(--accent);margin:0 0 .6rem;font-weight:640}
h3{font-size:1rem;margin:0 0 .3rem}
.qmeta{font-family:var(--mono);font-size:.72rem;color:var(--faint);margin-bottom:.5rem;display:flex;gap:.6rem;flex-wrap:wrap;align-items:center}
.qmeta a{color:var(--accent)}
.question{font-family:var(--serif);font-size:1.1rem;line-height:1.5;margin:0 0 1rem}
.answer{font-family:var(--serif);border-left:2px solid var(--accent);padding-left:.9rem;color:var(--soft);margin:0 0 1.1rem}
.ref{border:1px solid var(--rule);background:var(--panel);margin-bottom:.4rem}
.ref summary{cursor:pointer;padding:.38rem .6rem;font-family:var(--mono);font-size:.77rem;display:flex;gap:.5rem;align-items:center;list-style:none}
.ref summary::-webkit-details-marker{display:none}
.ref .body{padding:0 .6rem .5rem;font-size:.85rem;color:var(--soft);font-family:var(--serif)}
.dot{width:.5rem;height:.5rem;border-radius:50%;flex:none;background:var(--ok)}
.dot.bad{background:var(--bad)}.dot.warn{background:var(--accent)}
.pill{font-size:.66rem;padding:.08rem .32rem;border:1px solid var(--rule);color:var(--faint);border-radius:2px}
.draft{background:var(--panel);border:1px dashed var(--rule);padding:.55rem .7rem;margin-bottom:.85rem}
.draft ul{margin:.3rem 0 0;padding-left:1.05rem;color:var(--soft);font-size:.85rem}
label{display:block;font-size:.68rem;text-transform:uppercase;letter-spacing:.09em;color:var(--soft);margin:.85rem 0 .28rem;font-weight:620}
textarea,input[type=text],select{width:100%;background:var(--panel);border:1px solid var(--rule);padding:.45rem .55rem;border-radius:2px;font-family:var(--serif);font-size:.95rem}
textarea{resize:vertical;line-height:1.5}
input[type=text].mono{font-family:var(--mono);font-size:.85rem}
select{font-family:var(--sans);font-size:.85rem}
.hint{font-size:.72rem;color:var(--faint);margin-top:.22rem}
.row{display:flex;gap:.7rem;flex-wrap:wrap}.row>*{flex:1;min-width:11rem}
.actions{position:sticky;bottom:0;background:var(--paper);border-top:1px solid var(--rule);padding:.65rem 0 .5rem;margin-top:1.1rem;display:flex;gap:.45rem;align-items:center;flex-wrap:wrap}
button{cursor:pointer;border:1px solid var(--rule);background:var(--panel);padding:.42rem .8rem;border-radius:2px;font-size:.85rem}
button:hover{border-color:var(--accent)}button:disabled{opacity:.5;cursor:not-allowed}
button.primary{background:var(--accent);border-color:var(--accent);color:var(--paper);font-weight:620}
button.danger:hover{border-color:var(--bad);color:var(--bad)}
button:focus-visible,summary:focus-visible,a:focus-visible,input:focus-visible,textarea:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.msg{font-size:.8rem;margin-left:auto}.msg.err{color:var(--bad)}.msg.ok{color:var(--ok)}
.kbd{font-family:var(--mono);font-size:.66rem;color:var(--faint);border:1px solid var(--rule);padding:0 .22rem;border-radius:2px}
.card{border:1px solid var(--rule);background:var(--panel);padding:.75rem .85rem;margin-bottom:.5rem}
.card .top{display:flex;gap:.6rem;align-items:baseline;flex-wrap:wrap}
.card p{margin:.3rem 0 0;font-size:.85rem;color:var(--soft)}
.grp{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:var(--faint);margin:1.3rem 0 .5rem;font-weight:620}
.log{background:var(--panel);border:1px solid var(--rule);font-family:var(--mono);font-size:.76rem;
 line-height:1.5;padding:.7rem .8rem;overflow:auto;max-height:26rem;white-space:pre-wrap;word-break:break-word}
.status{font-family:var(--mono);font-size:.72rem;padding:.06rem .35rem;border-radius:2px;border:1px solid var(--rule)}
.status.running{color:var(--accent);border-color:var(--accent)}
.status.done{color:var(--ok);border-color:var(--ok)}
.status.failed,.status.cancelled{color:var(--bad);border-color:var(--bad)}
ul.problems{margin:.4rem 0 0;padding-left:1.1rem;color:var(--bad);font-size:.85rem}
.empty{padding:3rem 1rem;text-align:center;color:var(--soft)}
</style></head>
<body>
<header>
  <span class="brand">magic-llm</span>
  <nav>
    <a href="#/label" id="n-label">Label</a>
    <a href="#/new" id="n-new">New record</a>
    <a href="#/position" id="n-position">Position</a>
    <a href="#/scripts" id="n-scripts">Scripts</a>
  </nav>
  <div class="chips" id="chips"></div>
</header>
<div id="view"></div>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>{const d=document.createElement('div');d.textContent=s??'';return d.innerHTML};
const lines=el=>($(el).value||'').split('\n').map(s=>s.trim()).filter(Boolean);
let META=null,STATS=null,Q=[],IDX=0,CUR=null,JOB=null,POLL=null;

async function api(u,o){const r=await fetch(u,o);return r.json()}

function chips(){
  if(!STATS)return; const c=STATS.by_category,el=$('#chips');el.innerHTML='';
  const s=document.createElement('span');s.className='chip'+(STATS.stale?' low':'');
  s.innerHTML='stale <b>'+STATS.stale+'</b>';el.appendChild(s);
  const g=document.createElement('span');g.className='chip';
  g.innerHTML='gold <b>'+STATS.gold_total+'</b>';el.appendChild(g);
  Object.keys(c).sort((a,b)=>c[a]-c[b]).slice(0,4).forEach(k=>{
    const x=document.createElement('span');x.className='chip'+(c[k]<8?' low':'');
    x.innerHTML=esc(k.split(' ')[0])+' <b>'+c[k]+'</b>';el.appendChild(x)});
}

/* ---------------- router ---------------- */
async function route(){
  const h=location.hash||'#/label';
  $$('nav a').forEach(a=>a.classList.toggle('on',a.getAttribute('href')===h.split('?')[0]));
  if(POLL){clearInterval(POLL);POLL=null}
  if(!META)META=await api('/api/meta');
  if(h.startsWith('#/new'))return viewNew();
  if(h.startsWith('#/position'))return viewPosition();
  if(h.startsWith('#/scripts'))return viewScripts();
  return viewLabel();
}
addEventListener('hashchange',route);

/* ---------------- labeling ---------------- */
async function viewLabel(){
  const r=await api('/api/queue');Q=r.items;STATS=r.stats;chips();
  if(!Q.length){$('#view').innerHTML='<div class="empty">Queue is empty.</div>';return}
  const p=Q.findIndex(i=>!i.done);IDX=p===-1?0:p;
  $('#view').innerHTML='<div class="split"><div class="col" id="left"></div><div class="col" id="right"></div></div>';
  loadItem();
}
async function loadItem(){
  const it=Q[IDX];CUR=await api('/api/item/'+encodeURIComponent(it.id));
  const d=CUR;
  const refs=d.rules.length?d.rules.map(r=>'<details class="ref"><summary><span class="dot '+(r.ok?'':'bad')+'"></span>'+esc(r.id)+(r.ok?'':' — unresolved')+'</summary><div class="body">'+esc(r.text)+'</div></details>').join(''):'<p class="hint">No citations.</p>';
  const cards=d.card_checks.length?d.card_checks.map(c=>{const cls=c.ok===null?'warn':(c.ok?'':'bad');
    const note=c.ok===null?'unchecked':(c.ok?'exact':(c.resolved?('via '+c.how+' → '+c.resolved):'unresolved'));
    return '<details class="ref"><summary><span class="dot '+cls+'"></span>'+esc(c.name)+' <span class="pill">'+esc(note)+'</span></summary><div class="body">'+esc(c.text||'—')+'</div></details>'}).join(''):'<p class="hint">No cards.</p>';
  $('#left').innerHTML='<div class="qmeta"><span>'+(IDX+1)+' / '+Q.length+'</span><span>'+esc(d.id)+'</span>'+
    (d.url?'<a href="'+esc(d.url)+'" target="_blank" rel="noopener">source ↗</a>':'')+
    (d.level?'<span class="pill">level '+esc(d.level)+'</span>':'')+
    (d.authored?'<span class="pill">authored</span>':'')+'</div>'+
    '<p class="question">'+esc(d.question)+'</p><h2>Verified answer</h2><p class="answer">'+esc(d.answer)+'</p>'+
    '<h2>Cited rules — confirm each settles the question</h2>'+refs+
    '<h2 style="margin-top:1rem">Cards</h2>'+cards;
  const opts=(a,s)=>a.map(o=>'<option'+(o===s?' selected':'')+'>'+esc(o)+'</option>').join('');
  $('#right').innerHTML='<h2>Rubric</h2>'+
    '<div class="draft"><span class="hint">Machine draft — rewrite, don\'t keep</span><ul>'+
    (d.draft.length?d.draft.map(p=>'<li>'+esc(p)+'</li>').join(''):'<li>(none)</li>')+
    '</ul><button style="margin-top:.45rem" id="cp">Copy into editor</button></div>'+
    '<label for="kp">Key points — one per line</label><textarea id="kp" rows="6"></textarea>'+
    '<div class="hint">2–4 sharp points. Never a bare "Yes".</div>'+
    '<label for="ce">Common errors — one per line</label><textarea id="ce" rows="4"></textarea>'+
    '<div class="hint">The real trap, not a negated key point.</div>'+
    '<div class="row"><div><label for="cat">Category</label><select id="cat">'+opts(META.categories,d.category)+'</select></div>'+
    '<div><label for="dif">Difficulty</label><select id="dif">'+opts(META.difficulties,d.difficulty)+'</select></div></div>'+
    '<label for="cites">Rule citations</label><input type="text" class="mono" id="cites" value="'+esc(d.rule_citations.join(' '))+'">'+
    '<div class="hint" id="cc"></div>'+
    '<label for="nt">Notes</label><input type="text" id="nt" value="'+esc(d.notes||'')+'">'+
    '<div class="actions"><button class="primary" id="sv">Save &amp; next <span class="kbd">⌘⏎</span></button>'+
    '<button id="sk">Skip</button><button id="bk">Back</button>'+
    '<button class="danger" id="rj">Reject</button><span class="msg" id="m"></span></div>';
  $('#kp').value=(d.key_points||[]).join('\n');$('#ce').value=(d.common_errors||[]).join('\n');
  $('#cp').onclick=()=>{$('#kp').value=d.draft.join('\n');$('#kp').focus()};
  $('#sv').onclick=saveItem;$('#sk').onclick=()=>step(1);$('#bk').onclick=()=>step(-1);
  $('#rj').onclick=rejectItem;$('#cites').oninput=deb(checkCites,350);checkCites();
  window.scrollTo(0,0);
}
let T;const deb=(f,ms)=>(...a)=>{clearTimeout(T);T=setTimeout(()=>f(...a),ms)};
async function checkCites(){const v=($('#cites')||{}).value;if(v===undefined)return;
  if(!v.trim()){$('#cc').textContent='No citations.';return}
  const r=await api('/api/check?rules='+encodeURIComponent(v.trim()));
  const bad=r.rules.filter(x=>!x.ok).map(x=>x.id);
  $('#cc').innerHTML=bad.length?'<span style="color:var(--bad)">unresolved: '+esc(bad.join(', '))+'</span>':
    '<span style="color:var(--ok)">all '+r.rules.length+' resolve</span>'}
async function saveItem(){const m=$('#m');m.className='msg';m.textContent='saving…';
  const r=await api('/api/item/'+encodeURIComponent(CUR.id),{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key_points:lines('#kp'),common_errors:lines('#ce'),category:$('#cat').value,
      difficulty:$('#dif').value,rule_citations:$('#cites').value.replace(/,/g,' ').split(/\s+/).filter(Boolean),notes:$('#nt').value})});
  if(!r.ok){m.className='msg err';m.textContent=r.error;return}
  const q=await api('/api/queue');STATS=q.stats;chips();Q[IDX].done=true;step(1)}
async function rejectItem(){const why=prompt('Why reject?');if(why===null)return;
  await api('/api/reject/'+encodeURIComponent(CUR.id),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reason:why})});step(1)}
function step(n){let i=IDX+n;while(i>=0&&i<Q.length&&Q[i].done&&n>0)i++;IDX=Math.max(0,Math.min(Q.length-1,i));loadItem()}

/* ---------------- new record ---------------- */
function viewNew(){
  const opts=a=>a.map(o=>'<option>'+esc(o)+'</option>').join('');
  $('#view').innerHTML='<div class="page"><h2>New gold record</h2>'+
    '<p class="hint" style="margin-bottom:1rem">For rulings that come from a judge rather than the RulesGuru corpus — including <strong>definition recall</strong>, which has zero candidates there because RulesGuru is a scenario database.</p>'+
    '<label for="q">Question — as a player would ask it</label><textarea id="q" rows="3" placeholder="my 5/5 got hit by a 1/1 deathtouch creature. does it actually die?"></textarea>'+
    '<label for="pp">Paraphrases — one per line, optional</label><textarea id="pp" rows="2"></textarea>'+
    '<div class="hint">Same ruling, different wording. They share this rubric and test whether the model tracks meaning or surface form.</div>'+
    '<label for="a">Answer — the judge\'s own words, 1–4 sentences</label><textarea id="a" rows="3"></textarea>'+
    '<label for="kp">Key points — one per line</label><textarea id="kp" rows="4"></textarea>'+
    '<div class="hint">Fold the verdict into a substantive claim; one atomic assertion per line.</div>'+
    '<label for="ce">Common errors — one per line</label><textarea id="ce" rows="3"></textarea>'+
    '<label for="cites">Rule citations</label><input type="text" class="mono" id="cites" placeholder="704.5g 702.2b">'+
    '<div class="hint" id="cc"></div>'+
    '<label for="cards">Cards — one per line, optional</label><textarea id="cards" rows="2"></textarea>'+
    '<div class="hint" id="cdc"></div>'+
    '<div class="row"><div><label for="cat">Category</label><select id="cat">'+opts(META.categories)+'</select></div>'+
    '<div><label for="dif">Difficulty</label><select id="dif">'+opts(META.difficulties)+'</select></div>'+
    '<div><label for="src">Source</label><input type="text" id="src" placeholder="judge:LX"></div></div>'+
    '<div class="row"><div><label for="fc">Format context</label><input type="text" id="fc" placeholder="commander, two-player…"></div>'+
    '<div><label for="vb">Verified by</label><input type="text" id="vb" placeholder="second judge, optional"></div></div>'+
    '<label for="nt">Notes</label><input type="text" id="nt" placeholder="why players get this wrong, related interactions">'+
    '<div class="actions"><button class="primary" id="cr">Create record <span class="kbd">⌘⏎</span></button>'+
    '<button id="clr">Clear</button><span class="msg" id="m"></span></div><div id="probs"></div></div>';
  $('#cites').oninput=deb(async()=>{const v=$('#cites').value.trim();
    if(!v){$('#cc').textContent='';return}
    const r=await api('/api/check?rules='+encodeURIComponent(v));
    const bad=r.rules.filter(x=>!x.ok).map(x=>x.id);
    $('#cc').innerHTML=bad.length?'<span style="color:var(--bad)">unresolved: '+esc(bad.join(', '))+'</span>':'<span style="color:var(--ok)">all resolve</span>'},350);
  $('#cards').oninput=deb(async()=>{const v=lines('#cards');
    if(!v.length){$('#cdc').textContent='';return}
    const r=await api('/api/check?cards='+encodeURIComponent(v.join('|')));
    const bad=r.cards.filter(c=>c.ok===false).map(c=>c.name);
    $('#cdc').innerHTML=bad.length?'<span style="color:var(--bad)">did not resolve: '+esc(bad.join(', '))+'</span>':'<span style="color:var(--ok)">all resolve exactly</span>'},400);
  $('#clr').onclick=()=>viewNew();
  $('#cr').onclick=createRecord;
}
async function createRecord(){
  const m=$('#m');m.className='msg';m.textContent='saving…';$('#probs').innerHTML='';
  const body={question:$('#q').value,paraphrases:lines('#pp'),answer:$('#a').value,
    key_points:lines('#kp'),common_errors:lines('#ce'),
    rule_citations:$('#cites').value.replace(/,/g,' ').split(/\s+/).filter(Boolean),
    cards:lines('#cards'),category:$('#cat').value,difficulty:$('#dif').value,
    source:$('#src').value,format_context:$('#fc').value,verified_by:$('#vb').value,notes:$('#nt').value};
  const r=await api('/api/new',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){m.className='msg err';m.textContent='not saved';
    $('#probs').innerHTML='<ul class="problems">'+r.problems.map(p=>'<li>'+esc(p)+'</li>').join('')+'</ul>';return}
  const q=await api('/api/queue');STATS=q.stats;chips();
  m.className='msg ok';m.textContent='created '+r.id;
  ['q','pp','a','kp','ce','cites','cards','nt'].forEach(k=>$('#'+k).value='');
  $('#cc').textContent='';$('#cdc').textContent='';$('#q').focus();
}

/* ---------------- positions ---------------- */
const PF=['pid','pturn','pphase','pactive','ppriority','ylife','ylib','olife','ohand','olib',
  'ybf','obf','yhand','ygy','ogy','pmana','pknown','plegal','pans','pkp','pce','pcites',
  'pcat','pdif','psrc','pdeck'];
function posBody(){
  return {id:$('#pid').value,turn:$('#pturn').value,phase:$('#pphase').value,
    active_player:$('#pactive').value,priority:$('#ppriority').value,
    you_life:$('#ylife').value,you_library:$('#ylib').value,
    opp_life:$('#olife').value,opp_hand_count:$('#ohand').value,opp_library:$('#olib').value,
    you_battlefield:$('#ybf').value,opp_battlefield:$('#obf').value,
    you_hand:$('#yhand').value,you_graveyard:$('#ygy').value,opp_graveyard:$('#ogy').value,
    mana_available:$('#pmana').value,known_information:$('#pknown').value,
    legal_actions:$('#plegal').value,answer:$('#pans').value,
    key_points:$('#pkp').value,common_errors:$('#pce').value,
    rule_citations:$('#pcites').value,category:$('#pcat').value,
    difficulty:$('#pdif').value,source:$('#psrc').value,deck_note:$('#pdeck').value};
}
async function viewPosition(){
  const opts=a=>a.map(o=>'<option>'+esc(o)+'</option>').join('');
  const st=await api('/api/positions');
  const thin=Object.entries(st.by_category).sort((a,b)=>a[1]-b[1]).slice(0,3)
    .map(([k,v])=>esc(k)+' <b>'+v+'</b>').join(' · ');
  $('#view').innerHTML='<div class="split"><div class="col"><div class="page">'+
    '<h2>New board position</h2>'+
    '<p class="hint" style="margin-bottom:1rem"><b>'+st.total+'</b> positions so far. Thinnest: '+(thin||'—')+
    '.<br>A position is scored by the same rubric judge as a rules question — '+
    '<b>key points</b> are the correct line, <b>common errors</b> are the blunders, '+
    'and blunder rate is measured directly from them.</p>'+
    '<div class="row"><div><label for="pturn">Turn</label><input type="text" id="pturn" value="4"></div>'+
    '<div><label for="pphase">Phase</label><input type="text" id="pphase" value="precombat main"></div>'+
    '<div><label for="pactive">Active</label><select id="pactive"><option>you</option><option>opp</option></select></div>'+
    '<div><label for="ppriority">Priority</label><select id="ppriority"><option>you</option><option>opp</option></select></div></div>'+
    '<div class="row"><div><label for="ylife">Your life</label><input type="text" id="ylife" value="20"></div>'+
    '<div><label for="ylib">Your library</label><input type="text" id="ylib" value="0"></div>'+
    '<div><label for="olife">Opp life</label><input type="text" id="olife" value="20"></div>'+
    '<div><label for="ohand">Opp hand</label><input type="text" id="ohand" value="0"></div>'+
    '<div><label for="olib">Opp library</label><input type="text" id="olib" value="0"></div></div>'+
    '<label for="ybf">Your battlefield — one per line</label><textarea id="ybf" rows="4" class="mono" placeholder="Mountain x3&#10;Monastery Swiftspear 1/2"></textarea>'+
    '<div class="hint">Suffixes: <code>x3</code> for copies, <code>2/2</code> for current power/toughness, and <code>tapped</code> / <code>attacking</code> / <code>sick</code>.</div>'+
    '<label for="obf">Opponent battlefield</label><textarea id="obf" rows="4" class="mono" placeholder="Grizzly Bears 2/2 tapped attacking&#10;Island x3"></textarea>'+
    '<label for="yhand">Your hand — one card per line</label><textarea id="yhand" rows="3"></textarea>'+
    '<div class="row"><div><label for="ygy">Your graveyard</label><textarea id="ygy" rows="2"></textarea></div>'+
    '<div><label for="ogy">Opp graveyard</label><textarea id="ogy" rows="2"></textarea></div></div>'+
    '<div class="row"><div><label for="pmana">Mana available</label><input type="text" id="pmana" class="mono" placeholder="{R}{R}{G}"></div>'+
    '<div><label for="pdeck">Deck note</label><input type="text" id="pdeck" placeholder="mono-red aggro"></div></div>'+
    '<label for="pknown">Known information — one per line</label><textarea id="pknown" rows="2" placeholder="opp revealed Negate to a turn-3 Duress"></textarea>'+
    '<label for="plegal">Legal actions — one per line</label><textarea id="plegal" rows="4" class="mono" placeholder="PLAY Mountain&#10;CAST Lightning Strike TARGET Grizzly Bears&#10;PASS"></textarea>'+
    '<div class="hint" id="pgram"></div>'+
    '<label for="pans">The correct line</label><textarea id="pans" rows="2"></textarea>'+
    '<label for="pkp">Key points — one per line</label><textarea id="pkp" rows="4"></textarea>'+
    '<div class="hint">What a correct answer must say. Two minimum.</div>'+
    '<label for="pce">Common errors — the blunders, one per line</label><textarea id="pce" rows="3"></textarea>'+
    '<div class="hint">Required. Blunder rate is measured from these, so a position without one cannot contribute to the gate.</div>'+
    '<div class="hint" id="pce_warn"></div>'+
    '<label for="pcites">Rule citations — optional</label><input type="text" class="mono" id="pcites" placeholder="509.1a 510.1c">'+
    '<div class="row"><div><label for="pcat">Category</label><select id="pcat">'+opts(META.position_categories)+'</select></div>'+
    '<div><label for="pdif">Difficulty</label><select id="pdif">'+opts(META.difficulties)+'</select></div>'+
    '<div><label for="psrc">Source</label><input type="text" id="psrc" placeholder="judge:LX"></div></div>'+
    '<label for="pid">Id — blank to auto-number</label><input type="text" class="mono" id="pid">'+
    '<div class="actions"><button class="primary" id="pcr">Create position <span class="kbd">⌘⏎</span></button>'+
    '<button id="pclr">Clear</button><span class="msg" id="pm"></span></div><div id="pprobs"></div>'+
    '</div></div><div class="col"><div class="page">'+
    '<h2>What the model sees</h2>'+
    '<pre class="mono" id="ppreview" style="white-space:pre-wrap;font-size:.82rem"></pre>'+
    '<div id="pgrammar"></div></div></div></div>';
  $('#pgrammar').innerHTML='<details class="ref"><summary>Action grammar</summary><div class="body"><pre class="mono" style="white-space:pre-wrap">'+esc(META.action_grammar)+'</pre></div></details>';
  PF.forEach(k=>{const el=$('#'+k); if(el) el.oninput=deb(preview,400)});
  $('#pclr').onclick=()=>viewPosition();
  $('#pcr').onclick=createPosition;
  preview();
}
async function preview(){
  const r=await api('/api/position/preview',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(posBody())});
  $('#ppreview').textContent=r.rendered;
  // Only surface problems that are about the board itself while typing —
  // "needs >=2 key points" on a half-filled form is noise, not a finding.
  const live=r.problems.filter(p=>/card|legal_action|controller/.test(p));
  $('#pgram').innerHTML=live.length
    ?'<span style="color:var(--bad)">'+live.map(esc).join('<br>')+'</span>'
    :'<span style="color:var(--ok)">board and actions parse</span>';
  // Advisory, never blocking — see Section 16.12.
  $('#pce_warn').innerHTML=(r.warnings||[]).length
    ?'<span style="color:var(--warn,#b8860b)">⚠ '+r.warnings.map(esc).join('<br>⚠ ')+'</span>':'';
}
async function createPosition(){
  const m=$('#pm');m.className='msg';m.textContent='saving…';$('#pprobs').innerHTML='';
  const r=await api('/api/position',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(posBody())});
  if(!r.ok){m.className='msg err';m.textContent='not saved';
    $('#pprobs').innerHTML='<ul class="problems">'+r.problems.map(p=>'<li>'+esc(p)+'</li>').join('')+'</ul>';return}
  m.className='msg ok';m.textContent='created '+r.id+' ('+r.total+' total)';
  ['ybf','obf','yhand','ygy','ogy','pknown','plegal','pans','pkp','pce','pcites','pid','pmana'].forEach(k=>$('#'+k).value='');
  preview();$('#ybf').focus();
}

/* ---------------- scripts ---------------- */
function viewScripts(){
  const groups={};META.actions.forEach(a=>(groups[a.group]=groups[a.group]||[]).push(a));
  let html='<div class="page"><h2>Pipeline scripts</h2>'+
    '<p class="hint" style="margin-bottom:.6rem">One job at a time — several of these write the same files.</p>'+
    '<div id="jobbox"></div>';
  for(const g of Object.keys(groups)){
    html+='<div class="grp">'+esc(g)+'</div>';
    for(const a of groups[g]){
      html+='<div class="card"><div class="top"><h3>'+esc(a.label)+'</h3>'+
        '<span class="pill">'+esc(a.eta)+'</span>'+(a.writes?'<span class="pill">writes data</span>':'<span class="pill">read-only</span>')+'</div>'+
        '<p>'+esc(a.desc)+'</p><div class="row" style="margin-top:.5rem">';
      a.args.filter(x=>x.type!=='fixed').forEach((x,i)=>{
        const id='arg-'+a.id+'-'+i;
        if(x.type==='flag')html+='<div style="flex:0 0 auto"><label style="margin-top:.3rem">'+esc(x.label)+'</label><input type="checkbox" id="'+id+'" '+(x.default?'checked':'')+'></div>';
        else if(x.type==='choice')html+='<div><label>'+esc(x.label)+'</label><select id="'+id+'">'+x.choices.map(c=>'<option'+(c===x.default?' selected':'')+'>'+esc(c)+'</option>').join('')+'</select></div>';
        else html+='<div><label>'+esc(x.label)+'</label><input type="text" id="'+id+'" value="'+esc(String(x.default??''))+'"></div>';
      });
      html+='</div><div style="margin-top:.55rem"><button data-run="'+esc(a.id)+'">Run</button></div></div>';
    }
  }
  $('#view').innerHTML=html+'</div>';
  $$('[data-run]').forEach(b=>b.onclick=()=>runAction(b.dataset.run));
  refreshJobs();
}
async function runAction(id){
  const a=META.actions.find(x=>x.id===id),values={};
  a.args.filter(x=>x.type!=='fixed').forEach((x,i)=>{
    const el=document.getElementById('arg-'+id+'-'+i);if(!el)return;
    values[x.name||x.label]=x.type==='flag'?el.checked:el.value});
  const r=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:id,values})});
  if(!r.ok){$('#jobbox').innerHTML='<div class="card"><span style="color:var(--bad)">'+esc(r.error)+'</span></div>';return}
  JOB=r.job;watch();
}
function watch(){
  if(POLL)clearInterval(POLL);
  let seen=0;
  const tick=async()=>{
    const j=await api('/api/job/'+JOB+'?since='+seen);
    if(j.error)return;
    seen=j.total_lines;
    const box=$('#jobbox');
    if(box)box.innerHTML='<div class="card"><div class="top"><h3>'+esc(j.label)+'</h3>'+
      '<span class="status '+esc(j.status)+'">'+esc(j.status)+(j.returncode!==null?' ('+j.returncode+')':'')+'</span>'+
      (j.status==='running'?'<button id="cn" style="margin-left:auto">Cancel</button>':'')+'</div>'+
      '<p style="font-family:var(--mono);font-size:.72rem">'+esc(j.cmd)+'</p>'+
      '<div class="log" id="lg"></div></div>';
    const lg=$('#lg');if(lg){lg.textContent=(window._buf=(window._buf||'')+j.lines.map(l=>l+'\n').join(''));lg.scrollTop=lg.scrollHeight}
    const cn=$('#cn');if(cn)cn.onclick=()=>api('/api/job/'+JOB+'/cancel',{method:'POST'});
    if(j.status!=='running'){clearInterval(POLL);POLL=null;
      const q=await api('/api/queue');STATS=q.stats;chips()}
  };
  window._buf='';tick();POLL=setInterval(tick,700);
}
async function refreshJobs(){
  const r=await api('/api/jobs');
  if(r.active){JOB=r.active;watch();return}
  if(!r.jobs.length)return;
  $('#jobbox').innerHTML='<div class="card"><div class="top"><h3>Recent</h3></div>'+
    r.jobs.map(j=>'<p><span class="status '+esc(j.status)+'">'+esc(j.status)+'</span> '+esc(j.label)+
    ' <span class="hint">'+esc(j.started)+'</span></p>').join('')+'</div>';
}

document.addEventListener('keydown',e=>{
  if((e.metaKey||e.ctrlKey)&&e.key==='Enter'){e.preventDefault();
    if(location.hash.startsWith('#/new'))createRecord();else if($('#sv'))saveItem()}
});
(async()=>{const q=await api('/api/queue');STATS=q.stats;chips();route()})();
</script></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    p.add_argument("--lan", action="store_true", help="bind all interfaces so a phone can reach it")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--gold", type=Path, default=GOLD_PATH)
    p.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    p.add_argument("--rules", type=Path, default=RULES_PATH)
    p.add_argument("--category", default=None, help="restrict the labeling queue to one category")
    p.add_argument("--author", default="", help="recorded in rubric_source, e.g. 'judge:LX'")
    p.add_argument("--token", default=None, help="fixed access token instead of a generated one")
    p.add_argument("--no-auth", action="store_true", help="skip the token even when bound to the LAN")
    p.add_argument("--skip-cards", action="store_true")
    args = p.parse_args()

    host = args.host or ("0.0.0.0" if args.lan else "127.0.0.1")
    loopback = host in ("127.0.0.1", "localhost", "::1")
    token = None if (loopback or args.no_auth) else (args.token or secrets.token_urlsafe(6))

    print("loading corpora ...")
    store = Store(args.gold, args.candidates, args.rules, args.category, args.skip_cards)
    s = store.stats()
    items, fixes = store.queue()
    print(f"  {s['gold_total']} gold ({s['authored']} authored, {s['stale']} stale)")
    print(f"  {s['candidates_left']} candidates, queue {len(items)} ({fixes} to fix first)")

    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn is required: pip install uvicorn fastapi")

    suffix = f"?t={token}" if token else ""
    print()
    if loopback:
        print(f"  http://127.0.0.1:{args.port}/")
    else:
        ip, hn = lan_ip(), socket.gethostname().split(".")[0]
        print(f"  this machine   http://127.0.0.1:{args.port}/{suffix}")
        print(f"  phone / LAN    http://{ip}:{args.port}/{suffix}")
        print(f"  bonjour name   http://{hn}.local:{args.port}/{suffix}")
        if token:
            print(f"\n  Token {token} — open the link once and it is stored in a cookie.")
        else:
            print("\n  WARNING: --no-auth on a LAN binding. Anyone on this network can run "
                  "the allowlisted scripts and edit the gold set.")
    print()

    uvicorn.run(build_app(store, Runner(), args.author, token),
                host=host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
