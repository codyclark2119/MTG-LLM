"""A tiny web form for hand-authoring rubrics, deployable to a public host.

WHY THIS IS NOT PART OF `webui.py`
----------------------------------
`webui.py` is a *local* console. Its script runner executes training and
evaluation on the host, and its store writes `data/gold/gold_questions.jsonl`
directly. Both are fine behind a LAN token and both are unacceptable on a
public URL — the runner is remote code execution by design, and the gold set
is the project's highest-trust artifact.

So this is a separate program with a deliberately small surface:

  * it reads ONE self-contained file, `tasks.json`, produced by
    `author_rubrics.py --export-tasks` — questions, verified answers and
    machine drafts, nothing else;
  * it appends to ONE file, `submissions.jsonl`;
  * it has NO access to the gold set, the candidates, the rules corpus, the
    card index or any model, and imports none of them;
  * it runs on `fastapi` + `uvicorn` alone. No numpy, no mlx, no datasets —
    which is also why the container is small enough to be free-tier sized.

Promotion into the gold set stays a local, reviewed step:

    python scripts/author_rubrics.py --ingest-submissions submissions.jsonl --dry-run
    python scripts/author_rubrics.py --ingest-submissions submissions.jsonl

That preserves the invariant that makes the tier mean something: a rubric is
gold because a person reviewed it, not because someone typed it into a form.

WHY A FORM RATHER THAN A TEXT FILE
----------------------------------
Measured. Of seven plausible ways a person writes a bulleted list into a
worksheet, only a plain ASCII hyphen parsed; the other six produced zero key
points and were silently skipped (see BULLET_RE in author_rubrics.py). The
parser now accepts all of them, but the deeper fix is not to have a parser:
a form submits one field per key point, so there is no formatting to get
wrong and nothing to silently drop.

Usage:
    python scripts/rubric_server.py --tasks data/gold/worksheets/tasks.json
    python scripts/rubric_server.py --tasks tasks.json --host 0.0.0.0 --port 8080
"""

import argparse
import hashlib
import json
import os
import re
import secrets
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# The only project imports, all pure python — see the module docstring.
from common import (lint_common_errors, stray_names, templatize, untemplatize)

# One writer at a time. Submissions append, and two contributors finishing a
# question in the same instant would otherwise interleave a line.
_WRITE_LOCK = threading.Lock()

# Bumped whenever the adjudication form's WORDING changes in a way that could
# move a verdict. The first eight verdicts were collected under v1, which asked
# "which of these does it commit?" and did not say to judge the play rather than
# the wording; they under-fire on action-list answers as a result (Section
# 21.47). Segmenting on this is the only way to tell that apart from a real
# change in the answers, so it rides on every submission the way `author` does.
ADJUDICATION_FORM_VERSION = 2


def _answer_sha(answer: str) -> str:
    """12-char digest of an answer, matching `adjudicate.answer_sha`.

    Duplicated deliberately rather than imported: the server's only project
    import is `common`, and pulling in `adjudicate` would put the gold set, the
    queue and the run files on the import path of a public service. Twelve
    characters of sha256 is not a helper worth that.
    """
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()[:12]


def load_tasks(path: Path) -> tuple[list[dict], str]:
    """Tasks plus which KIND of work they are.

    The kind comes from the data, never from a flag. A `--mode` flag that
    disagreed with the file would serve the wrong form against the right
    tasks, and the failure would look like a rendering bug rather than a
    configuration one.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload["tasks"] if isinstance(payload, dict) else payload
    kind = payload.get("kind", "rubric") if isinstance(payload, dict) else "rubric"
    if not tasks:
        raise SystemExit(f"{path} has no tasks")
    if kind not in ("rubric", "adjudication"):
        raise SystemExit(f"{path}: unknown task kind {kind!r}")
    if kind == "adjudication":
        # The whole value of an adjudication is that it was made without
        # seeing the judge. If the export ever leaks that, the verdicts are
        # worthless and nothing downstream would be able to tell.
        # `source_run` joins the list because a run filename names its judge
        # (`pos_n24_verbose_32b.jsonl`), which is the thing `judge_model` is here
        # to keep out. Provenance belongs on the local queue, not on a task
        # served to a blind reviewer (Section 21.62).
        leaked = {k for t in tasks for k in t} & {
            "errors_made", "judge_model", "fired", "blundered", "run", "judge",
            "source_run"}
        if leaked:
            raise SystemExit(
                f"{path}: tasks carry judge output {sorted(leaked)}. A verdict "
                "collected while looking at the judge is not independent of it; "
                "re-export with `adjudicate.py --export-tasks`.")
    return tasks, kind


def read_submissions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_submission(path: Path, row: dict) -> None:
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def build_app(tasks: list[dict], submissions_path: Path, token: str | None,
              kind: str = "rubric"):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    app = FastAPI(title=f"magic-llm {kind}")
    id_key = "key" if kind == "adjudication" else "id"
    by_id = {t[id_key]: t for t in tasks}
    categories = sorted({t.get("category") or "" for t in tasks})

    @app.middleware("http")
    async def auth(request: Request, call_next):
        # /healthz is exempt on purpose: a platform health check arrives with
        # no token and no cookie, so gating it would leave every machine
        # marked unhealthy and the app permanently down. It exposes only a
        # task count, which is not sensitive.
        if not token or request.url.path == "/healthz":
            return await call_next(request)
        supplied = (request.query_params.get("t")
                    or request.cookies.get("mlrt")
                    or request.headers.get("x-token"))
        if supplied != token:
            return PlainTextResponse("This link needs its access token. Use the full "
                                     "URL you were sent.", 401)
        response = await call_next(request)
        if request.query_params.get("t") == token:
            response.set_cookie("mlrt", token, max_age=86400 * 30, samesite="lax")
        return response

    @app.get("/", response_class=HTMLResponse)
    def index():
        return ADJUDICATE_HTML if kind == "adjudication" else INDEX_HTML

    @app.get("/api/tasks")
    def api_tasks(author: str = ""):
        """Task list plus this author's own progress.

        Progress is per author so two people working the same category each
        see their own remaining count. `done_by_anyone` is separate and only
        dims a row — it is a hint, not a lock, because a second opinion on a
        rubric is useful rather than wasted.
        """
        subs = read_submissions(submissions_path)
        if kind == "adjudication":
            # The whole task inlines, because the reviewer needs every field at
            # once and there are only ~60 of them. Same per-author progress
            # rule as below: a verdict is "done" for the person who gave it.
            # Done means "this author has graded THIS TEXT", not "this key".
            # Comparing keys alone marked already-graded keys done after the
            # arms were regenerated, hiding exactly the tasks that most needed a
            # fresh verdict (Section 21.62).
            #
            # A submission with NO digest predates this field, which means it
            # was collected before the arms were regenerated — so it is about
            # older text by construction, and counts as NOT done. The asymmetry
            # is deliberate: treating it as done makes a reviewer silently skip
            # work that needs redoing, while treating it as undone at worst asks
            # for one duplicate verdict, visibly. Cheap and visible beats silent
            # and corrupting. It is also self-clearing — every submission
            # collected from here on carries a digest, so this branch stops
            # firing once the current log is superseded.
            live = {t["key"]: _answer_sha(t.get("answer") or "") for t in tasks}
            mine = {s["key"] for s in subs
                    if s.get("author") == author and "errors_present" in s
                    and s["key"] in live
                    and s.get("answer_sha") == live[s["key"]]}
            return {"tasks": [dict(t, done=t["key"] in mine) for t in tasks],
                    "done_by_me": len(mine), "total": len(tasks)}
        # Intersect with the CURRENT task list. The submissions log is
        # append-only and outlives any one export, so after an ingest the
        # questions it covers are promoted out of tasks.json while their rows
        # remain. Counting the raw log reported "10 / 28 done" against a task
        # list those ten had already left — phantom progress, and worst in
        # exactly the ingest -> re-export -> restart loop this is used in.
        live = {t["id"] for t in tasks}
        mine = {s["id"] for s in subs
                if s.get("author") == author and s.get("key_points")} & live
        anyone = {s["id"] for s in subs if s.get("key_points")} & live
        rows = [{"id": t["id"], "category": t["category"], "difficulty": t["difficulty"],
                 "question": t["question"][:110],
                 "done_by_me": t["id"] in mine,
                 "done_by_anyone": t["id"] in anyone} for t in tasks]
        return {"tasks": rows, "categories": categories,
                "done_by_me": len(mine), "total": len(tasks)}

    @app.get("/api/task/{tid}")
    def api_task(tid: str, author: str = ""):
        t = by_id.get(tid)
        if not t:
            return JSONResponse({"error": "unknown id"}, 404)
        prior = [s for s in read_submissions(submissions_path)
                 if s["id"] == tid and s.get("author") == author]
        last = prior[-1] if prior else None
        if last:
            # Stored as [[card1]], shown as the card's name. An author writes
            # about cards, not about slots; the slot is a storage detail and
            # showing it would be asking them to maintain the encoding by hand.
            slots = t.get("card_slots") or {}
            last = {**last,
                    "key_points": [untemplatize(x, slots) for x in last.get("key_points") or []],
                    "common_errors": [untemplatize(x, slots)
                                      for x in last.get("common_errors") or []]}
        return {**t, "prior": last}

    @app.post("/api/check")
    async def api_check(request: Request):
        """Live feedback while typing. Never blocks a save."""
        body = await request.json()
        t = by_id.get(body.get("id"), {})
        kp = [s.strip() for s in body.get("key_points") or [] if s.strip()]
        ce = [s.strip() for s in body.get("common_errors") or [] if s.strip()]
        notes = []
        if len(kp) < 2:
            notes.append(f"{len(kp)} key point so far — at least 2 are needed to save.")
        if not ce:
            notes.append("No common errors yet. That field is half the rubric's power "
                         "and cannot be derived from the answer.")
        for p in kp:
            if p.strip().lower().rstrip(".") in ("yes", "no"):
                notes.append(f'"{p}" is a bare verdict. Fold it into a claim that '
                             f"carries information.")
        warnings = lint_common_errors({"answer": t.get("answer", ""),
                                       "key_points": kp, "common_errors": ce})
        stray = stray_names(t.get("question", ""), t.get("answer", ""), kp + ce,
                            list((t.get("card_slots") or {}).values()))
        if stray:
            warnings.append(
                f"{', '.join(stray)} — not mentioned in the question or its answer. "
                f"Player names differ from question to question, so check you have "
                f"the right one and that the action is attributed to them.")
        slots = t.get("card_slots") or {}
        matched = sorted({name for name in slots.values()
                          for line in kp + ce
                          if re.search(r"\b" + re.escape(name) + r"\b", line)})
        return {"notes": notes, "warnings": warnings, "can_save": len(kp) >= 2,
                "cards_detected": matched}

    @app.post("/api/submit")
    async def api_submit(request: Request):
        body = await request.json()
        tid = body.get("id")
        if tid not in by_id:
            return JSONResponse({"error": "unknown id"}, 404)
        kp = [s.strip() for s in body.get("key_points") or [] if s.strip()]
        if len(kp) < 2:
            return JSONResponse({"error": "at least 2 key points are needed"}, 400)
        author = (body.get("author") or "").strip()[:60]
        ce = [s.strip() for s in body.get("common_errors") or [] if s.strip()]
        # Card names go in as slots. The author wrote "Castle Locthwain"; what
        # is stored is "[[card2]]", so this one rubric also scores the versions
        # of this question that RulesGuru builds on other cards. The form shows
        # what was detected, so the substitution is visible rather than silent.
        slots = by_id[tid].get("card_slots") or {}
        row = {
            "id": tid,
            "key_points": [templatize(x, slots) for x in kp],
            "common_errors": [templatize(x, slots) for x in ce],
            "author": author,
            "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        append_submission(submissions_path, row)
        return {"ok": True, "id": tid}

    @app.post("/api/adjudicate")
    async def api_adjudicate(request: Request):
        """One blind verdict: which listed errors does this answer commit?

        An EMPTY list is a real verdict and the most common one, so it is
        accepted and recorded; only a missing field is an error. Treating
        "commits none" as a non-answer would drop exactly the cases that
        measure a judge's false positives.
        """
        body = await request.json()
        tid = body.get("key")
        if tid not in by_id:
            return JSONResponse({"error": "unknown key"}, 404)
        present = body.get("errors_present")
        if not isinstance(present, list):
            return JSONResponse({"error": "errors_present must be a list"}, 400)
        n_errors = len(by_id[tid].get("common_errors") or [])
        try:
            nums = sorted({int(n) for n in present})
        except (TypeError, ValueError):
            return JSONResponse({"error": "errors_present must be numbers"}, 400)
        if any(n < 1 or n > n_errors for n in nums):
            return JSONResponse({"error": f"error numbers must be 1..{n_errors}"}, 400)
        append_submission(submissions_path, {
            "key": tid,
            "record_id": by_id[tid].get("record_id"),
            "arm": by_id[tid].get("arm"),
            # A digest of the exact answer this verdict was given about, taken
            # from the served task rather than from the client — a value the
            # client supplied would be its claim about what it was shown.
            #
            # The key `record_id::arm` is stable while the text behind it is
            # not: regenerate the arms and the same key names a different
            # answer. Without this, a reviewer who graded a key under an older
            # grammar sees it marked done and skips the one task that most needs
            # regrading, and `--score` matches the old verdict to the new text
            # (Section 21.62).
            "answer_sha": _answer_sha(by_id[tid].get("answer") or ""),
            "errors_present": nums,
            "blundered": bool(nums),
            "unsure": bool(body.get("unsure")),
            # "wrong, but none of the listed mistakes describe it" — the case
            # that produced the first disagreements: an answer of `PASS` or a
            # repeated action commits no CLAIM in the rubric while being
            # obviously bad. Recorded separately because it measures rubric
            # COVERAGE, which no judge number can (Section 21.47).
            "not_covered": bool(body.get("not_covered")),
            "note": (body.get("note") or "").strip()[:500],
            "author": (body.get("author") or "").strip()[:60],
            "form_version": ADJUDICATION_FORM_VERSION,
            "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return {"ok": True, "key": tid}

    @app.get("/api/export")
    def api_export():
        """The raw submissions log, for pulling down and ingesting locally."""
        if not submissions_path.exists():
            return PlainTextResponse("", media_type="application/x-ndjson")
        return PlainTextResponse(submissions_path.read_text(encoding="utf-8"),
                                 media_type="application/x-ndjson")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "tasks": len(tasks)}

    return app


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Rubric authoring</title>
<style>
:root{--paper:#F5F6F8;--panel:#fff;--ink:#131820;--soft:#59636F;--faint:#8A939E;
 --accent:#1C5A8C;--accent-bg:#EAF1F7;--ok:#1E7A4B;--warn:#8A6100;--warn-bg:#FBF3DF;
 --bad:#94382C;--rule:#D9DEE4;--rule-soft:#E8EBEF;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#0F1319;--panel:#161B22;
 --ink:#E3E7EC;--soft:#9BA5B1;--faint:#6B7683;--accent:#79B0DC;--accent-bg:#16242F;--ok:#68C08D;
 --warn:#D8B45E;--warn-bg:#2A2416;--bad:#D59286;--rule:#2A313A;--rule-soft:#1F252D}}
:root[data-theme="dark"]{--paper:#0F1319;--panel:#161B22;--ink:#E3E7EC;--soft:#9BA5B1;
 --faint:#6B7683;--accent:#79B0DC;--accent-bg:#16242F;--ok:#68C08D;--warn:#D8B45E;
 --warn-bg:#2A2416;--bad:#D59286;--rule:#2A313A;--rule-soft:#1F252D}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
 font-size:15px;line-height:1.55}
button,input,select,textarea{font:inherit;color:inherit}
header{position:sticky;top:0;z-index:10;background:var(--panel);border-bottom:1px solid var(--rule);
 padding:.55rem .9rem;display:flex;gap:.7rem;align-items:center;flex-wrap:wrap}
.brand{font-weight:650;letter-spacing:-.01em;white-space:nowrap}
.grow{flex:1}
select,input[type=text]{background:var(--panel);border:1px solid var(--rule);border-radius:3px;
 padding:.3rem .5rem}
.prog{font-size:.84rem;color:var(--soft);font-variant-numeric:tabular-nums}
main{display:grid;grid-template-columns:minmax(230px,300px) 1fr;gap:0;
 height:calc(100vh - 51px);min-height:0}
@media(max-width:820px){main{grid-template-columns:1fr;height:auto}
 #list{max-height:38vh;border-right:none;border-bottom:1px solid var(--rule)}
 #pane{height:auto}}
#list{overflow-y:auto;border-right:1px solid var(--rule);background:var(--panel)}
#list .row{padding:.5rem .75rem;border-bottom:1px solid var(--rule-soft);cursor:pointer;
 font-size:.87rem;display:flex;gap:.5rem;align-items:baseline}
#list .row:hover{background:var(--accent-bg)}
#list .row.on{background:var(--accent-bg);box-shadow:inset 3px 0 0 var(--accent)}
#list .row.mine{opacity:.55}
#list .row .q{flex:1;color:var(--soft)}
#list .row .tick{color:var(--ok);font-weight:700}
/* The save bar is a GRID ROW, not a sticky child of the scrolling content.
   As a sticky element it stopped sticking at the end of the scroll and came
   to rest on top of the notes — so the warnings a contributor most needs to
   read were hidden underneath the button they were about to press. A row
   cannot overlap its sibling. */
#pane{display:grid;grid-template-rows:1fr auto;overflow:hidden;min-height:0}
#scroll{overflow-y:auto;padding:1.1rem 1.3rem 1.6rem;min-height:0}
.wrap{max-width:62rem;margin:0 auto}
h2{font-family:var(--serif);font-size:1.22rem;margin:.1rem 0 .1rem;font-weight:600;
 text-wrap:balance}
.meta{font-size:.8rem;color:var(--faint);margin-bottom:1rem}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:4px;
 padding:.8rem .95rem;margin-bottom:1rem}
.card h3{margin:0 0 .45rem;font-size:.74rem;text-transform:uppercase;letter-spacing:.07em;
 color:var(--faint);font-weight:650}
.answer{font-family:var(--serif);font-size:1.02rem}
.draft{background:var(--rule-soft);border-radius:3px;padding:.5rem .7rem;
 font-family:var(--mono);font-size:.82rem;color:var(--soft);white-space:pre-wrap;
 user-select:text}
.draft .lbl{display:block;color:var(--faint);font-family:var(--sans);font-size:.75rem;
 margin-bottom:.3rem}
.fieldrow{display:flex;gap:.4rem;margin-bottom:.4rem;align-items:flex-start}
.fieldrow textarea{flex:1;min-height:2.4rem;padding:.4rem .55rem;border:1px solid var(--rule);
 border-radius:3px;background:var(--paper);resize:vertical}
.fieldrow button{background:none;border:1px solid var(--rule);border-radius:3px;color:var(--faint);
 padding:.25rem .5rem;cursor:pointer;line-height:1}
.fieldrow button:hover{color:var(--bad);border-color:var(--bad)}
.addbtn{background:none;border:1px dashed var(--rule);border-radius:3px;color:var(--soft);
 padding:.32rem .7rem;cursor:pointer;font-size:.85rem}
.addbtn:hover{border-color:var(--accent);color:var(--accent)}
.hint{font-size:.82rem;color:var(--faint);margin:.15rem 0 .5rem}
.notes{margin-top:.5rem}
.note{font-size:.83rem;padding:.4rem .6rem;border-radius:3px;margin-bottom:.3rem;
 background:var(--rule-soft);color:var(--soft);border-left:3px solid var(--rule)}
.note.warn{background:var(--warn-bg);color:var(--warn);border-left-color:var(--warn)}
.bar{background:var(--panel);border-top:1px solid var(--rule);
 padding:.6rem 1.3rem;display:flex;gap:.75rem;align-items:center}
.save{background:var(--accent);color:#fff;border:none;border-radius:3px;padding:.45rem 1.1rem;
 cursor:pointer;font-weight:600}
.save[disabled]{opacity:.4;cursor:not-allowed}
.saved{color:var(--ok);font-size:.85rem}
/* A count in the bar, because the notes themselves live at the end of the
   scroll and a contributor working near the top would not see them at all. */
.flags{font-size:.83rem;color:var(--soft);cursor:pointer;text-decoration:underline;
 text-underline-offset:2px}
.flags.warn{color:var(--warn);font-weight:600}
.flags:empty{display:none}
/* Card chips. An author writes card names, not slots — these are a typing
   aid, and the caption explains what storage does with them. */
.chips{display:flex;flex-wrap:wrap;gap:.35rem;margin:.1rem 0 .5rem}
.chip{background:var(--accent-bg);border:1px solid var(--rule);border-radius:12px;
 padding:.16rem .6rem;font-size:.82rem;cursor:pointer;color:var(--accent);white-space:nowrap}
.chip:hover{border-color:var(--accent)}
.chip.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.caption{font-size:.78rem;color:var(--faint);margin:-.2rem 0 .6rem}
.rules{font-size:.83rem;color:var(--soft)}
.rules li{margin-bottom:.2rem}
.empty{color:var(--faint);text-align:center;padding:4rem 1rem;font-size:.95rem}
a{color:var(--accent)}
</style></head><body>
<header>
  <span class="brand">Rubrics</span>
  <select id="cat"><option value="">All categories</option></select>
  <input type="text" id="author" placeholder="your name" size="12">
  <span class="grow"></span>
  <span class="prog" id="prog"></span>
</header>
<main>
  <div id="list"></div>
  <div id="pane"><div id="scroll"><div class="empty">Enter your name, then pick a
    question.</div></div></div>
</main>
<script>
const $ = s => document.querySelector(s);
let TASKS = [], CUR = null, DIRTY = false;

const author = () => $('#author').value.trim();

function saveName(){ localStorage.setItem('mlr_author', author()); }
function loadName(){ $('#author').value = localStorage.getItem('mlr_author') || ''; }

async function refresh(){
  const r = await fetch('/api/tasks?author=' + encodeURIComponent(author()));
  const d = await r.json();
  TASKS = d.tasks;
  const sel = $('#cat');
  if (sel.options.length <= 1)
    d.categories.forEach(c => sel.add(new Option(c + ' (' +
      d.tasks.filter(t => t.category === c).length + ')', c)));
  $('#prog').textContent = d.done_by_me + ' / ' + d.total + ' done';
  renderList();
}

function renderList(){
  const cat = $('#cat').value;
  const rows = TASKS.filter(t => !cat || t.category === cat);
  $('#list').innerHTML = rows.map(t =>
    '<div class="row' + (t.done_by_me ? ' mine' : '') + (CUR === t.id ? ' on' : '') +
    '" data-id="' + t.id + '">' +
    (t.done_by_me ? '<span class="tick">&#10003;</span>' : '<span style="width:.7em"></span>') +
    '<span class="q">' + esc(t.question) + '</span></div>').join('')
    || '<div class="empty">Nothing in this category.</div>';
  document.querySelectorAll('#list .row').forEach(el =>
    el.onclick = () => open(el.dataset.id));
}

function esc(s){ return (s||'').replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

let LAST = null;   // the textarea a chip should insert into

function fieldRow(val){
  const d = document.createElement('div');
  d.className = 'fieldrow';
  const ta = document.createElement('textarea');
  ta.rows = 2; ta.value = val || '';
  ta.oninput = () => { DIRTY = true; check(); };
  ta.onfocus = () => { LAST = ta; };
  const b = document.createElement('button');
  b.type = 'button'; b.textContent = '×'; b.title = 'remove';
  b.onclick = () => { d.remove(); DIRTY = true; check(); };
  d.append(ta, b);
  return d;
}

function vals(sel){
  return [...document.querySelectorAll(sel + ' textarea')]
    .map(t => t.value.trim()).filter(Boolean);
}

async function open(id){
  if (DIRTY && !confirm('You have unsaved edits. Discard them?')) return;
  DIRTY = false; CUR = id;
  const r = await fetch('/api/task/' + id + '?author=' + encodeURIComponent(author()));
  const t = await r.json();
  const prior = t.prior || {};
  $('#pane').innerHTML =
    '<div id="scroll"><div class="wrap">' +
    '<h2>' + esc(t.question) + '</h2>' +
    '<div class="meta">' + esc(t.category) + ' &middot; ' + esc(t.difficulty) +
      (t.url ? ' &middot; <a href="' + esc(t.url) + '" target="_blank" rel="noopener">source</a>' : '') +
      '</div>' +
    '<div class="card"><h3>Verified answer &mdash; this is correct, do not re-adjudicate</h3>' +
      '<div class="answer">' + esc(t.answer) + '</div></div>' +
    (t.draft && t.draft.length
      ? '<div class="card"><h3>Machine draft</h3><div class="draft"><span class="lbl">' +
        'Read-only. A sentence-split of the answer, shown so you can see what to avoid: ' +
        'compound points, bare verdicts, no errors at all.</span>' +
        t.draft.map(d => '• ' + esc(d)).join('\n') + '</div></div>'
      : '') +
    ((t.card_slots && Object.keys(t.card_slots).length)
      ? '<div class="card"><h3>Cards in this question</h3>' +
        '<div class="chips" id="chips"></div>' +
        '<div class="caption">Click to insert a name. Write card names normally &mdash; ' +
        'they are stored as slots so this rubric also scores the other versions of ' +
        'this question, which RulesGuru builds on different cards.</div></div>'
      : '') +
    '<div class="card"><h3>Key points</h3>' +
      '<div class="hint">The claims a correct answer must make. One checkable ' +
      'assertion each; 2–4 sharp points beat 5–8 soft ones. Never a bare ' +
      '&ldquo;Yes&rdquo; or &ldquo;No&rdquo;.</div>' +
      '<div id="kp"></div>' +
      '<button class="addbtn" type="button" id="addkp">+ key point</button></div>' +
    '<div class="card"><h3>Common errors</h3>' +
      '<div class="hint">The mistakes a real player makes &mdash; traps, not negations ' +
      'of the points above. Lead with what makes it wrong: an error that opens with ' +
      'words also true of the correct answer gets matched before the judge reaches ' +
      'the part that matters.</div>' +
      '<div id="ce"></div>' +
      '<button class="addbtn" type="button" id="addce">+ common error</button></div>' +
    '<div class="notes" id="notes"></div>' +
    '</div></div>' +
    '<div class="bar"><button class="save" id="save">Save</button>' +
      '<span class="flags" id="flags"></span>' +
      '<span class="saved" id="saved"></span></div>';

  const kp = prior.key_points || ['', ''];
  const ce = prior.common_errors || [''];
  kp.forEach(v => $('#kp').append(fieldRow(v)));
  ce.forEach(v => $('#ce').append(fieldRow(v)));
  $('#addkp').onclick = () => { $('#kp').append(fieldRow('')); };
  $('#addce').onclick = () => { $('#ce').append(fieldRow('')); };
  $('#save').onclick = submit;

  const chips = $('#chips');
  if (chips){
    SLOTS = t.card_slots || {};
    Object.keys(SLOTS).sort().forEach(slot => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'chip'; b.dataset.name = SLOTS[slot];
      b.textContent = SLOTS[slot];
      b.title = 'stored as [[' + slot + ']]';
      b.onclick = () => insertCard(SLOTS[slot]);
      chips.append(b);
    });
  }
  LAST = null;
  renderList();
  check();
}

let SLOTS = {};

function insertCard(name){
  // Into the field last focused, at the caret. Falls back to the first empty
  // key point, then the first key point, so a click before touching anything
  // still lands somewhere sensible rather than doing nothing.
  let ta = LAST;
  if (!ta || !document.body.contains(ta)){
    const all = [...document.querySelectorAll('#kp textarea')];
    ta = all.find(x => !x.value.trim()) || all[0];
  }
  if (!ta) return;
  const s = ta.selectionStart ?? ta.value.length, e = ta.selectionEnd ?? s;
  const before = ta.value.slice(0, s), after = ta.value.slice(e);
  const pad = (before && !/\s$/.test(before)) ? ' ' : '';
  ta.value = before + pad + name + after;
  const caret = (before + pad + name).length;
  ta.focus(); ta.setSelectionRange(caret, caret);
  LAST = ta; DIRTY = true; check();
}

let checkTimer = null;
function check(){
  clearTimeout(checkTimer);
  checkTimer = setTimeout(async () => {
    const r = await fetch('/api/check', {method:'POST',
      headers:{'content-type':'application/json'},
      body: JSON.stringify({id: CUR, key_points: vals('#kp'), common_errors: vals('#ce')})});
    const d = await r.json();
    $('#notes').innerHTML =
      d.warnings.map(w => '<div class="note warn">' + esc(w) + '</div>').join('') +
      d.notes.map(n => '<div class="note">' + esc(n) + '</div>').join('');
    $('#save').disabled = !d.can_save;
    const n = d.warnings.length + d.notes.length, f = $('#flags');
    f.textContent = n ? n + (n === 1 ? ' note' : ' notes') : '';
    f.className = 'flags' + (d.warnings.length ? ' warn' : '');
    f.onclick = () => $('#notes').scrollIntoView({behavior: 'smooth', block: 'end'});
    // Light up the cards actually found in the text, so the substitution that
    // happens on save is visible while writing rather than a surprise after.
    const found = new Set(d.cards_detected || []);
    document.querySelectorAll('.chip').forEach(c =>
      c.classList.toggle('on', found.has(c.dataset.name)));
  }, 250);
}

async function submit(){
  if (!author()){ alert('Put your name in the box at the top so your work is credited.'); return; }
  const r = await fetch('/api/submit', {method:'POST',
    headers:{'content-type':'application/json'},
    body: JSON.stringify({id: CUR, author: author(),
      key_points: vals('#kp'), common_errors: vals('#ce')})});
  if (!r.ok){ const e = await r.json(); alert(e.error || 'could not save'); return; }
  DIRTY = false;
  $('#saved').textContent = 'Saved.';
  setTimeout(() => { $('#saved').textContent = ''; }, 2500);
  await refresh();
}

$('#cat').onchange = renderList;
$('#author').onchange = () => { saveName(); refresh(); };
window.onbeforeunload = () => DIRTY ? '' : undefined;
loadName();
refresh();
</script></body></html>
"""


# The adjudication form. A separate page from the rubric one on purpose: they
# ask for different things and sharing a template would mean branching inside
# every block. Both are served by the same program, chosen by task kind.
#
# Deliberately mobile-first. The point of deploying this is doing the 60
# verdicts away from the machine that holds the gold set, which in practice
# means a phone.
ADJUDICATE_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Judge adjudication</title>
<style>
:root{--paper:#F5F6F8;--panel:#fff;--ink:#131820;--soft:#59636F;--faint:#8A939E;
 --accent:#1C5A8C;--accent-bg:#EAF1F7;--ok:#1E7A4B;--rule:#D9DEE4;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --paper:#0F1319;--panel:#161B22;--ink:#E6EAF0;--soft:#9BA5B2;--faint:#6B7683;
 --accent:#6FA8D6;--accent-bg:#16232E;--ok:#5BB98B;--rule:#2A313A}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
 font-size:16px;line-height:1.5}
header{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--rule);
 padding:.6rem .9rem;display:flex;gap:.7rem;align-items:center;flex-wrap:wrap;z-index:5}
header b{font-size:.9rem}
main{max-width:44rem;margin:0 auto;padding:1rem .9rem 5rem}
h2{font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);
 margin:1.3rem 0 .4rem;font-weight:600}
pre{font-family:var(--mono);font-size:.8rem;line-height:1.45;white-space:pre-wrap;
 background:var(--panel);border:1px solid var(--rule);padding:.7rem;margin:0;border-radius:4px}
.answer{border:0;border-left:3px solid var(--accent);border-radius:0;background:transparent;
 padding:.3rem 0 .3rem .8rem;font-size:.9rem}
ul{margin:.2rem 0;padding-left:1.2rem;color:var(--soft);font-size:.9rem}
label.opt{display:flex;gap:.6rem;align-items:flex-start;padding:.7rem .8rem;margin-bottom:.4rem;
 background:var(--panel);border:1px solid var(--rule);border-radius:4px;cursor:pointer}
label.opt:has(input:checked){border-color:var(--accent);background:var(--accent-bg)}
label.opt input{margin:.25rem 0 0;width:1.1rem;height:1.1rem;flex:none}
.hint{font-size:.8rem;color:var(--faint);margin:.5rem 0 0}
input[type=text]{width:100%;padding:.55rem .6rem;font:inherit;font-size:.9rem;
 background:var(--panel);color:var(--ink);border:1px solid var(--rule);border-radius:4px}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--panel);
 border-top:1px solid var(--rule);padding:.7rem .9rem;display:flex;gap:.6rem;
 max-width:44rem;margin:0 auto}
button{font:inherit;font-size:.9rem;padding:.6rem 1rem;border-radius:4px;
 border:1px solid var(--rule);background:var(--panel);color:var(--ink);cursor:pointer}
button.go{background:var(--accent);border-color:var(--accent);color:#fff;flex:1;font-weight:600}
.done{color:var(--ok);font-size:.8rem}
.mut{color:var(--faint);font-size:.78rem;font-family:var(--mono)}
</style></head><body>
<header><b id="prog">-</b><span class="mut" id="meta"></span>
  <input type="text" id="who" placeholder="your name" style="width:9rem;margin-left:auto">
</header>
<main id="main"></main>
<div class="bar"><button id="skip">Skip</button><button id="go" class="go">Save &amp; next</button></div>
<script>
const $=s=>document.querySelector(s);
const esc=s=>{const d=document.createElement('div');d.textContent=s??'';return d.innerHTML};
let T=[],i=0;
const who=()=>$('#who').value.trim();
$('#who').value=localStorage.getItem('adjWho')||'';
$('#who').oninput=()=>localStorage.setItem('adjWho',who());

async function load(){
  const r=await fetch('/api/tasks'+(who()?'?author='+encodeURIComponent(who()):''));
  const d=await r.json();T=d.tasks;
  const first=T.findIndex(t=>!t.done);i=first===-1?0:first;render();
}
function render(){
  const t=T[i];
  if(!t){$('#main').innerHTML='<p>Nothing queued.</p>';return}
  $('#prog').textContent=T.filter(x=>x.done).length+'/'+T.length;
  $('#meta').textContent=t.record_id+' · '+t.arm+(t.done?' · saved':'');
  $('#main').innerHTML=
    '<h2>The situation</h2><pre>'+esc(t.question)+'</pre>'+
    '<h2>The correct line</h2><ul>'+(t.key_points||[]).map(k=>'<li>'+esc(k)+'</li>').join('')+'</ul>'+
    '<h2>The answer under review</h2><pre class="answer">'+esc(t.answer)+'</pre>'+
    '<h2>Which of these mistakes does the answer make?</h2>'+
    '<p class="hint">Judge the <b>play</b>, not the wording. Check an item if the answer '+
    'does that thing. An action list like <code>PASS</code> still "takes 2 from the '+
    'Vanguard" even though it never says those words.</p>'+
    (t.common_errors||[]).map((e,n)=>
      '<label class="opt"><input type="checkbox" class="e" value="'+(n+1)+'">'+
      '<span><b>'+(n+1)+'.</b> '+esc(e)+'</span></label>').join('')+
    '<p class="hint"><b>Check none if it makes none of them</b> — a real verdict and a '+
    'common one, not a skip. You are not being asked whether a judge was right.</p>'+
    '<label class="opt"><input type="checkbox" id="notcovered"><span><b>Bad, but not '+
    'for any reason above.</b> The answer is wrong or useless — does nothing, repeats '+
    'itself, plays something illegal — and none of the listed mistakes describe it. '+
    'This measures gaps in the rubric, so flagging it is worth as much as the '+
    'checkboxes.</span></label>'+
    '<label class="opt"><input type="checkbox" id="unsure"><span>Genuinely ambiguous — '+
    'I could argue it either way</span></label>'+
    '<input type="text" id="note" placeholder="what the answer actually did, if a box '+
    'above does not capture it">';
}
$('#skip').onclick=()=>{i=Math.min(T.length-1,i+1);render();scrollTo(0,0)};
$('#go').onclick=async()=>{
  const t=T[i];if(!t)return;
  const present=[...document.querySelectorAll('.e')].filter(c=>c.checked).map(c=>+c.value);
  const r=await fetch('/api/adjudicate',{method:'POST',
    headers:{'content-type':'application/json'},
    body:JSON.stringify({key:t.key,errors_present:present,author:who(),
      unsure:$('#unsure').checked,not_covered:$('#notcovered').checked,
      note:$('#note').value})});
  const d=await r.json();
  if(!d.ok){alert(d.error||'save failed');return}
  t.done=true;
  const nxt=T.findIndex((x,n)=>n>i&&!x.done);
  i=nxt===-1?Math.min(T.length-1,i+1):nxt;render();scrollTo(0,0);
};
load();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path,
                        default=Path(os.environ.get("TASKS_FILE", "data/gold/worksheets/tasks.json")),
                        help="task file from `author_rubrics.py --export-tasks`")
    parser.add_argument("--submissions", type=Path,
                        default=Path(os.environ.get("SUBMISSIONS_FILE", "data/submissions.jsonl")),
                        help="append-only log of what contributors submit")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--token", default=os.environ.get("RUBRIC_TOKEN"),
                        help="required unless --no-token. Generated if not supplied "
                             "and the host is not loopback.")
    parser.add_argument("--no-token", action="store_true",
                        help="serve without auth. Only sane on loopback.")
    args = parser.parse_args()

    tasks, kind = load_tasks(args.tasks)

    # A public bind with no token would put the form — and every contributor's
    # work — behind nothing at all. Refuse rather than warn.
    public = args.host not in ("127.0.0.1", "localhost", "::1")
    token = None if args.no_token else (args.token or (secrets.token_urlsafe(12) if public else None))
    if public and args.no_token:
        raise SystemExit("--no-token with a public --host would expose the form to anyone. "
                         "Drop --no-token, or bind to 127.0.0.1.")

    print(f"{len(tasks)} tasks from {args.tasks}")
    print(f"submissions -> {args.submissions.resolve()}")
    base = f"http://{'localhost' if not public else args.host}:{args.port}/"
    print(f"\n  {base}{'?t=' + token if token else ''}\n")
    if token and not args.token:
        print(f"generated token: {token}  (set RUBRIC_TOKEN to keep it stable across restarts)\n")

    import uvicorn
    uvicorn.run(build_app(tasks, args.submissions, token, kind), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
