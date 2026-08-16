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


def load_tasks(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload["tasks"] if isinstance(payload, dict) else payload
    if not tasks:
        raise SystemExit(f"{path} has no tasks")
    return tasks


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


def build_app(tasks: list[dict], submissions_path: Path, token: str | None):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    app = FastAPI(title="magic-llm rubrics")
    by_id = {t["id"]: t for t in tasks}
    categories = sorted({t["category"] for t in tasks})

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
        return INDEX_HTML

    @app.get("/api/tasks")
    def api_tasks(author: str = ""):
        """Task list plus this author's own progress.

        Progress is per author so two people working the same category each
        see their own remaining count. `done_by_anyone` is separate and only
        dims a row — it is a hint, not a lock, because a second opinion on a
        rubric is useful rather than wasted.
        """
        subs = read_submissions(submissions_path)
        mine = {s["id"] for s in subs if s.get("author") == author and s.get("key_points")}
        anyone = {s["id"] for s in subs if s.get("key_points")}
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
        stray = stray_names(t.get("question", ""), t.get("answer", ""), kp + ce)
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

    tasks = load_tasks(args.tasks)

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
    uvicorn.run(build_app(tasks, args.submissions, token), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
