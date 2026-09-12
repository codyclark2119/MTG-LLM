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
import hmac
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
from common import (ACTION_GRAMMAR, PHASE_NAMES,  # noqa: E402
                    POSITION_CATEGORIES, POSITION_REVIEW_KINDS,
                    lint_common_errors, protocol_restatements, stray_names,
                    templatize, untemplatize, verdict_is_current)

# One writer at a time. Submissions append, and two contributors finishing a
# question in the same instant would otherwise interleave a line.
_WRITE_LOCK = threading.Lock()

MAX_AUTHOR_LENGTH = 60

# A scenario id becomes a record id and then a file key, so it is
# constrained at the door rather than sanitised later.
_SCENARIO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{2,63}$")
MAX_RUBRIC_ITEMS = 20
MAX_RUBRIC_ITEM_LENGTH = 1000
MAX_NOTE_LENGTH = 2000

# --- The public boundary ----------------------------------------------------
#
# This is the only server in this repo that is meant to be reachable from the
# internet, so its auth is the one that has to be right rather than merely
# present.

COOKIE_NAME = "mlrt"          # unchanged: existing contributors hold this one
COOKIE_MAX_AGE = 86400 * 30   # a contributor follows the link once a month

# Everything a public form accepts is attacker-controlled and arrives before
# any validator runs, so the size cap belongs at the door. 256 KB is far above
# the largest real submission (a 12-step scenario with full boards) and far
# below anything that costs a 256MB fly machine its memory.
MAX_BODY_BYTES = 256 * 1024

# A position's own bounds, expressed in the constants the rubric form already
# uses rather than a second vocabulary: a key point here is the same kind of
# object as a key point there, and two independently-maintained limits for one
# concept is how they end up disagreeing.
MAX_SCENARIO_STEPS = 12


def _secret_matches(supplied, expected: str | None) -> bool:
    """Constant-time comparison for a shared secret.

    `!=` leaks the length of the matching prefix through timing, which is the
    whole reason `chat_auth` compares its password and its session signature
    with `compare_digest`. This server had the same shared-secret check written
    with `!=`, which is the same bug with a public URL in front of it.

    A supplied value that cannot even be encoded (a lone surrogate from a
    hand-built query string) is a mismatch, not a 500.
    """
    if not expected or not isinstance(supplied, str):
        return False
    try:
        return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))
    except (UnicodeEncodeError, TypeError):
        return False


def cookie_flags(request=None, force_secure: bool = False) -> dict:
    """Flags for the contributor cookie. Public deployments get Secure.

    `httponly` keeps the token out of `document.cookie`, so an injected script
    cannot read the shared secret back out of the browser. `samesite="lax"`
    rather than `"strict"` deliberately: a contributor arrives by following a
    link from a mail or a chat window, and Strict would withhold the cookie on
    exactly that top-level navigation and lock them out of the form they were
    just sent. `path="/"` is explicit because the form spans `/rubric`,
    `/position`, `/adjudicate` and `/api/*`, and a default path derived from
    whichever URL happened to set the cookie would scope it to one of them.

    `secure` is decided from the request rather than from a flag someone has to
    remember: fly terminates TLS and forwards `x-forwarded-proto: https`, so
    the public deployment sets Secure and a LAN/loopback HTTP session does not
    — where Secure would mean the cookie is set and never sent back.
    """
    proto = ""
    if request is not None:
        proto = (request.headers.get("x-forwarded-proto")
                 or getattr(request.url, "scheme", "") or "").lower()
    return {"httponly": True, "samesite": "lax",
            "secure": bool(force_secure or proto == "https"),
            "max_age": COOKIE_MAX_AGE, "path": "/"}


def _text_list(value, field: str) -> tuple[list[str], list[str]]:
  """Normalize a submitted text list and report malformed or oversized input."""
  if not isinstance(value, list):
    return [], [f"{field} must be a list"]
  if len(value) > MAX_RUBRIC_ITEMS:
    return [], [f"{field} must contain at most {MAX_RUBRIC_ITEMS} items"]
  out, problems = [], []
  for i, item in enumerate(value, 1):
    if not isinstance(item, str):
      problems.append(f"{field}[{i}] must be text")
      continue
    item = item.strip()
    if len(item) > MAX_RUBRIC_ITEM_LENGTH:
      problems.append(f"{field}[{i}] exceeds {MAX_RUBRIC_ITEM_LENGTH} characters")
    elif item:
      out.append(item)
  return out, problems


def validate_rubric_input(body: dict, task_ids: set[str]) -> list[str]:
  """Validate the public rubric submission contract before persistence."""
  problems = []
  if body.get("id") not in task_ids:
    problems.append("unknown id")
  author = body.get("author")
  if not isinstance(author, str) or not author.strip():
    problems.append("author is required for attribution")
  elif len(author.strip()) > MAX_AUTHOR_LENGTH:
    problems.append(f"author exceeds {MAX_AUTHOR_LENGTH} characters")
  key_points, kp_problems = _text_list(body.get("key_points"), "key_points")
  common_errors, ce_problems = _text_list(body.get("common_errors"), "common_errors")
  problems.extend(kp_problems + ce_problems)
  if len(key_points) < 2:
    problems.append("at least 2 key points are needed")
  return problems


def validate_adjudication_input(body: dict, task: dict) -> list[str]:
  """Validate a blind adjudication submission before persistence."""
  problems = []
  if body.get("key") != task.get("key"):
    problems.append("unknown key")
  author = body.get("author")
  if not isinstance(author, str) or not author.strip():
    problems.append("author is required for attribution")
  elif len(author.strip()) > MAX_AUTHOR_LENGTH:
    problems.append(f"author exceeds {MAX_AUTHOR_LENGTH} characters")
  present = body.get("errors_present")
  if not isinstance(present, list):
    problems.append("errors_present must be a list")
  else:
    if len(present) > len(task.get("common_errors") or []):
      problems.append("errors_present contains too many entries")
    for n in present:
      if not isinstance(n, int) or isinstance(n, bool):
        problems.append("errors_present must contain integers")
        break
    if all(isinstance(n, int) and not isinstance(n, bool) for n in present):
      n_errors = len(task.get("common_errors") or [])
      if any(n < 1 or n > n_errors for n in present):
        problems.append(f"error numbers must be 1..{n_errors}")
  note = body.get("note", "")
  if not isinstance(note, str):
    problems.append("note must be text")
  elif len(note.strip()) > MAX_NOTE_LENGTH:
    problems.append(f"note exceeds {MAX_NOTE_LENGTH} characters")
  elif (body.get("errors_present") or body.get("not_covered")) and not note.strip():
    problems.append("a note is required when errors are selected or the answer is not covered")
  for field in ("unsure", "not_covered"):
    if field in body and not isinstance(body[field], bool):
      problems.append(f"{field} must be boolean")
  return problems


def validate_reference_input(body: dict, record_ids: set[str]) -> list[str]:
  """Validate reference-line submissions from the public form."""
  problems = []
  rid = (body.get("record_id") or "").strip()
  if rid not in record_ids:
    problems.append("unknown record")
  author = body.get("author")
  if not isinstance(author, str) or not author.strip():
    problems.append("author is required for attribution")
  elif len(author.strip()) > MAX_AUTHOR_LENGTH:
    problems.append(f"author exceeds {MAX_AUTHOR_LENGTH} characters")
  lines = body.get("reference_actions")
  if not isinstance(lines, list):
    problems.append("reference_actions must be a list")
    return problems
  clean = [str(x).strip() for x in lines if str(x).strip()]
  if len(clean) > 60:
    problems.append("at most 60 actions")
  if any(len(x) > 200 for x in clean):
    problems.append("an action line is over 200 characters")
  return problems


def validate_review_input(body: dict, record_ids: set[str]) -> list[str]:
  """Validate board-review flag submissions from the public form."""
  problems = []
  rid = (body.get("record_id") or "").strip()
  if rid not in record_ids:
    problems.append("unknown record")
  author = body.get("author")
  if not isinstance(author, str) or not author.strip():
    problems.append("author is required for attribution")
  elif len(author.strip()) > MAX_AUTHOR_LENGTH:
    problems.append(f"author exceeds {MAX_AUTHOR_LENGTH} characters")
  kind = (body.get("kind") or "").strip()
  if kind and kind not in POSITION_REVIEW_KINDS:
    problems.append("kind must be one of: " + ", ".join(sorted(POSITION_REVIEW_KINDS)))
  note = body.get("note")
  if note is not None and not isinstance(note, str):
    problems.append("note must be text")
    return problems
  if kind and not (note or "").strip():
    problems.append("say what needs changing")
  return problems

# Bumped whenever the adjudication form's WORDING changes in a way that could
# move a verdict. The first eight verdicts were collected under v1, which asked
# "which of these does it commit?" and did not say to judge the play rather than
# the wording; they under-fire on action-list answers as a result (Section
# 21.47). Segmenting on this is the only way to tell that apart from a real
# change in the answers, so it rides on every submission the way `author` does.
# 3: the form shows PROTOCOL_ERRORS alongside the position's own `common_errors`,
# in the judge's numbering, and records `n_shown` so `score_run` can tell which
# entries a verdict was actually offered (Section 21.75).
# 4: the note asks for the REASONING on every faulted verdict, where v3 asked
# for it only when no box applied. An absent note therefore means different
# things under the two versions -- "a box covered it" vs "the reviewer did not
# explain" -- so the version is what tells them apart (Section 21.77).
#
# Deliberately NOT read by `verdict_is_current`, which compares the entry count
# shown. The checkbox verdict a v3 reviewer gave is still a correct verdict, so
# bumping this must not discard their work; only a rubric that GREW invalidates.
ADJUDICATION_FORM_VERSION = 4


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


def group_adjudication_tasks(tasks: list[dict]) -> list[dict]:
    """Collapse per-arm rows into a single question page while keeping data keyed by arm.

    The public form shows one question and every model answer next to each other,
    but each submission still persists as its own per-arm verdict. This preserves
    the blind-review model while reducing the page churn for the reviewer.
    """
    grouped: dict[str, dict] = {}
    for task in tasks:
        rid = task.get("record_id") or (task.get("key") or "").split("::")[0]
        if rid not in grouped:
            grouped[rid] = {
                "record_id": rid,
                "question": task.get("question"),
                "key_points": task.get("key_points") or [],
                "n_strategy": task.get("n_strategy"),
                "common_errors": task.get("common_errors") or [],
                # Per-RECORD, not per-arm: the correct line is a property of the
                # board, not of any model's attempt at it.
                "legal_actions": task.get("legal_actions") or [],
                "reference_actions": task.get("reference_actions") or [],
                "done": False,
                "arms": [],
            }
        # A reference-only record carries no arm and no answer: it exists so the
        # correct line can be authored for a board the adjudication sample never
        # drew (21.88). Adding it to `arms` would render an empty grading panel
        # and invite a verdict on nothing.
        if not task.get("reference_only"):
            grouped[rid]["arms"].append(task)
    out = []
    for item in grouped.values():
        if item["arms"]:
            item["done"] = all(a.get("done") for a in item["arms"])
        else:
            # Nothing to grade, so "done" can only mean the reference is written.
            item["done"] = bool(item.get("reference_actions"))
            item["reference_only"] = True
        out.append(item)
    return out


def append_submission(path: Path, row: dict) -> None:
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def prioritize_tasks(tasks: list[dict], done_key: str = "done") -> list[dict]:
    """Keep work still needing a verdict at the front of the queue.

    The queue is intentionally ordered by current status first, with a stable
    sort preserving the original task order for items in the same bucket. That
    way a stale row cannot sit at the top of the list while a genuinely open
    item is hidden behind it, and the web client starts on the first task the
    reviewer still needs to do.
    """
    return sorted(tasks, key=lambda t: bool(t.get(done_key)))


# Shown at `/` only when BOTH forms are deployed. Deliberately plain: it exists
# to route, and every pixel of design here is a pixel not spent on the two forms
# that do the work. No fetch, no state — a wrong turn costs one click.
# Who the reviewer is, stored once. Three pages had three localStorage keys for
# the same identity — `adjWho`, `author`, `mlr_author` — so a name typed in one
# view was invisible in the next, and the first nine submissions from the new
# authoring page carried an empty author. Attribution rides on each submission
# (that is the invariant), so an empty one is a lost verdict, not a cosmetic
# gap. Legacy keys are read once and migrated (Section 21.91).
AUTHOR_JS = """
function mlAuthorGet(){
  const legacy=['mlAuthor','adjWho','author','mlr_author'];
  for(const k of legacy){
    try{const v=localStorage.getItem(k); if(v&&v.trim()){mlAuthorSet(v.trim());return v.trim()}}
    catch(e){}
  }
  return '';
}
function mlAuthorSet(v){
  try{localStorage.setItem('mlAuthor',v); localStorage.setItem('adjWho',v);
      localStorage.setItem('author',v); localStorage.setItem('mlr_author',v)}catch(e){}
}
"""

CHOOSE_HTML = """<meta name=viewport content="width=device-width,initial-scale=1">
<title>magic-llm</title>
<style>
 body{font:16px/1.5 system-ui,sans-serif;margin:0;padding:2rem 1.25rem;
      max-width:34rem;background:#fbfbfa;color:#1a1a1a}
 h1{font-size:1.25rem;margin:0 0 .25rem}
 p.sub{margin:0 0 1.75rem;color:#666;font-size:.9rem}
 a.card{display:block;padding:1rem 1.1rem;margin:0 0 .75rem;border:1px solid #ddd;
        border-radius:10px;background:#fff;text-decoration:none;color:inherit}
 a.card:hover{border-color:#999}
 a.card b{display:block;font-size:1.02rem;margin-bottom:.2rem}
 a.card span{color:#666;font-size:.87rem}
 @media(prefers-color-scheme:dark){
   body{background:#151515;color:#eee}
   a.card{background:#1e1e1e;border-color:#333}
   a.card:hover{border-color:#666}
   p.sub,a.card span{color:#9a9a9a}}
</style>
<h1>magic-llm</h1>
<p class=sub>Two forms are deployed. Pick one.</p>
<a class=card href="/adjudicate"><b>Judge adjudication</b>
<span>Read an answer and say which of the listed errors it commits.</span></a>
<a class=card href="/rubric"><b>Rubric authoring</b>
<span>Write the key points and common errors for a question.</span></a>
<a class=card href="/scenario"><b>Author a scenario</b><span>A turn as a sequence of boards. Each step is a whole board and adding one copies the last, so you edit what changed. The only way to write a line that crosses an opponent's window.</span></a>
<a class=card href="/reference"><b>Correct lines</b><span>Write the 100%-correct answer for a board, in the action grammar. Separate from grading so a line can be revisited until it is right.</span></a>
<a class=card href="/position"><b>Author a position</b>
<span>Build a board and the decision it tests. Previews what the model sees.</span></a>
"""


# Position authoring, mobile-first. A BLANK form: it serves no tasks, reads no
# gold set, and reaches nothing — which makes it the least exposed of the three
# surfaces here, not the most. Submissions land in the same append-only log and
# are promoted locally by `positions.py --ingest`, so "gold" keeps meaning a
# person reviewed it.
#
# Card names are NOT validated here. That needs the 25MB card index, which is
# deliberately not deployable — so the form previews the rendered board (the
# feedback that actually helps while authoring) and name checking happens at
# ingest, where it can fail loudly without costing anyone a lost draft.
POSITION_HTML = """<meta name=viewport content="width=device-width,initial-scale=1">
<title>author a position</title>
<style>
 :root{--bg:#fbfbfa;--fg:#1a1a1a;--rule:#ddd;--paper:#fff;--dim:#666}
 @media(prefers-color-scheme:dark){:root{--bg:#151515;--fg:#eee;--rule:#333;--paper:#1e1e1e;--dim:#9a9a9a}}
 *{box-sizing:border-box}
 body{font:16px/1.5 system-ui,sans-serif;margin:0;padding:1rem .9rem 4rem;background:var(--bg);color:var(--fg)}
 h1{font-size:1.1rem;margin:0 0 .15rem}
 p.sub{margin:0 0 1rem;color:var(--dim);font-size:.85rem}
 label{display:block;margin:.7rem 0 .2rem;font-size:.82rem;color:var(--dim)}
 input,textarea,select{width:100%;font:inherit;font-size:16px;padding:.5rem .55rem;
   border:1px solid var(--rule);border-radius:8px;background:var(--paper);color:inherit}
 textarea{min-height:4.2rem;resize:vertical;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:14px}
 .row{display:flex;gap:.5rem}.row>div{flex:1}
 fieldset{border:1px solid var(--rule);border-radius:10px;margin:1rem 0;padding:.3rem .8rem 1rem}
 legend{font-size:.78rem;color:var(--dim);padding:0 .35rem}
 button{font:inherit;padding:.6rem 1rem;border-radius:8px;border:1px solid var(--rule);
   background:var(--paper);color:inherit}
 button.primary{background:#2a6df4;color:#fff;border-color:#2a6df4}
 pre{white-space:pre-wrap;background:var(--paper);border:1px solid var(--rule);
   border-radius:8px;padding:.6rem;font-size:13px;overflow-x:auto}
 .bar{position:sticky;bottom:0;background:var(--bg);border-top:1px solid var(--rule);
   padding:.6rem 0;display:flex;gap:.5rem;align-items:center}
 .msg{font-size:.85rem}
</style>
<h1>Author a position</h1>
<p class=sub>One board, one decision. Preview renders exactly what the model will see.</p>

<label>Your name</label><input id=author placeholder="for attribution">

<fieldset><legend>situation</legend>
<div class=row>
 <div><label>Turn</label><input id=turn inputmode=numeric value="4"></div>
 <div><label>Phase</label><select id=phase>
   <option>precombat main</option><option>postcombat main</option>
   <option>declare attackers</option><option>declare blockers</option>
   <option>upkeep</option><option>opening hand, on the play</option>
   <option>beginning of combat</option><option>end step</option></select></div>
</div>
<div class=row>
 <div><label>Your life</label><input id=you_life inputmode=numeric value="20"></div>
 <div><label>Opp life</label><input id=opp_life inputmode=numeric value="20"></div>
</div>
<label>Mana available to you</label><input id=mana_available placeholder="{R}{G}{G}">
</fieldset>

<fieldset><legend>board — one permanent per line, e.g. <code>Forest untapped x2</code></legend>
<label>Your battlefield</label><textarea id=you_battlefield></textarea>
<label>Opponent's battlefield</label><textarea id=opp_battlefield></textarea>
<label>Your hand (one card per line)</label><textarea id=you_hand></textarea>
<div class=row>
 <div><label>Your library</label><input id=you_library inputmode=numeric value="30"></div>
 <div><label>Opp hand size</label><input id=opp_hand_count inputmode=numeric value="3"></div>
</div>
</fieldset>

<fieldset><legend>the decision</legend>
<label>Legal actions (one per line — every play available, not just the right one)</label>
<textarea id=legal_actions></textarea>
<label>Key points (the correct line, one per line)</label><textarea id=key_points></textarea>
<label>Common errors (plausible wrong answers, one per line)</label><textarea id=common_errors></textarea>
<label>Answer in prose</label><textarea id=answer></textarea>
<div class=row>
 <div><label>Category</label><input id=category placeholder="payment"></div>
 <div><label>Difficulty</label><select id=difficulty>
   <option>basic</option><option selected>intermediate</option><option>advanced</option></select></div>
</div>
</fieldset>

<div class=bar>
  <button onclick="preview()">Preview</button>
  <button class=primary onclick="save()">Submit</button>
  <span class=msg id=msg></span>
</div>
<pre id=out></pre>

<script>
const F=['turn','phase','you_life','opp_life','mana_available','you_battlefield',
 'opp_battlefield','you_hand','you_library','opp_hand_count','legal_actions',
 'key_points','common_errors','answer','category','difficulty'];
const $=s=>document.querySelector(s);
function body(){const b={};F.forEach(k=>b[k]=$('#'+k).value);b.author=$('#author').value;return b;}
function tok(){const m=location.search.match(/[?&]t=([^&]+)/);return m?'?t='+m[1]:'';}
async function post(path,b){
  const r=await fetch(path+tok(),{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify(b)});return r.json();}
async function preview(){
  const d=await post('/api/position/preview',body());
  $('#out').textContent=d.rendered||'';
  $('#msg').textContent=(d.problems&&d.problems.length)?d.problems.join(' | '):'looks well-formed';}
async function save(){
  if(!$('#author').value){$('#msg').textContent='add your name first';return;}
  const d=await post('/api/position',body());
  $('#msg').textContent=d.ok?('submitted — '+d.total+' on file'):(d.problems||['failed']).join(' | ');
  if(d.ok){['you_battlefield','opp_battlefield','you_hand','legal_actions','key_points',
            'common_errors','answer'].forEach(k=>$('#'+k).value='');$('#out').textContent='';}}
try{const w=localStorage.getItem('posWho');if(w)$('#author').value=w;}catch(e){}
$('#author').addEventListener('change',()=>{try{localStorage.setItem('posWho',$('#author').value);}catch(e){}});
</script>
"""


def validate_position_input(body: dict) -> list[str]:
  """Bounds for a publicly submitted position draft.

  Reuses MAX_AUTHOR_LENGTH, MAX_RUBRIC_ITEMS and MAX_RUBRIC_ITEM_LENGTH rather
  than introducing a second set of limits: a key point submitted here is the
  same kind of object as a key point submitted to the rubric form, and two
  independently maintained limits for one concept is how they end up
  disagreeing (this repo's "one name, two meanings" trap, in the numbers).

  `/api/position` previously checked only that an author was present and that
  there were two key points. Everything else — the author's length, how many
  key points, how long each one is, how large the board fields are — was
  unbounded on a public endpoint that appends every accepted body to a file.

  Returns problems; empty means acceptable. This is a SIZE gate, not a
  semantic one: what a position must MEAN is `validate_position`'s job at
  ingest, and it stays there because promotion is local and reviewed.
  """
  problems = []
  author = body.get("author")
  if not isinstance(author, str) or not author.strip():
    problems.append("name required for attribution")
  elif len(author.strip()) > MAX_AUTHOR_LENGTH:
    problems.append(f"author exceeds {MAX_AUTHOR_LENGTH} characters")

  for field in ("key_points", "common_errors", "legal_actions",
                "reference_actions"):
    value = body.get(field)
    if value is None or value == "":
      continue
    # The form posts these as newline-joined text; `position_from_form`
    # splits them. Bound both shapes, because the endpoint accepts JSON and a
    # client is not obliged to use the form.
    items = value.splitlines() if isinstance(value, str) else value
    if not isinstance(items, list):
      problems.append(f"{field} must be text or a list")
      continue
    if len(items) > MAX_RUBRIC_ITEMS:
      problems.append(f"{field} must contain at most {MAX_RUBRIC_ITEMS} items")
      continue
    for i, item in enumerate(items, 1):
      if not isinstance(item, str):
        problems.append(f"{field}[{i}] must be text")
      elif len(item) > MAX_RUBRIC_ITEM_LENGTH:
        problems.append(
          f"{field}[{i}] exceeds {MAX_RUBRIC_ITEM_LENGTH} characters")

  # The free-text board fields. Same per-item ceiling; a board is prose, so
  # it is bounded as one item rather than as a list.
  for field in ("question", "board", "you_battlefield", "opp_battlefield",
                "you_hand", "known_information", "notes", "answer"):
    value = body.get(field)
    if value is None:
      continue
    if not isinstance(value, str):
      problems.append(f"{field} must be text")
    elif len(value) > MAX_RUBRIC_ITEM_LENGTH * MAX_RUBRIC_ITEMS:
      problems.append(
        f"{field} exceeds {MAX_RUBRIC_ITEM_LENGTH * MAX_RUBRIC_ITEMS} characters")
  return problems


def build_app(task_sets, submissions_path: Path, token: str | None,
              kind: str | None = None, export_token: str | None = None,
              secure_cookies: bool = False):
    """One app, one or both forms.

    `task_sets` maps kind -> tasks. Both forms already used disjoint endpoints
    apart from `/` and `/api/tasks`, so serving both needs a chooser at the root
    and one extra route rather than two applications: a second app would mean a
    second port, a second token and a second volume mount for one shared
    submissions log.

    A list is still accepted for the single-form case, so existing callers and
    tests keep working — `kind` then says which form it is.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import (HTMLResponse, JSONResponse, PlainTextResponse,
                                   RedirectResponse)

    if isinstance(task_sets, list):
        task_sets = {kind or "rubric": task_sets}
    kinds = sorted(task_sets)

    app = FastAPI(title="magic-llm " + "+".join(kinds))
    rubric_tasks = task_sets.get("rubric") or []
    adj_tasks = task_sets.get("adjudication") or []
    # Two id spaces. Kept apart because a rubric task is keyed by `id` and an
    # adjudication task by `key`, and one dict would let a collision silently
    # serve the wrong form's task.
    by_id = {t["id"]: t for t in rubric_tasks}
    # Reference-only rows are excluded: they have no answer, so a verdict on one
    # would be a verdict about nothing.
    by_key = {t["key"]: t for t in adj_tasks if not t.get("reference_only")}
    # Reference lines are per-BOARD, so /api/reference validates the record id
    # against the same served tasks rather than trusting the client's string.
    by_record = {t.get("record_id") or (t.get("key") or "").split("::")[0]
                 for t in adj_tasks}

    @app.get("/reference", response_class=HTMLResponse)
    def reference_page():
        """Authoring the correct line, separate from grading it (21.90)."""
        return REFERENCE_HTML

    @app.get("/scenario", response_class=HTMLResponse)
    def scenario_page():
        """Authoring a turn as a sequence of boards (21.96)."""
        return SCENARIO_HTML

    @app.get("/api/categories")
    def api_categories():
        """The position category vocabulary, for the authoring dropdown."""
        return {"categories": list(POSITION_CATEGORIES)}

    @app.get("/api/reference-boards")
    def api_reference_boards():
        """One row per BOARD, deduplicated across arms.

        `tasks.json` is keyed (record, arm) because grading is per answer.
        Authoring is per board, so the arms collapse here rather than in the
        export — one file still ships, and the deploy boundary does not grow.

        Ordered so unwritten boards come first and scenario steps stay in
        sequence, which is the order someone authoring works in.
        """
        seen: dict[str, dict] = {}
        for t in adj_tasks:
            rid = t.get("record_id") or (t.get("key") or "").split("::")[0]
            if rid in seen:
                continue
            seen[rid] = {
                "record_id": rid,
                "question": t.get("question"),
                "key_points": t.get("key_points") or [],
                "legal_actions": t.get("legal_actions") or [],
                "reference_actions": t.get("reference_actions") or [],
                "category": t.get("category"),
                "review": t.get("review"),
            }
        rows = sorted(seen.values(),
                      key=lambda r: (bool(r["reference_actions"]), r["record_id"]))
        return {"boards": rows, "total": len(rows)}

    @app.get("/api/grammar")
    def api_grammar():
        """The action grammar, for the reference-authoring box.

        The MODEL is shown this in every gameplay prompt; the reviewer authoring
        the correct line was not, and the first reference submitted invented
        `END PHASE <step>` and used `BLOCK`'s arrow on `ATTACK` — both
        reasonable guesses, neither in the grammar (Section 21.86). Served from
        `common.ACTION_GRAMMAR` so the form cannot drift from what the model is
        told, which is the same one-definition rule `build_rag_messages` follows.
        """
        return {"grammar": ACTION_GRAMMAR, "phases": list(PHASE_NAMES),
                "review_kinds": POSITION_REVIEW_KINDS}
    tasks = rubric_tasks or adj_tasks
    categories = sorted({t.get("category") or "" for t in rubric_tasks})

    @app.middleware("http")
    async def auth(request: Request, call_next):
        # /healthz is exempt on purpose: a platform health check arrives with
        # no token and no cookie, so gating it would leave every machine
        # marked unhealthy and the app permanently down. It exposes only a
        # task count, which is not sensitive.
        if request.url.path == "/healthz":
            return await call_next(request)

        # Size cap first, and on every request including an unauthenticated
        # one: refusing a 50 MB body after parsing it is not a refusal.
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return PlainTextResponse(
                f"request body exceeds {MAX_BODY_BYTES} bytes", 413)

        if not token:
            return await call_next(request)

        query_token = request.query_params.get("t")
        supplied = (query_token
                    or request.cookies.get(COOKIE_NAME)
                    or request.headers.get("x-token"))
        if not _secret_matches(supplied, token):
            # compare_digest, not `!=`. See _secret_matches.
            return PlainTextResponse("This link needs its access token. Use the full "
                                     "URL you were sent.", 401)

        # BOOTSTRAP, THEN GET THE TOKEN OUT OF THE URL. It used to be left
        # there: the contributor's address bar, their history, every Referer
        # header the page emitted and any screenshot of the form all carried
        # the shared secret for the whole session. Exchange it for the cookie
        # and redirect to the same URL without `t`.
        if query_token is not None:
            if request.method in ("GET", "HEAD"):
                clean = request.url.remove_query_params("t")
                # 303, not 307: this is "the thing you asked for is over
                # there", and it must land as a GET.
                resp = RedirectResponse(str(clean), status_code=303)
                resp.set_cookie(COOKIE_NAME, token, **cookie_flags(request, secure_cookies))
                return resp
            # A non-GET carrying ?t= cannot be redirected without dropping the
            # body, so it is served and the cookie is set for next time.
            response = await call_next(request)
            response.set_cookie(COOKIE_NAME, token, **cookie_flags(request, secure_cookies))
            return response

        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def index():
        # The chooser whenever there is more than one thing to do — and
        # position authoring is always available, so a single-task deployment
        # still has two.
        return CHOOSE_HTML

    @app.get("/rubric", response_class=HTMLResponse)
    def rubric_form():
        if not rubric_tasks:
            return HTMLResponse("no rubric tasks were deployed", 404)
        return INDEX_HTML

    @app.get("/position", response_class=HTMLResponse)
    def position_form():
        # Always available: authoring needs no task file, which is why this is
        # the least exposed surface here rather than another thing to deploy.
        return POSITION_HTML

    @app.post("/api/position/preview")
    async def position_preview(request: Request):
        from common import position_from_form, render_position
        body = await request.json()
        pos = position_from_form(body)
        pos["id"] = pos.get("id") or "pos-preview"
        # Structural checks only. Card names are validated at INGEST, where the
        # 25MB index lives; doing it here would put the card corpus on a public
        # host to catch a typo that local promotion catches anyway.
        problems = []
        if not pos.get("legal_actions"):
            problems.append("no legal actions — the closed arm has nothing to choose from")
        if len(pos.get("key_points") or []) < 2:
            problems.append("a rubric needs at least 2 key points")
        if not pos.get("battlefield") and "opening hand" not in (pos.get("phase") or ""):
            problems.append("no permanents on either battlefield")
        return {"rendered": render_position(pos), "problems": problems}

    @app.post("/api/position")
    async def position_submit(request: Request):
        """Append a drafted position to the submissions log. Never the gold set.

        The same invariant the other two forms keep: promotion is local and
        reviewed (`positions.py --ingest`), so "gold" goes on meaning a person
        looked at it. A server that writes positions.jsonl directly is the one
        thing `webui.py` does that makes it undeployable.
        """
        from common import position_from_form, render_position
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse({"ok": False, "problems": ["expected an object"]}, 400)
        # Bounded BEFORE anything is parsed or written. See
        # validate_position_input: this endpoint appends whatever it accepts.
        problems = validate_position_input(body)
        if problems:
            return JSONResponse({"ok": False, "problems": problems}, 400)
        author = (body.get("author") or "").strip()
        pos = position_from_form(body)
        if not pos.get("legal_actions") or len(pos.get("key_points") or []) < 2:
            return JSONResponse({"ok": False, "problems":
                                 ["needs legal actions and at least 2 key points"]}, 400)
        pos["author"] = author
        pos["kind"] = "position_draft"
        # Set here so a draft is promotable without hand-editing. The ID is NOT
        # set: `positions.py --ingest` assigns it from the category, which keeps
        # numbering under local control and stops two people submitting the same
        # id from a form that cannot see the gold set.
        pos["source"] = f"hand-authored (web form, {author})"
        pos["rubric_source"] = f"hand-authored ({author})"
        pos["rendered_preview"] = render_position(pos)
        pos["submitted"] = datetime.now(timezone.utc).isoformat()
        append_submission(submissions_path, pos)
        total = sum(1 for x in read_submissions(submissions_path)
                    if x.get("kind") == "position_draft")
        return {"ok": True, "total": total}

    @app.get("/adjudicate", response_class=HTMLResponse)
    def adjudicate_form():
        if not adj_tasks:
            return HTMLResponse("no adjudication tasks were deployed", 404)
        return ADJUDICATE_HTML

    @app.get("/api/tasks")
    def api_tasks(author: str = "", kind: str = ""):
        """Task list plus this author's own progress.

        Progress is per author so two people working the same category each
        see their own remaining count. `done_by_anyone` is separate and only
        dims a row — it is a hint, not a lock, because a second opinion on a
        rubric is useful rather than wasted.

        `kind` selects the form when both are deployed. It defaults to the only
        one present, so a single-form deployment and every existing client keep
        working without passing it.
        """
        which = kind or (kinds[0] if len(kinds) == 1 else "rubric")
        subs = read_submissions(submissions_path)
        if which == "adjudication":
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
            #
            # The RUBRIC goes stale independently of the text, and checking the
            # digest alone caught only half. `PROTOCOL_ERRORS` was added to the
            # judge and not to this form, so 27 verdicts were given against four
            # entries where the judge saw eleven — and since the answers never
            # changed, every one of those tasks would show done and be skipped
            # (Section 21.75). `verdict_is_current` compares both, and lives in
            # `common` because `adjudicate --status` asks the same question.
            live = {t["key"]: (_answer_sha(t.get("answer") or ""),
                               len(t.get("common_errors") or []))
                    for t in adj_tasks}
            mine = {s["key"] for s in subs
                    if s.get("author") == author and "errors_present" in s
                    and s["key"] in live
                    and verdict_is_current(s, *live[s["key"]])}
            tasks = [dict(t, done=t["key"] in mine) for t in adj_tasks]
            grouped = group_adjudication_tasks(tasks)
            return {"tasks": prioritize_tasks(grouped, done_key="done"),
                    "done_by_me": len(mine), "total": len(adj_tasks)}
        # Intersect with the CURRENT task list. The submissions log is
        # append-only and outlives any one export, so after an ingest the
        # questions it covers are promoted out of tasks.json while their rows
        # remain. Counting the raw log reported "10 / 28 done" against a task
        # list those ten had already left — phantom progress, and worst in
        # exactly the ingest -> re-export -> restart loop this is used in.
        live = {t["id"] for t in rubric_tasks}
        mine = {s["id"] for s in subs
                if s.get("author") == author and s.get("key_points")} & live
        anyone = {s["id"] for s in subs if s.get("key_points")} & live
        rows = [{"id": t["id"], "category": t["category"], "difficulty": t["difficulty"],
                 "question": t["question"][:110],
                 "done_by_me": t["id"] in mine,
                 "done_by_anyone": t["id"] in anyone} for t in rubric_tasks]
        return {"tasks": rows, "categories": categories,
                "done_by_me": len(mine), "total": len(rubric_tasks)}

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
        # Only on a board. A rules question's `common_errors` never meets
        # `PROTOCOL_ERRORS`, so there is nothing to duplicate (Section 21.126),
        # and the entries the task carries are the ones this author was shown.
        if t.get("kind") == "position":
            warnings += protocol_restatements(ce)
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
        problems = validate_rubric_input(body, set(by_id))
        if problems:
            code = 404 if problems == ["unknown id"] else 400
            return JSONResponse({"error": "; ".join(problems)}, code)
        kp = [s.strip() for s in body["key_points"] if s.strip()]
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
        task = by_key.get(tid)
        if not task:
          return JSONResponse({"error": "unknown key"}, 404)
        problems = validate_adjudication_input(body, task)
        if problems:
          code = 404 if problems == ["unknown key"] else 400
          return JSONResponse({"error": "; ".join(problems)}, code)
        present = body.get("errors_present")
        n_errors = len(by_key[tid].get("common_errors") or [])
        nums = sorted(set(present))
        append_submission(submissions_path, {
            "key": tid,
            "record_id": by_key[tid].get("record_id"),
            "arm": by_key[tid].get("arm"),
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
            "answer_sha": _answer_sha(by_key[tid].get("answer") or ""),
            "errors_present": nums,
            # How many rubric entries this reviewer was actually shown. A judge
            # charge above this number cannot be scored against them — they were
            # never offered it — and `score_run` excludes it rather than
            # counting it as a false positive (Section 21.75).
            "n_shown": n_errors,
            "blundered": bool(nums),
            "unsure": bool(body.get("unsure")),
            # "wrong, but none of the listed mistakes describe it" — the case
            # that produced the first disagreements: an answer of `PASS` or a
            # repeated action commits no CLAIM in the rubric while being
            # obviously bad. Recorded separately because it measures rubric
            # COVERAGE, which no judge number can (Section 21.47).
            "not_covered": bool(body.get("not_covered")),
            # 2000, not 500: from form_version 4 this is the reviewer's
            # reasoning rather than a one-line fallback, and one v3 note already
            # reached 461 of the old cap.
            "note": (body.get("note") or "").strip()[:2000],
            "author": (body.get("author") or "").strip()[:60],
            "form_version": ADJUDICATION_FORM_VERSION,
            "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return {"ok": True, "key": tid}

    @app.post("/api/reference")
    async def api_reference(request: Request):
        """Record the 100%-correct line for a board, in the action grammar.

        A property of the POSITION, not of any arm — so it is keyed by
        `record_id` and carries no `errors_present`, which is also how the
        ingest filters tell the three row kinds apart in the shared log.

        **Validated locally, not here.** The deployed container has no card
        index, no positions file and no action parser — `common.py` must stay
        pure stdlib, and the grammar lives in `gameplay/actions.py`. So this
        endpoint checks only shape and size; `positions.py --ingest-references`
        parses each line, requires every one to be legal on this board, and
        refuses the whole submission otherwise. That keeps the invariant the
        rubric form has always had: **the server never writes the gold set**,
        promotion is local and reviewed.

        Re-submitting is allowed and appends. The ingest takes the newest row
        per (record, author), so a correction does not need the old row deleted.
        """
        body = await request.json()
        problems = validate_reference_input(body, by_record)
        if problems:
          code = 404 if problems == ["unknown record"] else 400
          return JSONResponse({"error": "; ".join(problems)}, code)
        rid = (body.get("record_id") or "").strip()
        clean = [str(x).strip() for x in (body.get("reference_actions") or []) if str(x).strip()]
        append_submission(submissions_path, {
            "kind": "reference",
            "record_id": rid,
            "reference_actions": clean,
            "author": (body.get("author") or "").strip()[:60],
            "form_version": ADJUDICATION_FORM_VERSION,
            "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return {"ok": True, "record_id": rid, "n": len(clean)}

    @app.post("/api/review")
    async def api_review(request: Request):
        """Flag a board for editing, or clear the flag.

        A separate claim from the reference line: one says what the correct
        answer is, the other says the board itself needs work. Kept apart so
        flagging does not require inventing a line for a board you are flagging
        BECAUSE you cannot write one (Section 21.94).

        Written to the submissions log like everything else — the server never
        writes the gold set — and promoted by
        `positions.py --ingest-references`, which validates the kind.
        """
        body = await request.json()
        problems = validate_review_input(body, by_record)
        if problems:
          code = 404 if problems == ["unknown record"] else 400
          return JSONResponse({"error": "; ".join(problems)}, code)
        rid = (body.get("record_id") or "").strip()
        kind = (body.get("kind") or "").strip()
        note = (body.get("note") or "").strip()[:1000]
        append_submission(submissions_path, {
            "kind": "review",
            "record_id": rid,
            # Empty `review_kind` clears the flag, so a board can be unflagged
            # once fixed without editing the log.
            "review_kind": kind,
            "note": note,
            "author": (body.get("author") or "").strip()[:60],
            "form_version": ADJUDICATION_FORM_VERSION,
            "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return {"ok": True, "record_id": rid, "review_kind": kind}

    @app.post("/api/scenario")
    async def api_scenario(request: Request):
        """A whole multi-step scenario, authored as one submission.

        A scenario is the only way to write a line that crosses an opponent's
        window: the next step's board STATES what happened in it, rather than
        the answer assuming it (21.89, 21.92). The form authors each step as a
        complete board because that is what a person can check; the stored
        record keeps only what CHANGES between steps, and that diff is done at
        ingest by `turns.scenario_from_submission` — the browser never builds
        the schema (Section 21.96).

        Stored raw here. `positions.py --ingest-scenarios` builds it, runs
        `validate_scenario` and `check_reference` on every step, and refuses the
        whole thing if any step fails. The server never writes the gold set.
        """
        body = await request.json()
        problems = []
        sid = (body.get("scenario_id") or "").strip()
        if not sid:
            problems.append("scenario_id is required")
        elif not _SCENARIO_ID_RE.match(sid):
            problems.append("scenario_id may use letters, digits and hyphens only")
        author = body.get("author")
        if not isinstance(author, str) or not author.strip():
            problems.append("author is required for attribution")
        elif len(author.strip()) > MAX_AUTHOR_LENGTH:
            problems.append(f"author exceeds {MAX_AUTHOR_LENGTH} characters")
        steps = body.get("steps")
        if not isinstance(steps, list):
            problems.append("steps must be a list")
        elif len(steps) < 2:
            # The same rule `validate_scenario` enforces, said here so the
            # author hears it before typing a second board rather than after.
            problems.append("a scenario needs at least 2 steps — one step is a position")
        elif len(steps) > MAX_SCENARIO_STEPS:
            problems.append(f"at most {MAX_SCENARIO_STEPS} steps")
        elif not all(isinstance(x, dict) for x in steps):
            problems.append("each step must be an object")
        else:
            # Each step is a whole board, so each step gets a board's bounds.
            # The step COUNT was capped and the step CONTENTS were not, which
            # bounds the list and not the payload.
            for n, step in enumerate(steps, 1):
                for problem in validate_position_input(
                        {**step, "author": author if isinstance(author, str) else ""}):
                    if problem == "name required for attribution":
                        continue  # the author is on the envelope, not the step
                    problems.append(f"step {n}: {problem}")
        if problems:
            return JSONResponse({"error": "; ".join(problems)}, 400)
        append_submission(submissions_path, {
            "kind": "scenario",
            "scenario_id": sid,
            "category": (body.get("category") or "").strip(),
            "difficulty": (body.get("difficulty") or "").strip(),
            "steps": steps,
            "author": author.strip()[:MAX_AUTHOR_LENGTH],
            "form_version": ADJUDICATION_FORM_VERSION,
            "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return {"ok": True, "scenario_id": sid, "steps": len(steps)}

    @app.get("/api/export")
    def api_export(request: Request):
        """The raw submissions log, for pulling down and ingesting locally.

        A SEPARATE CREDENTIAL from the one that gates the form. The contributor
        token is shared by everyone who was sent the link, and it used to be
        enough to download the complete submission log — every other
        contributor's rubrics, verdicts, notes and names.

        That is not only a disclosure question; it is a measurement one. This
        project's per-author agreement breakdown (`eval.py --compare`) is worth
        something precisely because reviewers are independent evidence, and
        21.74 is the section about a reviewer who sees the answer key first
        ceasing to be that. A contributor who can read everyone else's
        submissions before writing their own is the same failure with a wider
        blast radius. `deploy/README.md` establishes contributors as trusted to
        USE the form ("a handful of trusted people"); it nowhere establishes
        that each is meant to hold the whole log.

        Fails closed: with a contributor token in force and no
        RUBRIC_EXPORT_TOKEN set, export is refused rather than falling back to
        the weaker secret. An auth layer whose default is "off" is the shape of
        every accidental exposure, and this repo already has one 5.0 GB
        near-miss on that theme. With no token at all the server is
        loopback-only by construction (a public bind without one is refused in
        `main`), so local export is unaffected.
        """
        if token and not _secret_matches(request.headers.get("x-export-token"),
                                         export_token):
            return PlainTextResponse(
                "the submission log needs the export credential; set "
                "RUBRIC_EXPORT_TOKEN on the server and send it as "
                "x-export-token. The contributor token does not authorize this.",
                403)
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
""" + AUTHOR_JS + r"""

const $ = s => document.querySelector(s);
let TASKS = [], CUR = null, DIRTY = false;

const author = () => $('#author').value.trim();

function saveName(){ mlAuthorSet(author()); }
function loadName(){ $('#author').value = mlAuthorGet(); }

async function refresh(){
  const r = await fetch('/api/tasks?kind=rubric&author=' + encodeURIComponent(author()));
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
    // A board has no verified answer to argue with — the rules flow's whole
    // premise. Asserting an empty string "is correct" is worse than showing
    // nothing, so a position gets the board's own status instead (21.126).
    (t.kind === 'position'
      ? '<div class="card"><h3>This is a board, not a rules question</h3>' +
        '<div class="hint">There is no reference answer: you are writing what the ' +
        'right LINE is. <b>Key points</b> = the plays a correct answer must make, ' +
        'in order. <b>Common errors</b> = the strategy mistakes a real player makes ' +
        'on this board.</div>' +
        ((t.legal_actions && t.legal_actions.length)
          ? '<div class="draft">' + t.legal_actions.map(a => '• ' + esc(a)).join('\n') +
            '</div>'
          : '<div class="hint" style="color:#c0392b">No <code>legal_actions</code> yet &mdash; ' +
            'the engine path fills these in. Your rubric is still valid (the rendered ' +
            'board does not change), but this board cannot be scored until it has them.' +
            '</div>') +
        '</div>'
      : '<div class="card"><h3>Verified answer &mdash; this is correct, do not re-adjudicate</h3>' +
        '<div class="answer">' + esc(t.answer) + '</div></div>') +
    ((t.protocol_errors && t.protocol_errors.length)
      ? '<div class="card"><h3>Already covered &mdash; do not restate these</h3>' +
        '<div class="hint">The judge is given your errors <b>plus</b> these ' +
        t.protocol_errors.length + '. Repeating one here creates two entries for one ' +
        'mistake, and the copy is scored as a strategy error that no parser checks.</div>' +
        '<div class="draft">' +
        t.protocol_errors.map((e, i) => (i + 1) + '. ' + esc(e)).join('\n') +
        '</div></div>'
      : '') +
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
      '<div class="hint">' + (t.kind === 'position'
        ? 'The plays a correct answer must make, one per point, in order. Name the ' +
          'card and what it does &mdash; &ldquo;Casts Requiting Hex on Mudbutton ' +
          'Cursetosser&rdquo;, not &ldquo;removes the blocker&rdquo;. A board asks ' +
          'ONE question: the median line is 4.5 actions but only 1.5 plays, so if ' +
          'you are writing plays in two phases this is two positions.'
        : 'The claims a correct answer must make. One checkable ' +
          'assertion each; 2–4 sharp points beat 5–8 soft ones. Never a bare ' +
          '&ldquo;Yes&rdquo; or &ldquo;No&rdquo;.') + '</div>' +
      '<div id="kp"></div>' +
      '<button class="addbtn" type="button" id="addkp">+ key point</button></div>' +
    '<div class="card"><h3>Common errors</h3>' +
      '<div class="hint">' + (t.kind === 'position'
        ? 'STRATEGY mistakes only &mdash; a legal play that wins less. Everything ' +
          'protocol-shaped is already covered above. Write each as a CLAIM a wrong ' +
          'answer could assert (&ldquo;Requiting Hex should be held for the bigger ' +
          'threat&rdquo;), not as narration of a move (&ldquo;Casts Requiting ' +
          'Hex&hellip;&rdquo;) &mdash; the judge is asked which of these the answer ' +
          'asserted, and all 32 stored positions are written this way. Lead with ' +
          'what makes it wrong, so the judge cannot match the opening before it ' +
          'reaches the qualifier.'
        : 'The mistakes a real player makes &mdash; traps, not negations ' +
          'of the points above. Lead with what makes it wrong: an error that opens with ' +
          'words also true of the correct answer gets matched before the judge reaches ' +
          'the part that matters.') + '</div>' +
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
.arm-grid{display:grid;gap:.85rem}
.arm-panel{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:.8rem}
.arm-panel.done{opacity:.82}
.arm-head{display:flex;align-items:center;gap:.45rem;margin:0 0 .5rem}
.reopen{margin-left:auto;font:inherit;font-size:.78rem;padding:.2rem .5rem;
 border:1px solid var(--rule);border-radius:3px;background:var(--paper);
 color:var(--soft);cursor:pointer}
.reopen:hover{color:var(--ink);border-color:var(--accent)}
.answer-wrap{margin:0 0 .55rem}
ul{margin:.2rem 0;padding-left:1.2rem;color:var(--soft);font-size:.9rem}
label.opt{display:flex;gap:.6rem;align-items:flex-start;padding:.7rem .8rem;margin-bottom:.4rem;
 background:var(--panel);border:1px solid var(--rule);border-radius:4px;cursor:pointer}
label.opt:has(input:checked){border-color:var(--accent);background:var(--accent-bg)}
label.opt input{margin:.25rem 0 0;width:1.1rem;height:1.1rem;flex:none}
.hint{font-size:.8rem;color:var(--faint);margin:.5rem 0 0}
input[type=text],.note{width:100%;padding:.55rem .6rem;font:inherit;font-size:.9rem;
 background:var(--panel);color:var(--ink);border:1px solid var(--rule);border-radius:4px}
/* The note became a textarea at form_version 4 — it carries the reasoning now,
   not a one-line fallback — and input[type=text] does not match a textarea, so
   on its own it would render unstyled. */
.note{resize:vertical;min-height:5.5rem;line-height:1.5;font-size:16px}
.note:focus{outline:2px solid var(--accent);outline-offset:1px}
/* The reference line. #ref is a textarea, so it needs the same treatment .note
   needed — input[type=text] does not match one. Monospace because it is the
   action grammar, not prose. */
#ref{width:100%;padding:.55rem .6rem;background:var(--panel);color:var(--ink);
 border:1px solid var(--rule);border-radius:4px;resize:vertical;min-height:7rem;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:15px;line-height:1.5}
#ref:focus{outline:2px solid var(--accent);outline-offset:1px}
.refrow{display:flex;gap:.7rem;align-items:center;margin-top:.5rem}
.refrow .hint{margin:0}
details.legal{margin:.4rem 0 .6rem;font-size:.86rem;color:var(--soft)}
details.legal summary{cursor:pointer;color:var(--accent)}
details.legal ul{margin:.4rem 0 0;padding-left:1.2rem}
pre.gram{margin:.4rem 0 0;padding:.5rem .6rem;background:var(--panel);
 border:1px solid var(--rule);border-radius:4px;overflow-x:auto;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;
 line-height:1.45;white-space:pre}
h3.grp{margin:1.1rem 0 .35rem;font-size:.74rem;text-transform:uppercase;
 letter-spacing:.07em;color:var(--faint);font-weight:650;
 border-top:1px solid var(--rule);padding-top:.7rem}
h3.grp:first-of-type{border-top:0;padding-top:0;margin-top:.5rem}
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
""" + AUTHOR_JS + r"""

const $=s=>document.querySelector(s);
const esc=s=>{const d=document.createElement('div');d.textContent=s??'';return d.innerHTML};
let T=[],i=0;
const who=()=>$('#who').value.trim();
$('#who').value=mlAuthorGet();
$('#who').oninput=()=>mlAuthorSet(who());

function renderArm(arm){
  const n_strategy = arm.n_strategy || 0;
  const dis = arm.done ? ' disabled' : '';
  return '<div class="arm-panel'+(arm.done?' done':'')+'" data-key="'+esc(arm.key)+'">'+
    '<div class="arm-head"><b>'+esc(arm.arm)+'</b>'+(arm.done?'<span class="done">saved</span>'+
      '<button type="button" class="reopen">re-open</button>':'')+'</div>'+
    '<div class="answer-wrap"><pre class="answer">'+esc(arm.answer)+'</pre></div>'+
    '<h2>Which of these mistakes does this answer make?</h2>'+
    ((n_strategy>0)?'<h3 class="grp">Mistakes specific to this position</h3>':'')+
    '<p class="hint">Judge the <b>play</b>, not the wording. Check an item if the answer does that thing.</p>'+
    (arm.common_errors||[]).map((e,n)=>
      ((n===n_strategy)?'<h3 class="grp">Mistakes any answer can make</h3>'+
        '<p class="hint">These apply to every position. Check them the same way — by what the answer did, not by how it worded it.</p>':'')+
      '<label class="opt"><input type="checkbox" class="e" value="'+(n+1)+'"'+dis+'><span><b>'+(n+1)+'.</b> '+esc(e)+'</span></label>').join('')+
    '<p class="hint"><b>Check none if it makes none of them</b> — a real verdict and a common one, not a skip.</p>'+
    '<label class="opt"><input type="checkbox" class="notcovered"'+dis+'><span><b>Bad, but not for any reason above.</b> The answer is wrong or useless and none of the listed mistakes describe it.</span></label>'+
    '<label class="opt"><input type="checkbox" class="unsure"'+dis+'><span>Genuinely ambiguous — I could argue it either way</span></label>'+
    '<h3 class="grp">Why is this play wrong?</h3>'+
    '<p class="hint">The boxes above are the verdict; this is the reasoning.</p>'+
    '<textarea class="note" rows="4"'+dis+' placeholder="Explain what the answer got wrong and what the right line was."></textarea>'+
    '</div>';
}

let GRAMMAR='',PHASES=[];
async function load(){
  try{const g=await (await fetch('/api/grammar')).json();GRAMMAR=g.grammar||'';PHASES=g.phases||[]}catch(e){}
  const r=await fetch('/api/tasks?kind=adjudication'+(who()?'&author='+encodeURIComponent(who()):''));
  const d=await r.json();T=(d.tasks||[]).slice().sort((a,b)=>(a.done===b.done?0:(a.done?1:-1)));
  const first=T.findIndex(t=>!t.done);i=first===-1?0:first;render();
}
function render(){
  const t=T[i];
  if(!t){$('#main').innerHTML='<p>Nothing queued.</p>';return}
  $('#prog').textContent=T.filter(x=>x.done).length+'/'+T.length;
  const nArms=(t.arms||[]).length;
  $('#meta').innerHTML=esc(t.record_id)+' · '+
    (nArms?esc(nArms)+' arms':'reference only')+
    (t.done?' <span class="done">· saved</span>':'');
  $('#main').innerHTML=
    '<h2>The situation</h2><pre>'+esc(t.question)+'</pre>'+
    '<h2>The correct line</h2><ul>'+(t.key_points||[]).map(k=>'<li>'+esc(k)+'</li>').join('')+'</ul>'+
    ((t.arms||[]).length
      ? '<div class="arm-grid">'+(t.arms||[]).map(renderArm).join('')+'</div>'
      : '<p class="hint">No model answers were sampled for this board, so there '+
        'is nothing to grade here — it is in the queue so the correct line can '+
        'be written for it.</p>')+
    referenceBlock(t);
  bindReference();
  bindReopen();
}

// The 100%-correct line, in the action grammar, as a property of the BOARD
// rather than of any arm's answer. Rendered AFTER the grading panels and with
// the legal plays collapsed: that list is the answer key for protocol entry 3,
// and a reviewer who reads it before ticking boxes stops being independent
// evidence on the one entry the judge already gets right (21.74).
function referenceBlock(t){
  const have=(t.reference_actions||[]).join('\n');
  return '<h2>The 100% correct line</h2>'+
    '<p class="hint">Write the ideal answer in the action grammar, one action per '+
    'line, exactly as a perfect model would output it. This is stored as the '+
    'reference for this board \u2014 it is checked by the parser on import and '+
    'refused if any line is not legal here, so it can serve as a known-good '+
    'answer to measure against later.</p>'+
    '<details class="legal"><summary>Show the action grammar</summary>'+
    '<pre class="gram">'+esc(GRAMMAR||'(loading)')+'</pre>'+
    '<p class="hint">PHASE and END PHASE accept these step names only, so that '+
    'prose cannot become a declaration:</p>'+
    '<pre class="gram">'+esc((PHASES||[]).join(' · '))+'</pre></details>'+
    '<details class="legal"><summary>Show the legal plays on this board ('+
      (t.legal_actions||[]).length+')</summary>'+
    '<ul>'+((t.legal_actions||[]).map(a=>'<li><code>'+esc(a)+'</code></li>').join('')
            || '<li><i>none \u2014 PASS is the only response</i></li>')+'</ul></details>'+
    '<textarea id="ref" rows="6" spellcheck="false" placeholder="PHASE Declare '+
    'Attackers Step&#10;TAP Forest FOR {G}&#10;CAST Ambush Viper&#10;PASS">'+
    esc(have)+'</textarea>'+
    '<div class="refrow"><button id="saveref" type="button">Save the correct line'+
    '</button><span id="refmsg" class="hint"></span></div>';
}
// Bound after every render, because render() replaces #main wholesale and any
// handler attached to the previous node is discarded with it.
function bindReference(){
  const btn=$('#saveref'); if(!btn) return;
  btn.onclick=async()=>{
    const t=T[i]; if(!t) return;
    const lines=$('#ref').value.split('\n').map(x=>x.trim()).filter(Boolean);
    const msg=$('#refmsg');
    msg.textContent='saving...';
    const r=await fetch('/api/reference',{method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({record_id:t.record_id,reference_actions:lines,author:who()})});
    const d=await r.json();
    if(!d.ok){msg.textContent=d.error||'save failed';return}
    t.reference_actions=lines;
    if(!(t.arms||[]).length) t.done=lines.length>0;
    msg.textContent=lines.length?('saved '+lines.length+' actions'):'saved (cleared)';
    $('#prog').textContent=T.filter(x=>x.done).length+'/'+T.length;
  };
}
// Re-opening a saved arm, in place. Re-rendering the record would discard
// whatever is typed in its OTHER panels, which is the whole reason the page
// groups them — so this clears the lock on one panel and leaves the rest alone.
//
// Re-adjudication is a supported act, not an accident: a verdict is identified
// by (key, author, answer_sha, n_shown) precisely so a second one on the same
// answer is kept beside the first rather than colliding with it (21.78). The
// lock exists to stop a grouped save re-posting arms nobody touched, which is a
// different problem from a reviewer changing their mind.
function bindReopen(){
  [...document.querySelectorAll('.reopen')].forEach(btn=>{
    btn.onclick=()=>{
      const card=btn.closest('.arm-panel'); if(!card) return;
      const group=T[i]; if(!group) return;
      const arm=(group.arms||[]).find(x=>x.key===card.dataset.key); if(!arm) return;
      arm.done=false;
      group.done=(group.arms||[]).length
        ? (group.arms||[]).every(a=>a.done)
        : ((group.reference_actions||[]).length>0);
      card.classList.remove('done');
      [...card.querySelectorAll('input,textarea')].forEach(el=>{el.disabled=false});
      const tag=card.querySelector('.arm-head .done'); if(tag) tag.remove();
      btn.remove();
      $('#prog').textContent=T.filter(x=>x.done).length+'/'+T.length;
    };
  });
}
$('#skip').onclick=()=>{i=Math.min(T.length-1,i+1);render();scrollTo(0,0)};
$('#go').onclick=async()=>{
  const group=T[i];if(!group)return;
  const cards=[...document.querySelectorAll('.arm-panel')];
  for (const card of cards) {
    const key = card.dataset.key;
    const arm = (group.arms||[]).find(x => x.key === key);
    if (!arm) continue;
    if (arm.done) continue;
    const present=[...card.querySelectorAll('.e')].filter(c=>c.checked).map(c=>+c.value);
    const note = card.querySelector('.note');
    const unsure = card.querySelector('.unsure');
    const notcovered = card.querySelector('.notcovered');
    const faulted=present.length||notcovered.checked;
    if (faulted && !note.value.trim() && !confirm('Submit without saying why this play is wrong?')) return;
    const r=await fetch('/api/adjudicate',{method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({key:arm.key,errors_present:present,author:who(),
        unsure:!!unsure.checked,not_covered:!!notcovered.checked,note:note.value})});
    const d=await r.json();
    if(!d.ok){alert(d.error||'save failed');return}
    arm.done=true;
  }
  // `every` on an empty list is true, which would mark a board with nothing to
  // grade as done without a reference having been written.
  group.done = (group.arms||[]).length
    ? (group.arms||[]).every(a => a.done)
    : ((group.reference_actions||[]).length > 0);
  const nxt=T.findIndex((x,n)=>n>i&&!x.done);
  i=nxt===-1?Math.min(T.length-1,i+1):nxt;render();scrollTo(0,0);
};
load();
</script></body></html>
"""


# A view for AUTHORING the correct line, separate from grading it.
#
# The reference box lived inside the adjudication form, which made editing one
# a matter of finding the record in a queue ordered by grading status. The two
# jobs have different shapes: grading walks a sample once, authoring revisits a
# board until the line is right (Section 21.90).
REFERENCE_HTML = r"""<!doctype html>
<meta charset="utf-8"><meta name=viewport content="width=device-width,initial-scale=1">
<title>Correct lines</title>
<style>
:root{--paper:#F5F6F8;--panel:#fff;--ink:#131820;--soft:#59636F;--faint:#8A939E;
 --accent:#1C5A8C;--accent-bg:#EAF1F7;--ok:#1E7A4B;--bad:#A32B2B;--rule:#D9DEE4;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --paper:#0F1319;--panel:#161B22;--ink:#E6EAF0;--soft:#9BA5B2;--faint:#6B7683;
 --accent:#6FA8D6;--accent-bg:#16232E;--ok:#5BB98B;--bad:#E0736D;--rule:#2A313A}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
 font-size:16px;line-height:1.5}
header{position:sticky;top:0;z-index:5;background:var(--panel);
 border-bottom:1px solid var(--rule);padding:.6rem .9rem;display:flex;
 gap:.8rem;align-items:center;flex-wrap:wrap}
header b{font-variant-numeric:tabular-nums}
header .who{margin-left:auto;display:flex;gap:.4rem;align-items:center}
input[type=text]{background:var(--panel);color:var(--ink);border:1px solid var(--rule);
 border-radius:4px;padding:.35rem .5rem;font:inherit;font-size:.9rem}
main{display:grid;grid-template-columns:minmax(220px,290px) 1fr;
 height:calc(100vh - 53px);min-height:0}
nav{border-right:1px solid var(--rule);overflow-y:auto;background:var(--panel)}
nav button{display:block;width:100%;text-align:left;border:0;background:none;
 color:inherit;font:inherit;font-size:.87rem;padding:.45rem .7rem;cursor:pointer;
 border-bottom:1px solid var(--rule)}
nav button:hover{background:var(--accent-bg)}
nav button.on{background:var(--accent-bg);box-shadow:inset 3px 0 0 var(--accent)}
nav .tick{color:var(--ok);font-weight:700}
nav .cat{color:var(--faint);font-size:.78rem}
nav .flag{color:var(--bad);font-weight:700}
select{font:inherit;font-size:.9rem;padding:.35rem .5rem;background:var(--panel);
 color:var(--ink);border:1px solid var(--rule);border-radius:4px}
#rnote{width:100%;margin-top:.5rem}
section{overflow-y:auto;padding:1rem 1.1rem 4rem;max-width:52rem}
h2{font-size:1rem;margin:1.2rem 0 .4rem;text-transform:uppercase;
 letter-spacing:.06em;color:var(--faint)}
h2:first-child{margin-top:0}
pre.board{background:var(--panel);border:1px solid var(--rule);border-radius:4px;
 padding:.7rem .8rem;overflow-x:auto;font-family:var(--mono);font-size:13.5px;
 white-space:pre;margin:0}
ul{margin:.3rem 0;padding-left:1.2rem;color:var(--soft);font-size:.92rem}
details{margin:.5rem 0;font-size:.88rem;color:var(--soft)}
summary{cursor:pointer;color:var(--accent)}
pre.gram{background:var(--panel);border:1px solid var(--rule);border-radius:4px;
 padding:.5rem .6rem;overflow-x:auto;font-family:var(--mono);font-size:12.5px;
 white-space:pre;margin:.4rem 0 0}
textarea{width:100%;min-height:9rem;padding:.6rem .7rem;background:var(--panel);
 color:var(--ink);border:1px solid var(--rule);border-radius:4px;resize:vertical;
 font-family:var(--mono);font-size:15px;line-height:1.55}
textarea:focus{outline:2px solid var(--accent);outline-offset:1px}
.row{display:flex;gap:.7rem;align-items:center;margin-top:.6rem;flex-wrap:wrap}
button.go{font:inherit;font-size:.92rem;padding:.5rem 1rem;border-radius:4px;
 border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
button.ghost{font:inherit;font-size:.92rem;padding:.5rem .9rem;border-radius:4px;
 border:1px solid var(--rule);background:var(--panel);color:inherit;cursor:pointer}
.msg{font-size:.88rem;color:var(--soft)}
.msg.bad{color:var(--bad)}
.hint{font-size:.85rem;color:var(--faint);margin:.4rem 0}
</style>
<header><b id="prog">0/0</b><span class="hint" id="meta"></span>
<span class="who">author <input type="text" id="author" placeholder="your name"></span>
</header>
<main><nav id="list"></nav><section id="pane"><p class="hint">Loading…</p></section></main>
<script>
""" + AUTHOR_JS + r"""
const $=s=>document.querySelector(s);
const esc=s=>{const d=document.createElement('div');d.textContent=s??'';return d.innerHTML};
let R=[],i=0,GRAMMAR='',PHASES=[],KINDS={};
function who(){return $('#author').value.trim()}
$('#author').value=mlAuthorGet();
$('#author').oninput=()=>mlAuthorSet(who());

function done(r){return (r.reference_actions||[]).length>0}
function drawList(){
  $('#prog').textContent=R.filter(done).length+'/'+R.length;
  $('#list').innerHTML=R.map((r,n)=>
    '<button class="'+(n===i?'on':'')+'" data-n="'+n+'">'+
    (done(r)?'<span class="tick">✓</span> ':'   ')+
    esc(r.record_id)+'<br><span class="cat">'+esc(r.category||'')+'</span></button>').join('');
  [...document.querySelectorAll('#list button')].forEach(b=>{
    b.onclick=()=>{i=+b.dataset.n;draw()};
  });
}
function draw(){
  const r=R[i];
  if(!r){$('#pane').innerHTML='<p class="hint">Nothing to author.</p>';return}
  $('#meta').textContent=r.record_id+(r.record_id.includes('::step')?' · scenario step':'');
  $('#pane').innerHTML=
    '<h2>The situation</h2><pre class="board">'+esc(r.question)+'</pre>'+
    '<h2>The correct line, in words</h2><ul>'+
      (r.key_points||[]).map(k=>'<li>'+esc(k)+'</li>').join('')+'</ul>'+
    '<details><summary>Legal plays on this board ('+(r.legal_actions||[]).length+')</summary>'+
      '<ul>'+((r.legal_actions||[]).map(a=>'<li><code>'+esc(a)+'</code></li>').join('')
        ||'<li><i>none — PASS is the only response</i></li>')+'</ul></details>'+
    '<details><summary>Action grammar and step names</summary>'+
      '<pre class="gram">'+esc(GRAMMAR||'(loading)')+'</pre>'+
      '<p class="hint">PHASE and END PHASE accept these step names only:</p>'+
      '<pre class="gram">'+esc(PHASES.join(' · '))+'</pre></details>'+
    '<h2>The 100% correct line</h2>'+
    '<p class="hint">One action per line, exactly as a perfect answer would emit '+
    'it. Checked by the parser on import and refused if any line is not legal '+
    'here.</p>'+
    '<textarea id="ref" spellcheck="false"></textarea>'+
    '<div class="row"><button class="go" id="save">Save</button>'+
    '<button class="ghost" id="next">Save and next</button>'+
    '<span class="msg" id="msg"></span></div>'+
    '<h2>Flag this board for editing</h2>'+
    '<p class="hint">The reference says what the right answer is; this says the '+
    'BOARD needs work. A flagged board is still evaluated and still counted \u2014 '+
    'runs report how many carry a flag, so the caveat travels with the number.</p>'+
    '<div class="row"><select id="rkind"><option value="">not flagged</option>'+
    Object.keys(KINDS).map(k=>'<option value="'+esc(k)+'"'+
      ((r.review&&r.review.kind===k)?' selected':'')+'>'+esc(k)+'</option>').join('')+
    '</select><span class="hint" id="rwhy"></span></div>'+
    '<input type="text" id="rnote" placeholder="what needs changing" value="'+
      esc((r.review&&r.review.note)||'')+'">'+
    '<div class="row"><button class="ghost" id="rsave">Save flag</button>'+
    '<span class="msg" id="rmsg"></span></div>';
  $('#ref').value=(r.reference_actions||[]).join('\n');
  $('#save').onclick=()=>save(false);
  $('#next').onclick=()=>save(true);
  const why=()=>{$('#rwhy').textContent=KINDS[$('#rkind').value]||''};
  $('#rkind').onchange=why; why();
  $('#rsave').onclick=saveReview;
  drawList();
}
async function save(advance){
  const r=R[i],msg=$('#msg');
  const lines=$('#ref').value.split('\n').map(x=>x.trim()).filter(Boolean);
  msg.className='msg';msg.textContent='saving…';
  let d;
  try{
    d=await (await fetch('/api/reference',{method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({record_id:r.record_id,reference_actions:lines,author:who()})
    })).json();
  }catch(e){d={error:String(e)}}
  if(!d.ok){msg.className='msg bad';msg.textContent=d.error||'save failed';return}
  r.reference_actions=lines;
  msg.textContent=lines.length?('saved '+lines.length+' actions'):'saved (cleared)';
  drawList();
  if(advance){
    const nxt=R.findIndex((x,n)=>n>i&&!done(x));
    i=nxt===-1?Math.min(R.length-1,i+1):nxt;draw();
    document.querySelector('section').scrollTop=0;
  }
}
async function saveReview(){
  const r=R[i],msg=$('#rmsg');
  const kind=$('#rkind').value,note=$('#rnote').value.trim();
  msg.className='msg';msg.textContent='saving…';
  let d;
  try{
    d=await (await fetch('/api/review',{method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({record_id:r.record_id,kind:kind,note:note,author:who()})
    })).json();
  }catch(e){d={error:String(e)}}
  if(!d.ok){msg.className='msg bad';msg.textContent=d.error||'save failed';return}
  r.review=kind?{kind:kind,note:note}:null;
  msg.textContent=kind?('flagged: '+kind):'flag cleared';
  drawList();
}
async function load(){
  try{const g=await (await fetch('/api/grammar')).json();
      GRAMMAR=g.grammar||'';PHASES=g.phases||[];KINDS=g.review_kinds||{}}catch(e){}
  const d=await (await fetch('/api/reference-boards')).json();
  R=d.boards||[];
  const first=R.findIndex(x=>!done(x));i=first===-1?0:first;
  draw();
}
load();
</script>
"""


# Authoring a whole scenario: a turn written as a sequence of boards.
#
# 21.89 found the grammar has one player in it, so an opponent's action cannot
# be written inside an answer. A scenario is where it goes instead — the next
# step's BOARD states what happened, and the answer to that step is one
# checkable decision (Section 21.96).
#
# Each step is authored as a complete board because that is what a person can
# read and check. Adding a step CLONES the previous one, so authoring a turn is
# editing what changed rather than retyping a battlefield per step; the stored
# record keeps only the diff, computed server-side by
# `turns.scenario_from_submission`.
SCENARIO_HTML = r"""<!doctype html>
<meta charset="utf-8"><meta name=viewport content="width=device-width,initial-scale=1">
<title>Author a scenario</title>
<style>
:root{--paper:#F5F6F8;--panel:#fff;--ink:#131820;--soft:#59636F;--faint:#8A939E;
 --accent:#1C5A8C;--accent-bg:#EAF1F7;--ok:#1E7A4B;--bad:#A32B2B;--rule:#D9DEE4;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --paper:#0F1319;--panel:#161B22;--ink:#E6EAF0;--soft:#9BA5B2;--faint:#6B7683;
 --accent:#6FA8D6;--accent-bg:#16232E;--ok:#5BB98B;--bad:#E0736D;--rule:#2A313A}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
 font-size:16px;line-height:1.5;padding:0 0 5rem}
header{position:sticky;top:0;z-index:5;background:var(--panel);
 border-bottom:1px solid var(--rule);padding:.6rem .9rem;display:flex;gap:.7rem;
 align-items:center;flex-wrap:wrap}
main{max-width:54rem;margin:0 auto;padding:1rem .9rem}
h2{font-size:.8rem;margin:1.3rem 0 .4rem;text-transform:uppercase;
 letter-spacing:.06em;color:var(--faint)}
label{display:block;font-size:.82rem;color:var(--soft);margin:.5rem 0 .15rem}
input,textarea,select{width:100%;font:inherit;font-size:15px;padding:.4rem .55rem;
 background:var(--panel);color:var(--ink);border:1px solid var(--rule);border-radius:4px}
textarea{min-height:3.6rem;resize:vertical;font-family:var(--mono);font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:.5rem}
.step{background:var(--panel);border:1px solid var(--rule);border-radius:6px;
 padding:.8rem .9rem;margin:.9rem 0}
.step h3{margin:0 0 .3rem;font-size:.95rem}
.step .drop{float:right;font:inherit;font-size:.78rem;padding:.15rem .5rem;
 border:1px solid var(--rule);border-radius:3px;background:var(--paper);
 color:var(--soft);cursor:pointer}
.hint{font-size:.83rem;color:var(--faint);margin:.3rem 0}
.bar{position:fixed;left:0;right:0;bottom:0;background:var(--panel);
 border-top:1px solid var(--rule);padding:.6rem .9rem;display:flex;gap:.6rem;
 align-items:center}
button.go{font:inherit;font-size:.92rem;padding:.5rem 1.1rem;border-radius:4px;
 border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
button.ghost{font:inherit;font-size:.92rem;padding:.5rem .9rem;border-radius:4px;
 border:1px solid var(--rule);background:var(--panel);color:inherit;cursor:pointer}
.msg{font-size:.87rem;color:var(--soft)}
.msg.bad{color:var(--bad)}
pre.gram{background:var(--panel);border:1px solid var(--rule);border-radius:4px;
 padding:.5rem .6rem;overflow-x:auto;font-family:var(--mono);font-size:12.5px;
 white-space:pre;margin:.4rem 0 0}
details{margin:.5rem 0;font-size:.86rem;color:var(--soft)}
summary{cursor:pointer;color:var(--accent)}
</style>
<header><b>Author a scenario</b>
<span class="hint">a turn as a sequence of boards</span>
<span style="margin-left:auto">author <input type="text" id="author"
 style="width:11rem" placeholder="your name"></span></header>
<main>
<h2>The scenario</h2>
<div class="grid">
<div><label for="scenario_id">id</label><input type="text" id="scenario_id"
 placeholder="turn-hold-removal-0001"></div>
<div><label for="category">category</label><select id="category"></select></div>
<div><label for="difficulty">difficulty</label><select id="difficulty">
<option>basic</option><option selected>intermediate</option><option>advanced</option>
</select></div>
</div>
<details><summary>Action grammar and step names</summary>
<pre class="gram" id="gram">(loading)</pre>
<pre class="gram" id="phases"></pre></details>
<p class="hint">Every step is a whole board so you can read it. Adding a step
copies the one before it &mdash; edit only what changed. Only the differences
are stored.</p>
<div id="steps"></div>
</main>
<div class="bar"><button class="ghost" id="add">Add a step</button>
<button class="go" id="save">Submit scenario</button>
<span class="msg" id="msg"></span></div>
<script>
""" + AUTHOR_JS + r"""
const $=s=>document.querySelector(s);
const esc=s=>{const d=document.createElement('div');d.textContent=s??'';return d.innerHTML};
const FIELDS=['turn','phase','active_player','priority','you_life','you_hand',
 'you_library','opp_life','opp_hand_count','opp_library','you_battlefield',
 'opp_battlefield','mana_available','known_information','legal_actions',
 'key_points','common_errors','reference_actions'];
let STEPS=[{}],KINDS={};
$('#author').value=mlAuthorGet();
$('#author').oninput=()=>mlAuthorSet($('#author').value.trim());

function field(n,key,label,area,ph){
  const id='s'+n+'_'+key;
  return '<label for="'+id+'">'+esc(label)+'</label>'+
    (area?'<textarea id="'+id+'" placeholder="'+esc(ph||'')+'"></textarea>'
         :'<input type="text" id="'+id+'" placeholder="'+esc(ph||'')+'">');
}
function drawSteps(){
  $('#steps').innerHTML=STEPS.map((st,n)=>
    '<div class="step"><h3>Step '+(n+1)+
    (n?'<button type="button" class="drop" data-n="'+n+'">remove</button>':'')+'</h3>'+
    (n?'<p class="hint">Copied from step '+n+'. Change only what the previous '+
       'step\'s line and the opponent made different.</p>':'')+
    '<div class="grid">'+
      field(n,'turn','turn')+field(n,'phase','phase','','declare attackers')+
      field(n,'active_player','active player (you/opp)')+
      field(n,'priority','priority (you/opp)')+
      field(n,'you_life','your life')+field(n,'opp_life','opponent life')+
      field(n,'you_library','your library')+field(n,'opp_hand_count','opp hand count')+
      field(n,'opp_library','opp library')+
      field(n,'mana_available','mana available','','{R}{U}')+
    '</div>'+
    field(n,'you_battlefield','your battlefield',1,'Mountain\nIsland')+
    field(n,'opp_battlefield',"opponent's battlefield",1,'Goblin Guide 2/2 attacking')+
    field(n,'you_hand','your hand',1,'Lightning Strike\nShock')+
    field(n,'known_information','known information',1)+
    field(n,'legal_actions','legal plays (one per line)',1,
          'CAST Shock TARGET Goblin Guide')+
    field(n,'key_points','key points (two or more)',1)+
    field(n,'common_errors','common errors',1)+
    field(n,'reference_actions','the 100% correct line for THIS step',1,
          'PHASE declare attackers\nTAP Mountain FOR {R}\nCAST Shock TARGET Goblin Guide\nPASS')+
    '</div>').join('');
  STEPS.forEach((st,n)=>FIELDS.forEach(k=>{
    const el=document.getElementById('s'+n+'_'+k);
    if(el){el.value=st[k]||'';el.oninput=()=>{STEPS[n][k]=el.value};}
  }));
  [...document.querySelectorAll('.drop')].forEach(b=>{
    b.onclick=()=>{STEPS.splice(+b.dataset.n,1);drawSteps()};
  });
}
$('#add').onclick=()=>{
  // Clone, do not blank: consecutive steps of one turn differ in a few fields,
  // and retyping a battlefield per step is both work and a place to drift.
  const prev=STEPS[STEPS.length-1]||{};
  const next=Object.assign({},prev);
  next.reference_actions='';       // the line is what each step is asking for
  next.key_points='';
  next.common_errors='';
  STEPS.push(next);drawSteps();
  window.scrollTo(0,document.body.scrollHeight);
};
$('#save').onclick=async()=>{
  const msg=$('#msg');msg.className='msg';msg.textContent='submitting…';
  let d;
  try{
    d=await (await fetch('/api/scenario',{method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({scenario_id:$('#scenario_id').value.trim(),
        category:$('#category').value,difficulty:$('#difficulty').value,
        author:$('#author').value.trim(),steps:STEPS})})).json();
  }catch(e){d={error:String(e)}}
  if(!d.ok){msg.className='msg bad';msg.textContent=d.error||'submit failed';return}
  msg.textContent='submitted '+d.scenario_id+' ('+d.steps+' steps) — validated on import';
};
async function load(){
  try{const g=await (await fetch('/api/grammar')).json();
    $('#gram').textContent=g.grammar||'';
    $('#phases').textContent='steps: '+(g.phases||[]).join(' · ');
    KINDS=g.review_kinds||{};
  }catch(e){}
  try{const c=await (await fetch('/api/categories')).json();
    $('#category').innerHTML=(c.categories||[]).map(x=>'<option>'+esc(x)+'</option>').join('');
  }catch(e){}
  drawSteps();
}
load();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path,
                        default=Path(os.environ.get("TASKS_FILE", "data/gold/worksheets/tasks.json")),
                        help="task file from `author_rubrics.py --export-tasks`")
    parser.add_argument("--also-tasks", type=Path, nargs="*",
                        default=([Path(os.environ["ALSO_TASKS_FILE"])]
                                 if os.environ.get("ALSO_TASKS_FILE") else None),
                        help="a second task file, so one deployment serves both forms. "
                             "Its kind comes from the file, so order does not matter.")
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
    parser.add_argument("--export-token", default=os.environ.get("RUBRIC_EXPORT_TOKEN"),
                        help="separate credential for GET /api/export, sent as the "
                             "x-export-token header. The contributor token does NOT "
                             "authorize the export: it is shared by everyone who has "
                             "the link, and the log holds every other contributor's "
                             "work. Unset with a contributor token in force means "
                             "export is refused (fail closed); unset on a loopback "
                             "server with no token at all is fine.")
    parser.add_argument("--secure-cookies", action="store_true",
                        help="force the Secure flag on the contributor cookie. Not "
                             "normally needed: it is set automatically whenever the "
                             "request arrived over HTTPS (fly forwards "
                             "x-forwarded-proto). Use it behind a proxy that does not.")
    args = parser.parse_args()

    # One file per form. Both may be given; the kind of each comes from the
    # file, never from the order or a flag (see load_tasks).
    paths = [args.tasks] + list(args.also_tasks or [])
    task_sets: dict[str, list[dict]] = {}
    for path in paths:
        loaded, k = load_tasks(path)
        if k in task_sets:
            raise SystemExit(f"{path}: a {k!r} task file was already given — one per kind")
        task_sets[k] = loaded

    # A public bind with no token would put the form — and every contributor's
    # work — behind nothing at all. Refuse rather than warn.
    public = args.host not in ("127.0.0.1", "localhost", "::1")
    token = None if args.no_token else (args.token or (secrets.token_urlsafe(12) if public else None))
    if public and args.no_token:
        raise SystemExit("--no-token with a public --host would expose the form to anyone. "
                         "Drop --no-token, or bind to 127.0.0.1.")

    for k, t in sorted(task_sets.items()):
        print(f"{len(t)} {k} tasks")
    print(f"submissions -> {args.submissions.resolve()}")
    base = f"http://{'localhost' if not public else args.host}:{args.port}/"
    print(f"\n  {base}{'?t=' + token if token else ''}\n")
    if token and not args.token:
        print(f"generated token: {token}  (set RUBRIC_TOKEN to keep it stable across restarts)\n")
    if token and not args.export_token:
        print("  note: RUBRIC_EXPORT_TOKEN is unset, so GET /api/export is refused.\n"
              "        Set one to collect submissions:\n"
              "          fly secrets set RUBRIC_EXPORT_TOKEN=\"$(openssl rand -hex 32)\"\n")

    import uvicorn
    uvicorn.run(build_app(task_sets, args.submissions, token,
                          export_token=args.export_token,
                          secure_cookies=args.secure_cookies),
                host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
