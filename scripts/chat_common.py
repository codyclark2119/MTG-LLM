"""Pieces of the chat surface that carry no model dependency.

Split out of `chat_server.py` when `chat_relay.py` needed the same pages and
validators. The alternative was to ship `chat_server.py` itself in the public
image, where its `Engine` would be inert (mlx is not installed there) but a
model loader would still be sitting on the deployable side. CLAUDE.md's rule
is that a growing COPY list is the moment to ask whether the new thing belongs
on the public side; a module that loads a 4.4 GB model does not, inert or not.

Everything here is stdlib plus the served HTML. Both servers import it; neither
redefines any of it, which is the point — the login page's behaviour must not
be able to differ between the local and deployed surfaces.
"""

import json
import os
import threading
from pathlib import Path

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


LOGIN_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Sign in</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  margin:0;background:#f6f7f9;color:#1a1a1a;display:flex;min-height:100vh;
  align-items:center;justify-content:center}
.card{background:#fff;border:1px solid #e2e2e2;border-radius:12px;padding:28px;
  width:320px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
h1{font-size:17px;margin:0 0 6px}
.sub{color:#666;font-size:13px;margin-bottom:18px}
input{width:100%;box-sizing:border-box;padding:9px 10px;font:inherit;
  border:1px solid #ccc;border-radius:8px}
button{width:100%;margin-top:12px;font:inherit;padding:9px 16px;border-radius:8px;
  border:1px solid #1a1a1a;background:#1a1a1a;color:#fff;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.err{color:#b00020;font-size:13px;margin-top:10px;min-height:18px}
</style></head><body>
<div class="card">
<h1>MTG Rules Assistant</h1>
<div class="sub">Enter the access password to continue.</div>
<form id="f"><input id="pw" type="password" placeholder="Password" autofocus
  autocomplete="current-password"><button id="go" type="submit">Sign in</button></form>
<div class="err" id="err"></div>
</div>
<script>
function el(id){return document.getElementById(id);}
function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}
el('f').onsubmit=function(ev){
  ev.preventDefault();
  el('go').disabled=true;
  el('err').textContent='';
  fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({password:el('pw').value})})
    .then(function(r){return r.json().then(function(j){
      if(!r.ok)throw new Error((j.errors||['sign in failed']).join('; '));
      location.href='/';});})
    .catch(function(e){
      el('err').innerHTML=esc(e.message);
      el('go').disabled=false;
    });
};
</script></body></html>
"""
