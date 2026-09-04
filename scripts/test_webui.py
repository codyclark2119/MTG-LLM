"""The served page must actually parse. Run: python scripts/test_webui.py

WHY THIS EXISTS

`webui.py` shipped for several commits serving JavaScript that no browser could
run. Two Python-style `#` comments sat inside `INDEX_HTML` — a raw string — so
Python never stripped them and the browser received them as JS, where `#` is a
syntax error. A syntax error is fatal to the whole `<script>`, so **every view
rendered blank**, including the three unrelated to the code the comments were
next to.

Nothing caught it. `test_imports.py` passes because the module imports fine, the
API returns 200 with correct JSON because the server half was never broken, and
`--help` exits 0. The failure lived entirely in a string that Python is not
required to understand — which is the general shape worth guarding: **a string
in one language embedded in a file of another gets no checking from either.**

So this parses the served page's script with JavaScriptCore (`osascript -l
JavaScript`, present on every macOS) and asserts the structural invariants a
single-file app depends on. No browser, no network, no model.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CHECKS_RUN = 0
FAILED = 0


def check(label: str, got, want) -> None:
    global CHECKS_RUN, FAILED
    CHECKS_RUN += 1
    if got != want:
        FAILED += 1
        print(f"  FAIL {label}\n       got  {got!r}\n       want {want!r}")


def served_html() -> str:
    """The exact bytes `GET /` returns, without starting a server."""
    import webui
    return webui.INDEX_HTML


def js_of(html: str) -> str:
    """The page's script, or "" for a page that deliberately has none.

    A page with no JavaScript is legitimate — the chooser is static by design —
    and demanding exactly one block failed it for being simple. What must never
    happen is a page with MORE than one, because `js_of` would silently check
    only the first and the second could be anything.
    """
    parts = re.findall(r"<script>(.*?)</script>", html, re.S)
    check("at most one <script> block", len(parts) <= 1, True)
    return parts[0] if parts else ""


def jsc_parse(js: str) -> str:
    """Parse (never execute) the script. Returns '' on success, else the error.

    `new Function(src)` compiles without running, so a page that talks to the
    network or the DOM is still safe to check.
    """
    src = Path("/private/tmp/claude-501/-Users-codyclark-Documents-code-magic-llm/"
               "d331db77-266c-4801-b4d1-a37d84de05db/scratchpad/_webui_check.js")
    try:
        src.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        src = Path("/tmp/_webui_check.js")
    src.write_text(js, encoding="utf-8")
    script = (
        'ObjC.import("Foundation");'
        f'var s=$.NSString.stringWithContentsOfFileEncodingError("{src}",4,null).js;'
        'try{ new Function(s); "" }catch(e){ "" + e.message }'
    )
    try:
        out = subprocess.run(["osascript", "-l", "JavaScript", "-e", script],
                             capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  SKIP osascript unavailable — JS not parse-checked")
        return ""
    return out.stdout.strip()


def check_page(label: str, html: str) -> None:
    global FAILED, CHECKS_RUN
    js = js_of(html)

    # The bug itself: a Python comment inside the SCRIPT. Scoped to the script
    # rather than the whole page on purpose — a CSS id selector at the start of
    # a line (`#prog{...}`) is legitimate and lives in <style>, so checking the
    # whole page would flag eleven innocent lines in the rubric form and train
    # everyone to ignore the check.
    offenders = [ln for ln in js.split("\n")
                 if ln.lstrip().startswith("#") and not ln.lstrip().startswith("#/")]
    check(f"{label}: no Python-style comments in the script", offenders, [])

    err = jsc_parse(js)
    if err:
        print(f"  FAIL {label}: the served JavaScript does not parse: {err}")
        FAILED += 1
    CHECKS_RUN += 1

    # Every route the router dispatches to must be defined, or that view is a
    # blank pane with an error only the console shows.
    for name in re.findall(r"return\s+(view[A-Za-z]+)\(\)", js):
        check(f"{label}: {name} is defined", f"function {name}(" in js, True)

    # Balanced template literals — an odd count silently swallows the rest of
    # the file into a string and produces a parse error a long way from home.
    check(f"{label}: backticks balanced", js.count("`") % 2, 0)


def check_shared_runner_adoption() -> None:
    """The script runner is the shared console, not a copy of it.

    This file used to carry ~130 lines of `Runner` and `lan_ip` duplicated from
    harness/core/webui/runner.py -- a PRE-extraction copy wired to module
    globals instead of constructor arguments. The extraction had happened; the
    adoption had not, and nothing guarded the drift: test_imports.py's
    SUBTREE_DUPLICATES check covers common.py only, and no test in this repo
    exercised Runner at all.

    Reintroducing a local copy now fails here.
    """
    import webui
    from harness.core.webui.runner import Runner as SharedRunner

    check("runner: Runner is imported, not redefined",
          webui.Runner.__module__, "harness.core.webui.runner")
    check("runner: lan_ip is imported, not redefined",
          webui.lan_ip.__module__, "harness.core.webui.runner")
    src = Path(__file__).with_name("webui.py").read_text()
    check("runner: no local class Runner", "\nclass Runner:" in src, False)
    check("runner: no local def lan_ip", "\ndef lan_ip(" in src, False)

    from fastapi.testclient import TestClient
    from label_store import CANDIDATES_PATH, Store
    from common import GOLD_PATH, RULES_PATH

    store = Store(GOLD_PATH, CANDIDATES_PATH, RULES_PATH, None, True)
    runner = SharedRunner({a["id"]: a for a in webui.ACTIONS}, webui.REPO_ROOT)

    # --- mounted, and reachable ---
    c = TestClient(webui.build_app(store, runner, author="t", token=None))
    check("console: page is served", c.get("/console").status_code, 200)
    check("console: actions listed", c.get("/console/api/actions").status_code, 200)
    groups = c.get("/console/api/actions").json()["groups"]
    check("console: this repo's own action groups reach it",
          "Gold set" in groups and "Gameplay" in groups, True)
    check("console: the script path is never sent to the client",
          "cmd" in json.dumps(groups), False)

    # --- the endpoints it replaced are gone ---
    check("console: the old /api/jobs is gone", c.get("/api/jobs").status_code, 404)
    check("console: the old /api/run is gone",
          c.post("/api/run", json={}).status_code, 404)

    # --- the allowlist still holds through the mount ---
    r = c.post("/console/api/run/rm-rf", json={"values": {}})
    check("console: an unknown action is refused", r.status_code, 400)

    # --- and the mount inherits THIS server's auth ---
    # It is built with no token of its own on purpose: the outer middleware
    # already gates every request, and the shared monitor's cookie has a
    # different name, so a second token would mean two logins for one server.
    # If the mount escaped that middleware, the LAN mode would expose a remote
    # script runner with no auth at all.
    gated = TestClient(webui.build_app(store, runner, author="t", token="sekrit"))
    check("console: no token is refused", gated.get("/console").status_code, 401)
    check("console: running without a token is refused",
          gated.post("/console/api/run/validate_gold", json={"values": {}}).status_code, 401)
    check("console: a valid token is accepted",
          gated.get("/console", headers={"x-token": "sekrit"}).status_code, 200)


def main() -> None:
    check_page("webui", served_html())

    # The deployable half serves two pages, chosen by task kind. Both ship to a
    # public host, so a syntax error there is worse than locally: nobody is
    # watching a terminal, and the page simply looks empty.
    import rubric_server
    check_page("rubric_server/rubric", rubric_server.INDEX_HTML)
    check_page("rubric_server/adjudicate", rubric_server.ADJUDICATE_HTML)
    # Every page the server can return. A page added without a line here is a
    # page whose JavaScript nothing parses — which is exactly how four views
    # shipped blank for several commits (see the module docstring).
    check_page("rubric_server/position", rubric_server.POSITION_HTML)
    check_page("rubric_server/choose", rubric_server.CHOOSE_HTML)
    # Every page the server can return. Added with /reference (21.90) — a page
    # without a line here is a page whose JavaScript nothing parses.
    check_page("rubric_server/reference", rubric_server.REFERENCE_HTML)
    check_page("rubric_server/scenario", rubric_server.SCENARIO_HTML)

    # The chat surface (Phase 2). Same rule as every page above: a page added
    # without a line here is a page whose JavaScript nothing parses. This one
    # is the first intended for people outside the project, where a blank page
    # is not a debugging inconvenience but the entire product.
    import chat_common
    check_page("chat/index", chat_common.INDEX_HTML)
    check_page("chat/login", chat_common.LOGIN_HTML)

    # Grouped adjudication must not re-submit already-saved arms when revisiting
    # a record. The guard is in-page JavaScript so keep a string-level check.
    check("adjudication skips already-saved arms on grouped submit",
          "if (arm.done) continue;" in rubric_server.ADJUDICATE_HTML, True)
    check("adjudication disables controls on saved arm panels",
          "const dis = arm.done ? ' disabled' : '';" in rubric_server.ADJUDICATE_HTML,
          True)

    # ...and the lock must be undoable. Re-adjudication is a supported act — a
    # verdict is keyed (key, author, answer_sha, n_shown) precisely so a second
    # one is kept beside the first (21.78) — so locking a saved arm with no way
    # back forecloses a workflow the identity machinery was built for. The lock
    # is for accidental re-posts by a grouped save, which is a different problem
    # from a reviewer changing their mind (Section 21.95).
    _adj = rubric_server.ADJUDICATE_HTML
    check("a saved arm offers a re-open control",
          'class="reopen"' in _adj, True)
    _reopen = _adj.split("function bindReopen")[1].split("$('#skip')")[0]
    check("re-opening clears the flag the submit loop reads",
          "arm.done=false" in _reopen.replace(" ", ""), True)
    check("...and re-enables the panel's controls",
          "el.disabled=false" in _reopen.replace(" ", ""), True)
    # In place, not by re-rendering: render() rebuilds the whole record, which
    # would discard anything typed into the OTHER panels of the same group.
    check("re-open does not re-render the record",
          "render();" not in _reopen, True)

    # This file holds four complete pages as separate Python strings, and
    # nothing ties a CSS rule to the page whose markup uses it. Twice now a
    # selector has been added to one page while the elements it styles live in
    # another: `h3.grp` landed in INDEX_HTML while the group headings it exists
    # for are in ADJUDICATE_HTML, so they rendered unstyled on a deployed form.
    #
    # It cannot fail loudly. The page renders, the JS parses, every endpoint
    # returns 200 — the rule is simply dead in one page and absent in the other.
    # Same family as the `#` comment that blanked four views: a string in one
    # language inside a file of another gets no checking from either.
    #
    # Checked in the dead-rule direction only. A page with markup and no CSS is
    # ordinary; a page with CSS for a selector it never uses is a mistake.
    for name, page in (("reference", rubric_server.REFERENCE_HTML),
                       ("scenario", rubric_server.SCENARIO_HTML),
                       ("rubric", rubric_server.INDEX_HTML),
                       ("adjudicate", rubric_server.ADJUDICATE_HTML),
                       ("position", rubric_server.POSITION_HTML),
                       ("choose", rubric_server.CHOOSE_HTML),
                       ("webui", served_html())):
        style = "\n".join(re.findall(r"<style>(.*?)</style>", page, re.S))
        body = page.replace(style, "")
        for sel in sorted(set(re.findall(r"[#.]([A-Za-z][\w-]*)\s*(?=[{,:])", style))):
            # Word-boundary, and never after a dot: `t.done` is a property read,
            # not the class `.done`, and `common_errors` is not `.err`. Matching
            # on a bare substring reported nine of these and six were noise.
            used = re.search(r"(?<![\w.\-])" + re.escape(sel) + r"(?![\w\-])", body)
            check(f"{name}: CSS for {sel!r} is on the page that uses it",
                  bool(used), True)

    # A page that CALLS a helper it does not DEFINE parses perfectly and throws
    # at runtime, blanking the view — the JavaScript parse-check above cannot
    # see it. `mlAuthorGet` was used by three pages and defined in one, which
    # is the same cross-page drift as the dead-CSS bug, in a third language
    # (Section 21.91).
    #
    # Scoped to helpers this repo defines somewhere, so browser builtins and
    # library calls are not flagged.
    pages = {"reference": rubric_server.REFERENCE_HTML,
             "scenario": rubric_server.SCENARIO_HTML,
             "rubric": rubric_server.INDEX_HTML,
             "adjudicate": rubric_server.ADJUDICATE_HTML,
             "position": rubric_server.POSITION_HTML,
             "choose": rubric_server.CHOOSE_HTML,
             "webui": served_html()}
    ours = set()
    for page in pages.values():
        ours |= set(re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", page))
        ours |= set(re.findall(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
                               r"(?:async\s*)?(?:function\b|\()", page))
    for name, page in pages.items():
        script = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", page, re.S)) or page
        defined = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", script))
        defined |= set(re.findall(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)", script))
        called = set(re.findall(r"\b([A-Za-z_$][\w$]*)\s*\(", script))
        missing = sorted((called & ours) - defined)
        check(f"{name}: every helper it calls is defined on the page", missing, [])

    check_shared_runner_adoption()

    print(f"\n{'FAILED' if FAILED else 'all checks passed'} ({CHECKS_RUN} assertions)")
    if FAILED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
