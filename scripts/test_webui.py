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

    print(f"\n{'FAILED' if FAILED else 'all checks passed'} ({CHECKS_RUN} assertions)")
    if FAILED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
