"""What may leave this machine. Run: python scripts/test_deploy.py

WHY

`webui.py` is LAN-only: its script runner executes training and evaluation on
the host, and its store writes the gold set. `rubric_server.py` is the
deployable half, and the boundary between them is enforced by exactly two
files — `deploy/Dockerfile`'s COPY list and the root `.dockerignore`.

Both have already failed once. `fly launch` auto-detected the repo as a generic
Python app and generated a root Dockerfile with `COPY . .`: 5.0 GB including the
gold set, the adjudication queue, 3.8 GB of adapters and `scripts/webui.py`,
plus a GitHub workflow that would have deployed it on the next push to main.
Nothing shipped, but nothing stopped it either.

So this asserts the boundary rather than trusting it:

  * the build context contains EXACTLY the files the Dockerfile copies — no
    more (the gold set must not travel) and no fewer (a COPY of a file the
    context excludes fails the build at deploy time, which is the worst moment
    to find out);
  * the COPY list contains nothing that must stay local;
  * `common.py` is pure stdlib, which is what makes the deployable half
    deployable at all;
  * no root fly.toml or Dockerfile has reappeared.

The context check matters independently of the image: fly uploads the whole
context to its remote builder BEFORE any COPY runs, so an over-broad context
transmits the gold set even when the image never contains it.
"""

import fnmatch
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

REPO = Path(__file__).resolve().parent.parent
CHECKS_RUN = 0
FAILED = 0

# Things that must never reach a public host, by substring of their path.
FORBIDDEN = ("webui.py", "label_store.py", "gold_questions.jsonl",
             "positions.jsonl", "judge_adjudications.jsonl",
             "adjudication_queue.json", "data/processed", "eval/runs", "models/")


def check(label: str, got, want) -> None:
    global CHECKS_RUN, FAILED
    CHECKS_RUN += 1
    if got != want:
        FAILED += 1
        print(f"  FAIL {label}\n       got  {got!r}\n       want {want!r}")


def copied_paths() -> set[str]:
    """Source paths in deploy/Dockerfile's COPY lines."""
    out = set()
    for line in (REPO / "deploy/Dockerfile").read_text().splitlines():
        m = re.match(r"\s*COPY\s+(?!--from)(.+)", line)
        if not m:
            continue
        parts = m.group(1).split()
        out.update(parts[:-1])          # last token is the destination
    return out


def context_files() -> set[str]:
    """Files Docker would upload, by last-matching-pattern.

    Every parent directory is handled explicitly in `.dockerignore`, so this
    does not depend on the builder's 'descend into an excluded directory'
    behaviour and last-match-wins models the file decisions faithfully.
    """
    pats = []
    for line in (REPO / ".dockerignore").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            neg = line.startswith("!")
            pats.append((neg, line[1:] if neg else line))

    def keep(rel: str) -> bool:
        decided, parts = True, rel.split("/")
        for neg, pat in pats:
            for i in range(1, len(parts) + 1):
                sub = "/".join(parts[:i])
                if fnmatch.fnmatch(sub, pat) or (
                        pat.startswith("**/") and fnmatch.fnmatch(parts[i - 1], pat[3:])):
                    decided = neg
                    break
        return decided

    found = set()
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs
                   if d not in (".git", "mlx_env", "models", "__pycache__", "node_modules")]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), REPO).replace(os.sep, "/")
            if keep(rel):
                found.add(rel)
    return found


def main() -> None:
    copied, context = copied_paths(), context_files()

    # The two lists must agree exactly. Extra context is an exposure; missing
    # context is a build that fails at deploy time.
    check("context == the Dockerfile's COPY list", sorted(context), sorted(copied))
    check("context is not empty", bool(context), True)

    for group, name in ((copied, "COPY list"), (context, "build context")):
        for item in sorted(group):
            for bad in FORBIDDEN:
                check(f"{name} excludes {bad} (saw {item})", bad in item, False)

    # common.py is rubric_server's only project import; a non-stdlib import
    # there would drag the model stack onto a 256 MB public VM.
    import ast
    src = (REPO / "scripts/common.py").read_text()
    tree = ast.parse(src)
    mods = {n.module.split(".")[0] for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module}
    mods |= {a.name.split(".")[0] for n in ast.walk(tree)
             if isinstance(n, ast.Import) for a in n.names}
    check("common.py is pure stdlib", sorted(mods - set(sys.stdlib_module_names)), [])

    reqs = [l for l in (REPO / "deploy/requirements.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    check("deploy/requirements.txt stays tiny", len(reqs) <= 4, True)

    # `fly launch` without --copy-config regenerates these at the root and they
    # override deploy/. Their reappearance is the failure, not a nuisance.
    for stray in ("fly.toml", "Dockerfile", ".github/workflows/fly-deploy.yml"):
        check(f"no auto-generated {stray} at the repo root",
              (REPO / stray).exists(), False)

    print(f"\n{'FAILED' if FAILED else 'all checks passed'} ({CHECKS_RUN} assertions)")
    if FAILED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
