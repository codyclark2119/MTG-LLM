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


def check_requirement_tiers() -> None:
    """The dependency tiers must agree with the lock, version for version.

    Splitting one requirements file into four creates four chances for a
    version to drift, which is the same shape as every other duplicated-value
    bug this repo has paid for. So the lock stays the single source of truth
    and this asserts the others quote it rather than restate it.

    What is deliberately NOT asserted: that the lock contains only what is
    imported. `requirements/research.txt` documents five pins nothing imports,
    and they stay in the lock on purpose — a reproducibility artifact describes
    the environment the numbers were measured in, not today's import graph.
    """
    import re

    def pins(path: Path) -> dict[str, str]:
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z0-9._-]+)(?:\[[^\]]*\])?==(.+)$", line)
            if m:
                out[m.group(1).lower().replace("_", "-")] = m.group(2).strip()
        return out

    lock = pins(REPO / "requirements.txt")
    check("the lock is a real freeze, not a short list", len(lock) > 50, True)

    for tier in ("base", "ci", "research"):
        path = REPO / "requirements" / f"{tier}.txt"
        check(f"requirements/{tier}.txt exists", path.exists(), True)
        if not path.exists():
            continue
        for name, version in pins(path).items():
            check(f"{tier}.txt pins {name} to the lock's version",
                  version, lock.get(name))

    # base.txt is the import graph written down; the six packages CLAUDE.md
    # names as directly imported must all be in it.
    base = pins(REPO / "requirements" / "base.txt")
    for direct in ("mlx-lm", "mlx-embeddings", "numpy", "datasets",
                   "fastapi", "uvicorn"):
        check(f"base.txt lists the directly-imported {direct}",
              direct in base, True)

    # ci.txt must be installable WITHOUT Apple Silicon, so no mlx in it.
    ci = pins(REPO / "requirements" / "ci.txt")
    check("ci.txt pulls in no mlx package",
          [n for n in ci if n.startswith("mlx")], [])
    check("ci.txt has httpx, which the async chat tests need",
          "httpx" in ci, True)

    # And the workflow must install that file rather than its own list.
    workflow = REPO / ".github" / "workflows" / "ci.yml"
    check("a CI workflow exists", workflow.exists(), True)
    if workflow.exists():
        body = workflow.read_text(encoding="utf-8")
        check("CI installs requirements/ci.txt",
              "requirements/ci.txt" in body, True)
        check("CI does not install the Apple-Silicon lock",
              re.search(r"-r\s+requirements\.txt", body) is not None, False)
        check("CI runs the canonical runner",
              "scripts/run_tests.sh" in body, True)


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

    check_requirement_tiers()

    # `fly launch` without --copy-config regenerates these at the root and they
    # override deploy/. Their reappearance is the failure, not a nuisance.
    for stray in ("fly.toml", "Dockerfile", ".github/workflows/fly-deploy.yml"):
        check(f"no auto-generated {stray} at the repo root",
              (REPO / stray).exists(), False)

    # The `dockerfile` key resolves against THIS FILE's directory, while COPY
    # paths resolve against the build context (the cwd at deploy time). Two
    # bases, and `--copy-config` mixed them: it rewrote 'Dockerfile' to
    # 'deploy/Dockerfile', which fly resolved to deploy/deploy/Dockerfile and
    # failed at deploy time. Cheap to check here, expensive to find there.
    import tomllib
    cfg_path = REPO / "deploy/fly.toml"
    cfg = tomllib.loads(cfg_path.read_text())
    dockerfile = cfg.get("build", {}).get("dockerfile", "Dockerfile")
    resolved = (cfg_path.parent / dockerfile).resolve()
    check(f"fly.toml's dockerfile resolves to a real file ({dockerfile} -> "
          f"{resolved.relative_to(REPO) if resolved.is_relative_to(REPO) else resolved})",
          resolved.is_file(), True)
    check("...and it is the narrow one, not a root Dockerfile",
          resolved == (REPO / "deploy/Dockerfile").resolve(), True)

    # A mount the app declares but the platform has no volume for is a machine
    # that will not start; worse, without it a redeploy discards submissions.
    check("submissions are on a mounted volume",
          [m["destination"] for m in cfg.get("mounts", [])], ["/data"])
    check("SUBMISSIONS_FILE lives on that mount",
          cfg["env"]["SUBMISSIONS_FILE"].startswith("/data/"), True)

    # The reviewer must never see what the judge said about the answer they are
    # grading. `source_run` was added to the queue for provenance (21.62) and
    # very nearly travelled with the task — a run is called
    # `pos_n24_verbose_32b.jsonl`, which names the judge, which is exactly what
    # the assertion exists to withhold. Untested until now: the check had no
    # case, so widening it could not be verified and narrowing it would have
    # gone unnoticed.
    import json as _json
    import tempfile

    from rubric_server import load_tasks

    ok_task = {"key": "p::a", "record_id": "p", "arm": "a", "question": "Q",
               "answer": "PASS", "common_errors": ["x"], "key_points": []}

    def _write(tasks):
        f = Path(tempfile.mkdtemp()) / "tasks.json"
        f.write_text(_json.dumps({"kind": "adjudication", "tasks": tasks}), encoding="utf-8")
        return f

    def _loads(tasks) -> bool:
        try:
            load_tasks(_write(tasks))
            return True
        except SystemExit:
            return False

    check("a clean adjudication task loads", _loads([ok_task]), True)
    for leak in ("errors_made", "judge_model", "blundered", "run", "judge", "source_run"):
        check(f"a task carrying {leak!r} is refused",
              _loads([dict(ok_task, **{leak: "anything"})]), False)

    # "Done" must mean "this author graded THIS TEXT", not "this key". The key
    # record_id::arm survives a regeneration of the arms while the answer behind
    # it does not, so comparing keys alone marks already-graded keys done and
    # hides exactly the tasks needing a fresh verdict (Section 21.62).
    from rubric_server import (_answer_sha as _sha, group_adjudication_tasks,
                               validate_reference_input, validate_review_input)

    grouped = group_adjudication_tasks([
        {"record_id": "p", "question": "Q", "key": "p::a", "arm": "a",
         "answer": "PLAY Swamp\nPASS", "done": False},
        {"record_id": "p", "question": "Q", "key": "p::b", "arm": "b",
         "answer": "PLAY Forest\nPASS", "done": True},
        {"record_id": "q", "question": "Q2", "key": "q::a", "arm": "a",
         "answer": "PASS", "done": False},
    ])
    check("tasks group by record_id with per-arm rows preserved",
          [{g["record_id"]: [a["key"] for a in g["arms"]] for g in grouped}],
          [{"p": ["p::a", "p::b"], "q": ["q::a"]}])

    # Reference/review rows are attribution-bearing judgements and must never
    # accept empty authors, mirroring rubric/adjudication validation.
    recs = {"p", "q"}
    check("reference submission requires an author",
          validate_reference_input({"record_id": "p",
                                    "reference_actions": ["PASS"],
                                    "author": ""}, recs),
          ["author is required for attribution"])
    check("reference submission validates unknown records",
          validate_reference_input({"record_id": "z",
                                    "reference_actions": ["PASS"],
                                    "author": "me"}, recs),
          ["unknown record"])
    check("review submission requires an author",
            validate_review_input({"record_id": "p", "kind": "split",
                                 "note": "two decisions", "author": ""}, recs),
          ["author is required for attribution"])
    check("review submission requires note when flagging",
            validate_review_input({"record_id": "p", "kind": "split",
                                 "note": "", "author": "me"}, recs),
          ["say what needs changing"])

    _tasks = [{"key": "p::a", "answer": "PLAY Swamp\nPASS"}]
    _live = {t["key"]: _sha(t["answer"]) for t in _tasks}

    def _done(sub):
        return (sub.get("author") == "me" and "errors_present" in sub
                and sub["key"] in _live and sub.get("answer_sha") == _live[sub["key"]])

    check("a verdict on the same text is done",
          _done({"key": "p::a", "author": "me", "errors_present": [1],
                 "answer_sha": _sha("PLAY Swamp\nPASS")}), True)
    check("a verdict on different text is NOT done",
          _done({"key": "p::a", "author": "me", "errors_present": [1],
                 "answer_sha": _sha("PHASE upkeep\nPLAY Swamp\nPASS")}), False)
    # No digest means it predates the field, so it is about older text by
    # construction. Reopening asks for one visible duplicate verdict; the
    # alternative silently skips work that needs redoing.
    check("a verdict with no digest reopens",
          _done({"key": "p::a", "author": "me", "errors_present": [1]}), False)
    check("the server digest matches the local one",
          _sha("PLAY Swamp\nPASS"), __import__("hashlib").sha256(
              "PLAY Swamp\nPASS".encode()).hexdigest()[:12])

    from rubric_server import prioritize_tasks
    check("unfinished tasks sort before done ones",
          [x["key"] for x in prioritize_tasks(
              [{"key": "done::a", "done": True},
               {"key": "open::b", "done": False},
               {"key": "open::a", "done": False}],
              done_key="done")],
          ["open::b", "open::a", "done::a"])

    print(f"\n{'FAILED' if FAILED else 'all checks passed'} ({CHECKS_RUN} assertions)")
    if FAILED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
