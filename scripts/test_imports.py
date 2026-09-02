"""Assert no script calls a name it never imported.

Why this exists. `eval.py` called `build_rag_messages` without importing it
from commit 3ee1ea5 onward. Every cheap check passed the whole time:

    import eval            -> fine, the name is only looked up when called
    python eval.py --help  -> fine, argparse exits before generation
    --rescore-from         -> fine, that path never builds a prompt

So the script's MAIN path — generate answers, then judge them — raised
`NameError` on its first real invocation, three commits and one repository
review later. CLAUDE.md's "verify by running, not by reading" says a --help
that exits 0 proves almost nothing; this is the check that makes that
concrete, because the expensive path can cost hours to reach.

Static, so it costs nothing and needs no model:

    python scripts/test_imports.py
"""

import ast
import builtins
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
SKIP = {"test_imports.py"}


def bound_names(tree: ast.AST) -> set[str]:
    """Every name the module binds anywhere — import, def, class, assignment.

    Deliberately generous about scope: a name bound inside one function and
    used in another would not be caught. That is fine. This is a smoke test
    for the specific failure of calling something never brought into the
    module at all, not a type checker.
    """
    out: set[str] = set(dir(builtins)) | {
        "__doc__", "__name__", "__file__", "__spec__", "__package__", "__builtins__",
    }
    for n in ast.walk(tree):
        if isinstance(n, ast.alias):
            out.add(n.asname or n.name.split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.arguments):
            out |= {a.arg for a in (*n.posonlyargs, *n.args, *n.kwonlyargs)}
            for extra in (n.vararg, n.kwarg):
                if extra:
                    out.add(extra.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global):
            out |= set(n.names)
    return out


def unresolved(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(used - bound_names(tree))


def duplicated_vocabularies(files) -> dict[str, list[str]]:
    """Collection constants DEFINED at module level in more than one file.

    Consolidating the four that existed does not stop a fifth appearing, and a
    duplicate is invisible until the copies drift — at which point the symptom
    is a value that is valid in one half of the program and rejected by the
    other (Section 21.97).

    Only ASSIGNMENTS count, so a re-export (`from common import CATEGORIES`) is
    correctly not a definition. That is the point of the pattern: one home, any
    number of doors.

    Names are compared, not contents. Two lists that happen to agree today are
    exactly the case worth failing on, because agreeing today is what all four
    of these did. `_STOP` is allowed: those two are genuinely different lists
    for different jobs — contamination overlap and rubric lint — and the right
    fix there is distinct names, not a merge.
    """
    from collections import defaultdict

    allowed = {"_STOP"}
    seen = defaultdict(list)
    # Tests are excluded: a fixture named after the thing it stands in for is a
    # fixture, not a second home. `test_eval_positions` defines its own `ARMS`
    # precisely so it does not depend on the real one.
    for path in [p for p in files if not p.name.startswith("test_")]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Name) and target.id.isupper()
                        and len(target.id) > 3 and target.id not in allowed):
                    seen[target.id].append(path.name)
    return {k: v for k, v in seen.items() if len(v) > 1}


# Names that legitimately recur: CLI entry points, and helpers whose two copies
# are genuinely different code that happens to share a generic verb. Each is
# here because it was checked, not because the check was noisy.
DUPLICATE_FUNCTION_ALLOWED = {
    "main", "check", "build_app", "emit", "normalize", "validate",
    "export_tasks", "parse_worksheet", "read_submissions", "stratified_split",
    "compare_judges",
}


def duplicated_helpers(files: list[Path], threshold: float = 0.85) -> dict:
    """Module-level functions defined twice with NEARLY THE SAME BODY.

    `duplicated_vocabularies` above catches a CONSTANT with two homes, and it
    caught a real one. It does not catch a duplicated FUNCTION, because it
    only inspects `ast.Assign` targets that are uppercase — and a duplicated
    helper is precisely the trap this repo records: five jsonl readers, three
    of which crashed on a trailing blank line because the guard was added to
    some copies and not others.

    Similarity is compared on the UNPARSED body, so formatting and comments do
    not hide a copy. A shared name over genuinely different code is a weaker
    smell and is allowed above by name; near-identical bodies are the thing
    that rots, so the threshold is high and the failure is loud.
    """
    import difflib

    bodies: dict[str, list[tuple[str, str]]] = {}
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in DUPLICATE_FUNCTION_ALLOWED or node.name.startswith("_"):
                    continue
                bodies.setdefault(node.name, []).append((path.name, ast.unparse(node)))

    out = {}
    for name, seen in bodies.items():
        if len(seen) < 2:
            continue
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                ratio = difflib.SequenceMatcher(None, seen[i][1], seen[j][1]).ratio()
                if ratio >= threshold:
                    out[name] = (seen[i][0], seen[j][0], ratio)
    return out


def main() -> None:
    files = sorted(p for p in [*SCRIPTS.glob("*.py"), *SCRIPTS.glob("gameplay/*.py")]
                   if p.name not in SKIP)
    failed = 0
    dupes = duplicated_vocabularies(files)
    for name, where in sorted(dupes.items()):
        failed += 1
        print(f"  FAIL {name} is defined in {len(where)} modules: "
              f"{', '.join(where)} — give it one home and re-export")
    for name, (a, b, ratio) in sorted(duplicated_helpers(files).items()):
        failed += 1
        print(f"  FAIL {name}() is {ratio:.0%} identical in {a} and {b} — "
              f"import one copy; a guard added to only one is the documented trap")
    for path in files:
        missing = unresolved(path)
        if missing:
            failed += 1
            rel = path.relative_to(SCRIPTS.parent)
            print(f"  FAIL {rel}: uses but never binds {missing}")
    if failed:
        print(f"\n{failed} problem(s): a name used but never bound, "
              "or a vocabulary with more than one home.")
        raise SystemExit(1)
    print(f"all {len(files)} scripts resolve every name they use, "
          f"and no vocabulary is defined twice")


if __name__ == "__main__":
    main()
