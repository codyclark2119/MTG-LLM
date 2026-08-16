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


def main() -> None:
    files = sorted(p for p in [*SCRIPTS.glob("*.py"), *SCRIPTS.glob("gameplay/*.py")]
                   if p.name not in SKIP)
    failed = 0
    for path in files:
        missing = unresolved(path)
        if missing:
            failed += 1
            rel = path.relative_to(SCRIPTS.parent)
            print(f"  FAIL {rel}: uses but never binds {missing}")
    if failed:
        print(f"\n{failed} file(s) call a name they never import.")
        raise SystemExit(1)
    print(f"all {len(files)} scripts resolve every name they use")


if __name__ == "__main__":
    main()
