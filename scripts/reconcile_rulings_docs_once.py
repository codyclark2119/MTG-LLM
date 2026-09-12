"""One-time branch helper for the code-review documentation reconciliation.

This file is deleted before merge. It exists only because GitHub's contents API
replaces whole files, while the two target files are large and need exact
in-place edits.
"""

from pathlib import Path


def replace_exact(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"{path}: expected text not found exactly")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_exact(
    "scripts/common.py",
    """# `ruling_chunks.jsonl` is pinned but currently READ BY NOTHING: it is ingested
# and never wired into retrieval. Pinned anyway, because the moment it is wired
# in is the moment nobody will think to pin it.
""",
    """# `ruling_chunks.jsonl` is also on the active retrieval path:
# `retrieve_hybrid.RulingIndex` reads it for official card rulings, and the chat
# server constructs that index for served answers. Pinning it therefore protects
# both evaluation provenance and the live serving surface from silent corpus drift.
""",
)

replace_exact(
    "STRUCTURAL_AUDIT.md",
    """**`ruling_chunks.jsonl` is read by no script.** 19,726 chunks ingested and never
wired into retrieval. It backs no published number; it is simply not doing
anything.
""",
    """> **Historical finding, now resolved.** At the time of this audit,
> `ruling_chunks.jsonl` had been ingested but was not yet wired into retrieval.
> That is no longer the current state: `retrieve_hybrid.RulingIndex` reads the
> pinned corpus, and the chat server constructs that index for official-ruling
> retrieval. The original finding is retained here as provenance rather than
> presented as a current limitation.
""",
)

docs = Path("scripts/test_docs.py")
text = docs.read_text(encoding="utf-8")
anchor = "    # 2. No current-state file may claim the chat surface serves the 7B.\n"
guard = """    # The two files that previously carried the opposite claim must not
    # regress to saying the rulings corpus is unused.
    stale_rulings_claims = (
        "ruling_chunks.jsonl` is read by no script",
        "ruling_chunks.jsonl` is pinned but currently READ BY NOTHING",
        "never wired into retrieval",
    )
    for name in ("scripts/common.py", "STRUCTURAL_AUDIT.md"):
        body = (REPO_ROOT / name).read_text(encoding="utf-8")
        for claim in stale_rulings_claims:
            if claim in body:
                problems.append(
                    f"{name} still contains the superseded rulings-corpus claim "
                    f"{claim!r}; retrieve_hybrid.RulingIndex is on the active path")

"""
if guard not in text:
    if anchor not in text:
        raise SystemExit("scripts/test_docs.py: insertion anchor not found")
    docs.write_text(text.replace(anchor, guard + anchor, 1), encoding="utf-8")
