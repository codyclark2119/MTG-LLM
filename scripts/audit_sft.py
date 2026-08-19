"""Audit a training set before training on it. Contamination is an assertion.

    python scripts/audit_sft.py data/datasets/verified
    python scripts/audit_sft.py data/datasets            # the synthetic set
    python scripts/audit_sft.py data/datasets/verified --report-out eval/reports/sft_verified.md

WHY

Nothing checked training data before it was trained on. The 18% refusal rate
that explains run 3's fine-tuned arm (Section 21.6) was found by reading the
file, months after the adapter it produced was published.

Five properties, one of which is a hard failure:

  contamination     eval questions present in the training set. THE assertion.
                    It is the only failure here that invalidates everything
                    downstream *and looks like an improvement while doing it*,
                    so it exits non-zero rather than printing a warning.
  refusals          targets that teach the model to decline (Section 21.6)
  grounding         targets citing a comprehensive-rule id inline, which is the
                    largest clean effect in the project (Section 19.1)
  duplicates        exact and near-duplicate targets, which inflate the
                    effective epoch count on whatever they repeat
  lengths           the distribution, because a set of one-liners and a set of
                    paragraphs train differently at identical line counts

MATCHING BY TEXT, NOT BY ID

`build_sft_verified.py` already excludes eval records by id and asserts it. This
checks something that cannot: a question that reaches the training set under a
*different* id, or with the card names swapped. RulesGuru re-randomizes card and
player names per request, so the same underlying ruling legitimately appears
twice with different nouns — which means an id filter can pass while the model
still sees the eval question.

So two thresholds, deliberately different in kind:

  exact (normalized) match   -> CONTAMINATION, exit 1
  high token overlap         -> reported for review, never auto-failed

The second is a judgement call a person has to make. Two questions about the
same card are often genuinely different questions, and hard-failing on overlap
would make the check something people route around. The first is not a
judgement call.
"""

import argparse
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (  # noqa: E402
    GOLD_PATH,
    REFUSAL_RE,
    REPO_ROOT,
    RULE_ID_RE,
    read_jsonl,
)

EVAL_SETS_DIR = REPO_ROOT / "eval/sets"

# Files in eval/sets/ that are SOURCE POOLS, not eval sets, and are therefore
# excluded from the contamination check by default.
#
# `rulesguru_candidates.jsonl` is the 1,202-record pool `build_sft_verified.py`
# draws from. 72 of those were promoted into the gold set and are excluded by
# id; training on the other 1,130 is the entire point. Checked as an eval set it
# reports 1,130 of 1,130 lines contaminated, which is true, useless, and the
# kind of alarming-looking number that gets a check disabled rather than read.
#
# The real consequence is worth stating rather than hiding: a model trained on
# this pool can never again be evaluated on the pool. Only the 99 promoted gold
# records are a valid eval set for it, which is exactly what `--gold` defaults
# to. Pass `--eval-sets` explicitly to override.
SOURCE_POOLS = {"rulesguru_candidates.jsonl"}

# Content words only. Magic prose is dense with these, so dropping them stops
# two unrelated questions scoring 0.3 similarity on grammar alone.
_STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "do",
    "does", "did", "have", "has", "had", "i", "you", "it", "its", "he", "she",
    "they", "them", "his", "her", "their", "of", "to", "in", "on", "at", "for",
    "with", "from", "by", "as", "into", "my", "me", "we", "can", "will",
    "would", "what", "when", "how", "why", "which", "who", "does", "so", "not",
}
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1}


def normalize(text: str) -> str:
    """Case- and whitespace-insensitive, punctuation-insensitive form.

    Equality on this is what triggers the contamination assertion, so it is
    deliberately conservative: it forgives reformatting and nothing else.
    """
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def question_of(rec: dict) -> str:
    """The question inside a training line.

    `build_rag_messages` emits either a bare question or
    "<context>\\n\\nQuestion: <question>". Splitting on the last marker
    recovers the question in both shapes; a context block that happens to
    contain the marker itself would otherwise take the wrong half.
    """
    msgs = rec.get("messages") or []
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    return user.rpartition("\n\nQuestion: ")[2] if "\n\nQuestion: " in user else user


def target_of(rec: dict) -> str:
    msgs = rec.get("messages") or []
    return next((m["content"] for m in reversed(msgs) if m.get("role") == "assistant"), "")


def eval_questions(paths: list[Path]) -> list[tuple[str, str]]:
    """[(source label, question text)] from gold records and eval sets.

    Both shapes appear: gold records store the question in `messages[1]`, and
    some eval sets carry a flat `question` field.
    """
    out = []
    for p in paths:
        for rec in read_jsonl(p):
            q = rec.get("question")
            if not q:
                msgs = rec.get("messages") or []
                q = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            if q and q.strip():
                out.append((f"{p.name}:{rec.get('id') or rec.get('gold_id') or '?'}", q))
    return out


def near_duplicate_pairs(texts: list[str], threshold: float, rare_max: int = 40):
    """Pairs above `threshold`, without the O(n^2) comparison.

    An inverted index over RARE tokens — those appearing in at most `rare_max`
    records — supplies the candidate pairs. Two texts similar enough to matter
    necessarily share an uncommon word; two that share only common words cannot
    reach a high Jaccard. This is a prune on the REPORT path only. The
    contamination assertion below never uses it, because a prune that is wrong
    once is a missed contamination, and that is the failure this file exists to
    prevent.
    """
    toks = [tokens(t) for t in texts]
    index: dict[str, list[int]] = {}
    for i, ts in enumerate(toks):
        for t in ts:
            index.setdefault(t, []).append(i)

    seen, pairs = set(), []
    for t, ids in index.items():
        if len(ids) > rare_max:
            continue
        for a_i in range(len(ids)):
            for b_i in range(a_i + 1, len(ids)):
                key = (ids[a_i], ids[b_i])
                if key in seen:
                    continue
                seen.add(key)
                s = jaccard(toks[key[0]], toks[key[1]])
                if s >= threshold:
                    pairs.append((s, key[0], key[1]))
    pairs.sort(reverse=True)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path,
                    help="directory holding train.jsonl and valid.jsonl")
    ap.add_argument("--gold", type=Path, default=GOLD_PATH)
    ap.add_argument("--eval-sets", type=Path, nargs="*", default=None,
                    help="default: every .jsonl in eval/sets/")
    ap.add_argument("--near-threshold", type=float, default=0.75,
                    help="token overlap at which a pair is reported for review")
    ap.add_argument("--report-out", type=Path, default=None)
    ap.add_argument("--allow-contamination", action="store_true",
                    help="report contamination without exiting non-zero. For "
                         "inspecting a set you already know is dirty — never "
                         "for one you intend to train on.")
    args = ap.parse_args()

    splits = {}
    for name in ("train", "valid"):
        p = args.dataset_dir / f"{name}.jsonl"
        if p.exists():
            splits[name] = read_jsonl(p)
    if not splits:
        raise SystemExit(f"no train.jsonl or valid.jsonl in {args.dataset_dir}")

    if args.eval_sets is not None:
        eval_paths, skipped = list(args.eval_sets), []
    else:
        found = sorted(EVAL_SETS_DIR.glob("*.jsonl"))
        eval_paths = [p for p in found if p.name not in SOURCE_POOLS]
        skipped = [p for p in found if p.name in SOURCE_POOLS]
    eval_paths = [args.gold] + [p for p in eval_paths if p != args.gold]
    eval_paths = [p for p in eval_paths if p.exists()]
    evals = eval_questions(eval_paths)
    eval_norm = {}
    for label, q in evals:
        eval_norm.setdefault(normalize(q), label)

    L = [f"# SFT audit — `{args.dataset_dir}`", ""]
    L.append(f"Checked against {len(evals)} eval questions from "
             + ", ".join(f"`{p.name}`" for p in eval_paths) + ".")
    if skipped:
        L.append("")
        L.append("Excluded as a **source pool, not an eval set**: "
                 + ", ".join(f"`{p.name}`" for p in skipped)
                 + ". The training set is drawn from it by design. The consequence is "
                 "real and worth stating: a model trained here can never be evaluated "
                 "on that pool — only on the records promoted into the gold set.")
    L.append("")

    all_rows = [(s, r) for s, rows in splits.items() for r in rows]
    print(f"{args.dataset_dir}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # --- 1. contamination: exact, brute force, no pruning -------------------
    hits = []
    for split, rec in all_rows:
        n = normalize(question_of(rec))
        if n and n in eval_norm:
            hits.append((split, eval_norm[n], question_of(rec)[:90]))

    # --- 2. contamination: near, reported ------------------------------------
    eval_toks = [(label, tokens(q)) for label, q in evals]
    near = []
    for split, rec in all_rows:
        qt = tokens(question_of(rec))
        if not qt:
            continue
        best = max(((jaccard(qt, et), label) for label, et in eval_toks), default=(0.0, ""))
        if best[0] >= args.near_threshold:
            near.append((best[0], split, best[1], question_of(rec)[:90]))
    near.sort(reverse=True)

    L += ["## Contamination", ""]
    if hits:
        L.append(f"**{len(hits)} training questions are eval questions verbatim.**")
        L.append("")
        L += ["| Split | Eval record | Question |", "| --- | --- | --- |"]
        L += [f"| {s} | `{lbl}` | {q}… |" for s, lbl, q in hits[:20]]
    else:
        L.append(f"**No exact matches** across {len(all_rows)} training lines "
                 f"x {len(evals)} eval questions (brute force, no pruning).")
    L.append("")
    if near:
        L.append(f"{len(near)} pairs above {args.near_threshold:.0%} token overlap — "
                 "**for review, not an automatic failure.** RulesGuru re-randomizes "
                 "card and player names, so the same ruling legitimately appears "
                 "twice with different nouns.")
        L += ["", "| Overlap | Split | Nearest eval record | Question |", "| --- | --- | --- | --- |"]
        L += [f"| {s:.0%} | {sp} | `{lbl}` | {q}… |" for s, sp, lbl, q in near[:20]]
    else:
        L.append(f"No pair reaches {args.near_threshold:.0%} token overlap with an eval question.")
    L.append("")

    # --- 3. target quality ---------------------------------------------------
    L += ["## Targets", "", "| Split | Lines | Refusal-shaped | Cites a rule inline | Empty |",
          "| --- | --- | --- | --- | --- |"]
    for split, rows in splits.items():
        tg = [target_of(r) for r in rows]
        n = len(tg) or 1
        ref = sum(1 for t in tg if REFUSAL_RE.search(t or ""))
        cit = sum(1 for t in tg if RULE_ID_RE.search(t or ""))
        empty = sum(1 for t in tg if not (t or "").strip())
        L.append(f"| {split} | {len(tg)} | {ref} ({ref / n:.1%}) | {cit} ({cit / n:.0%}) | {empty} |")
    L.append("")
    L.append("> Section 21.6 measured **18%** refusals in the synthetic set and **0.1%** in "
             "the verified one. A refusal target teaches the model to decline.")
    L.append("")

    # --- 4. duplicates -------------------------------------------------------
    train = splits.get("train", [])
    targets = [target_of(r) for r in train]
    dup_exact = [(t, c) for t, c in Counter(normalize(t) for t in targets).items() if c > 1 and t]
    dup_lines = sum(c - 1 for _, c in dup_exact)
    near_t = near_duplicate_pairs(targets, args.near_threshold) if targets else []

    L += ["## Duplicate targets (train)", ""]
    L.append(f"- **{len(dup_exact)} distinct targets repeat**, accounting for "
             f"{dup_lines} redundant lines ({dup_lines / max(1, len(targets)):.1%} of the split).")
    L.append(f"- {len(near_t)} pairs above {args.near_threshold:.0%} token overlap.")
    if dup_lines:
        L.append("")
        L.append("> A repeated target is trained on once per copy, so the effective epoch "
                 "count over that content is higher than the configured one — the same "
                 "arithmetic error as a config comment claiming an epoch it never ran.")
    L.append("")

    # --- 5. lengths ----------------------------------------------------------
    lens = sorted(len(t or "") for t in targets)
    if lens:
        q = statistics.quantiles(lens, n=4) if len(lens) > 3 else [lens[0]] * 3
        L += ["## Target length (train, characters)", "",
              f"- min {lens[0]} · p25 {q[0]:.0f} · median {q[1]:.0f} · p75 {q[2]:.0f} · max {lens[-1]}",
              f"- mean {statistics.mean(lens):.0f}", ""]
        short = sum(1 for x in lens if x < 80)
        if short:
            L.append(f"- {short} targets ({short / len(lens):.0%}) are under 80 characters. "
                     "The fine-tuned arm produces the shortest answers in every run; a set "
                     "of one-liners is one explanation for that.")
            L.append("")

    text = "\n".join(L) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text, encoding="utf-8")
        print(f"-> {args.report_out}")
    print()
    print(text)

    if hits and not args.allow_contamination:
        raise SystemExit(
            f"CONTAMINATION: {len(hits)} training questions appear in the eval set. "
            "Every number produced by a model trained on this set would describe "
            "questions it was trained on. Refusing to pass."
        )


if __name__ == "__main__":
    main()
