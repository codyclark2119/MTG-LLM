#!/usr/bin/env python3
"""Compare two matched eval runs that differ only in rules-retrieval k.

This is the pre-committed analysis for the gold-set replication of Sections
21.144/21.158. It deliberately reports the same two quantities that made the
53-question result ambiguous:

* paired correctness: wins / losses / ties and a two-sided exact sign test;
* grounding cost: fabricated-citation counts for the card+rulings arm.

The script refuses mismatched question sets, arm sets, base models, judges, or
judge prompts. The point is to make it difficult to accidentally call two runs
"k=0 vs k=3" when another consequential variable changed too.

`--check-historical` is a model-free regression check against the two committed
53-question runs. It proves this analyzer reproduces the evidence that motivated
the replication before it is used on the new 99-question result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ARM = "base_rag_cards_rulings"
HISTORICAL_A = REPO / "eval/runs/deconf32b_k0_card_ruling53.jsonl"
HISTORICAL_B = REPO / "eval/runs/deconf32b_k3_card_ruling53.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record_id(row: dict) -> str:
    value = row.get("gold_id") or row.get("id")
    if not value:
        raise ValueError("run row has neither gold_id nor id")
    return str(value)


def exact_sign_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign-test p-value, conditioning on non-ties."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def one_value(rows: list[dict], key: str):
    values = {r.get(key) for r in rows}
    if len(values) != 1:
        raise ValueError(f"{key} is not constant within a run: {sorted(map(str, values))}")
    return next(iter(values))


def validate_pair(rows_a: list[dict], rows_b: list[dict], arm: str) -> tuple[list[str], dict[str, dict], dict[str, dict]]:
    if not rows_a or not rows_b:
        raise ValueError("both run files must contain at least one row")

    a = {record_id(r): r for r in rows_a}
    b = {record_id(r): r for r in rows_b}
    if len(a) != len(rows_a) or len(b) != len(rows_b):
        raise ValueError("duplicate record id in a run")
    if set(a) != set(b):
        raise ValueError(
            f"question sets differ: {len(set(a) - set(b))} only in A, "
            f"{len(set(b) - set(a))} only in B"
        )

    for label, rows in (("A", rows_a), ("B", rows_b)):
        if arm not in rows[0].get("arms", {}):
            raise ValueError(f"{label} does not contain arm {arm!r}")
        expected = set(rows[0]["arms"])
        for row in rows:
            if set(row.get("arms", {})) != expected:
                raise ValueError(f"{label} changes arm set within the run")

    if set(rows_a[0]["arms"]) != set(rows_b[0]["arms"]):
        raise ValueError(
            f"arm sets differ: {sorted(rows_a[0]['arms'])} vs {sorted(rows_b[0]['arms'])}"
        )

    for key in ("judge_model", "judge_prompt"):
        va, vb = one_value(rows_a, key), one_value(rows_b, key)
        if va != vb:
            raise ValueError(f"{key} differs: {va!r} vs {vb!r}")

    # Base-model provenance lives on every row in current runs, but older runs
    # may omit it. When present on either side it must be present and equal on both.
    model_a = {r.get("base_model") for r in rows_a if r.get("base_model") is not None}
    model_b = {r.get("base_model") for r in rows_b if r.get("base_model") is not None}
    if model_a or model_b:
        if len(model_a) != 1 or len(model_b) != 1 or model_a != model_b:
            raise ValueError(f"base_model differs or is inconsistently stamped: {model_a} vs {model_b}")

    return sorted(a), a, b


def summarize(path_a: Path, path_b: Path, arm: str = DEFAULT_ARM) -> dict:
    rows_a, rows_b = read_jsonl(path_a), read_jsonl(path_b)
    ids, a, b = validate_pair(rows_a, rows_b, arm)

    wins = losses = ties = unjudged = 0
    deltas: list[float] = []
    fabricated_a = fabricated_b = 0
    grounded_a = grounded_b = 0

    for rid in ids:
        da = a[rid]["arms"][arm]
        db = b[rid]["arms"][arm]
        ca, cb = da.get("correctness"), db.get("correctness")
        if ca is None or cb is None:
            unjudged += 1
        else:
            delta = float(ca) - float(cb)
            deltas.append(delta)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1

        citation_a = da.get("citation") or {}
        citation_b = db.get("citation") or {}
        fabricated_a += bool(citation_a.get("has_fabricated"))
        fabricated_b += bool(citation_b.get("has_fabricated"))
        grounded_a += bool(citation_a.get("cited"))
        grounded_b += bool(citation_b.get("cited"))

    return {
        "n_questions": len(ids),
        "arm": arm,
        "a": str(path_a),
        "b": str(path_b),
        "wins_a": wins,
        "wins_b": losses,
        "ties": ties,
        "unjudged_pairs": unjudged,
        "mean_correctness_delta_a_minus_b": (sum(deltas) / len(deltas)) if deltas else None,
        "sign_test_p_two_sided": exact_sign_p(wins, losses),
        "fabricated_a": fabricated_a,
        "fabricated_b": fabricated_b,
        "grounded_a": grounded_a,
        "grounded_b": grounded_b,
    }


def historical_check() -> None:
    got = summarize(HISTORICAL_A, HISTORICAL_B)
    expected = {
        "n_questions": 53,
        "wins_a": 15,
        "wins_b": 6,
        "ties": 32,
        "unjudged_pairs": 0,
        "fabricated_a": 7,
        "fabricated_b": 3,
        "grounded_a": 34,
        "grounded_b": 29,
    }
    bad = {k: (got[k], want) for k, want in expected.items() if got[k] != want}
    if bad:
        details = ", ".join(f"{k}: got {g!r}, want {w!r}" for k, (g, w) in bad.items())
        raise SystemExit(f"historical k-rules analysis changed: {details}")
    p = got["sign_test_p_two_sided"]
    if not math.isclose(p, exact_sign_p(15, 6), rel_tol=0.0, abs_tol=1e-15):
        raise SystemExit(f"historical sign-test changed: {p}")
    print(
        "historical k-rules pair reproduced: 15/6/32, "
        "fabricated 7/53 vs 3/53, grounded 34/53 vs 29/53"
    )


def markdown(summary: dict, label_a: str, label_b: str) -> str:
    delta = summary["mean_correctness_delta_a_minus_b"]
    delta_s = "n/a" if delta is None else f"{delta:+.3f}"
    p = summary["sign_test_p_two_sided"]
    n = summary["n_questions"]
    return "\n".join([
        "# Paired k-rules replication",
        "",
        f"- arm: `{summary['arm']}`",
        f"- questions: {n}",
        f"- {label_a}: `{summary['a']}`",
        f"- {label_b}: `{summary['b']}`",
        "",
        "| Measure | Result |",
        "| --- | ---: |",
        f"| {label_a} wins | {summary['wins_a']} |",
        f"| {label_b} wins | {summary['wins_b']} |",
        f"| ties | {summary['ties']} |",
        f"| unjudged pairs | {summary['unjudged_pairs']} |",
        f"| mean correctness delta ({label_a} - {label_b}) | {delta_s} |",
        f"| exact sign-test p (two-sided, ties excluded) | {p:.6f} |",
        f"| fabricated citations: {label_a} | {summary['fabricated_a']}/{n} |",
        f"| fabricated citations: {label_b} | {summary['fabricated_b']}/{n} |",
        f"| answers citing at least one rule: {label_a} | {summary['grounded_a']}/{n} |",
        f"| answers citing at least one rule: {label_b} | {summary['grounded_b']}/{n} |",
        "",
        "Interpret correctness and fabrication together. A correctness gain that is paid for by "
        "more invented rule citations is the exact trade-off this replication was designed to test.",
        "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", type=Path, nargs="?", help="first matched run (for this experiment: k=0)")
    ap.add_argument("b", type=Path, nargs="?", help="second matched run (for this experiment: k=3)")
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--label-a", default="k=0")
    ap.add_argument("--label-b", default="k=3")
    ap.add_argument("--out", type=Path, default=None, help="optional Markdown report path")
    ap.add_argument("--check-historical", action="store_true",
                    help="reproduce the committed 53-question k=0 vs k=3 evidence and exit")
    args = ap.parse_args()

    if args.check_historical:
        if args.a or args.b or args.out:
            ap.error("--check-historical takes no run paths or --out")
        historical_check()
        return

    if args.a is None or args.b is None:
        ap.error("A and B run paths are required unless --check-historical is used")

    summary = summarize(args.a, args.b, args.arm)
    text = markdown(summary, args.label_a, args.label_b)
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
