#!/usr/bin/env python3
"""Analyze matched keyword-rule injection runs on the MTG rules gold set.

Primary safety outcome: fabricated citations, tested as paired binary outcomes
with an exact McNemar/binomial test on discordant question pairs.
Secondary outcome: rubric correctness, reported as paired wins/losses/ties with
an exact sign test and mean score delta.

The companion runner writes a manifest beside each run. This analyzer refuses
pairs whose manifests differ on any precommitted setting except
`keyword_rules`, so a model/judge/k/arm drift cannot masquerade as an injection
comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ARM = "base_rag_cards_rulings"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record_id(row: dict) -> str:
    value = row.get("gold_id") or row.get("id")
    if not value:
        raise ValueError("run row has neither gold_id nor id")
    return str(value)


def exact_two_sided(wins: int, losses: int) -> float:
    """Two-sided exact binomial p-value, conditioning on discordant/non-tied pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def manifest_path(run: Path) -> Path:
    return run.with_suffix(".manifest.json")


def load_manifest(run: Path) -> dict:
    path = manifest_path(run)
    if not path.exists():
        raise ValueError(f"missing experiment manifest beside {run}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifests(off_run: Path, on_run: Path) -> tuple[dict, dict]:
    off = load_manifest(off_run)
    on = load_manifest(on_run)
    required = {
        "experiment", "gold", "base_model", "judge_model", "judge_prompt",
        "k_rules", "with_cards", "with_rulings", "base_only", "no_plain_rag",
        "max_tokens", "seed", "keyword_rules",
    }
    for label, m in (("off", off), ("on", on)):
        missing = required - set(m)
        if missing:
            raise ValueError(f"{label} manifest missing {sorted(missing)}")
    if off["experiment"] != "keyword-rules-gold99-v1" or on["experiment"] != off["experiment"]:
        raise ValueError("unexpected experiment id")
    if off["keyword_rules"] is not False or on["keyword_rules"] is not True:
        raise ValueError("expected OFF manifest keyword_rules=false and ON manifest keyword_rules=true")
    for key in sorted(required - {"keyword_rules"}):
        if off[key] != on[key]:
            raise ValueError(f"manifest setting {key} differs: {off[key]!r} vs {on[key]!r}")
    return off, on


def summarize(off_run: Path, on_run: Path) -> dict:
    validate_manifests(off_run, on_run)
    off_rows = read_jsonl(off_run)
    on_rows = read_jsonl(on_run)
    if not off_rows or not on_rows:
        raise ValueError("both runs must be non-empty")

    off = {record_id(r): r for r in off_rows}
    on = {record_id(r): r for r in on_rows}
    if len(off) != len(off_rows) or len(on) != len(on_rows):
        raise ValueError("duplicate record id in a run")
    if set(off) != set(on):
        raise ValueError(
            f"question sets differ: {len(set(off)-set(on))} only off, "
            f"{len(set(on)-set(off))} only on"
        )

    for label, rows in (("off", off_rows), ("on", on_rows)):
        arms = set(rows[0].get("arms", {}))
        if ARM not in arms:
            raise ValueError(f"{label} run lacks {ARM}")
        if any(set(r.get("arms", {})) != arms for r in rows):
            raise ValueError(f"{label} arm set changes within run")
    if set(off_rows[0]["arms"]) != set(on_rows[0]["arms"]):
        raise ValueError("arm sets differ across runs")

    on_wins = off_wins = ties = unjudged = 0
    deltas: list[float] = []
    fabricated_off = fabricated_on = 0
    fixed_by_on = broken_by_on = 0
    cited_off = cited_on = 0

    for rid in sorted(off):
        a = off[rid]["arms"][ARM]
        b = on[rid]["arms"][ARM]
        ca, cb = a.get("correctness"), b.get("correctness")
        if ca is None or cb is None:
            unjudged += 1
        else:
            delta = float(cb) - float(ca)
            deltas.append(delta)
            if delta > 0:
                on_wins += 1
            elif delta < 0:
                off_wins += 1
            else:
                ties += 1

        c_off = a.get("citation") or {}
        c_on = b.get("citation") or {}
        f_off = bool(c_off.get("has_fabricated"))
        f_on = bool(c_on.get("has_fabricated"))
        fabricated_off += f_off
        fabricated_on += f_on
        cited_off += bool(c_off.get("cited"))
        cited_on += bool(c_on.get("cited"))
        if f_off and not f_on:
            fixed_by_on += 1
        elif f_on and not f_off:
            broken_by_on += 1

    return {
        "n": len(off),
        "on_wins": on_wins,
        "off_wins": off_wins,
        "ties": ties,
        "unjudged": unjudged,
        "mean_delta_on_minus_off": sum(deltas) / len(deltas) if deltas else None,
        "correctness_p": exact_two_sided(on_wins, off_wins),
        "fabricated_off": fabricated_off,
        "fabricated_on": fabricated_on,
        "fixed_by_on": fixed_by_on,
        "broken_by_on": broken_by_on,
        "fabrication_p": exact_two_sided(fixed_by_on, broken_by_on),
        "cited_off": cited_off,
        "cited_on": cited_on,
    }


def markdown(s: dict, off_run: Path, on_run: Path) -> str:
    delta = s["mean_delta_on_minus_off"]
    delta_s = "n/a" if delta is None else f"{delta:+.3f}"
    return "\n".join([
        "# Keyword-rule injection gold replication",
        "",
        f"- questions: {s['n']}",
        f"- arm: `{ARM}`",
        f"- OFF: `{off_run}`",
        f"- ON: `{on_run}`",
        "",
        "| Measure | Result |",
        "| --- | ---: |",
        f"| keyword ON correctness wins | {s['on_wins']} |",
        f"| keyword OFF correctness wins | {s['off_wins']} |",
        f"| correctness ties | {s['ties']} |",
        f"| unjudged pairs | {s['unjudged']} |",
        f"| mean correctness delta (ON - OFF) | {delta_s} |",
        f"| correctness exact sign-test p | {s['correctness_p']:.6f} |",
        f"| fabricated citations OFF | {s['fabricated_off']}/{s['n']} |",
        f"| fabricated citations ON | {s['fabricated_on']}/{s['n']} |",
        f"| fabrication fixed by ON (OFF yes, ON no) | {s['fixed_by_on']} |",
        f"| fabrication introduced by ON (OFF no, ON yes) | {s['broken_by_on']} |",
        f"| fabrication exact McNemar/binomial p | {s['fabrication_p']:.6f} |",
        f"| answers citing at least one rule OFF | {s['cited_off']}/{s['n']} |",
        f"| answers citing at least one rule ON | {s['cited_on']}/{s['n']} |",
        "",
        "Decision rule was fixed before the run: keep keyword injection enabled only if it "
        "significantly reduces paired fabrication (p < 0.05, with more fixes than newly "
        "introduced fabrications) and does not show significant correctness harm "
        "(p < 0.05 with OFF winning more paired questions).",
        "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("off", type=Path)
    ap.add_argument("on", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    s = summarize(args.off, args.on)
    text = markdown(s, args.off, args.on)
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
