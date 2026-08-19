"""Score board positions: blunder rate, legality, and the three gates.

The judge is not reimplemented here. `judge_batch_rubric` and
`rubric_correctness` are imported from scripts/eval.py, so a position is scored
by exactly the same code that scores a rules question — the rubric extraction,
the anonymized A/B/C/D labels, the length-neutrality instruction, and the
Python-side arithmetic. `common_errors` IS the blunder list, so blunder rate
falls straight out of `errors_made`.

Kept as its own script rather than a flag on eval.py for one reason: the rules
eval is mid-measurement (the n~100 two-judge comparison), and its output must
stay byte-identical while this is built. Sharing the judge by import gets the
reuse without touching that path.

Arms are declarative, so a new checkpoint or a larger base model is a config
line rather than an edit:

    python scripts/gameplay/eval_positions.py --positions data/gold/positions_seed.jsonl
    python scripts/gameplay/eval_positions.py --base-model mlx-community/... --limit 4
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from actions import legality, match_to_legal, parse_output  # noqa: E402
from common import (  # noqa: E402
    POSITIONS_PATH,
    REPO_ROOT,
    build_position_messages,
    pearson_r,
    render_position,
)
from positions import load_positions, position_card_names  # noqa: E402

# Four arms, matching the judge prompt's "labeled A, B, C, D". Each of the
# three pre-registered predictions is a difference between two of them:
#   closed vs open        base_closed      vs base_open
#   card retrieval helps  base_cards_open  vs base_open
#   adapter is out of dbn ft_cards_open    vs base_cards_open
ARMS = [
    {"name": "base_open", "adapter": None, "retrieval": False, "closed": False},
    {"name": "base_closed", "adapter": None, "retrieval": False, "closed": True},
    {"name": "base_cards_open", "adapter": None, "retrieval": True, "closed": False},
    {"name": "ft_cards_open", "adapter": "ADAPTER", "retrieval": True, "closed": False},
    # Added, not substituted. base_open differs from this arm ONLY in the
    # deliberation instruction, so the pair isolates one variable — the same
    # control discipline the no-card subset gave the card experiment.
    {"name": "base_open_think", "adapter": None, "retrieval": False, "closed": False,
     "deliberate": True},
]


def build_position_context(pos: dict, card_index, embed_model, k_rules: int = 2) -> str:
    """Card text for every card in the position, plus a little rules text.

    Cards are resolved by NAME rather than by embedding the board: the position
    already tells us exactly which cards are in play, so there is nothing to
    guess. That is a strictly better setup than the rules-question case in
    Section 13.5, where only 16 of 60 questions named a card at all.

    Rules retrieval is k=2 rather than the usual 3 — a position plus ~8 card
    texts is already most of the context budget (see configs/phase1_lora_v3.yaml
    on the 2048-token training window).
    """
    parts = []
    seen, texts = set(), []
    for name in position_card_names(pos):
        card, how = card_index.resolve(name)
        if card and how == "exact" and card["name"] not in seen:
            seen.add(card["name"])
            texts.append(card["text"])
    if texts:
        parts.append("Cards referenced:\n" + "\n\n".join(texts))

    if embed_model is not None and k_rules:
        from rag import retrieve
        hits = retrieve(render_position(pos), k=k_rules, model_and_tokenizer=embed_model)
        parts.append("Rules text:\n" + "\n\n".join(h["text"] for h in hits))
    return "\n\n".join(parts)


def generate(positions, arms, base_model_id, adapter_path, contexts, max_tokens):
    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm

    answers: dict[str, list[str]] = {}
    # Group by adapter so each model is loaded once, not once per arm.
    for adapter in sorted({a["adapter"] for a in arms}, key=lambda x: x is not None):
        path = adapter_path if adapter == "ADAPTER" else None
        print(f"loading {base_model_id} (adapter={path}) ...")
        model, tokenizer = load_lm(base_model_id, adapter_path=path)
        for arm in [a for a in arms if a["adapter"] == adapter]:
            print(f"generating arm: {arm['name']}")
            out = []
            for i, pos in enumerate(positions, 1):
                ctx = contexts[i - 1] if arm["retrieval"] else None
                messages = build_position_messages(
                    pos, ctx, closed=arm["closed"],
                    deliberate=arm.get("deliberate", False))
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                out.append(lm_generate(model, tokenizer, prompt=prompt,
                                       max_tokens=max_tokens, verbose=False))
                if i % 5 == 0 or i == len(positions):
                    print(f"  {arm['name']}: {i}/{len(positions)}")
            answers[arm["name"]] = out
        del model, tokenizer
    return answers


def compare_judges(path_a: Path, path_b: Path, report_out: Path) -> None:
    """Inter-judge agreement on the blunder call, over identical stored answers.

    Blunder rate is a binary call, and raw percent agreement flatters a skewed
    one: if both judges say "blundered" 80% of the time, they agree ~68% by
    chance alone. Cohen's kappa subtracts that, so it is the number reported
    first. Section 9.9 measured judge disagreement large enough to reverse an
    arm ranking on the rules eval; there is no reason to assume positions are
    kinder, and this is the cheapest way to find out BEFORE 8-12 hours go into
    authoring 40 of them.
    """
    def load(p):
        return {r["id"]: r for r in
                (json.loads(line) for line in p.open(encoding="utf-8") if line.strip())}

    a, b = load(path_a), load(path_b)
    shared_ids = sorted(set(a) & set(b))
    if not shared_ids:
        raise SystemExit("the two files share no position ids")
    arms = [x for x in a[shared_ids[0]]["arms"] if x in b[shared_ids[0]]["arms"]]

    pairs, corr_pairs = [], []
    for rid in shared_ids:
        for arm in arms:
            xa, xb = a[rid]["arms"][arm], b[rid]["arms"][arm]
            if xa.get("blundered") is not None and xb.get("blundered") is not None:
                pairs.append((bool(xa["blundered"]), bool(xb["blundered"]), rid, arm))
            if xa.get("correctness") is not None and xb.get("correctness") is not None:
                corr_pairs.append((xa["correctness"], xb["correctness"]))

    n = len(pairs)
    agree = sum(1 for x, y, *_ in pairs if x == y)
    po = agree / n if n else float("nan")
    pa1 = sum(x for x, _, *_ in pairs) / n if n else 0
    pb1 = sum(y for _, y, *_ in pairs) / n if n else 0
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

    r = pearson_r(corr_pairs)  # one definition, in common.py

    lines = ["# Position Judge Agreement\n",
             f"`{path_a.name}` vs `{path_b.name}` — identical stored answers, judge varied.\n",
             f"- {n} arm-position blunder calls over {len(shared_ids)} positions x {len(arms)} arms",
             f"- **Cohen's kappa on the blunder call: {kappa:+.2f}** "
             f"(raw agreement {po:.0%}, chance {pe:.0%})",
             f"- correctness correlation: r = {r:+.2f}",
             f"- blunder rate: {path_a.stem} {pa1:.0%}, {path_b.stem} {pb1:.0%}\n"]

    verdict = ("kappa below 0.20 means the two judges are barely agreeing beyond chance, so "
               "blunder rate is not yet a usable gate metric — fix the rubrics or the judge "
               "prompt before authoring more positions"
               if kappa < 0.20 else
               "kappa between 0.20 and 0.40 is weak agreement; usable for large effects only"
               if kappa < 0.40 else
               "kappa at or above 0.40 is moderate agreement — comparable to the r = +0.62 "
               "hand-authored rubrics reached on the rules eval (Section 14.6)")
    lines.append(f"**Reading:** {verdict}.\n")

    lines.append("| Arm | " + f"{path_a.stem} blunder | {path_b.stem} blunder | agree |")
    lines.append("| --- | --- | --- | --- |")
    for arm in arms:
        sub = [(x, y) for x, y, _, m in pairs if m == arm]
        if sub:
            lines.append(f"| {arm} | {sum(x for x, _ in sub) / len(sub):.0%} | "
                         f"{sum(y for _, y in sub) / len(sub):.0%} | "
                         f"{sum(1 for x, y in sub if x == y) / len(sub):.0%} |")

    disputed = [(rid, arm) for x, y, rid, arm in pairs if x != y]
    if disputed:
        lines.append(f"\n## {len(disputed)} disputed calls\n")
        lines.append("Each is a position where one judge saw a blunder and the other did not. "
                     "These are the cases to read by hand — they show whether the rubric is "
                     "ambiguous or a judge is simply wrong.\n")
        for rid, arm in disputed[:15]:
            lines.append(f"- `{rid}` / {arm}")

    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nreport -> {report_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", type=Path, default=POSITIONS_PATH)
    parser.add_argument("--limit", type=int, default=None)
    # Comparing two BASE MODELS needs the adapter arm dropped -- a LoRA trained
    # on Qwen2.5-7B cannot be applied to a different base at all. But Section
    # 21.5 established that arm count changes every arm's score, because the
    # judge grades all candidates in one batched call. So the baseline has to be
    # re-run at the SAME reduced arm count rather than compared against the
    # existing four-arm numbers. This flag is what makes that matched pair
    # possible; it is not a convenience.
    parser.add_argument("--arms", nargs="+", default=None,
                        metavar="NAME",
                        help="run only these arms (default: all). Comparing across runs "
                             "requires the same arm COUNT — see Section 21.5.")
    parser.add_argument("--base-model", default=None,
                        help="defaults to eval.BASE_MODEL_ID; a larger 4-bit model fits "
                             "at 36GB for inference")
    parser.add_argument("--adapter-path", default=None, help="defaults to eval.ADAPTER_PATH")
    parser.add_argument("--judge-model", default=None, help="defaults to --base-model")
    parser.add_argument("--second-judge", default=None,
                        help="also score with this judge and emit an agreement report. "
                             "Section 16.12: Gates 2 and 3 both reversed between judges on "
                             "identical answers, so this is how a gate run should be done.")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--judge-max-tokens", type=int, default=600)
    parser.add_argument("--k-rules", type=int, default=2)
    parser.add_argument("--no-retrieval", action="store_true",
                        help="drop the retrieval arms (skips loading the embedder and card index)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval/runs/positions_latest.jsonl")
    parser.add_argument("--report-out", type=Path, default=REPO_ROOT / "eval/reports/positions_latest.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rescore-from", type=Path, default=None,
                        help="re-judge the answers stored in a previous results file instead "
                             "of regenerating them. This is how a second judge is run: the "
                             "answers stay byte-identical and only the judge varies, which is "
                             "the only way an agreement number means anything (Section 9.9).")
    parser.add_argument("--compare", type=Path, nargs=2, default=None,
                        metavar=("A.jsonl", "B.jsonl"),
                        help="report inter-judge agreement between two results files")
    args = parser.parse_args()

    if args.compare:
        compare_judges(args.compare[0], args.compare[1], args.report_out)
        return

    import eval as rules_eval  # the judge lives there; do not reimplement it

    base_model = args.base_model or rules_eval.BASE_MODEL_ID
    adapter_path = args.adapter_path or rules_eval.ADAPTER_PATH
    judge_model_id = args.judge_model or base_model

    positions = load_positions(args.positions)
    if not positions:
        raise SystemExit(f"no positions in {args.positions}")

    if args.rescore_from:
        # Answers come from disk; the position file supplies the rubric that
        # both judges score against. Arms are whatever the stored run used, so
        # a rescore cannot silently drop or add one.
        stored = [json.loads(line) for line in
                  args.rescore_from.open(encoding="utf-8") if line.strip()]
        by_id = {p["id"]: p for p in positions}
        missing = [r["id"] for r in stored if r["id"] not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} stored ids are not in {args.positions}: "
                             f"{', '.join(missing[:3])}")
        positions = [by_id[r["id"]] for r in stored]
        arm_names = list(stored[0]["arms"])
        answers = {arm: [r["arms"][arm]["answer"] for r in stored] for arm in arm_names}
        closed_arms = {a["name"] for a in ARMS if a["closed"] and a["name"] in arm_names}
        print(f"re-judging {len(stored)} stored positions x {len(arm_names)} arms "
              f"from {args.rescore_from}")
    else:
        if args.limit:
            positions = positions[:args.limit]
        arms = [a for a in ARMS if not (args.no_retrieval and a["retrieval"])]
        if args.arms:
            known = {a["name"] for a in ARMS}
            unknown = [n for n in args.arms if n not in known]
            if unknown:
                raise SystemExit(f"unknown arm(s): {unknown}. Known: {sorted(known)}")
            arms = [a for a in arms if a["name"] in set(args.arms)]
        print(f"{len(positions)} positions x {len(arms)} arms")

        contexts = [None] * len(positions)
        if any(a["retrieval"] for a in arms):
            from card_lookup import CardIndex
            from mlx_embeddings import load as load_embedder
            from rag import MODEL_ID as EMBED_MODEL_ID
            card_index = CardIndex()
            embed_model = load_embedder(EMBED_MODEL_ID)
            contexts = [build_position_context(p, card_index, embed_model, args.k_rules)
                        for p in positions]
            lens = [len(c) for c in contexts]
            print(f"context: {min(lens)}-{max(lens)} chars (mean {sum(lens) / len(lens):.0f})")

        answers = generate(positions, arms, base_model, adapter_path, contexts, args.max_tokens)
        arm_names = [a["name"] for a in arms]
        closed_arms = {a["name"] for a in arms if a["closed"]}

    def judge_with(model_id: str) -> list[dict]:
        """Score every stored answer with one judge. Split out so a second judge
        is a loop iteration rather than a separate run — Section 16.12 showed the
        gate verdict reversing between judges, so one judge is never enough."""
        return _judge_all(rules_eval, model_id, positions, answers, arm_names, args)

    results = judge_with(judge_model_id)
    _write_results(results, args.out)
    _write_report(results, positions, arm_names, closed_arms, args,
                  base_model, adapter_path, judge_model_id, args.report_out)

    if args.second_judge:
        second_out = args.out.with_name(args.out.stem + "_judge2" + args.out.suffix)
        second_report = args.report_out.with_name(args.report_out.stem + "_JUDGE2.md")
        print(f"\n=== second judge: {args.second_judge} ===")
        results2 = judge_with(args.second_judge)
        _write_results(results2, second_out)
        _write_report(results2, positions, arm_names, closed_arms, args,
                      base_model, adapter_path, args.second_judge, second_report)
        agreement = args.report_out.with_name(args.report_out.stem + "_AGREEMENT.md")
        compare_judges(args.out, second_out, agreement)
    return


def _judge_all(rules_eval, judge_model_id, positions, answers, arm_names, args) -> list[dict]:
    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm
    print(f"loading {judge_model_id} as judge ...")
    judge_model, judge_tokenizer = load_lm(judge_model_id)

    rng = random.Random(args.seed)
    results = []
    for i, pos in enumerate(positions):
        candidates = {arm: answers[arm][i] for arm in arm_names}
        judged = rules_eval.judge_batch_rubric(
            lm_generate, judge_model, judge_tokenizer, render_position(pos),
            pos["key_points"], pos.get("common_errors") or [], candidates,
            args.judge_max_tokens, rng,
        )
        per_arm = {}
        for arm in arm_names:
            parsed = parse_output(candidates[arm])
            legal_set = pos.get("legal_actions") or []
            # Goes through legality() rather than re-deriving all_legal here.
            # The inline version was a second copy of the rule and missed the
            # once-per-turn check entirely when that was added to actions.py --
            # the "helper duplicated with a guard in only some copies" trap.
            legal_info = legality(parsed, legal_set)
            matched = [a for a in parsed.actions if match_to_legal(a, legal_set)]
            j = judged.get(arm, {})
            per_arm[arm] = {
                "answer": candidates[arm],
                "correctness": j.get("correctness"),
                "errors_made": j.get("errors_made"),
                # The gate metric. None when the judge failed to return usable
                # JSON, so a parse failure is never silently counted as "clean".
                "blundered": (bool(j.get("errors_made")) if j.get("errors_made") is not None
                              else None),
                "points_hit": j.get("points_hit"),
                "n_actions": len(parsed.actions),
                "n_parse_failures": len(parsed.failures),
                "parsed_ok": parsed.ok,
                "degenerate": parsed.degenerate,
                "repeats_collapsed": parsed.repeats_collapsed,
                "n_legal": len(matched),
                "all_legal": legal_info["all_legal"],
                "illegal": legal_info["illegal"],
                "actions": parsed.keys(),
            }
        results.append({"id": pos["id"], "category": pos["category"],
                        "difficulty": pos["difficulty"], "arms": per_arm})
        print(f"  judged {i + 1}/{len(positions)}")
    return results


def _write_results(results: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_report(results, positions, arm_names, closed_arms, args,
                  base_model, adapter_path, judge_model_id, report_out) -> None:
    # ---- aggregate ---------------------------------------------------------
    def rate(arm, key):
        vals = [r["arms"][arm][key] for r in results if r["arms"][arm][key] is not None]
        return (sum(bool(v) for v in vals) / len(vals), len(vals)) if vals else (float("nan"), 0)

    lines = ["# Position Evaluation Report\n"]
    lines.append(f"{len(positions)} positions from `{args.positions}`")
    lines.append(f"\n- base model: `{base_model}`\n- adapter: `{adapter_path}`\n"
                 f"- judge: `{judge_model_id}`"
                 + ("  (**same model as the base arms** — self-preference bias not ruled out)"
                    if judge_model_id == base_model else ""))
    seeded = [p for p in positions if p.get("seed_note")]
    if seeded:
        lines.append(f"\n> **{len(seeded)}/{len(positions)} positions are machine-drafted seed "
                     "fixtures.** They verify the pipeline; they are not Gate 3 evidence. "
                     "Section 14.6 measured hand-authored rubrics beating machine drafts "
                     "(inter-judge r +0.30 → +0.62).\n")

    lines.append("\n| Arm | Blunder rate | Correctness | Parsed ok | Actions/answer "
                 "| All legal | Degenerate |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    summary = {}
    for arm in arm_names:
        blunder, n_b = rate(arm, "blundered")
        cs = [r["arms"][arm]["correctness"] for r in results
              if r["arms"][arm]["correctness"] is not None]
        corr = sum(cs) / len(cs) if cs else float("nan")
        parsed_ok, _ = rate(arm, "parsed_ok")
        all_legal, _ = rate(arm, "all_legal")
        degen, _ = rate(arm, "degenerate")
        acts = sum(r["arms"][arm]["n_actions"] for r in results) / len(results)
        summary[arm] = {"blunder": blunder, "corr": corr, "parsed_ok": parsed_ok,
                        "all_legal": all_legal, "n_judged": n_b, "degenerate": degen}
        lines.append(f"| {arm} | {blunder:.0%} (n={n_b}) | {corr:.2f} | {parsed_ok:.0%} "
                     f"| {acts:.1f} | {all_legal:.0%} | {degen:.0%} |")

    # ---- the three gates ---------------------------------------------------
    lines.append("\n## Gates\n")
    worst_parse = min(summary[a]["parsed_ok"] for a in arm_names)
    g1_closed = [summary[a]["all_legal"] for a in arm_names if a in closed_arms]
    g1 = worst_parse >= 0.90 and (not g1_closed or min(g1_closed) >= 0.95)
    lines.append(f"**Gate 1 — protocol works: {'PASS' if g1 else 'FAIL'}.** "
                 f"Worst-arm parse rate {worst_parse:.0%} (need ≥90%)"
                 + (f"; worst closed-arm legality {min(g1_closed):.0%} (need ≥95%)"
                    if g1_closed else "; no closed arm in this run") + ".")

    blunders = [summary[a]["blunder"] for a in arm_names]
    spread = max(blunders) - min(blunders)
    g2 = spread >= 0.15
    lines.append(f"\n**Gate 2 — the eval discriminates: {'PASS' if g2 else 'FAIL'}.** "
                 f"Blunder rate spans {min(blunders):.0%}–{max(blunders):.0%} "
                 f"(spread {spread:.0%}). A spread near zero means the positions are not "
                 "separating the arms, and more positions will not fix that.")

    easy = [r for r in results if r["difficulty"] in ("basic", "intermediate")]
    best_arm, best_rate = None, 1.0
    for arm in arm_names:
        vals = [r["arms"][arm]["blundered"] for r in easy
                if r["arms"][arm]["blundered"] is not None]
        if vals:
            v = sum(vals) / len(vals)
            if v < best_rate:
                best_arm, best_rate = arm, v
    g3 = best_arm is not None and best_rate <= 0.25
    lines.append(f"\n**Gate 3 — blunder rate ≤25% on basic+intermediate: "
                 f"{'PASS' if g3 else 'FAIL'}.** Best arm `{best_arm}` at {best_rate:.0%} "
                 f"over {len(easy)} positions."
                 + (" Seed fixtures cannot settle this gate — it needs judge-authored positions."
                    if seeded else ""))

    # Gate 3 is one number over basic+intermediate, and that number hides which
    # band it came from: a set weighted toward `basic` can pass on positions too
    # easy to separate the arms at all. The band table is reported ALONGSIDE the
    # gate, never instead of it — at ~13 positions per band a proportion cannot
    # resolve a difference on its own, so each row is a composition fact and the
    # combined figure above stays the gate.
    lines.append("\n## Blunder rate by difficulty band\n")
    lines.append("| Difficulty | n | " + " | ".join(arm_names) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in arm_names) + " |")
    for diff in ("basic", "intermediate", "advanced"):
        rows = [r for r in results if r["difficulty"] == diff]
        if not rows:
            continue
        cells = []
        for arm in arm_names:
            vals = [r["arms"][arm]["blundered"] for r in rows
                    if r["arms"][arm]["blundered"] is not None]
            cells.append(f"{sum(vals) / len(vals):.0%}" if vals else "—")
        lines.append(f"| {diff} | {len(rows)} | " + " | ".join(cells) + " |")
    thin = [d for d in ("basic", "intermediate", "advanced")
            if 0 < sum(1 for r in results if r["difficulty"] == d) < 25]
    if thin:
        lines.append(f"\n> Bands under 25 positions ({', '.join(thin)}) are too small to "
                     "resolve a blunder-rate difference on their own. Read them as "
                     "composition, and take the gate from the combined figure above.\n")
    if not any(r["difficulty"] == "basic" for r in results):
        lines.append("\n> **No basic positions in this set.** Gate 3 is then an "
                     "intermediate-only number wearing a basic+intermediate label.\n")

    lines.append("\n## Blunder rate by category\n")
    cats = sorted({r["category"] for r in results})
    lines.append("| Category | n | " + " | ".join(arm_names) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in arm_names) + " |")
    for cat in cats:
        rows = [r for r in results if r["category"] == cat]
        cells = []
        for arm in arm_names:
            vals = [r["arms"][arm]["blundered"] for r in rows
                    if r["arms"][arm]["blundered"] is not None]
            cells.append(f"{sum(vals) / len(vals):.0%}" if vals else "—")
        lines.append(f"| {cat} | {len(rows)} | " + " | ".join(cells) + " |")

    n_unjudged = sum(1 for r in results for a in arm_names
                     if r["arms"][a]["blundered"] is None)
    if n_unjudged:
        lines.append(f"\n{n_unjudged} arm-position pairs went unjudged (judge returned "
                     "unparseable JSON) and are excluded rather than counted as clean.\n")

    if judge_model_id == base_model:
        lines.append("\n> **The judge is the same model as the base arms**, so self-preference "
                     "bias is not ruled out (Section 9.9).\n")
    if not args.second_judge:
        lines.append("\n> **One judge only.** Section 16.12 measured Gates 2 and 3 BOTH "
                     "reversing between judges on byte-identical answers, so a gate verdict "
                     "from a single judge is a statement about the judge. Re-run with "
                     "`--second-judge mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`.\n")

    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[3:]))
    print(f"\nresults -> {args.out}\nreport -> {report_out}")

    by_cat = Counter(p["category"] for p in positions)
    print("categories: " + ", ".join(f"{c}={n}" for c, n in sorted(by_cat.items())))


if __name__ == "__main__":
    main()
