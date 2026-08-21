"""Record which prompt an adapter was trained under, and check it at eval time.

    python scripts/stamp_adapter.py models/mtg-rules-adapter-v4 --dataset data/datasets/verified
    python scripts/stamp_adapter.py models/mtg-rules-adapter-v2-best --verify

WHY

`build_rag_messages` is the single definition of a prompt's shape, used by both
training and inference, which is what stops them drifting *within* a run.
Nothing records which prompt an adapter was trained under, so an edit to
`common.py` silently invalidates every adapter already on disk — and the failure
presents as a capability result. The model looks worse. That is the project's #1
documented failure mode (Section 8.7) and it cost a full re-run.

This makes the invariant checkable: a stamp beside the weights, compared at eval
time, turning the most expensive failure this project has had into an error
message.

STAMPING IS EVIDENCE, NOT ASSERTION

`--dataset` is how a stamp is earned. The training data contains the exact
prompts the adapter saw, so the stamp is verified against the file rather than
asserted by whoever ran the command. Stamping without `--dataset` records only
"the prompts as they are right now", which is a claim about the present, not
about the training run — allowed, and labelled as such in the stamp.

Run it immediately after `mlx_lm.lora` finishes. `mlx_lm` writes the adapter
directory itself and knows nothing about this repo's prompts, so this cannot be
folded into training.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (  # noqa: E402
    CARDS_RAG_SYSTEM_PROMPT,
    PROMPT_STAMP_FILE,
    RAG_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    prompt_fingerprint,
    read_jsonl,
)

KNOWN = {"SYSTEM_PROMPT": SYSTEM_PROMPT,
         "RAG_SYSTEM_PROMPT": RAG_SYSTEM_PROMPT,
         "CARDS_RAG_SYSTEM_PROMPT": CARDS_RAG_SYSTEM_PROMPT}


def dataset_prompts(dataset_dir: Path) -> tuple[Counter, list[str]]:
    """Which system prompts a dataset actually contains, and which are unknown."""
    seen: Counter = Counter()
    for name in ("train.jsonl", "valid.jsonl"):
        p = dataset_dir / name
        if not p.exists():
            continue
        for rec in read_jsonl(p):
            for m in rec.get("messages") or []:
                if m.get("role") == "system":
                    seen[m["content"]] += 1
    unknown = [s for s in seen if s not in KNOWN.values()]
    return seen, unknown


# Which system prompt each generated arm puts in front of the adapter. Only the
# `finetuned*` arms load it; `base*` arms are the same model either way.
ARM_PROMPTS = {"finetuned": "SYSTEM_PROMPT",
               "finetuned_rag": "RAG_SYSTEM_PROMPT",
               "finetuned_rag_cards": "CARDS_RAG_SYSTEM_PROMPT"}


def unseen_arms(adapter_dir: Path, arms: list[str]) -> list[str]:
    """Arms whose system prompt appears ZERO times in the adapter's training set.

    The fingerprint check asks whether the prompts were *edited* since training.
    This asks the other question, which it cannot see: whether the adapter ever
    saw the prompt an arm is about to hand it.

    Section 8.7 is usually told as a prompt edit, but its mechanism was a
    training set that contained one shape while inference used another. That is
    reachable with no edit at all — `data/datasets/verified` is 1,112 examples
    under `SYSTEM_PROMPT` and **zero** under `RAG_SYSTEM_PROMPT`, deliberately
    (`build_sft_verified.build_examples` attaches no retrieved context), so the
    fingerprints match perfectly and the `finetuned_rag` arm is still evaluated
    on a shape the weights never saw.

    A warning, not an error: measuring that arm anyway is legitimate. What is
    not legitimate is reading the result as a statement about the training
    *data* when it is a statement about the training *shape*, so the caller is
    expected to carry this into the report rather than only print it.

    Empty list when the stamp carries no `dataset_prompt_counts` — a stamp
    without `--dataset` records the prompts as they are now, not what was
    trained on, and cannot answer this.
    """
    stamp = read_stamp(adapter_dir)
    counts = (stamp or {}).get("dataset_prompt_counts")
    if not counts:
        return []
    return [a for a in arms
            if a in ARM_PROMPTS and not counts.get(ARM_PROMPTS[a])]


def read_stamp(adapter_dir: Path) -> dict | None:
    p = adapter_dir / PROMPT_STAMP_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def check(adapter_dir: Path) -> tuple[bool, str]:
    """(ok, message). Missing stamp is not a failure — older adapters predate this."""
    stamp = read_stamp(adapter_dir)
    if stamp is None:
        return True, (f"no {PROMPT_STAMP_FILE} in {adapter_dir} — cannot verify the prompt "
                      f"this adapter was trained under. Stamp it with "
                      f"`python scripts/stamp_adapter.py {adapter_dir} --dataset <dir>`.")
    now = prompt_fingerprint()
    if stamp.get("prompt_fingerprint") == now["prompt_fingerprint"]:
        return True, f"prompt fingerprint matches ({now['prompt_fingerprint'][:12]})"

    changed = [k for k, v in now["parts"].items() if stamp.get("parts", {}).get(k) != v]
    return False, (
        f"PROMPT MISMATCH: {adapter_dir} was trained under a different prompt.\n"
        f"  stamped : {stamp.get('prompt_fingerprint', '?')[:12]}\n"
        f"  current : {now['prompt_fingerprint'][:12]}\n"
        f"  changed : {', '.join(changed) or 'unknown'}\n"
        "An adapter is only valid for the format it saw (Section 8.7). Either revert the "
        "prompt change or retrain — evaluating across it measures the mismatch, not the "
        "model, and reads as the model having got worse."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter_dir", type=Path)
    ap.add_argument("--dataset", type=Path, default=None,
                    help="training set to verify the prompts against before stamping")
    ap.add_argument("--verify", action="store_true", help="check only, write nothing")
    args = ap.parse_args()

    if not args.adapter_dir.is_dir():
        raise SystemExit(f"no such adapter directory: {args.adapter_dir}")

    if args.verify:
        ok, msg = check(args.adapter_dir)
        print(msg)
        raise SystemExit(0 if ok else 1)

    fp = prompt_fingerprint()
    stamp = {**fp, "verified_against": None}

    if args.dataset:
        seen, unknown = dataset_prompts(args.dataset)
        if not seen:
            raise SystemExit(f"{args.dataset} holds no system prompts to verify against")
        if unknown:
            print(f"{len(unknown)} system prompt(s) in {args.dataset} match nothing in common.py:")
            for s in unknown[:3]:
                print(f"  {s[:120]!r}")
            raise SystemExit(
                "Refusing to stamp. The dataset was built under a prompt this code no "
                "longer defines, so an adapter trained on it is NOT valid for the current "
                "format — which is exactly what the stamp would otherwise assert."
            )
        stamp["verified_against"] = str(args.dataset)
        stamp["dataset_prompt_counts"] = {
            next(k for k, v in KNOWN.items() if v == s): n for s, n in seen.items()}
        print(f"verified against {args.dataset}: "
              + ", ".join(f"{k}x{n}" for k, n in stamp["dataset_prompt_counts"].items()))
    else:
        print("no --dataset given: stamping the CURRENT prompts, which records what they "
              "are now and not what this adapter was trained on.")

    out = args.adapter_dir / PROMPT_STAMP_FILE
    out.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    print(f"prompt fingerprint {fp['prompt_fingerprint'][:12]} -> {out}")


if __name__ == "__main__":
    main()
