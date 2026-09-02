"""Measure generation speed per model size — the Phase 2 hosting gate.

Section 21.139 established that the 32B is worth +0.50 correctness (p = 0.014)
over the 7B on the card/ruling benchmark. That is a real gain bought with ~4x
the parameters, and whether it is worth paying for is a PRODUCT decision that
depends on numbers nobody had measured: how long a user actually waits, and
how many concurrent requests one machine can hold.

The plan's Phase 2 gate names three quantities, and this script reports all
three, per model:

  * tokens/sec at the settings a chatbot would really use (retrieval context
    attached, --max-tokens 800 — the `base_rag_cards_rulings` shape that won
    at both sizes);
  * wall-clock time to FIRST token and to a COMPLETE answer, because that is
    what a person experiences — a model can have good throughput and still
    feel unusable if the prefill is long;
  * peak RSS, which bounds concurrency: a 17.6GB resident model on a 64GB box
    holds roughly two instances, a 4GB one holds many.

Prompts come from the real benchmark with real retrieved context, because
prefill cost scales with context length and a short synthetic prompt would
flatter the big model exactly where it is weakest. Reported per question and
as a median, since one slow outlier should not set the headline.

Usage:
    python scripts/measure_throughput.py --n 5
    python scripts/measure_throughput.py --models mlx-community/Qwen2.5-7B-Instruct-4bit
"""

import argparse
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import build_rag_messages, k_rules_arg, read_jsonl

CARD_RULING_PATH = Path(__file__).parent.parent / "data" / "gold" / "card_ruling_candidates.jsonl"
DEFAULT_MODELS = [
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-32B-Instruct-4bit",
]


def peak_rss_gb() -> float:
    """Peak RSS of this process, in GB.

    macOS reports ru_maxrss in BYTES; Linux reports KILOBYTES. Getting this
    wrong is a 1024x error that still looks like a plausible number, so it is
    branched explicitly rather than guessed.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 ** 3) if sys.platform == "darwin" else raw / (1024 ** 2)


def build_prompts(n: int, k_rules: int | str) -> list[tuple[str, str]]:
    """(question, context) pairs with REAL retrieved context attached."""
    from card_lookup import CardIndex
    from retrieve_hybrid import RulingIndex, build_context

    records = read_jsonl(CARD_RULING_PATH)[:n]
    card_index, ruling_index = CardIndex(), RulingIndex()
    out = []
    for rec in records:
        ctx = build_context(rec["question"], card_index, embed_model=None,
                            ruling_index=ruling_index, k_rules=k_rules,
                            card_names=rec.get("cards") or None)["context"]
        out.append((rec["question"], ctx))
    return out


def measure(model_id: str, prompts: list[tuple[str, str]], max_tokens: int) -> dict:
    from mlx_lm import load as load_lm
    from mlx_lm import stream_generate

    # `ru_maxrss` is a process-wide HIGH WATER MARK, not a current reading, so
    # a smaller model measured after a larger one in the same process inherits
    # the larger one's peak and reports a confidently wrong number. Recorded
    # before the load so that case is DETECTABLE (peak unchanged => this model
    # never set it) rather than silently plausible — the same discipline as
    # refusing a one-sided control (Section 21.43).
    rss_before = peak_rss_gb()

    print(f"\nloading {model_id} ...", flush=True)
    t_load = time.perf_counter()
    model, tokenizer = load_lm(model_id)
    load_s = time.perf_counter() - t_load

    first_tok, total_s, out_toks, prompt_toks = [], [], [], []
    for i, (question, context) in enumerate(prompts, 1):
        messages = build_rag_messages(question, context, preformatted=True)
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt_toks.append(len(tokenizer.encode(prompt)) if isinstance(prompt, str) else len(prompt))

        t0 = time.perf_counter()
        ttft, n = None, 0
        for _ in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
        elapsed = time.perf_counter() - t0
        first_tok.append(ttft if ttft is not None else elapsed)
        total_s.append(elapsed)
        out_toks.append(n)
        print(f"  q{i}: {n} tok in {elapsed:.1f}s "
              f"(ttft {first_tok[-1]:.1f}s, {n / elapsed:.1f} tok/s)", flush=True)

    del model, tokenizer
    rss_after = peak_rss_gb()
    return {
        "model": model_id,
        "load_s": load_s,
        # None means "this model did not set the process peak", i.e. a bigger
        # model ran earlier and this figure would be that one's, not this one's.
        "peak_rss_gb": rss_after if rss_after > rss_before + 0.05 else None,
        "median_prompt_tokens": statistics.median(prompt_toks),
        "median_output_tokens": statistics.median(out_toks),
        "median_ttft_s": statistics.median(first_tok),
        "median_total_s": statistics.median(total_s),
        "median_tok_per_s": statistics.median(
            n / t for n, t in zip(out_toks, total_s)),
    }


def verdict(median_total_s: float) -> str:
    """The plan's Phase 2 decision rule, written BEFORE the numbers existed.

    Stated in the plan so it could not be rationalised after seeing them:
    under ~15s use the 32B, over ~45s ship the 7B, in between ship the 7B for
    interactivity and revisit.
    """
    if median_total_s < 15:
        return "USE IT — under the 15s bar; accuracy is worth it for a rules bot"
    if median_total_s > 45:
        return "TOO SLOW — over the 45s bar; ship the 7B, keep this as a batch path"
    return "BORDERLINE — 15-45s; ship the 7B for interactivity and revisit"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n", type=int, default=5,
                        help="questions to time per model (default 5)")
    parser.add_argument("--max-tokens", type=int, default=800,
                        help="must match the eval runs being compared (default 800)")
    parser.add_argument("--k-rules", type=k_rules_arg, default=3,
                        help='CR chunks per question, or "auto" (Section 21.156). '
                             'Routing changes PROMPT LENGTH, so a throughput number '
                             'is only comparable to another at the same setting.')
    args = parser.parse_args()

    print(f"building {args.n} prompts with real retrieved context ...")
    prompts = build_prompts(args.n, args.k_rules)

    rows = [measure(m, prompts, args.max_tokens) for m in args.models]

    print("\n" + "=" * 78)
    print(f"{'model':34s} {'tok/s':>7s} {'ttft':>7s} {'total':>7s} {'peakRSS':>8s}")
    print("-" * 78)
    for r in rows:
        rss = f"{r['peak_rss_gb']:6.1f}G" if r["peak_rss_gb"] else "     n/a"
        print(f"{r['model'].split('/')[-1]:34s} {r['median_tok_per_s']:7.1f} "
              f"{r['median_ttft_s']:6.1f}s {r['median_total_s']:6.1f}s {rss:>8s}")
    if any(r["peak_rss_gb"] is None for r in rows):
        print("  n/a = a larger model set the process peak first; "
              "rerun that model alone with --models for a clean figure")
    print("-" * 78)
    print(f"medians over n={args.n}, max_tokens={args.max_tokens}, "
          f"k_rules={args.k_rules}, median prompt {rows[0]['median_prompt_tokens']:.0f} tokens")
    print("\nPhase 2 gate (rule fixed in advance):")
    for r in rows:
        print(f"  {r['model'].split('/')[-1]:34s} {verdict(r['median_total_s'])}")


if __name__ == "__main__":
    main()
