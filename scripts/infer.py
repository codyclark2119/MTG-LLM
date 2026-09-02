"""One local command for asking the model a rules question, offline after setup.

WHY THIS EXISTS

Every other way to run this project's model is either a measurement harness
(`eval.py`, `eval_positions.py` — built to score many answers against a
reference, not to answer one question for a person) or a server with a much
larger surface than a single question needs (`webui.py` is LAN-only and runs
training/eval jobs on the host; `rubric_server.py` is public-facing and knows
nothing about models at all, deliberately). Nothing existing is a small,
local, "just ask it something" command — this is that command, and only that.

It shares its prompt construction with `eval.py` (`common.build_rag_messages`)
and its retrieval with `rag.py` / `retrieve_hybrid.py`, rather than a fourth,
slightly-different copy of the same logic that could quietly drift from what
actually gets measured — Section 8.7 is the whole reason a second copy of a
prompt is treated as a bug waiting to happen in this repo, not a convenience.

USAGE

    python scripts/infer.py "When are state-based actions checked?"
    python scripts/infer.py "Does [[Chatterfang]] double token creation?" --with-cards
    python scripts/infer.py "..." --with-cards --with-rulings
    python scripts/infer.py "..." --adapter-path models/mtg-rules-adapter-v2-best
    python scripts/infer.py "..." --no-retrieval   # base model, no context at all

WORKS OFFLINE, AFTER SETUP

Nothing here calls a network API. The base model, the adapter (if any), and
the rules/card/ruling corpora and embedding model must already be present
locally — the same requirement `eval.py` already has. If `mlx_lm.load` needs
to fetch a model it does not have cached, that is a network call setup did
not finish, not something this script asks for.

WHAT IT PRINTS BEFORE ANSWERING, AND WHY

Model, adapter, and corpus identity — never silently. A stale default
evaluating the wrong thing without saying so is the most-repeated failure
mode in this repo's history (`ADAPTER_PATH` pointing at an old checkpoint,
`--judge-model` accepted and ignored on one path); the fix every time was
printing the default that actually ran, not trusting it unstated. An
adapter's prompt-fingerprint mismatch is reported the same way `eval.py`
reports it (Section 8.7) — as a loud warning before generation, not a
silent difference in output quality afterward.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (CARD_PIN, CR_PIN, CR_VERSION, build_rag_messages,  # noqa: E402
                    k_rules_arg, prompt_fingerprint)
from eval import BASE_MODEL_ID  # noqa: E402


def print_fingerprints(base_model_id: str, adapter_path: str | None, with_cards: bool) -> None:
    print(f"base model      : {base_model_id}")
    print(f"adapter         : {adapter_path or '(none — base model only)'}")
    if adapter_path:
        from stamp_adapter import check as prompt_stamp_check
        ok, msg = prompt_stamp_check(Path(adapter_path))
        print(f"  {msg}")
    fp = prompt_fingerprint()
    print(f"prompt fingerprint: {fp['prompt_fingerprint'][:12]}")
    print(f"CR pin          : {CR_VERSION} ({CR_PIN['n_rules']} rules)")
    if with_cards:
        print(f"card pin        : {CARD_PIN['n_card_chunks']} card chunks, "
              f"{CARD_PIN['n_ruling_chunks']} rulings")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--adapter-path", default=None,
                    help="LoRA adapter under test; omit for the base model alone")
    ap.add_argument("--base-model", default=None, help=f"default: {BASE_MODEL_ID}")
    ap.add_argument("--no-retrieval", action="store_true",
                    help="ask the model with no retrieved context at all")
    ap.add_argument("--with-cards", action="store_true",
                    help="resolve [[Card Name]] references and add card text")
    ap.add_argument("--with-rulings", action="store_true",
                    help="with --with-cards, also add pinned official WotC rulings")
    ap.add_argument("--k-rules", type=k_rules_arg, default=3,
                    help='CR chunks per question, or "auto" to route on whether a '
                         'card resolved (Section 21.156). Only meaningful with '
                         '--with-cards: without it nothing resolves and `auto` is '
                         'always the card-free branch.')
    ap.add_argument("--max-tokens", type=int, default=600)
    args = ap.parse_args()

    base_model_id = args.base_model or BASE_MODEL_ID
    print_fingerprints(base_model_id, args.adapter_path, args.with_cards)

    context = None
    preformatted = False
    if not args.no_retrieval:
        from mlx_embeddings import load as load_embedder
        from rag import MODEL_ID as EMBED_MODEL_ID
        print(f"loading {EMBED_MODEL_ID} for retrieval ...")
        embed_model = load_embedder(EMBED_MODEL_ID)

        if args.with_cards:
            from card_lookup import CardIndex
            from retrieve_hybrid import RulingIndex, build_context
            card_index = CardIndex()
            ruling_index = RulingIndex() if args.with_rulings else None
            result = build_context(args.question, card_index, embed_model=embed_model,
                                   k_rules=args.k_rules, ruling_index=ruling_index)
            context = result["context"]
            preformatted = True
            if result["card_names"]:
                print(f"cards resolved  : {', '.join(result['card_names'])}")
            if args.with_rulings and result["ruling_card_names"]:
                print(f"rulings resolved: {', '.join(result['ruling_card_names'])}")
        else:
            from rag import retrieve
            hits = retrieve(args.question, k=args.k_rules, model_and_tokenizer=embed_model)
            context = "\n\n".join(h["text"] for h in hits)
        print()

    print(f"loading {base_model_id} ...")
    from mlx_lm import generate as lm_generate
    from mlx_lm import load as load_lm
    model, tokenizer = load_lm(base_model_id, adapter_path=args.adapter_path)

    messages = build_rag_messages(args.question, context, preformatted)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    answer = lm_generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)

    print()
    print(answer)


if __name__ == "__main__":
    main()
