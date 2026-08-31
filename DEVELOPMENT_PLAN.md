# Development Plan — Training an LLM to Understand Magic: The Gathering

> Design rationale, experiment log, and results history. For setup, code structure, and how to run things, see [README.md](README.md).

## Phase 1: Rules Foundation

This walkthrough covers everything from base model selection through ingesting the comprehensive rules, and stops **before** card data input. The goal of this phase is a model that understands *how Magic works* — the rules engine, turn structure, priority, the stack, zones, and gameplay mechanics — as a foundation for later card-aware, deck-based text gameplay.

> **Target hardware for this build: an M3 Pro MacBook Pro with 36GB unified memory, macOS Tahoe 26.5.2.** Every recommendation below is tailored to Apple Silicon, using Apple's **MLX** framework (not the NVIDIA/CUDA `bitsandbytes` stack). With 36GB you're comfortably out of the tightest constraints: you can fine-tune **4-bit quantized 7–8B models with real headroom** (larger batches, more LoRA layers, longer sequences), and even attempt a **13–14B** fine-tune if you want. macOS Tahoe (26.x) is fully supported by current MLX. See Section 1.5 for the full hardware picture.

---

## Status (as of 2026-08-18)

**Phase 1 (Rules Foundation):**

- [x] Repository and MLX toolchain initialized (Section 2)
- [x] Comprehensive Rules corpus acquired, pinned to the 2026-08-07 release (Section 3)
- [x] Rules parsed into structured section/rule/subrule records; chunked for retrieval and training with reference-aware bundling and glossary injection (Sections 4–5)
- [x] RAG baseline stood up over the chunks — `mlx-community/all-MiniLM-L6-v2-4bit` embeddings, brute-force cosine retrieval (Section 6)
- [x] SFT dataset generated via RAG-grounded, category-balanced question/answer synthesis over `mlx-community/Qwen2.5-7B-Instruct-4bit` (Section 7) — ~700 examples across all 8 categories from 7.1, split into `data/datasets/train.jsonl` / `valid.jsonl` and a held-out `eval/sets/rules_questions.jsonl`
- [x] Fine-tuning run (Section 8) — QLoRA over `mlx-community/Qwen2.5-7B-Instruct-4bit`, 600 iterations, rank-16 LoRA on 16 layers. **Validation loss bottomed out at iteration 200 (1.159) and climbed steadily afterward (1.631 by iteration 600)** — classic overfitting on a 569-example train set. The iteration-200 checkpoint is promoted to `models/mtg-rules-adapter-best/`. **Spot-checks after training found the adapter fabricating rule citations more confidently than the un-tuned base model, including ignoring correct RAG-retrieved context it was explicitly told to use** — see the caveat in Section 8.6. Fine-tuning as currently trained is not yet a net improvement; needs the real Section 9 eval to quantify, and likely a larger/more diverse SFT set.
- [x] Evaluation against the RAG-only and base-model baselines (Section 9) — 110 questions (70 synthetic + 40 reddit) × 4 arms, judge-scored. Run 1: `finetuned_rag` 2.37 vs `base_rag` 3.19 (Section 9.5).
- [x] Second fine-tune + re-eval after fixing a train/inference format mismatch and growing the dataset 569 → 2,646 examples (Sections 8.7, 9.6). `finetuned_rag` improved **+0.43 (2.37 → 2.80)** against a measured judge-noise floor of ±0.09, and bare-`finetuned` citation fabrication fell 58% (26 → 11). Still below `base_rag` (3.25) — but that comparison is confounded by a judge length bias (r = +0.21 between answer length and score), so it reads as "not yet demonstrated better," not "worse."
- [x] Recalibrated the judge (Section 9.7) — separate correctness/citation scores, explicit length-neutrality, anonymized candidates; both runs re-scored under it. This reversed a false result: `base_rag` (3.75) in fact beats `base` (3.04), where the old judge showed the opposite. Retrieval is worth **+0.71 to +0.85**, the v2 format fix is worth **+0.69** (not +0.43), and measured judge noise is ±0.16.
- [x] Split eval scores by question source (Section 9.8) — **65% of synthetic questions retrieve their own source chunk**, so that subset flatters RAG (`base_rag` 4.23 synthetic vs 2.92 reddit). On *real* questions `finetuned_rag` and `base_rag` are tied within noise (2.88 vs 2.92); the apparent RAG win is largely an artifact of how the synthetic set was built.
- [x] Card-augmented retrieval measured (Section 13.5) — on card-referencing questions `base_rag_cards` gains **+0.69** over `base_rag` with citation quality 3.25 → 4.00, but n=16 gives a 95% CI of [−0.17, +1.54], so it is promising rather than established. A no-card control subset (+0.00) caught a prompt-formatting bug that had produced a convincing false result.
- [x] Settled the card question at n=100 (Section 13.5) — the +0.69 **did not replicate**: −0.01, CI [−0.33, +0.31], 25 better / 28 worse / 47 tied. It was small-sample noise, correctly flagged at the time as unestablished.
- [x] Tested judge validity with an independent Llama-3.1-8B judge (Section 9.9) — **the two judges rank the arms in opposite order** on identical answers, and agree only r = +0.43 (37% exact, mean disagreement 1.18 points). The earlier ±0.16 "noise floor" measured within-judge reproducibility only; judge-choice uncertainty is far larger, so previously reported effects of +0.4 to +0.7 are provisional.
- [x] Ingested 77,918 official WotC rulings covering 19,726 cards (Section 13.6) — the most authoritative corpus available, and the fix for eval references that currently cite a rule only 21% of the time.
- [x] Built the gold set on RulesGuru's verified Q&A and measured what actually drives judge disagreement (Sections 14, 14.6) — **hand-authored rubrics lift inter-judge agreement from r = +0.30 to +0.62** on identical answers, resolving the Section 9.9 ranking inversion. Rubric quality does *not* reduce the sample size needed: per-question variance is unchanged, so ~100–150 questions are still required to resolve a 0.4-point effect.
- [x] Built the Phase 3 gameplay scaffolding alongside the open Phase 1 work (Section 16) — position schema, action grammar and parser, position authoring in the console, and a gate-reporting eval that **reuses the V3 rubric judge unchanged**, because `common_errors` is the blunder list and `errors_made` was already being computed per question. Model choice is now a `--base-model` flag rather than a constant (Section 16.8), which also caught `--judge-model` being silently ignored on the generate-and-judge path. On 8 machine-drafted seed fixtures **all three gates fail**, which is the correct result for a pipeline whose fixtures are seeds: the closed arm lifts legality (88% vs 62%) but every arm blunders at exactly 75%, so the eval does not yet discriminate. See Section 16.11 — at n=8 with a single judge these numbers move substantially between runs, and a spot-check found the judge marking a false blunder.
- [x] Re-ran the fine-tune at a **full epoch** and evaluated it under both judges (Sections 18.1, 18.3) — the under-training confound is removed and **the verdict does not move**: `finetuned_rag` gains +0.04 under Qwen (CI [−0.26, +0.33]) and +0.02 under Llama (CI [−0.33, +0.37]), while the two control arms — byte-identical answers that cannot have changed — moved up to 0.23 on judge variance alone. Both judges now produce the same arm ranking (r = +0.60). Section 9.4's rule stands: fix the SFT data, not the hyperparameters.
- [x] Gold set finished to n=39, **39/39 hand-authored** with card slots and normalized player names. Caught before it mattered: templated rubrics were reaching the judge as literal `[[card2]]` text on 35 of 39 eval rows (Section 18.5).
- [x] Gold set grown 39 → **99, all hand-authored**, in five reviewed batches — 330 key points, 264 common errors, every category at 9–13, 0 lint warnings. `definition recall` was filled from the CR glossary, since RulesGuru is a scenario database and never asks "what does X mean?". Player normalization was retired back to a miss-rather-than-guess rule after `_PLAYER_PREP` turned "refers to Sand Warriors" into a player named Sand.
- [x] **A3 run at n=99 under both judges (Section 19) — the Phase 1 verdict.** `finetuned_rag` trails `base_rag` by **0.50** under Qwen (1.88 vs 2.38) and **0.80** under Llama (2.36 vs 3.16), and **both judges produce the same arm ranking**. That is well outside the 0.4-point effect Section 14.6 sized this sample to resolve. **Fine-tuning did not beat retrieval** — now a measurement at adequate sample size on hand-authored rubrics, not the n=8 and n=16 gestures that had to be walked back twice. Inter-judge r = +0.49 over 367 pairs. Caveat that must travel with the table: `base` scores nominally highest under Qwen while fabricating a citation on **35 of 99** questions, because the rubric judge does not price fabrication (Section 19.1).
- [x] Gave positions a reviewed batch path (`positions.py --ingest`) and pinned `ORDER TRIGGERS`' operand order, which meant "resolution order" in the grammar shown to the model and merely "order matters" in the parser's own docs — the `CROSS_REF_RE`/`PASS` trap again, caught on the first position that ever used the verb.
- [x] Built the position set and ran the gates on it (Section 20) — 22 positions, 18 hand-adjudicated, under both judges. **Gate 2 PASSES for the first time**: blunder rate spans 50–82% (Qwen) and 45–82% (Llama) against the seed set's zero spread, so hand-authored positions do separate the arms. Gates 1 and 3 fail on the models, not the harness. The fine-tuned arm is worst on every measure, matching Section 19. Two open problems: the closed arm makes the model *worse* (Section 20.1), and `land sequencing` is arm-invariant at 100% (Section 20.2).
- [ ] **Next:** (a) grow the position set toward ~40 and rewrite the 31 disputed calls and the 3 land-sequencing rubrics (Section 20.2, 20.3); (b) decide whether the rubric judge should price fabricated citations, given Section 19.1; (c) use the rulings corpus for eval references and SFT targets; (d) rebuild the synthetic eval so it stops testing retrieval of its own source chunk; (e) Section 9.4's standing instruction is unchanged and now well-supported — **fix the SFT data, not the hyperparameters**.

**Phase 2 (Card Data):** started early, ahead of finishing Phase 1 — see Section 13. The Standard-only pool proved far too narrow (it covered just 4% of cards players actually ask about), so the corpus is now Scryfall's full Oracle set: 34,933 playable cards chunked and linked to the rules governing their keywords, with card names resolved by lookup at 99%.

---

## 0. Scope and Guiding Principles

Before touching a model, be clear about what "understand Magic" means for Phase 1.

- **In scope:** Comprehensive Rules comprehension, turn/phase/step structure, priority and the stack, zones, the type system, keyword mechanics, state-based actions, the layer system, combat, and templating conventions.
- **Out of scope (Phase 2+):** Individual card text, deck lists, actual game-state simulation, legality/format rules beyond the core, and reinforcement from self-play.

A model that "knows the rules" but has never seen a card is exactly the target. Think of it as teaching a rules judge, not a player, first.

Two truths worth internalizing early:

1. **Magic's rules are famously edge-case-heavy.** The Comprehensive Rules are ~280 pages and full of interlocking references. Chunking and cross-referencing matter enormously.
2. **You almost certainly do not need to pretrain from scratch.** A strong instruction-tuned base plus targeted fine-tuning and retrieval will vastly outperform a from-scratch attempt at any reasonable budget.

---

## 1. Base Model Selection

### 1.1 Selection criteria

Pick a base by weighing these factors against your constraints:

- **Context window.** Rules reasoning requires pulling in multiple rules sections at once. Favor models with at least a 32K token context; 128K+ is comfortable.
- **Reasoning quality.** Magic rules interactions are logic puzzles. Prioritize models that score well on reasoning/instruction-following benchmarks over raw knowledge benchmarks.
- **Licensing.** For a hobby/commercial project you want a permissive license (Apache 2.0 or similar) if self-hosting.
- **Fine-tuning support.** Confirm the ecosystem supports LoRA/QLoRA on your hardware.
- **Size vs. hardware.** Match parameter count to your GPU memory (see below).

### 1.2 Practical tiers (and what fits in 36GB)

On an Apple Silicon Mac, the unified memory pool holds the model weights, optimizer/adapter state, and the training batch *all at once*. With **36GB** you have a genuinely comfortable envelope for local LoRA fine-tuning — the 7–8B tier is easy, and 13–14B becomes realistic.

| Tier | Param range | Fits on 36GB M3 Pro for LoRA? | Notes |
|------|-------------|-------------------------------|-------|
| Small | 3–4B | Trivially | Use for fast pipeline debugging |
| **Target** | **7–8B (4-bit)** | **Yes, with headroom** | Your Phase 1 workhorse; run bigger batches / more LoRA layers than a 16GB machine could |
| Larger | 13–14B (4-bit) | Yes, realistic | Viable if you want more reasoning capacity; slower per step |
| Big | 30–34B (4-bit) | Marginal | Possibly with tiny batch + few layers; expect slow steps, keep other apps closed |
| Huge | 70B+ | No (local training) | Cloud-only |

**Recommendation for a first pass:** Debug your pipeline on a **3–4B** instruct model, then do your real Phase 1 training on a **4-bit quantized 7–8B instruct** model — with 36GB you can afford a larger batch size and more LoRA layers than the bare minimum, which improves adapter quality. Once the pipeline is proven, a **13–14B** run is a reasonable "quality upgrade" experiment your hardware can handle.

### 1.3 Concrete model candidates

Choose a 7–8B instruct model that (a) has a strong reasoning reputation and (b) is available as pre-converted 4-bit MLX weights (the Hugging Face **MLX Community** hosts many). Good families to look at: Llama 3.1 8B Instruct, Qwen 2.5 7B Instruct, Mistral 7B Instruct, and Ministral/Gemma small-instruct variants. Prefer whichever currently benchmarks best on reasoning/instruction-following at 7–8B — verify current standings when you start, since the leaderboard shifts.

### 1.4 Base vs. instruct

Use an **instruction-tuned** ("instruct"/"chat") variant, not a raw base completion model. You want to preserve conversational and instruction-following behavior; you're layering rules knowledge on top, not teaching language from zero.

### 1.5 The hardware reality check (36GB M3 Pro, macOS Tahoe)

A few hardware truths that shape everything downstream:

- **MLX format required.** MLX fine-tuning needs Hugging Face safetensors weights (or pre-converted MLX weights). GGUF files won't work for *training* — GGUF is for inference (Ollama/LM Studio).
- **OS is fully supported.** macOS Tahoe 26.x runs current MLX without issue on M-series chips; keep `mlx`/`mlx-lm` updated via pip.
- **36GB is a comfortable envelope, not a tight one.** Unlike a 16GB machine, you're not fighting memory pressure at every step. You can raise batch size, LoRA layers, and sequence length for better adapters — just watch Activity Monitor when pushing a 13–14B model.
- **Memory bandwidth is the pacing factor, not correctness.** An M3 Pro's bandwidth is below a discrete GPU's, so each training step is slower — a run that's minutes on an A100 may be tens of minutes to a couple of hours locally. It *works*; it's just patient work. (The M3 Pro sits below the M3 Max on bandwidth, so expect somewhat slower steps than Max-chip benchmarks you'll see online.)
- **Context length costs memory.** Longer training sequences use more memory; 36GB gives you room to go beyond the minimal 1024 tokens if your rules chunks need it (Section 5.3).
- **Zero cloud cost, fully local, private.** No bills, no data leaving your machine, and silent overnight runs are entirely practical.

---

## 2. Environment and Tooling Setup

### 2.1 Core stack (Apple Silicon / MLX)

- **Training:** **`mlx-lm`** (Apple's MLX framework), using its built-in `mlx_lm.lora` command for LoRA/QLoRA. This is the Apple-Silicon-native path — it uses the unified memory pool directly with no CPU↔GPU copies, and is faster and lighter than PyTorch's MPS backend on Mac. Do **not** use `bitsandbytes`/CUDA tooling; it's NVIDIA-only and won't run here.
- **Quantization:** handled by MLX automatically — pass a 4-bit model to training and it does QLoRA (quantized base, full-precision adapter) with no extra flags.
- **Model conversion/download:** `huggingface_hub` (+ `hf_transfer` for speed) to pull weights; use pre-converted 4-bit MLX weights from the MLX Community when available, or convert with `mlx_lm.convert`.
- **Embeddings/RAG:** a small local embedding model (runnable via MLX or `sentence-transformers` on MPS) plus a lightweight vector store (e.g., FAISS or a local Chroma). RAG is CPU/GPU-light and fits easily.
- **Data handling:** plain Python + `datasets` for your ETL scripts.
- **Serving/inference (for eval + gameplay):** `mlx_lm.generate`/`mlx_lm.server` (OpenAI-compatible endpoint), or **LM Studio**/**Ollama** if you export the fused model to GGUF afterward.
- **Experiment tracking:** log runs to a simple file/CSV or Weights & Biases; you'll iterate often.

### 2.1.1 Quick install

```bash
python3 -m venv mlx_env
source mlx_env/bin/activate
pip install mlx mlx-lm huggingface_hub hf_transfer datasets
# verify — expect a recent version (0.30+ as of early 2026)
python -c "import mlx_lm; print('mlx-lm ready')"
```

Requires Python 3.10+ and a recent macOS. No CUDA toolkit, no driver juggling.

### 2.2 Repository layout

```text
magic-llm/
├── CLAUDE.md          # Orientation for Claude Code: architecture, traps, conventions
├── README.md          # How to run things
├── DEVELOPMENT_PLAN.md # This file — design rationale and the full experiment log
├── data/
│   ├── raw/           # Original rules documents, pinned by date
│   ├── processed/     # Parsed rules, chunks, RAG index
│   ├── datasets/      # SFT train/valid JSONL
│   ├── cards/         # Scryfall Oracle pull + derived chunks (Section 13)
│   └── gold/          # Hand-reviewed data: rules Q&A, board positions, RulesGuru
│       ├── SCHEMA.md      # How to author a gold record
│       └── POSITIONS.md   # How to author a board position (Section 16)
├── scripts/           # Every stage is a CLI; see CLAUDE.md for the map
│   ├── common.py      # Shared prompts, CR pinning, rule-id patterns, paths, jsonl IO
│   ├── ingest.py chunk.py rag.py            # rules → chunks → retrieval
│   ├── fetch_cards.py chunk_cards.py card_lookup.py retrieve_hybrid.py
│   ├── build_sft.py eval.py                 # training data and evaluation
│   ├── webui.py label_store.py              # the local authoring console
│   └── gameplay/     # Phase 3: positions, action grammar, gate scoring (Section 16)
├── configs/           # mlx_lm.lora settings, one file per training run
├── models/            # Adapters and checkpoints (gitignored)
└── eval/
    ├── sets/          # Inputs: the question sets, with their manifests
    ├── runs/          # Outputs: per-question scored results (.jsonl)
    └── reports/       # Outputs: human-readable reports, same stem as their run
```

### 2.3 Reproducibility

Pin all dependency versions, set random seeds, and log every training run's config. You will run this pipeline many times; a run you can't reproduce is a run you can't learn from.

---

## 3. Acquiring the Rules Corpus

### 3.1 Primary sources

1. **Comprehensive Rules (CR)** — the authoritative, exhaustive ruleset. This is your backbone document. It's published as a plain-text/HTML file and updated with each set release.
2. **Basic Rulebook** — a gentler, prose introduction useful for teaching intuitive framing.
3. **Glossary** — the CR includes a large glossary of defined terms; treat it as a first-class source.

> **Important:** Magic's rules text is owned by Wizards of the Coast. Confirm you have the right to use it for your intended purpose (personal/research use differs from redistribution or commercial deployment). Keep this in mind before publishing any model trained on it.

### 3.2 Versioning

Pin to a **specific dated release** of the Comprehensive Rules. Rules change over time; you want every training example traceable to one known version. Record the version string (e.g., the effective date) in your data manifest.

---

## 4. Ingesting and Cleaning the Rules

### 4.1 Parse structure, don't flatten it

The CR has a rigid, numbered hierarchy you must preserve:

- **Sections** (1–9, e.g., "1. Game Concepts")
- **Rules** (e.g., `100.` "General")
- **Subrules** (e.g., `100.1`, `100.1a`)

Parse into structured records rather than a wall of text. A good record schema:

```json
{
  "rule_id": "509.1a",
  "section": "5",
  "section_title": "Turn Structure",
  "parent_rule": "509.1",
  "text": "...",
  "cross_refs": ["508.1", "509.2"],
  "glossary_terms": ["attacking creature", "block"]
}
```

### 4.2 Cleaning steps

- Strip page headers/footers, page numbers, and formatting artifacts.
- Normalize whitespace and Unicode (mana symbols, em-dashes, etc.).
- Preserve rule numbers exactly — they are the primary keys of the entire system.
- Detect and record **cross-references** (any `NNN.N` pattern inside rule text) programmatically.
- Extract glossary entries into their own records linked back to the rules that use them.

### 4.3 Validate the parse

Sanity checks before proceeding: every subrule has a parent; no orphaned cross-references (a referenced rule ID should exist); glossary term count and section counts match expectations for the pinned version. A broken parse here poisons everything downstream.

---

## 5. Chunking Strategy

### 5.1 Why chunking matters

Rules interactions span multiple, often non-adjacent rules. Naive fixed-size chunking splits related material and destroys the reference graph. Chunk **semantically**, respecting the hierarchy.

### 5.2 Recommended approach

- **Base unit:** one rule and its subrules kept together (e.g., all of `509.x` in one chunk when short enough).
- **Overflow handling:** if a rule group exceeds your target chunk size, split on subrule boundaries, never mid-sentence.
- **Reference-aware bundling:** when building training/retrieval chunks, optionally append the *text* of directly cross-referenced rules so a chunk is self-contained.
- **Glossary injection:** attach short definitions of the defined terms a chunk uses.

### 5.3 Target sizes

Aim for chunks that comfortably fit alongside a question and answer within your training context — often a few hundred to ~1,000 tokens. Keep a metadata trail (`rule_id`s included) on every chunk for later retrieval and for building citations.

---

## 6. Choosing the Training Approach

You have three complementary levers. For Phase 1, use them in this priority order.

### 6.1 Retrieval-Augmented Generation (RAG) — do this first

Before any fine-tuning, stand up a retrieval system over your chunked rules:

- Embed every chunk, store in a vector database.
- At query time, retrieve the most relevant rules and put them in context.
- The model reasons over *provided* rules rather than recalled ones.

**Why first:** RAG gives you accurate, citable rules answers immediately, with zero training, and becomes your evaluation baseline and your source of ground-truth answers for building fine-tuning data. It also sidesteps memorization errors on exact rule numbers.

**Still important with 36GB:** even a 7–8B (or 13–14B) model has limited capacity to memorize ~280 pages of dense, cross-referential rules *reliably* — and exact rule-number recall is exactly where small models hallucinate. RAG offloads *recall* to the retrieval system so the model spends its capacity on *reasoning*. Your extra memory lets you train a stronger adapter, but it doesn't remove the case for retrieval: the strongest Phase 1 system is still "fine-tuned reasoner + rules retrieval," not "model that memorized everything." Treat RAG as a permanent part of the architecture.

### 6.2 Supervised Fine-Tuning (SFT) — do this second

Fine-tune (via LoRA/QLoRA) on curated instruction/response pairs so the model internalizes rules *reasoning patterns* and templating conventions — the things RAG alone handles clumsily. Details in Section 7.

### 6.3 Continued pretraining — usually skip for Phase 1

Raw next-token training on the rules text can help a model absorb terminology, but it's data-hungry, easy to overfit on a small corpus, and less sample-efficient than SFT+RAG for this goal. Consider it only if SFT+RAG plateaus.

---

## 7. Building the Supervised Fine-Tuning Dataset

This is the highest-leverage part of Phase 1. The model's rules ability will only be as good as this dataset.

### 7.1 Example categories

Cover the full spectrum of rules understanding:

- **Definition recall:** "What is the stack?" → concise, correct definition with rule citation.
- **Turn-structure walkthroughs:** "List the steps of the combat phase in order."
- **Priority reasoning:** "After I cast a spell, who gets priority and when does it resolve?"
- **Interaction puzzles:** "Player A does X, Player B responds with Y — what happens?"
- **State-based actions:** "A creature has 0 toughness. What happens and when is it checked?"
- **Zone transitions:** "Where does a countered spell go?"
- **Layer-system questions:** effect ordering and dependency.
- **Templating/keyword meaning:** "What does 'trample' mean mechanically?"

### 7.2 Generation workflow

1. **Seed with real questions.** Judge rulings, rules FAQs, and comprehension-question banks are excellent seeds.
2. **Generate answers via your RAG system**, so every answer is grounded in retrieved rules text.
3. **Cite rule IDs in every answer.** Train the model to reference `NNN.Nx` so answers are checkable.
4. **Human-review a sample.** Rules are unforgiving; spot-check for subtle errors before they get baked in.
5. **Format as instruction/response JSONL**, matching your model's chat template exactly.

### 7.3 Example record

```json
{
  "messages": [
    {"role": "system", "content": "You are a Magic: The Gathering rules expert. Answer precisely and cite comprehensive rule numbers."},
    {"role": "user", "content": "When exactly are state-based actions checked?"},
    {"role": "assistant", "content": "State-based actions are checked whenever a player would receive priority (rule 704.3). They are not checked in response to specific events but continuously before priority is granted; if any apply, they're all performed simultaneously, then the check repeats until none apply. Only then does the player receive priority."}
  ]
}
```

### 7.4 Dataset size and balance

- Start in the low thousands of high-quality examples rather than tens of thousands of noisy ones.
- Balance across the categories in 7.1 so the model doesn't over-index on definitions and under-learn interactions.
- Hold out a stratified slice for evaluation (Section 9); never let eval questions leak into training.

---

## 8. Running the Fine-Tune

### 8.1 Method

Use **QLoRA** via MLX: pass a **4-bit quantized** 7–8B model to `mlx_lm.lora` with `--train`, and MLX keeps the base quantized while training full-precision LoRA adapters — no extra flags needed. This is what makes a 7–8B fine-tune fit in 16GB at all.

### 8.2 The command

```bash
mlx_lm.lora \
  --model <4bit-mlx-model-id-or-path> \
  --train \
  --data ./data/datasets \
  --iters 600 \
  --batch-size 4 \
  --num-layers 16 \
  --learning-rate 1e-4 \
  --max-seq-length 2048 \
  --adapter-path ./models/mtg-rules-adapter
```

MLX expects `train.jsonl` / `valid.jsonl` in the `--data` directory (see 7.3 for record format).

### 8.3 Starting hyperparameters — tuned for 36GB

With 36GB you can be generous relative to a 16GB machine. Sensible starting points for a 7–8B 4-bit fine-tune:

- **`--batch-size 4`** to start (drop to 2 for a 13–14B model, or if you see memory pressure). This is still the biggest memory lever — tune it first.
- **`--num-layers` (LoRA layers): 16** for a good-quality adapter; you can push toward 24–32 on a 7–8B model if memory allows. More layers = more trainable parameters and better fine-tune quality at the cost of speed/memory.
- **`--max-seq-length`: ~2048.** You have room to fit longer rules chunks + Q&A than a 16GB machine could; raise only as far as your data actually needs.
- **Learning rate:** ~1e-4 (LoRA tolerates 1e-4–2e-4).
- **Iterations:** start a few hundred; watch validation loss rather than fixing epochs blindly on a small corpus.
- **LoRA rank/alpha:** rank 16–32, alpha ≈ 2× rank; the extra memory makes a higher rank painless on 7–8B models.

For a **13–14B** run, halve the batch size and start LoRA layers back at 8–16, then increase if memory holds.

### 8.4 After training: fuse and (optionally) export

- **Fuse** the adapter into the base for easy inference: `mlx_lm.fuse --model <base> --adapter-path ./models/mtg-rules-adapter`.
- For everyday local chat/eval you can serve directly with `mlx_lm.server`, or **export the fused model to GGUF** and run it in Ollama/LM Studio if you prefer that workflow.

### 8.5 Guardrails

- Track train **and** validation loss; stop when validation stops improving (MLX reports both during training).
- **Watch memory pressure** in Activity Monitor, especially on 13–14B runs — if macOS starts swapping heavily, reduce batch size, LoRA layers, or sequence length. On 7–8B with 36GB you'll rarely hit this.
- Closing other heavy apps (browsers especially) during runs is good hygiene, though 36GB is forgiving enough that it's not the constant concern it is on 16GB.
- Save adapter checkpoints periodically and evaluate several, not just the last.
- Keep the base frozen (LoRA only) so you can swap adapters cheaply and keep each adapter tiny (a few MB).

### 8.6 First run: results and an open problem

Ran per `configs/phase1_lora.yaml`: `mlx-community/Qwen2.5-7B-Instruct-4bit`, rank-16 LoRA on 16 layers, batch size 4, 600 iterations, checkpointed every 100. ~45 minutes on the M3 Pro, peak memory 12.75GB (well within the 36GB budget — this guardrail was never close to binding).

**Validation loss confirms the "save several checkpoints" guardrail was necessary, not optional:**

| Iter | 1 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 |
| ---- | - | -- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Val loss | 3.49 | 1.43 | 1.21 | 1.16 | **1.16** | 1.18 | 1.33 | 1.29 | 1.30 | 1.44 | 1.47 | 1.44 | 1.63 |

Val loss bottoms out at iteration 200 and climbs steadily afterward while train loss keeps falling toward ~0.15 — textbook overfitting on a 569-example train set repeated ~4 epochs. The iteration-200 checkpoint was promoted to `models/mtg-rules-adapter-best/` (copy `0000200_adapters.safetensors` over `adapters.safetensors` in a clean directory, since `mlx_lm.fuse`/`load_adapters` always read the unprefixed filename — the final checkpoint isn't automatically the one you want).

**A more serious problem showed up in spot-checks, independent of which checkpoint was used.** Asked "When exactly are state-based actions checked?":

- The fine-tuned adapter answered confidently and cited **rule 704.11 — which doesn't exist** in the pinned CR. The real answer is rule 704.3.
- Given the *correct* rule text directly in the prompt (via RAG retrieval) and instructed to answer only from it, the adapter **still ignored the provided text** and produced a different fabricated citation (704.1, a real but wrong rule).
- The **un-tuned base model**, given that same RAG-retrieved context, answered correctly and quoted the real rule 704.3 text verbatim.

So on this spot-check, fine-tuning made citation behavior *worse*, not better, and degraded the model's willingness to defer to provided context — likely the adapter learned the surface pattern "sound confident, end with a rule citation" from repeated exposure to a training set that's still an order of magnitude smaller than what Section 7.4 calls for ("low thousands"), rather than learning which citations are actually correct. This is exactly the failure mode Section 6.1 warns about ("exact rule-number recall is exactly where small models hallucinate") — it just turned out fine-tuning on a modest dataset can make it worse before it makes it better.

This isn't a final verdict — it's two anecdotal spot-checks, not the real exam from Section 9. But it means: **don't trust this adapter over base+RAG without running Section 9 properly first**, and the likely fix is a larger, more diverse SFT set (Section 7.4's "low thousands" floor was undershot at 569 train examples) rather than different training hyperparameters.

---

### 8.7 Second run: dataset fix, and two machine crashes worth documenting

Run 2 (`configs/phase1_lora_v2.yaml`) addressed Section 9.5's verdict by fixing the data, not the hyperparameters:

- **Train/inference format mismatch.** Run 1's examples were bare `question → answer`, but RAG inference wraps the question in `"Rules text: {context}\n\nQuestion: ..."` under a different system prompt. The adapter had never seen an example where the answer should come from prompt text — the leading explanation for it ignoring retrieved context. Every example whose grounding context is known now also appears in the exact inference shape, reusing existing answers at zero extra generation cost.
- **Volume and diversity.** 569 → 2,646 training lines, including 800 real r/MTGRules player questions (messy phrasing the rules-derived synthetic set never produces) with freshly generated RAG-grounded answers.
- **A leakage bug caught in the process.** 81 held-out eval questions were sitting in the training pool — they had been carved from the same generation pool by an earlier run. Training on them would have silently invalidated every Section 9 comparison. `build_sft.py` now excludes held-out questions by exact text and supports `--freeze-eval` so the exam doesn't drift between runs.

Validation loss: 1.986 → 1.231 (150) → 0.990 (300) → **0.966 (450)** → 0.975 (600). No overfitting blowup, unlike run 1's climb from 1.16 to 1.63. These numbers are *not* comparable to run 1's — the validation set itself changed — which is exactly why Section 9.6's re-run is the real verdict.

**Two config lessons.** `steps_per_eval: 150` and `save_every: 100` don't share a common cadence, so iteration 450 (best validation) had no saved checkpoint; iteration 600 was used instead. Align those next time.

**And a real failure worth recording:** the first two attempts at this run hard-crashed the machine. Peak memory hit **65.7GB on a 36GB box** and train loss went `NaN` by iteration 20. Both traced to one root cause — the grounded Reddit examples concatenated three retrieved chunks, producing sequences up to 3,625 tokens:

- The answer sits at the *end* of each example, so anything over `max_seq_length: 2048` had its training targets truncated away entirely, leaving nothing to compute loss against → `NaN`.
- Attention memory scales with the square of sequence length, so those long sequences blew past physical memory and forced swap.

Fixed at the data layer (`truncate_context()` caps grounding context on chunk boundaries: max sequence now 1,737 tokens, zero examples over the limit, 2 grounded copies dropped), plus `batch_size: 2` and `grad_checkpoint: true` for margin. Verified by measurement before relaunching: **peak memory 9.2GB, loss finite**. A short smoke run to check peak memory is cheap insurance before any long training job whose data shape has changed.

---

## 9. Evaluating Rules Comprehension

Do not judge by loss alone. Build a real rules exam.

### 9.1 Evaluation set

Curate a held-out set of rules questions across all categories from 7.1, with reference answers and the rule IDs that support them. Include deliberately tricky interaction puzzles.

**Real-world addendum:** alongside the synthetic held-out set (`eval/sets/rules_questions.jsonl`, Section 7.4), `eval/sets/reddit_questions.jsonl` adds 200 actual questions Magic players asked on r/MTGRules with community-vetted answers (`scripts/build_reddit_eval.py`, source: `Javier-Jimenez99/reddit-mtgrules-qa`, CC-BY-SA-4.0). Reddit's upvote score alone isn't a quality signal — it rewards jokes as readily as correct rulings (a top-scored reply to a real rules question was "Isn't Marty's cause to get back to the future?") — so candidates are filtered through an LLM judge for topical relevance before inclusion, and explicit rule citations are checked against the currently-pinned CR. Real questions surface phrasing, ambiguity, and multi-card interactions a synthetic set generated from rules text alone tends not to reproduce.

### 9.2 Scoring dimensions

- **Correctness:** Is the ruling right?
- **Citation accuracy:** Are the referenced rule numbers real and relevant?
- **Completeness:** Did it cover the required steps/edge cases?
- **Consistency:** Same question, stable answer across runs.

### 9.3 Methods

- **Automated:** exact-match on rule IDs cited; keyword/step coverage checks.
- **Model-graded:** a strong judge model scoring answers against references (spot-checked by you).
- **Human:** you or an experienced rules person reviewing the hardest cases.

### 9.4 Compare against baselines

Always compare fine-tuned model vs. (a) base model alone and (b) base model + RAG. If fine-tuning doesn't beat RAG-only on your exam, your SFT data needs work — fix the data, not the hyperparameters.

### 9.5 First run: results

`scripts/eval.py` runs every eval question through four arms — `base`, `base_rag`, `finetuned` (the Section 8.6 adapter), and `finetuned_rag` — and scores each with the base model as judge (1-5 vs. the reference), plus automated citation checks. 110 questions (70 synthetic + 40 reddit-sourced), ~2 hours end to end (one run stalled for an hour when the machine slept mid-job — MLX's Metal context doesn't reliably survive system sleep — then resumed on its own; a future run should checkpoint per-question rather than writing results only at the end).

| Arm | Avg score (1-5) | Fabricated citation | Citation matches reference |
| --- | --- | --- | --- |
| base | 3.62 | 33/110 | 5/107 |
| base_rag | 3.19 | 1/110 | 65/107 |
| finetuned | 2.47 | 26/110 | 5/107 |
| finetuned_rag | 2.37 | 5/110 | 58/107 |

**The robust finding:** `finetuned` and `finetuned_rag` both score below their non-fine-tuned counterparts, on both judge score and fabrication rate. This holds up at full scale, not just in the Section 8.6 anecdotes — of the 10 lowest-scoring cases in the report, half involve `finetuned_rag` producing a wrong or contradicted answer despite being handed the correct rule text. Per Section 9.4's decision rule, this is unambiguous: **the SFT dataset needs work** (very likely just more of it — 569 train examples is well short of Section 7.4's "low thousands" floor), not different LoRA hyperparameters.

**A finding that needs a caveat, not a headline:** `base` outscoring `base_rag` (3.62 vs. 3.19) looks like "RAG hurts," but spot-checking the disagreements says otherwise. On "What does 'Reveal' mean?", `base_rag` quoted the actual rule 701.20a text verbatim and still scored a point below `base`'s vaguer, more general answer — docked for "missing detail" despite being the more precisely grounded response. On "Archenemy Commander," `base_rag`'s retrieval (k=3) surfaced the general Archenemy rules but missed the specific compound concept, producing a real but incomplete answer. Two distinct issues are tangled together here: the LLM judge appears to have a verbosity/completeness bias that isn't well calibrated against precise-but-concise grounded answers, and k=3 retrieval sometimes misses the ideal chunk for compound or unusual questions. Neither of these is "RAG doesn't work" — but neither should be papered over either. Section 9.3's "spot-checked by you" caveat on model-graded scoring is doing real work here; a human pass on a sample of judge disagreements (starting with `eval/reports/rules_v1.md`'s lowest-scoring cases) is the natural next step before trusting the aggregate numbers further.

Consistency (15 questions rerun on `finetuned_rag`): 15/15 identical. This mostly confirms the generation config is deterministic (no temperature/sampling variance) rather than telling us much about the model's actual stability — a more meaningful consistency check would vary phrasing of the same underlying question, not literally repeat it.

### 9.6 Second run: the format fix, measured

Run 2 changed two things vs. run 1 (see `configs/phase1_lora_v2.yaml`): the dataset grew 569 → 2,646 training lines, and every example whose grounding context is known now also appears in the exact RAG prompt shape used at inference. Run 1 trained only on bare `question → answer` while the production path wraps the question in retrieved rules text — the adapter had literally never seen an example where the answer was supposed to come from the prompt.

`base` and `base_rag` are unchanged systems across both runs, so their movement is pure judge noise and gives a free control: **±0.09**. Anything smaller than that is not a result.

| Arm | v1 | v2 | Δ | Fabricated citations v1 → v2 |
| --- | --- | --- | --- | --- |
| base | 3.62 | 3.53 | −0.09 *(noise floor)* | 33 → 33 |
| base_rag | 3.19 | 3.25 | +0.05 *(noise floor)* | 1 → 1 |
| finetuned | 2.47 | 2.49 | +0.02 | **26 → 11** |
| finetuned_rag | 2.37 | **2.80** | **+0.43** | 5 → 6 |

**What improved, genuinely:** `finetuned_rag` gained +0.43, about 5x the noise floor — the format fix worked. Fabricated citations from the bare `finetuned` arm dropped 58% (26 → 11). The Section 8.6 failure case is fixed: asked when state-based actions are checked, v1 invented rule 704.11 and ignored the retrieved text; v2 answers correctly from context and cites the real 704.3.

**What did not:** `finetuned_rag` (2.80) still trails `base_rag` (3.25). By Section 9.4's rule, fine-tuning still isn't earning its place.

**But that gap is now partly a measurement artifact, and this matters more than the headline.** Judge score correlates with answer length across all 440 scored answers (**r = +0.21**), and the arms differ enormously in verbosity:

| Arm | avg answer length | avg score |
| --- | --- | --- |
| base | 1224 chars | 3.53 |
| base_rag | 1043 chars | 3.25 |
| finetuned_rag | 503 chars | 2.80 |
| finetuned | 450 chars | 2.49 |

The ranking by score is exactly the ranking by length. The clearest single case: asked what "Reveal" means, `finetuned_rag` answered *"To show a card to all players for a brief time. See rule 701.20a."* — essentially verbatim correct against a reference that reads *"To show a card to all players for a brief time. See rule 701.20"* — and was scored **4, "correct but missing detail"**, while `base_rag`'s longer restatement of the same fact scored 5. The fine-tuned model was trained on terse glossary-style targets, so it is being penalized for the concision it was taught, not for being wrong. (The judge is not simply broken: on the same question the bare `finetuned` arm invented rule 701.5 and "rotate a card so the back face is revealed," and was correctly scored 1.)

So the honest reading: the format fix produced a real improvement well clear of noise, and the remaining `finetuned_rag` vs. `base_rag` gap is **confounded by a length bias the current judge can't separate from correctness**. Fixing the judge — scoring factual correctness and citation validity while explicitly ignoring length and style — is now the highest-value next step, because every other decision is gated on these numbers being trustworthy. Until then, treat 2.80 vs. 3.25 as "not yet demonstrated better," not as "demonstrated worse."

### 9.7 Recalibrating the judge — and what it changed

Section 9.6 ended with the numbers themselves under suspicion, so the judge was rebuilt before any further model work:

- **Correctness and citation validity are scored separately**, so a right-but-terse answer can't be docked for thoroughness the question never asked for.
- **Length-neutrality is an explicit instruction**, not an implicit hope.
- **Candidates are anonymized** behind randomized A/B/C/D labels. The old judge saw real system names ("finetuned_rag") in a fixed order, leaving it free to reward a name or a position rather than an answer.
- Re-scoring reuses stored answers (`--rescore-from`) rather than regenerating, since only the judge changed — ~1.5 hours saved per pass. Both runs were then re-scored under this *same* judge, which is what makes v1 vs. v2 an honest comparison at all.

**Noise floor, measured not assumed:** the `base` arm's answers are byte-identical across both result files (110/110), yet scored 3.20 and 3.04 — so **±0.16** is judge variance (the randomized label shuffle is the likely source). Treat anything smaller as nothing.

| Arm | v1 adapter | v2 adapter | Δ | Citation (v2) |
| --- | --- | --- | --- | --- |
| base | 3.20 | 3.04 | *(identical answers — noise)* | 3.57 |
| base_rag | 3.87 | 3.75 | *(identical answers — noise)* | 4.37 |
| finetuned | 2.01 | 2.37 | **+0.36** | 3.11 |
| finetuned_rag | 2.53 | **3.22** | **+0.69** | 3.95 |

**Three things this settles.**

1. **The judge really was distorting the picture.** Under the old judge `base` (3.62) appeared to *beat* `base_rag` (3.19) — an inversion that never made sense, since RAG demonstrably cuts fabricated citations from 33/110 to 1/110. Length-neutral scoring reverses it: `base_rag` 3.75 vs `base` 3.04. Retrieval adds **+0.71** for the base model and **+0.85** for the fine-tuned one. RAG's value was real all along and the measurement was hiding it.

2. **The format fix is bigger than it first looked.** `finetuned_rag` gained **+0.69** (2.53 → 3.22), over 4x the noise floor, versus the +0.43 the biased judge reported.

3. **Fine-tuning still doesn't beat RAG-only — and that verdict is now firm.** `finetuned_rag` (3.22) trails `base_rag` (3.75) by 0.53, more than 3x noise. Section 9.6 could reasonably wave this away as style bias; it can't be waved away now, because the judge no longer rewards length (r = −0.005 on the v2 answers). Section 9.4's decision stands: keep fixing the data.

One nuance worth not overclaiming: the v1 answers still show r = +0.27 between length and correctness under the new judge. That is not residual bias — in the v1 run the fine-tuned answers were both short *and* genuinely bad (26 fabricated citations), so length and quality genuinely covaried in that data. The same measurement on v2's answers gives r = −0.005.

### 9.9 Judge validity: two judges, and how much they disagree

`base` scoring highest on real questions under a judge that *is* the base model was suspicious enough to test directly. The same 570 stored answers were re-scored by `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`, an independent family, changing nothing else.

| Arm | Qwen judge | Llama judge |
| --- | --- | --- |
| base | **3.40** | 3.60 |
| base_rag | 3.36 | 3.68 |
| base_rag_cards | 3.35 | **3.79** |
| finetuned | 3.11 | 3.05 |
| finetuned_rag | 3.08 | 3.07 |
| finetuned_rag_cards | 3.17 | 3.15 |

**The ranking inverts.** Qwen ranks plain `base` above both retrieval arms — i.e. "retrieval doesn't help." Llama ranks them the other way, with card-augmented retrieval on top. Identical answers, opposite conclusions about the core architecture.

**Agreement is weak in absolute terms:** Pearson **r = +0.43**, exact agreement **37%**, within-one-point 68%, mean disagreement **1.18 points** on a 1-5 scale.

This forces a correction to how earlier numbers were framed. The ±0.16 "noise floor" in Sections 9.6-9.7 measured only *within-judge* reproducibility — the same judge re-scoring the same text. It never measured *judge-choice* uncertainty, which is roughly an order of magnitude larger. Effects previously reported against that floor (+0.43 format fix, +0.69 cards, +0.71 retrieval) all sit inside the band where two reasonable judges routinely disagree. They should be read as provisional, not established.

**What survives both judges:** the fine-tuned arms score below the base arms under Qwen *and* Llama, and under Llama `base_rag` → `finetuned_rag` is −0.61 with a 95% CI of [−1.10, −0.12] — significant, from a judge with no stake in the outcome. That is the one model-quality conclusion this project can currently defend: **fine-tuning as done here makes things worse, not better.**

Practical consequence: single-judge scoring at 7-8B is too noisy for the effect sizes being chased. Future comparisons should require agreement between at least two independent judges, and a human-labeled gold subset (Section 9.3 always called for this) is needed to establish which judge is closer to right, rather than only how much they differ.

---

## 10. Iteration Loop

Phase 1 is a loop, not a line:

1. Run the eval, find the weakest category.
2. Inspect failures — usually a data gap, not a model flaw.
3. Add/repair SFT examples targeting those failures.
4. Retrain (cheap, thanks to LoRA), re-eval.
5. Repeat until rules comprehension clears your bar across all categories.

Freeze the winning adapter + its exact data manifest and config as your **Phase 1 release**.

---

## 11. Exit Criteria for Phase 1

You're ready to move on when the model can, without card data:

- Correctly walk through turn structure, priority, and the stack.
- Resolve multi-step interaction puzzles with correct sequencing.
- Explain state-based actions, zones, and the layer system accurately.
- Cite correct comprehensive rule numbers.
- Do all of the above consistently and beat the RAG-only baseline on your exam.

---

## 12. What Comes Next

Phase 2 begins card data ingestion: structured card representations, mapping card text to the rules mechanics learned here, deck-list handling, and eventually game-state tracking for text-based play with provided decks. That work builds directly on the rules foundation established above — which is exactly why we established it first.

Kickoff on the data-acquisition side has started early (Section 13) while Phase 1 fine-tuning and evaluation (Sections 8–9) are still outstanding — pulling card data doesn't depend on the adapter being trained, so there's no reason to block one on the other.

---

## 13. Phase 2 Kickoff: Card Data Acquisition

This section covers only the first step of Phase 2 — getting real card data onto disk. Card-to-rules mapping, deck-list handling, and game-state tracking (the rest of Phase 2) are still ahead.

### 13.1 Source

Card data comes from the **Scryfall REST API** (`api.scryfall.com`), a free, no-auth-required public API with comprehensive, actively-maintained Magic card data. Two ways to pull from it:

- **Bulk data downloads** — full database dumps (all ~30,000+ unique cards across Magic's history,100+ MB). Best for eventually building the complete card corpus.
- **`/cards/search` endpoint** — paginated, query-filtered results (175 cards/page). Best for pulling a scoped subset, like one format's legal pool, without downloading everything else first.

> **Important:** as with the Comprehensive Rules (Section 3.1), card text and templating are Wizards of the Coast IP. Scryfall's API terms ask for attribution ("Data courtesy of Scryfall") and restrict bulk commercial redistribution — confirm your intended use is covered before publishing anything built on this data.

**API etiquette:** every request needs a descriptive `User-Agent` and `Accept` header (missing/generic headers get throttled); stay under Scryfall's documented 10 requests/second guidance; back off 30 seconds on an HTTP 429 before retrying.

### 13.2 Format-scoped rollout strategy

Rather than pulling the entire card database up front, start with the **smallest actively-legal format pool** to validate the ingestion pipeline cheaply, then widen the net:

| Order | Format | Why |
| ----- | ------ | --- |
| **1 (start here)** | **Standard** | Smallest currently-legal pool — cheapest way to prove out schema, storage, and downstream processing before committing to a bigger pull |
| 2 | Pioneer | Meaningfully larger (all Standard-legal sets since 2012), next natural step up |
| 3 | Modern | Larger still (back to 2003), broader templating history to handle |
| 4 | Legacy/Vintage/full database | Everything, including reserved-list cards, un-sets, and the oldest templating conventions — do this last since it's the biggest surface area for parsing edge cases |

Standard is fetched via `/cards/search?q=legal:standard`, which correctly reflects real tabletop legality (Arena-only digital rebalances aren't paper-legal and are excluded automatically — no extra filtering needed).

### 13.3 Versioning

Like the Comprehensive Rules, a format's legal pool is a moving target — Standard rotates roughly yearly and cards get banned/unbanned between rotations. Every pull is a **point-in-time snapshot**: record the fetch timestamp and card count in a manifest alongside the data, and re-fetch (with a new dated manifest) before relying on it for anything legality-sensitive. Don't silently overwrite an old snapshot without updating its manifest.

### 13.4 What's deliberately deferred

Out of scope for this kickoff, still ahead in Phase 2 proper:

- Cleaning/normalizing the raw Scryfall JSON into a project-specific card schema
- Mapping card text to the rules mechanics learned in Phase 1 (e.g. resolving a card's keyword abilities to their Section 702 definitions)
- Deck-list handling and legality validation
- Game-state tracking for text-based play

### 13.5 Card-augmented retrieval: first measurement

`scripts/card_lookup.py` resolves card names by dictionary rather than embedding (99% of bracketed references on the eval set, vs. 4% when the corpus was Standard-only), and `scripts/retrieve_hybrid.py` keeps cards and rules in separate retrieval budgets. Evaluated as two extra arms (`base_rag_cards`, `finetuned_rag_cards`) on 60 real r/MTGRules questions — synthetic questions were excluded, since 0% of them name a card and their retrieval is partly circular (Section 9.8).

Only 16 of 60 questions resolved a card. For the other 44 both arms receive byte-identical context, so those act as a **control**: any difference there is a bug, not an effect.

| Subset | `base_rag` → `base_rag_cards` | `finetuned_rag` → `finetuned_rag_cards` |
| --- | --- | --- |
| Control, no card (n=44) | 3.66 → 3.66 (**+0.00**) | 3.20 → 3.18 (−0.02) |
| Card-referencing (n=16) | 2.94 → **3.62** (+0.69) | 3.38 → 3.19 (−0.19) |

**The control earned its keep.** The first run of this experiment showed −0.34 on the control subset, which is impossible if contexts are identical. The cause was in the harness, not the model: the hybrid builder emits its own `"Rules text:"` header, and the eval wrapper re-wrapped it, so every no-card prompt read `"Rules text:\nRules text:\n..."`. Without the control that would have been reported as a plausible-sounding finding ("cards slightly hurt non-card questions"). Fixed by passing an explicit `preformatted` flag instead of sniffing for a string prefix; the control now reads +0.00.

**What the card result does and does not show.** +0.69 is over 4x the ±0.16 systematic noise floor, and the citation sub-score rises 3.25 → 4.00. But per-question variance is large — 7 questions better, 3 worse, 6 tied — giving a 95% CI of **[−0.17, +1.54]**, which crosses zero. So: *directionally positive with a meaningful effect size, not statistically established at n=16.* Confirming it needs a bigger card-referencing sample; ~28% of the 200-question reddit set names a card, and the source dataset has 12,834 pairs to draw more from.

**Cards help the base model but not the fine-tuned one**, which is a coherent finding rather than noise. The v2 adapter was trained exclusively on `"Rules text: ..."`-shaped context and has never seen a `"Cards referenced:"` section — so card context is out-of-distribution for it. That is the *same* train/inference format mismatch diagnosed in Section 8.7, reappearing one level up: fixing the format made the adapter better at that format and more brittle outside it. If card retrieval becomes part of the architecture, card-formatted examples have to be in the training mix too.

## 14. RulesGuru: a labeled, human-authored question corpus

The gold set was being grown by hand-pasting questions from
[RulesGuru](https://rulesguru.org), a curated database of MTG rules questions.
RulesGuru publishes a [documented public API](https://rulesguru.org/api/documentation/),
so `scripts/fetch_rulesguru.py` replaces the copy-paste round trip.

This is the best-shaped source the project has found. Every question is
human-written and cleared through a review queue before publication (~1,500
finished against ~5,600 pending), and — critically — **labeled**: each carries a
difficulty level, a complexity rating, and topic tags. 1,402 questions were
pulled; **89% cite at least one CR rule inline**, against 21% for the reddit set.

### 14.1 The snapshot is frozen, and that is a correctness requirement

The API randomizes player names *and the cards themselves* per request. Fetching
question #3518 twice returns the same ruling about different cards:

```text
Nickolas controls Trinisphere. Amiya casts Surgical Extraction ...
Nikolas   controls Trinisphere. Alice casts Mental Misstep      ...
```

Question **ids** are stable; question **text** is not. So records are written once
and never rewritten, and re-runs are strictly additive. If refetching overwrote,
a rubric already authored against the stored wording would silently start
describing a question that no longer exists — exactly the failure the gold set
exists to prevent.

Raw responses are archived under `data/gold/rulesguru/raw/` (gitignored); the
tracked artifact is the slimmed `questions.jsonl`, which drops the embedded
MTGJSON printing data because it duplicates the Oracle pool already in
`data/cards/` (9.0MB → 4.1MB).

The randomization has one useful side effect: it is a free source of *controlled*
paraphrase. The same ruling under two card substitutions tests whether the model
tracks the rule or the card name.

### 14.2 Verified answers, unverified rubrics — kept in separate tiers

The distinction that decides where this data may be used:

- the **answer** is human-authored and rules-verified. Better ground truth than
  anything here except the judge-authored records.
- the **rubric is not.** `key_points` is drafted by sentence-splitting the answer
  and `category` is inferred from topic tags — machine guesses at the two fields
  the gold set exists to get right.

So `scripts/rulesguru_to_gold.py` writes to
`data/gold/rulesguru/gold_candidates.jsonl`, not to `gold_questions.jsonl`.
Promotion is a human act: read the drafted rubric, fix it, then `--promote`.
Bulk-importing 1,202 machine-rubriced records as gold would dissolve the one
property that makes the gold set worth having.

1,202 of 1,402 converted cleanly. The 200 held back split into 150 whose answer
cites no CR rule that resolves against the pinned version, and 50 whose answer is
a single sentence — the validator refuses a one-point rubric because one point
cannot separate a partially correct answer from a wrong one.

### 14.3 What it fixes

**The turn-structure blind spot is gone at the candidate tier.** That category sat
at 0 questions for the whole project; the tags `Turn structure`, `Turn-based
actions`, and `Cleanup Step` yield 62. Every category now clears the 8-question
target except `definition recall`:

| Category | Candidates |
| --- | --- |
| priority reasoning | 265 |
| interaction puzzle | 243 |
| layer-system question | 236 |
| templating/keyword meaning | 162 |
| zone transition | 161 |
| state-based actions | 73 |
| turn-structure walkthrough | 62 |
| definition recall | 0 |

`definition recall` reads 0 because RulesGuru is a *scenario* database — questions
are "Alice controls X, what happens?", never "what does X mean?". That is
complementary rather than a gap: the existing eval is already 54% definition
recall. Categories remain inferred and are marked `category_inferred: true`.

**It gives the judge-validity question a real instrument.** Section 9.9 established
that two reasonable LLM judges agree at only r = +0.43, which made every effect
below ~0.5 unmeasurable and downgraded three previously reported results to
provisional. Deciding *which judge tracks reality* needs human-authored answers at
a scale the 19-record gold set cannot supply; 1,202 verified answers with
citations can.

### 14.4 Cross-linking, and what it caught

Masking card names and player names produces a dedupe key stable across
substitutions. Run against the hand-pasted gold records, it matched **18 of 19**
back to their source ids at 0.98–1.00 similarity, with citation sets agreeing
exactly. That confirmed the hand-pastes were faithful, supplied provenance
(`rulesguru_id`, tags, level) for records that had none, and let the five newest
records be categorized from their twins' tags rather than left at the ingest
default.

Three defects surfaced in the process:

- The API writes multi-citations as `([603.3], [117.2a])`, a bracket form the
  rubric stripper did not handle, leaving citation noise in 50 drafted key points.
  Parenthesized groups are now dropped, but a bracketed rule id *inside* a
  sentence is unbracketed instead of deleted — removing it turns "none of the
  exceptions in [601.3] apply here" into "none of the exceptions in apply here".
- Blank `Question: ""` templates left at the end of a collection file parsed into
  a real record with an empty id. Now skipped and counted.
- One record (Elesh Norn + Omnath) carries no citation upstream either. 603.2d
  governs an ability that makes another trigger additional times; it was verified
  against the pinned CR, supplied during curation, and flagged as such in `notes`
  rather than presented as sourced.

The gold set now stands at **19 records, 19/19 cited, validation passing**.

### 14.5 The V3 judge: stop asking for a score, ask what the answer said

Building the rubrics exposed that nothing was consuming them. `key_points` and
`common_errors` rode along in the eval rows, but `scripts/eval.py` still scored
every answer against the prose reference — so the fix designed in SCHEMA.md for
the Section 9.6–9.9 judge problems had never actually been applied.

Sections 9.6 and 9.7 treated judge disagreement as a prompt-wording problem and
rewrote the rubric language. That helped the length bias but not the agreement
rate, because the disagreement is structural: "how close is this to my one
phrasing?" has no objective answer, so two competent judges legitimately land in
different places.

The V3 judge does not ask for a score. It asks two extraction questions with
checkable answers — *which of these enumerated claims did the candidate assert,
and which of these known misconceptions did it fall into* — and the 1–5 score is
computed from the counts in Python:

```text
correctness = 1 + 4 × (points_hit / points_total)     # halved if any common error is asserted
```

Two judges can still disagree about whether a claim was asserted, but they can no
longer disagree about the arithmetic. Scores land on fractions of the rubric size
(3 points → 1.0, 2.33, 3.67, 5.0) rather than on whichever integer felt right.
Mapping onto 1–5 keeps the numbers comparable with Sections 9.5–9.9 instead of
starting a fresh, incomparable scale. Asserting a common error halves credit
rather than zeroing it: an answer can state the right ruling and attach a wrong
reason, which is worse than a clean answer but better than a wrong ruling.

Routing is per question — anything carrying `key_points` goes to V3, everything
else falls back to V2 — so the synthetic and reddit sets still work unchanged, and
`--rescore-from` now carries the rubric forward so a rescore does not silently
downgrade rubric questions to prose comparison.

A 4-question smoke run shows the intended discrimination: on the Assassin's Trophy
question both RAG arms hit all three key points (5.0) while both no-context arms
hit only the verdict (2.33), despite writing *longer* answers. That is the length
bias of Section 9.6 inverted — verbosity earns nothing when the score is a count
of specific claims.

**Caveat carried forward:** auto-drafted rubrics often make the bare verdict
("Yes.", "Tapped.") key point 1, and any answer opening with the right verdict
collects it for free — worth 1/n of the score. Hand-written rubrics should fold
the verdict into a substantive claim, and this is one more reason promotion out of
the candidate tier stays a human act.

Not yet run at scale. n=4 establishes that the mechanism works, nothing about
which judge tracks reality.

### 14.6 Two noise sources, and only one of them is fixable

A 45-question pilot (stratified 6–7 per category, both judges) tested the V3
design. **It failed at what it was built for, and the failure was instructive.**

| | Prose (V2) judges | V3 rubric judges |
| --- | --- | --- |
| Inter-judge correlation | +0.43 | +0.39 |
| Exact agreement | 37% | 45% |
| Mean disagreement | 1.18 pts | 1.11 pts |
| SD of paired per-question difference | 1.48 | 1.53 |

The premise was that "which enumerated claims did this answer assert" is an
extraction question with a checkable answer, so judges could not disagree about
it. They can: the two judges produced **identical `points_hit` sets only 37% of
the time**, with Qwen systematically crediting fewer points. The subjectivity
survived being moved from "what score" to "did it say this".

**The diagnosis was rubric quality, not judge design.** The auto-drafted points
are compound ("...and applying the Mark of the Oni's effect would not change how
the Steal Enchantment's effect is applied, so..."), point 1 is usually a bare
verdict that 26% of answers scored on alone, and `common_errors` was empty
everywhere, so half the V3 mechanism had never executed.

That predicted a clean A/B: hold the questions, the stored answers, and both
judges fixed; change only the rubric. Twenty pilot questions were re-rubriced by
hand and re-judged — no regeneration, so nothing else could move.

| | Machine-drafted | Hand-authored |
| --- | --- | --- |
| Inter-judge correlation | +0.30 | **+0.62** |
| Mean disagreement | 1.26 pts | **0.80 pts** |
| Exact agreement | 41% | 49% |
| Identical `points_hit` sets | 36% | 48% |
| Judge parse failures | 4 | 1 |
| Key points per question | 3.5 | 2.5 |
| `common_errors` total | 0 | 37 |

**Rubric craft roughly doubles judge agreement.** The four rules that produced it
are now recorded in `data/gold/SCHEMA.md`.

**But sample-size requirements did not move** (SD 1.70 → 1.63). These are two
independent noise sources, and Sections 9.6–9.9 conflated them:

- *Per-question spread* — genuine variation in how much the arms differ from one
  question to the next. No judge or rubric design touches it. It sets n, and
  n ≈ 100–150 for a 0.4-point effect regardless.
- *Judge-choice uncertainty* — the kind that makes conclusions invert between
  judges. This is what rubric quality fixes.

Scaling to n=200 before fixing rubrics would have bought precision on an
instrument whose conclusions still flipped. The two investments are
complementary, not substitutes.

**The Section 9.9 inversion is resolved.** Under hand-authored rubrics both
judges rank `base` above `base_rag` — the pair they previously disagreed about.
Rules retrieval does not improve *correctness* on these card-heavy scenario
questions. This does not overturn "retrieval works": the fabrication result
(33/110 → 1/110) is a separate measurement and still stands. The claim sharpens
to *retrieval suppresses fabrication without improving correctness here*, which
fits Section 9.8's finding that the synthetic set flattered RAG.

Twenty records were promoted into the gold set with their authored rubrics
(19 → 39 records; turn-structure 0 → 3). Their `rubric_source` records that the
answer is RulesGuru-verified while the rubric is a non-judge decomposition of it
— this measured rubric *craft*, not rubric *authority*.

Caveat worth carrying: with real `common_errors` present, 62/80 and 54/80
arm-answers were charged with one. Spot-checking found the charges mostly
legitimate — on the Kruphix question every arm genuinely claims the wrong player
gains the mana — but one charge cited an error its own note did not describe.
Error attribution carries noise; it is not systematic enough to cancel the
agreement gain.

## 15. Repository review: consistency pass over the scaffolding

With the pipeline shape settled, a full read of the repository looking for
inconsistencies. Two were live bugs; the rest were latent hazards created by
copying values between scripts.

### 15.1 The destructive default

`chunk_cards.py` defaulted `--cards` to `standard_cards.jsonl` (4,887 cards) while
defaulting `--out` to `card_chunks.jsonl` — the committed 34,933-chunk artifact
derived from the full Oracle pool. The README's own quick start said:

```text
python scripts/fetch_cards.py --format standard
python scripts/chunk_cards.py
```

So following the documented path **silently replaced the full card corpus with a
Standard subset**, reverting the project to the 4% card coverage Section 13.5 was
written to fix. It degrades quietly rather than failing: `card_lookup`,
`retrieve_hybrid`, `ingest_qa_pastes`, `rulesguru_to_gold`, and `validate_gold`
all resolve names against that file, so gold-set validation would simply start
reporting cards that "did not resolve."

Fixed three ways: the default is now the full Oracle pool; a missing input prints
the command to fetch it rather than a traceback; and the writer refuses to shrink
an existing corpus by more than half without `--force`. Verified — pointing it at
the Standard file now stops with *"refusing to shrink card_chunks.jsonl from 34933
to 4887 chunks."*

`fetch_cards.py --bulk oracle_cards` was added at the same time, because the
*recommended* corpus was the one path the README asked you to download by hand.
`ingest_rulings.py` had its own near-identical copy of that downloader; it now
imports the shared one.

### 15.2 The stale adapter default

`eval.py` had `ADAPTER_PATH = "models/mtg-rules-adapter-best"` — the **v1**
adapter, long after v2 superseded it. Every documented invocation passes
`--adapter-path` explicitly, so this only bites someone running `eval.py` bare,
who would silently evaluate the wrong model and compare it against current
numbers. Now points at `mtg-rules-adapter-v2-best`.

### 15.3 Values that were copied instead of shared

Three classes of constant were duplicated across the codebase, each one edit away
from a silent inconsistency. They now live in `scripts/common.py`.

| Constant | Copies | Why it matters |
| --- | --- | --- |
| `SYSTEM_PROMPT` / RAG variants | 4 | Section 8.7's fine-tune failure *was* a train/inference format mismatch. Nothing kept the copies in agreement except that nobody had edited one yet. |
| `CROSS_REF_RE` | 8 | One name, **two incompatible meanings** — six used a search pattern to pull ids from prose, two used an anchored pattern to validate a whole string. Copying one into a file wanting the other changes behaviour silently. Now `RULE_ID_RE` and `RULE_ID_EXACT_RE`. |
| CR version | 3 defaults + a filename + prose | A gold record claiming `cr_version: 2026-08-07` while validated against a different corpus is a silent correctness failure. |

Verified after the refactor: every system prompt in `train.jsonl`, `valid.jsonl`,
and both eval sets still matches the shared constants exactly — 0 mismatches
across 3,080 records — so the training contract is unchanged.

### 15.4 The RAG index could go stale undetected

`build_index` stored `model_id` in the `.npz`; `retrieve` never read it. Nothing
compared the index against the chunks either. Chunk ids are positional, so
re-running `chunk.py` with different budgets leaves an index whose vectors still
*resolve* — retrieval returns text that was never embedded, with no error.

`retrieve` now refuses to run when the embedding model differs, and stores a
content fingerprint of the chunk set to detect drift. Verified: mutating one
chunk's text produces *"chunk_embeddings.npz is stale ... rebuild"*. Indexes built
before this change keep working, since the check is skipped when the key is
absent.

### 15.5 Smaller things

- `ingest_rulings.py --wotc-only` was `action="store_true"` with `default=True`,
  so it could never be false — a flag that read like a toggle and did nothing.
  Replaced with `--include-non-wotc`, which is the choice a caller would actually
  make.
- Documentation counts had drifted badly: the README advertised a **6**-record
  gold set against an actual 39, and described card acquisition that no longer
  matched the scripts.

### 15.6 The v2 adapter trained for 0.45 epochs, not ~1

`configs/phase1_lora_v2.yaml` documents its own reasoning in a comment block,
and that block no longer described the file it sits in:

> *"iters 600 -> 700. At batch size 4, one epoch is now ~662 iterations ... ~1
> epoch is deliberately modest"*

The actual values are `batch_size: 2` and `iters: 600`. The batch size was
halved during the Section 8.7 crash fixes — correctly, since grounded examples
had driven peak memory to 65GB on a 36GB machine — but `iters` was never raised
to compensate, and the comment kept describing the pre-crash plan.

At batch 2 over 2,644 training lines, one epoch is ~1,322 iterations. So:

| | batch | iters | steps/epoch | epochs |
| --- | --- | --- | --- | --- |
| What the comment describes | 4 | 700 | 661 | 1.06 |
| What actually ran | 2 | 600 | 1,322 | **0.45** |

**The adapter every Section 9 comparison evaluates has seen less than half the
training data once.** That matters beyond documentation hygiene: "fine-tuning
currently hurts" has been attributed to a self-distillation ceiling (the SFT
data was generated by base+RAG, so the student cannot exceed its teacher), and
that explanation may well be right. But **under-training is now an equally live
explanation**, and it is the cheaper one to rule out — raise `iters` to ~1,300
and re-run.

Until that is done, the honest statement is *"fine-tuning at 0.45 epochs does
not beat retrieval"*, which is a weaker claim than the one the Status section
currently makes.

**v2's validation curve was never recorded** (noticed 2026-08-14, while the v3
run was in flight). No training log was kept and there is no v2 loss table
anywhere in this document — the 1.159 / 1.631 figures that `phase1_lora_v3.yaml`
attributed to v2 are **run 1's**, from Section 8.6, on a different dataset at a
different batch size. So the check the v3 config specified for itself, "the
first 600 iterations should reproduce v2's loss curve", had no reference data.

Compare the **weights** instead, which needs nothing that was not saved:
`models/mtg-rules-adapter-v2/0000600_adapters.safetensors` against v3's, mean
absolute difference over the 23,068,672 LoRA parameters. For scale, 100
iterations of real training moves them by 8.1e-4 (measured, v2's own 0000500 vs
0000600). That is a stronger check than matching a summary statistic, since two
different runs can share a loss value without being the same run.

**Result: every shared checkpoint is bit-identical.**

| ckpt | 100 | 200 | 300 | 400 | 500 | 600 |
| --- | --- | --- | --- | --- | --- | --- |
| mean abs diff | 0 | 0 | 0 | 0 | 0 | 0 |

Not "within tolerance" — exactly zero, on all 224 tensors at all six
checkpoints. mlx-lm training is fully deterministic given the same seed, data
and batch size, so v3 is a **strict superset** of v2 in the literal sense: it
re-derives v2's weights step for step and then continues.

**Which recovers the lost curve rather than working around it.** Identical
weights on an identical validation set under a deterministic evaluation give
identical losses, so v3's log *is* v2's log for the first 600 iterations:

| Iter | 1 | 150 | 300 | 450 | 600 |
| --- | --- | --- | --- | --- | --- |
| Val loss | 1.986 | 1.231 | 0.990 | 0.966 | 0.975 |

Read that table with Section 18.2's error bars attached: the drop from 1.99 to
~0.97 over iterations 1–450 is roughly four times the sampling noise and is
real, but the difference between 0.966 and 0.975 is a third of it and is not.
**v2 did not overfit** — run 1 climbed from 1.16 at iteration 200 to 1.63 by
600, and nothing here goes up by anything approaching that. Beyond "it fell and
then stopped falling", this curve does not support a finer reading, and an
earlier draft of this section gave it one.

The prediction that v3's val loss would get *worse* was imported from run 1 and
is wrong for this dataset: run 1 bottomed out at 1.41 epochs, having cycled its
569 examples more than once, while iteration 600 here is 0.454 epochs and
nothing has been seen twice. Overfitting needs repetition.

It also sharpens Section 15.6's own claim. v2 did not stop mid-descent — it
stopped at the plateau. So "0.45 epochs" understates v2 less than feared on the
val-loss axis, and whether the remaining 0.55 epochs buys anything is exactly
what iterations 600–1322 are for.

The lesson is cheap and general: **redirect the training log to a file.** v3
writes to `models/mtg-rules-adapter-v3/train.log`. The recovery worked only
because the checkpoints happened to be kept; `save_every: 100` paid for itself
a second time.

### 15.7 Left for a decision

`data/cards/raw/standard_cards.jsonl` is tracked at ~26MB while the equivalent
`oracle_cards.jsonl` is gitignored, and nothing depends on it now that
`chunk_cards.py` defaults to the Oracle pool. Untracking it
(`git rm --cached` plus a `.gitignore` entry) would make the raw-data policy
consistent, but it changes what a fresh clone gets, so it is a repo-policy call
rather than a fix.

The `eval/` directory has also accumulated 8 result files and 9 reports under
inconsistent names (`cards_n100_judge2.md`). They are the record
of the experiments in Sections 9 and 13–14 and worth keeping, but a naming
convention would help before the next run adds more.

---

## 16. Phase 3 Kickoff: from explaining rules to piloting a deck

Phase 1 built a model that **explains** rules. The gameplay goal — hand it a
deck, a board, a hand, and known information, and let it choose a play — is a
different task with a different output shape and a different notion of correct.
This section covers the scaffolding for it, built alongside the open Phase 1
work rather than after it, because none of it depends on which checkpoint wins.

### 16.1 The gap was game state, not rules knowledge

Nothing in `scripts/` represented a game: no zones, no mana, no stack, no
priority. A model that can quote 603.3 still cannot say whether it may cast a
card right now. That is an engineering problem, and it is the whole of the gap.

The corollary shapes the architecture: **the rules model is a component of the
agent, not the agent itself.** The durable design is state tracker + legal-move
generator + a model choosing among moves. Every step that moves work from the
model to the environment makes the task more tractable — which matters more
here than usual, because the hardware is fixed at 36GB for now.

### 16.2 The existing judge already measures blunders

The most useful discovery of this kickoff is that almost none of the evaluation
apparatus needed rebuilding. `score_one_question` routes on `key_points`, and
nothing below it is specific to rules Q&A — `judge_batch_rubric` takes a
question string, key points, common errors, and candidate answers.

So a board position is a gold record whose "question" is a board:

| Gameplay concept | Existing gold-set field |
| --- | --- |
| the position | `question` (rendered from structured state) |
| the correct line | `answer` |
| what a correct line must say | `key_points` |
| **the blunders** | **`common_errors`** |

`rubric_correctness` already returns `errors_made`, so **blunder rate is the
fraction of positions where `errors_made` is non-empty** — it was being
computed per position all along and simply never aggregated. The rubric-craft
result from Section 14.6 therefore transfers directly, and it transfers to the
half that matters: `common_errors` is exactly where the inter-judge agreement
gain came from.

`scripts/gameplay/eval_positions.py` imports the judge from `eval.py` rather
than reimplementing it. It is a separate script only because the rules eval is
mid-measurement and its output must stay byte-identical.

### 16.3 Positions are a separate file, on purpose

`data/gold/positions.jsonl` is not `gold_questions.jsonl`, and the seven
position categories are not in `label_store.CATEGORIES`. Both of those lists
drive `stratified_sample` and `validate_gold`, and the n≈100 two-judge
comparison depends on the gold set's composition. Mixing positions in would
corrupt a measurement that is still running.

The prompt is **rendered, never authored** (`common.render_position`). Storing
rendered text would let a renderer improvement apply only to positions written
after it — the same trap Section 8.7 hit when a prompt's shape lived in two
files.

### 16.4 The action grammar, and a parser bug worth recording

A play has to be compared mechanically, so the model is given a grammar
(`PLAY`, `CAST`, `ACTIVATE`, `ATTACK`, `BLOCK`, `ORDER TRIGGERS`, `MULLIGAN`,
`KEEP`, `PASS`). Two properties earn their keep:

- **Liberal on input.** Real output arrives in markdown bullets, numbered
  lists, bold, and code fences. Rejecting that would measure formatting
  compliance rather than play quality.
- **Never silently drop a line.** Every line is an action, a parse failure, or
  ignored prose. A line that opens with a known verb and then fails to parse is
  a real failure; a line that never claimed to be an action is narration.
  Collapsing those two would either punish models for explaining themselves or
  hide malformed actions in the "prose" bucket.

The first implementation matched verbs with `startswith`, which matched the
present participles models narrate with. `"Attacking with Swiftspear is
correct"` parsed as an ATTACK on a creature named `"ing with Swiftspear is
correct"`, and — worse — `"Blocking the Bears with Elves would be bad"` parsed
as a BLOCK **with its operands inverted**. Phantom actions manufactured out of
prose inflate the action count and sink the legality rate, so this would have
surfaced as a capability finding rather than the parser bug it was. Verbs must
be whole words; the cases are asserted in `test_actions.py`.

`BLOCK` is also the one place where two natural phrasings mean opposite things
if read positionally — `BLOCK <blocker> -> <attacker>` versus `BLOCK <attacker>
with <blocker>`. Both are matched explicitly and neither falls back to the
other, because getting it backwards would silently invert every blocking
position in the set.

A second bug the schema caught: attacking creatures were first modelled as
`stack` entries. An attacking creature is a battlefield permanent with the
attacking status (506.3), not a stack object, and the Oracle name check
rejected `"Serra Angel is attacking you"` as a card that does not exist.

### 16.5 The gates

Three, in order, each able to end the avenue cheaply:

1. **Protocol works** — ≥90% of outputs parse, and ≥95% of closed-arm actions
   come from the enumerated list. Measurable at n≈8.
2. **The eval discriminates** — arms must separate. If every arm blunders at
   the same rate the positions are not measuring play quality, and more
   positions will not fix that. The most likely silent failure.
3. **Blunder rate ≤25%** on basic+intermediate positions, holding under both
   judges (Section 9.9's protocol).

**Sizing is n≈40, not the n≈100 the rules comparison needs.** Blunder rate is a
proportion and the expected effects are large: separating 60% from 30% needs
~42 per group unpaired, and this design is paired. That matters because a
position costs far more to author than a rubric — a whole board must be written
out — so 40 is 8–12 hours of authoring.

Gates 1 and 2 do not wait for 40. They run on the seed fixtures.

### 16.6 Seed fixtures are not gate evidence

`data/gold/positions_seed.jsonl` holds 8 machine-drafted positions that verify
the renderer, parser, validator and eval path end to end. They are in their own
file and are labelled `seed_note` in every record, because Section 14.6
measured machine-drafted rubrics scoring materially worse than hand-authored
ones (inter-judge r +0.30 → +0.62). The position report prints a warning
whenever seeds are present. **Gate 3 needs judge-authored positions.**

### 16.7 Working within 36GB

The constraint is asymmetric, and the asymmetry is the opportunity.

**Training is where it bites.** `max_seq_length: 2048` was set alongside
`batch_size: 2` after grounded examples drove peak memory to 65GB. A rendered
board is only ~350 characters, but board + ~8 card texts + rules chunks lands
close to the window. Retrieval for positions therefore uses `k=2` rules chunks
instead of 3, and cards are resolved **by name** rather than by embedding —
the position already states exactly which cards are in play, so there is
nothing to guess. That is a strictly better setup than the rules-question case
in Section 13.5, where only 16 of 60 questions named a card at all.

**Inference has room.** Qwen2.5-7B-4bit is ~4.5GB of weights; a 13–14B 4-bit
model is ~8GB. A larger base model for gameplay inference fits at 36GB today —
it is training an adapter for one that does not. Section 16.8 makes that a flag.

**The biggest lever is not the model.** Letting the environment enumerate legal
actions turns generation into ranking. That buys more on constrained hardware
than any parameter count, and it is why Forge and EDHplay are attractive later.

### 16.8 Model choice is now a late binding (and a flag that did nothing)

`BASE_MODEL_ID` was a module constant in `eval.py` read at five sites. Adopting
a new checkpoint or trying a larger base model meant editing code — the same
bug class as the stale `ADAPTER_PATH` default in Section 15.2.

While threading `--base-model` through, one of those five sites turned out to
be a live bug: the judge was loaded from `BASE_MODEL_ID` and **`--judge-model`
was ignored entirely** on the generate-and-judge path. `eval.py --judge-model
<other>` produced a run judged by the base model and labelled as if it had used
the other one. Only the `--rescore-from` path read the flag correctly.

**The published two-judge results are unaffected**: Section 9.9 and both Llama
reports were produced by re-scoring stored answers, which is the correct design
anyway and is the path that worked. Reports now also record the base model,
adapter and judge in the body — previously the only record of which judge
scored a run was the filename someone chose for it
(`cards_n100_judge2.md`), which put the most important variable
of a two-judge study outside the document.

### 16.9 Predictions, registered before the first run

1. **The adapter will underperform base on positions.** A board state is
   maximally out of distribution for something trained only on
   `"Rules text: ..."` — the same effect Section 13.5 measured for card context.
   Read a poor finetuned score as evidence about the training data, not the
   method.
2. **The closed arm will beat the open arm substantially** — most early failure
   should be generation, not judgment.
3. **Card retrieval will help here where it did not in Section 13.5**, because
   every position names cards by construction.

### 16.10 Substrate: now and later

Play starts as a **human text match** — the board is typed into the console,
the model answers. It reuses the existing plumbing and is the same input path
the position eval uses.

| Option | Read | Write | Note |
| --- | --- | --- | --- |
| **EDHplay** | spike | spike | Supports all play types despite the Commander branding, so Standard 1v1 vs. bot is available. Even browser-only with no API it pays off: play there, type the state in, execute the model's line. That is the advisory architecture against a live opponent with zero integration. |
| **Forge / XMage** | yes | yes | Open-source engines that **enumerate legal actions** — they hand over the closed arm for free and give an automated opponent for thousands of games. Strongest long-term substrate; a real Java integration spike. |
| **MTG Arena** | yes | **no** | Reading `Player.log` is established practice (17Lands, Untapped). **Sending input is bot play and violates WotC's ToS** — account-ban risk. Arena's viable form is read-only advisory. |

The Arena split is worth designing around now: if the agent's interface is
*state in, recommended line out*, the advisory product and the autonomous
player are the same code, and only the substrate decides who executes.

### 16.11 First run: three gate failures, and why that is the right answer

Run on the 8 seed fixtures, four arms, single judge:

| Arm | Blunder | Correctness | Parsed ok | All legal |
| --- | --- | --- | --- | --- |
| `base_open` | 75% | 2.21 | 100% | 62% |
| `base_closed` | 75% | 2.33 | 100% | **88%** |
| `base_cards_open` | 75% | 2.08 | 100% | 62% |
| `ft_cards_open` | 75% | 2.46 | 100% | 38% |

**Gate 1 fails** on closed-arm legality (88%, needs ≥95%), though parse rate is
100% everywhere. **Gate 2 fails** outright: every arm blunders at exactly 75%,
so the set does not separate them. **Gate 3 fails** at 83%.

> **Superseded — see Section 16.13.** The Gate 1 failure was a harness bug, not
> a model failure: the mandated trailing `PASS` was scored against a mulligan
> position that correctly does not list it. Closed-arm legality is 100% and
> Gate 1 passes. Gates 2 and 3 stand as written.

That is the correct outcome for machine-drafted fixtures, and Gate 2 is doing
precisely the job it was added for — flagging that this set is not yet
measuring play quality, so authoring 32 more like it would not help.

**The one result that held across both runs**: the closed arm improves legality
and does *nothing* for blunder rate. Enumerating the legal actions fixes what
the model knows is **possible** without touching what it knows is **good**.
That is a useful split, and it argues for the environment enumerating moves in
the product while treating judgment as the open research problem.

#### The run is not stable at n=8

Two runs differing only in prompt wording and parser tolerance:

| | run 1 | run 2 |
| --- | --- | --- |
| `base_open` legality | 38% | 62% |
| `base_closed` legality | 100% | 88% |
| `ft_cards_open` blunder | 100% | 75% |
| `ft_cards_open` correctness | 2.00 | 2.46 |
| Gate 2 | pass (25% spread) | fail (0% spread) |

Some of that is the fixes working (the bracket repair is what lifted open-arm
legality). But `ft_cards_open` went from emitting `PASS` 190 times to producing
the *best* correctness of any arm, on a prompt edit as small as "End with a
single PASS." **A run whose conclusions flip on prompt wording is not evidence
about a model.** Prediction 1 — that the adapter would collapse on
out-of-distribution board states — looked dramatically confirmed in run 1 and
is unsupported in run 2; it stays open.

#### Blunder rate is currently measuring the judge

Spot-checking `pos-seed-0001`: the model answered `CAST Lightning Strike TARGET
Grizzly Bears` and the judge recorded error 2 — *"casts Lightning Strike at the
opponent's face instead of removing the blocker."* It removed the blocker. That
is a false positive on the one metric the gate is built from.

With every arm pinned at 75% and a confirmed judge error in the sample, the
honest reading is that blunder rate is currently reporting more about the judge
than about the models. Section 9.9's protocol is the fix and the report now
prints the warning whenever judge and base arm are the same model.

**What this means for sequencing.** The scaffolding is verified end to end —
render, parse, validate, score, report, all four arms, gates computed. What is
not verified is that the *measurement* works, and that cannot be settled with
machine-drafted positions or a single judge. The next two steps are
judge-authored positions and a second judge, in that order.

### 16.12 A second judge, and the gate verdict reverses

Section 9.9's protocol applied to positions: the same stored answers, re-judged
by `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`, changing nothing else.
`eval_positions.py --rescore-from` exists for exactly this, because a second
judge run that regenerates the answers measures two things at once and settles
neither.

**The verdict is not robust to the judge.**

| | Qwen judge | Llama judge |
| --- | --- | --- |
| blunder rate, best arm | 75% | 25% |
| Gate 2 (discriminates) | **FAIL** — 0% spread | **PASS** — 38% spread |
| Gate 3 (≤25%) | **FAIL** — 83% | **PASS** — 17% |

Two of the three gates flip on identical answers. Any statement of the form
"the model blunders X% of the time" is currently a statement about the judge.

**Agreement is moderate, and comparable to the rules eval at its best:**
Cohen's kappa **+0.48** on the binary blunder call (75% raw, 52% by chance),
correctness r **+0.69**. Kappa rather than raw agreement because the call is
skewed — two judges each blundering 60% of the time agree half the time by
chance alone. For reference, Section 9.9 measured r = +0.43 between these same
two judges on rules questions, and hand-authored rubrics reached +0.62.

#### The judge is deterministic, which makes every difference real

Re-judging with the *same* judge over the same answers and the same rubrics
reproduced **24/24 blunder calls exactly, with 0.00 mean correctness drift**.
So there is no sampling noise to hide behind: every number here is exactly
reproducible, and the run-to-run shifts in Section 16.11 were caused by the
prompt and parser fixes changing the answers, not by variance.

#### Disagreement is concentrated in the positions, not the phrasing

Six of the eight disputed calls fall on two of the eight positions. Both had a
`common_errors` line whose leading clause was also true of the correct line —
the correct play *is* to cast Lightning Strike, and the error read "**Casts
Lightning Strike** at the opponent's face instead of…", so a judge extracting
claims can match the opening before reaching the qualifier that makes it wrong.

Rewriting those three lines to lead with what makes them wrong ("**Leaves
Grizzly Bears alive**, pointing Lightning Strike at the opponent instead"),
holding answers and judges fixed:

| | before | after |
| --- | --- | --- |
| judge blunder-rate gap | 75% vs 47% (**28 points**) | 62% vs 56% (**6 points**) |
| disputed calls | 9 | 8 |
| kappa | +0.45 | +0.48 |

**It fixed the systematic bias and not the per-call disagreement.** The gap in
*level* — the thing a gate threshold actually reads — closed by a factor of
four, because the rewrite stopped one judge over-firing. But the judges still
disagree on nearly as many individual calls; they simply now disagree in both
directions rather than one. So leading-clause overlap is a real defect worth
avoiding, and it is not the whole story: those two positions are genuinely
harder to adjudicate than the six that drew no dispute at all.

#### What this changes about the plan

**Do not author 40 positions against a single judge.** The gate was specified
as "≤25% blunder rate holding under both judges," and this run shows why the
second half of that sentence is the load-bearing part. Concretely:

1. Run both judges from the first authored position, not at the end.
2. Treat a position where the two judges disagree as a **position that needs
   rewriting**, the same way Section 14.6 treated a machine-drafted rubric —
   disagreement localizes to specific positions, so it is fixable.
3. Avoid `common_errors` whose opening clause is also true of the correct line.
   Lead with the thing that makes the play wrong.

That is a rule worth having before eight to twelve hours go into authoring, and
it cost one rescore to learn. All three are now enforced rather than advised:
`--second-judge` runs both judges and the agreement report in one invocation
and is the console's default, a one-judge report prints why it should not be
trusted, and `lint_common_errors` flags the wording defect in the authoring
form. The guide is [data/gold/POSITIONS.md](data/gold/POSITIONS.md).

The lint took two attempts, which is worth recording because the first one was
the more tempting design. Scoring word overlap anywhere in the correct line
missed the canonical case — "Casts Lightning Strike **at the opponent's face**"
has non-matching words that veto the match — while firing on two positions the
judges never disputed. A warning that is wrong in both directions gets ignored,
so it now requires a contiguous verbatim restatement: it catches the canonical
case, stays silent on all six undisputed positions, and warns without blocking.

Evidence files: `eval/runs/positions_seed.jsonl` and `_judge2` are the current
run under both judges, `_v0rubrics_judge2` is the pre-rewrite Llama run behind
the 28-point figure.

### 16.13 Gate 1 was failing on an obedient answer

Gate 1's single failure was ours, not the model's.

`GAMEPLAY_SYSTEM_PROMPT` ends with *"End with a single PASS"*, so every
well-formed answer contains a `PASS`. `match_to_legal` then scored that `PASS`
against the position's `legal_actions`. Seven of eight seed fixtures list
`PASS`, so it matched and nothing looked wrong.

`pos-seed-0006` is a mulligan decision. It happens before the game begins, when
no player has priority (103.4), so its legal set is exactly
`["MULLIGAN", "KEEP"]` — **correctly**, because there is no priority to pass.
The model answered:

```
KEEP
PASS
```

which is precisely what it was instructed to do, and the mandated terminator
scored illegal. Closed-arm legality read 88% against a ≥95% bar, and Gate 1
failed.

This is the **one name, two meanings** trap, the same shape as `CROSS_REF_RE`
and `JUDGE_SYSTEM_PROMPT` in Section 17.3. `PASS` was simultaneously a protocol
terminator ("I am done listing actions", mandated on every answer) and a game
action ("pass priority", legal only when priority exists). The two readings
agree everywhere except the one position where they don't — and that position
is not exotic, it is the entire `mulligan` category.

The fix is in `match_to_legal`: `PASS` always matches, listed or not.
Enumerating it is now optional and the seed fixtures were left as they are.
Declining to act is genuinely always available, so this is not just a protocol
escape hatch, and it does not open a dodge — an all-`PASS` answer hits no key
points and scores as a blunder, which is the gate that matters.

Recomputed from the stored answers, changing nothing else:

| Arm | All legal, before | after |
| --- | --- | --- |
| `base_open` | 62% | 62% |
| `base_closed` | 88% | **100%** |
| `base_cards_open` | 62% | 75% |
| `ft_cards_open` | 38% | 38% |

**Gate 1 now passes**: 100% parse against a ≥90% bar, 100% closed-arm legality
against ≥95%. Gates 2 and 3 are untouched — legality is computed from the
stored answers and never enters the blunder call, and both judge files produce
identical legality numbers, which is the check that it is judge-independent.

The stored `positions_seed*.jsonl` runs still carry the pre-fix `all_legal`
field. They are the record of what was measured at the time and are left
alone; this section is the correction.

#### The open arm's 62% is real

Worth separating, because "illegal" means different things in the two arms. In
the closed arm the model is shown the list, so an off-list action is a failure
to follow it. In the open arm there is no list, so an off-list action might
just as easily mean our enumeration was incomplete.

It doesn't. All three open-arm misses are genuine:

| Position | Proposed | Why it is wrong |
| --- | --- | --- |
| `pos-seed-0004` | `CAST Counterspell TARGET Mountain` | Counterspell is not in hand, and it targets spells, not lands |
| `pos-seed-0005` | `CAST Lightning Strike TARGET Grizzly Bears, Elite Vanguard` | one Lightning Strike, two targets |
| `pos-seed-0006` | `PLAY Mountain`, `CAST Lightning Strike` | taking turn actions during a mulligan decision |

A hallucinated card, a targeting-rules error, and a phase-confusion error. So
the open/closed split is now clean — 62% versus 100% on identical positions,
with the harness artifact removed from both. That is **prediction 2**
("`closed` will beat `open` substantially, because most early failure is
generation, not judgment") confirmed, and it is the strongest argument yet for
having the environment enumerate moves rather than asking the model to invent
them.

---

## 17. Repository review: the same consolidation failure, one level up

Section 15 consolidated values that had been copied into four to eight files.
This pass found the identical pattern in the *helpers*, plus three defects that
only appear when you run things the way a new contributor would.

### 17.1 Five jsonl readers, and three of them crashed on a blank line

The same six-line function existed five times — `load_jsonl` in three scripts,
`read_jsonl` in `label_store`, `load_positions` in `positions` — plus eight
more inline comprehensions. They did **not** agree:

```python
[json.loads(line) for line in f]                    # chunk.py, build_sft.py, rag.py
[json.loads(line) for line in f if line.strip()]    # label_store.py, positions.py
```

A trailing blank line is what appending by hand or saving in an editor leaves
behind, and the unguarded form raises
`JSONDecodeError: Expecting value: line 2 column 1`. Eight call sites carried
that bug.

This is exactly what Section 15.3 fixed for `load_rule_ids` — "six callers each
had their own copy, four as inline comprehensions that would raise on a trailing
newline" — reappearing because the *lesson* was applied to one function rather
than to the class of problem. There is now one `common.read_jsonl`, and
`write_jsonl_atomic` moved there too, which also removed a function-level
`sys.path` insert that `positions.py` used to reach back into `scripts/`.

### 17.2 Every script required being run from the repo root

```
$ cd /tmp && python ~/magic-llm/scripts/card_lookup.py "Llanowar Elves"
FileNotFoundError: data/cards/processed/card_chunks.jsonl
```

Every canonical path was relative to the current working directory. Nothing
documented it; the web console's runner only worked because it happened to
inherit the right cwd. Paths are now anchored to `REPO_ROOT`, computed from
`common.py`'s own location. Paths a user supplies on the command line stay
relative to their cwd, which is what anyone would expect.

The same pass found **30 argparse defaults re-hardcoding paths `common.py`
already defined** — `data/processed/rules.jsonl` appeared six separate times,
`gold_questions.jsonl` three. Those are now the constants, which fixes the
duplication and the cwd bug in one change.

### 17.3 Dead code and a name collision

`judge_batch` — the v1 judge — had been unreachable since Section 9.7 replaced
it and re-scored both earlier runs under v2. Removed, along with the prompt only
it used. The v1 *results* remain in `eval/`; the code remains in git history.

Removing it also surfaced a collision: `JUDGE_SYSTEM_PROMPT` existed in two
files meaning different things — "grade this answer against a reference" in
`eval.py`, "classify whether this Reddit comment is a genuine ruling" in
`build_reddit_eval.py`. That is the `CROSS_REF_RE` hazard from Section 15.3
verbatim. The survivor is renamed `RULING_FILTER_PROMPT`.

### 17.4 Documentation checked against the artifacts

Every corpus count in the README was verified by counting the file. Thirteen of
fourteen matched. The exception: official rulings were documented as 77,931 and
the tracked chunks contain **77,918** across 19,726 cards.

`requirements.txt` turned out to be a full `pip freeze` — 73 pins, of which the
project imports six directly (`mlx-lm`, `mlx-embeddings`, `numpy`, `datasets`,
`fastapi`, `uvicorn`). It also pins `mlx-audio`, `mlx-vlm`, `opencv-python`,
`miniaudio` and `sounddevice`, which no script here uses. Trimming is safe but
changes what a fresh clone installs, so the file now says what it is and the
decision is left open rather than made silently.

`CLAUDE.md` was added: architecture, the evaluation rules that make a number
trustworthy, and a list of the traps this repository has actually fallen into,
since those recur in new code.

### 17.5 The eval/ directory, reorganized

33 files sat at one level under four competing naming schemes, and Section 15.7
had flagged it. It is now split by role, because the roles were genuinely
indistinguishable by name:

```text
eval/sets/      inputs — the question sets, with their manifests
eval/runs/      outputs — per-question scored results
eval/reports/   outputs — human-readable summaries
```

**A report now shares its run's stem**, which was not previously true and was
actively misleading. `EVAL_REPORT_CARDS.md` documented
`eval_results_cards_v2.jsonl`, while the similarly-named
`eval_results_cards.jsonl` was the *discarded* run — the one whose no-card
control read −0.34, which is impossible when both arms receive byte-identical
context, and which is why the control exists at all. Pairing report to data by
name would have analyzed the wrong file. Determining which was which needed the
data rather than the names: the corrected run has 44 of 60 no-card answers
identical between arms, exactly matching Section 13.5; the discarded one has 5.
It is now `runs/cards_n60_superseded_headerbug.jsonl`.

Two suffixes, consistently:

| Suffix | Means | What varied |
| --- | --- | --- |
| `_rejudged` | re-scored under the v2 judge **prompt** (9.7) | the prompt |
| `_judge2` | re-scored by an independent judge **model** (9.9) | the model |

Both mean the answers are byte-identical and only the judge changed — the only
design under which an agreement number means anything.

`eval/README.md` documents the convention. Every reference was rewritten across
13 files, and 28 of the 33 moves used `git mv` so history follows.

**One hazard the rename created and then removed.** Mapping
`eval_results.jsonl` onto `runs/rules_v1.jsonl` left `eval.py`'s default `--out`
pointing at an archived experiment, so a bare `python scripts/eval.py` would
have overwritten the Section 9.5 run. Output defaults are now neutral
(`runs/latest.jsonl`) and anchored to `REPO_ROOT`. That is the same destructive-
default shape as the `chunk_cards.py` bug in Section 15.1, reintroduced by a
cleanup and caught by checking what every default resolved to.

`REPO_MAP.md` was added — every file and its purpose in one document. It is
deliberately **untracked**: a file-by-file inventory goes stale on every commit,
and a stale map is worse than none.

### 17.6 What the review did not change

`data/cards/raw/standard_cards.jsonl` is still tracked at ~26MB with nothing
depending on it (Section 15.7). Still a repo-policy call.

---

## 18. Run 3: the full epoch, and an instrument that could not read it

### 18.1 The run

`configs/phase1_lora_v3.yaml`, launched 2026-08-14. 1,322 iterations at batch 2
over 2,644 training lines — **exactly 1.00 epochs**, against v2's 0.454. Only
`iters` differs from v2. 6h27m wall clock, peak memory 10.17GB of 36, clean
exit, 13 checkpoints plus final weights.

**The one-variable claim is verified, not assumed.** Every checkpoint v2 and v3
share is *bit-identical* — mean absolute difference exactly 0.0 across all 224
tensors at iterations 100 through 600 (Section 15.6). mlx-lm is deterministic
given the same seed, data and batch size, so v3 re-derives v2 step for step and
then continues. Whatever this run shows, it is not confounded by a second
change.

| Iter | 1 | 150 | 300 | 450 | 600 | 750 | 900 | 1050 | 1200 | 1322 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Val loss | 1.986 | 1.231 | 0.990 | 0.966 | 0.975 | 0.673 | 0.665 | 0.770 | 0.723 | 0.657 |

### 18.2 Most of that curve is sampling noise

The curve invites a story — fell, plateaued around 450–600, dropped sharply
past v2's stopping point, wobbled, finished low. Three separate readings of it
were written into this document and the config header before anyone asked what
the number is computed on.

It is computed on **16 of the 327 validation examples**. `val_batches: 8` at
`batch_size: 2` is 8 batches, and worse, mlx-lm's `iterate_batches` **sorts the
dataset by length** before batching and `evaluate` calls it **with no seed** —
so `np.random.permutation` runs off the global numpy RNG state, which the
training loop keeps advancing. Every validation pass scores a *different* 16
examples, drawn from different length strata.

Measured directly, at checkpoint 1300 with the **weights frozen**, so every bit
of spread is sampling and nothing else:

| | |
| --- | --- |
| 8 draws of 16 examples | 0.818, 0.562, 0.581, 0.657, 0.569, 0.672, 0.678, 0.634 |
| range / SD | **0.256** / 0.084 |
| full 327-example set | **0.690** |
| logged at iteration 1322 | 0.657 |

**The noise band at a single frozen checkpoint is 0.256. The entire training
curve from iteration 600 onward spans 0.113.** Every value from iteration 750 to
the end falls inside the scatter a stationary model produces, so that half of
the curve carries no signal at all.

What survives: iterations 1–450 fall by ~1.0, about four times the noise band.
The model genuinely learned in the first third of an epoch. Beyond "it fell and
then stopped falling," this instrument cannot resolve anything.

Note also that the 8 draws average 0.646 against a full-set 0.690 — the draws
are biased low, not merely noisy, because sampling 8 of 163 *length-sorted*
batches is not sampling examples uniformly.

**This does not settle the under-training question either way.** It removes the
evidence, leaving Section 15.6's hypothesis exactly where it was: open, and
answerable only by the judge eval. Val loss here measures agreement with a set
generated *by* base+RAG, so even a clean reading would measure teacher
imitation rather than answer quality.

**For run 4:** `val_batches: -1` (score all 327), and a fixed validation seed so
the sample is at least stable across iterations. v3's config is left alone —
editing it would break the reproducibility property just confirmed.

The general shape is one this repo keeps finding: **a number that looks like a
measurement of the model, and is partly a measurement of its own sampling.**
`--judge-model` accepted and silently ignored, `ADAPTER_PATH` pointing at v1,
the double `"Rules text:"` header — same family. The tell is the same each time:
the parameter that controls it reads like a thoroughness knob (`val_batches:
8`) rather than a term in the result.

### 18.3 The judge eval: a full epoch changes nothing, under either judge

Section 18.2 left the under-training question open and answerable only by the
judge eval. It is now answered. Checkpoint 1322 was run through the standard
Section 9 harness — same 110 questions (70 synthetic + 40 reddit), same four
arms, same V2 anonymized judge — and then every stored answer was re-scored by
`Meta-Llama-3.1-8B-Instruct-4bit`, which is the two-judge protocol Section 9.9
made mandatory.

| Arm | v2 Qwen | v3 Qwen | Δ | v2 Llama | v3 Llama | Δ | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 3.04 | 3.21 | +0.17 | 2.94 | 2.93 | −0.01 | **control** |
| base_rag | 3.75 | 3.65 | −0.10 | 3.71 | 3.94 | +0.23 | **control** |
| finetuned | 2.37 | 2.40 | +0.03 | 2.06 | 2.20 | +0.15 | treated |
| finetuned_rag | 3.22 | **3.25** | **+0.04** | 3.29 | **3.32** | **+0.02** | treated |

Paired over all 110 questions: **+0.04, 95% CI [−0.26, +0.33]** under Qwen and
**+0.02, CI [−0.33, +0.37]** under Llama.

**The control is the argument, not the CI.** `base` and `base_rag` never touch
the adapter; their answers are byte-identical between the two runs, verified
110/110 on all four arms. Those arms moved by up to 0.23 anyway — that is the
judge re-scoring the same text. The treated arm moved *less than that*, in both
judges. Doubling the training epoch is not merely a small effect here; it is
smaller than re-reading the same answers.

**What this settles.** Section 15.6 hypothesized that "fine-tuning does not beat
retrieval" was confounded with training for 0.45 of an epoch. It was not. The
confound is removed and the verdict is unchanged, so Section 9.4's decision rule
stands where it has stood since run 1: **fix the SFT data, not the
hyperparameters.** More gradient steps over the same 2,644 examples is not the
missing ingredient.

**The checkpoint sweep is already two-thirds done.** `models/mtg-rules-adapter-v2-best`
and `models/mtg-rules-adapter-v3-ckpt600` are bit-identical — 224 tensors, max
absolute difference exactly 0.0 — because v3 re-derives v2 step for step
(Section 18.1). So the v2 column above *is* the iteration-600 evaluation, and
iteration 900 sits between two endpoints that a two-judge comparison cannot tell
apart. It was not run, and the ~1.5 GPU-hours went to the second judge instead,
which is the caveat that could actually have changed the answer.

**Two judges agree here, which is new.** Inter-judge Pearson on the 436 scored
answers is **r = +0.60** (44% exact, 74% within one point, mean disagreement
0.97), against the **+0.43** Section 9.9 measured — and for the first time both
judges produce the *identical* arm ranking, `base_rag > finetuned_rag > base >
finetuned`, with the `base_rag` → `finetuned_rag` gap excluding zero under each
(Qwen +0.40 [+0.09, +0.71], Llama +0.61 [+0.21, +1.02]). Section 9.9's ranking
inversion was measured on a different set (the n=100 reddit card questions), so
this does not overturn it; it does mean the run-3 conclusion does not depend on
which judge is asked, which is the only thing being claimed.

**One movement not to over-read:** fabricated citations from `finetuned_rag`
fell 6/110 → 2/110. At these counts that is not distinguishable from chance, and
it is recorded as an observation rather than a result.

**Promotion decision: nothing was promoted.** `models/mtg-rules-adapter-v3-best`
deliberately does not exist, and `eval.ADAPTER_PATH` stays at
`models/mtg-rules-adapter-v2-best`. Promoting ckpt1322 would assert it is
better; two judges say it is not distinguishable, and the project's rule is
that a number inside the noise band is not a result. Keeping v2-best also costs
nothing in currency: it *is* iteration 600 of the v3 run, bit-identical, so the
default points at the earlier of two indistinguishable checkpoints on the same
curve rather than at a superseded experiment.

This is written down because the opposite failure has happened here —
`ADAPTER_PATH` pointed at v1 long after v2 superseded it and a bare
`python scripts/eval.py` silently scored the wrong adapter (Section 15.2). The
distinction between "stale" and "deliberate" lives in the reason, so the reason
is recorded next to the default itself.

### 18.4 Gate 1 passes; Gate 2 still depends on which judge is asked

The `PASS`-matching fix from Section 16.13 was re-run against the stored seed
answers, changing only the parser — no regeneration, and the judge is
deterministic, so the correctness numbers reproduce exactly and any movement is
attributable to the fix alone.

| Arm | all-legal before | after |
| --- | --- | --- |
| base_open | 62% | 62% |
| **base_closed** | **88%** | **100%** |
| base_cards_open | 62% | 75% |
| ft_cards_open | 38% | 38% |

**Gate 1 now PASSES** — worst-arm parse rate 100%, closed-arm legality 100%
against a ≥95% bar. The gate had been failing on obedient answers.

**Gate 2 still splits between judges**, exactly as Section 16.12 found: Qwen
reports a blunder-rate spread of 0% (FAIL), Llama 50% (PASS). What the
per-position calls add is *why*, and it is sharper than a spread number:

| | positions where all four arms get the same blunder verdict |
| --- | --- |
| Qwen judge | **8 of 8** |
| Llama judge | 4 of 8 |

Under Qwen the blunder call is fully determined by the position and not at all
by the answer — including for the fine-tuned arm, whose answers differ
substantially from base. A judge that returns the same verdict for four
different answers is not grading answers. That is a rubric problem on
machine-drafted seeds, which is what Section 14.6 predicts (hand-authored
rubrics took inter-judge agreement from r +0.30 to +0.62), and it is why Gate 2
cannot be settled on the seed set no matter how many seeds are added.

Cohen's kappa on the blunder call is **+0.48** and correctness correlation
r = +0.69, both unchanged from the pre-fix run, confirming the parser fix touched
legality and nothing else.

**One signal worth carrying forward:** `ft_cards_open` has the *worst* legality
of any arm at 38%, against 100% for `base_closed`. Small n, but it points the
same direction as pre-registered prediction 1 — a board state is out of
distribution for an adapter trained on `"Rules text: ..."` — and it is the
cheapest evidence yet that the closed protocol is the lever, not the checkpoint.

### 18.5 A templated rubric would have reached the judge as literal text

Found while preparing the n≈100 comparison, not by a failing run.

The gold set stores rubrics templated — `"[[card2]] enters tapped"` — so one
rubric scores every instantiation RulesGuru builds of the same ruling
(Section 14). `validate_gold.py --to-eval` copied that through verbatim, and
nothing downstream expanded it. The judge would have been asked which of these
claims the answer made, against a claim naming `[[card2]]` while the answer
says "Urborg, Tomb of Yawgmoth". **35 of 39 eval rows carried a slot in
`key_points`, 28 of 39 in `common_errors`, and 0 of 39 carried the `card_slots`
needed to resolve them.**

Nothing would have crashed. Every affected record would simply have lost the key
points that named a card, deflating scores on precisely the hand-authored
rubrics the comparison exists to evaluate.

This is the `[TARGET <x>]` failure a third time: meta-syntax placed in front of
a model that reads it as literal text. It was already fixed once for
`--ingest-submissions`, where templating had blinded `lint_common_errors`. The
eval path was missed because a templated rubric is still valid JSON and still
produces a score — just a lower one.

`to_eval_records` now expands slots and carries `card_slots` onto the eval row
so a future variant run can re-template against a different instantiation's
cards. Validation gained a check for the failure hiding underneath: a rubric
referencing `[[card3]]` when `card_slots` has no `card3` survives expansion and
reaches the judge as literal text, so that is now an error rather than a silent
lost point. Regeneration was diffed field by field — only `key_points`,
`common_errors` and `card_slots` changed, and `messages` is byte-identical, so
generation is unaffected.

The mana-symbol hazard shows up plainly in the output, vindicating brackets over
braces:

```diff
- The payment of {b} is made as [[card2]]'s trigger resolves
+ The payment of {b} is made as Nihil Spellbomb's trigger resolves
```

---

## 19. A3: the n=99 verdict on hand-authored rubrics

Every judge number before this section was measured on one of two things: the
110-question synthetic+reddit set, whose references are machine-derived, or a
hand-authored subset too small to resolve the effect it was asked about. Section
14.6 sized the instrument at ~100–150 questions for a 0.4-point effect, and A3
has been blocked on that number since run 1. The gold set reached 99 (Section
14.7), so it ran.

99 questions, four arms, generated once and scored twice — Qwen first, then
`Meta-Llama-3.1-8B-Instruct-4bit` via `--rescore-from`. Same rubrics, same
answers, same judge prompt; the judge model is the only thing that varies, which
is what Section 9.9 requires.

| Arm | Qwen (V3 rubric) | Llama (V3 rubric) |
| --- | --- | --- |
| `base_rag` | **2.38** | **3.16** |
| `base` | 2.46 | 3.09 |
| `finetuned_rag` | 1.88 | 2.36 |
| `finetuned` | 1.69 | 2.18 |

**Both judges produce the same arm ranking.** `finetuned_rag` trails `base_rag`
by 0.50 under Qwen and 0.80 under Llama — well outside the 0.4-point effect this
sample was built to resolve, in the same direction, under two models that have
previously disagreed enough to reverse a ranking outright (Section 9.9) and to
reverse two of three gameplay gates (Section 16.12).

So the project's headline negative is now a measurement rather than a gesture:
**fine-tuning did not beat retrieval**, at adequate sample size, on
hand-authored rubrics, under two independent judges. Sections 8 and 9 reached
the same conclusion twice on instruments that could not support it. This one
can.

Inter-judge agreement is r = +0.49 over 367 arm-answer pairs, 30% exact. Broken
down by rubric author, `Cody Clark` r = +0.51 (n = 283 pairs) and `Steve Steve`
r = +0.40 (n = 84). That is the same direction and roughly the same size as the
n=12 pilot in Section 14.8, whose interval was [−0.19, +0.53] — so it remains
**unresolved, not confirmed**. Steve's 24 questions are still below the
25-per-segment line `compare_judges` warns at.

Llama returned an unparseable score on 7–8 questions per arm, so its n is 91–92
against Qwen's 99/99.

### 19.1 The best-scoring arm is the one that makes things up

Under Qwen, `base` — no retrieval at all — scores 2.46, nominally the highest of
the four. It also fabricates a rule citation on **35 of 99** questions:

| Arm | Fabricated citation | Matched the reference |
| --- | --- | --- |
| `base_rag` | 1/99 | 10/99 |
| `base` | **35/99** | 2/99 |
| `finetuned_rag` | 4/99 | 5/99 |
| `finetuned` | 9/99 | 0/99 |

The V3 judge scores which enumerated claims an answer made. Inventing a
plausible-looking rule id is not one of the claims, so **fabrication is very
nearly free under this metric**. Retrieval's measurable contribution shows up
almost entirely in the citation columns and almost not at all in the score.

Promoting the check from a column to a first-class metric changes what the run
says. Grounding — cited at least one rule id and fabricated none — is computed
mechanically against the pinned CR, with no judge call and therefore no judge
noise:

| Arm | Score | **Grounded** | Fabricated |
| --- | --- | --- | --- |
| `base_rag` | 2.38 | **86/99** | 1/99 |
| `base` | **2.46** | 45/99 | 35/99 |
| `finetuned_rag` | 1.88 | 68/99 | 4/99 |
| `finetuned` | 1.69 | 62/99 | 9/99 |

`base` wins the score column by 0.08 while grounding its answer *half as often*
as `base_rag`. So retrieval's contribution is not small and inside the noise, as
the score column suggests — it is 86 against 45, the largest and cleanest effect
anywhere in this run, and the holistic score was hiding all of it.

The report now prints this beside the score and refuses to let the score be read
alone: when the top-scoring arm is not also the least-fabricating one, it says so
in the output rather than trusting the reader to remember. Grounding is
deliberately NOT folded into correctness — blending them would rebuild the
confounded single number V3 exists to take apart.

This is a property of the instrument, not a result about the models, and it is
recorded here because the ranking invites exactly one misreading: that
retrieval is unnecessary because `base` scored highest. Any reading of the score
column without the fabrication column beside it is wrong. Whether the rubric
judge *should* price fabrication is an open question — pricing it would make the
metric a blend of two things again, which is what V3 was built to stop.

### 19.2 The rescore report described a judge that never ran

Found while checking whether the two-judge comparison was clean, which is the
one thing that makes the headline result meaningful.

`rescore()` built its report header from a hardcoded string reading "with the v2
judge: correctness and citation scored separately … candidates anonymized behind
randomized A/B/C/D labels." None of that describes what happens once the gold
set is fully rubric-backed: `score_one_question` routes to the V3 rubric judge
for any record carrying `key_points`, and `rescore()` deliberately preserves
that — its own comment says so.

Verified against the stored data rather than against the report: **all 396
arm-answers in both runs carry `scored_by="rubric"`.** The comparison is clean.
The report said it was not.

The failure mode is worth naming because it is the inverse of the usual one. A
stale default normally makes a bad result look good. This one made a *sound*
result look unusable: a reader comparing the judge-2 header against the V3
first-pass report would conclude the study had varied model **and** prompt, and
would discount it. Same family as the stale `ADAPTER_PATH` and the accepted-then-
ignored `--judge-model` — the default was correct, printing it without checking
was not.

The header is now derived: it counts `scored_by` across the run, names the V3
rubric judge, the V2 prose judge, or the mix, and states explicitly that only
the model differs when that is true.

---

## 20. The first gate run on hand-authored positions

Sections 16.11 and 18.4 both ran the gates on the 8 machine-drafted seed
fixtures and both said the same thing: the eval did not discriminate. Every arm
blundered at exactly 75%, and the band split added in Section 19 shows why that
was worse than it looked — the four arms were identical in *every* difficulty
band, which is the signature of a judge grading the position rather than the
answer.

`data/gold/positions.jsonl` now exists: 22 positions, 18 with the line
adjudicated by hand, 8 basic / 9 intermediate / 5 advanced across all seven
categories. Run through both judges.

| Arm | Blunder (Qwen) | Blunder (Llama) | Correctness | Parsed ok | All legal |
| --- | --- | --- | --- | --- | --- |
| `base_open` | **50%** | **45%** | 2.62 | 100% | 68% |
| `base_closed` | 77% | 82% | 2.55 | 100% | 82% |
| `base_cards_open` | 64% | 50% | 2.14 | 100% | 64% |
| `ft_cards_open` | 82% | 64% | 1.95 | 77% | 36% |

**Gate 2 PASSES, for the first time in the project.** Blunder rate spans 50–82%
under Qwen and 45–82% under Llama — a 32- and 37-point spread against the seed
set's zero. The positions separate the arms. That is the single thing
hand-authoring was supposed to buy and it is the first direct evidence that it
does.

Gate 1 **fails**: `ft_cards_open` parses 77% against a ≥90% bar, and closed-arm
legality is 82% against ≥95%. Gate 3 **fails**: the best arm blunders 47% on
basic+intermediate against a ≤25% bar. Both are real failures of the models,
not of the harness — Gate 1 passed on the seeds after the Section 16.13 fix, and
what breaks it here is `ft_cards_open` emitting 4.7 actions per answer with only
36% of them legal.

The arm ranking is the same under both judges and matches Section 19: the
fine-tuned arm is worst on every measure — highest blunder rate, lowest
correctness, worst parse rate, worst legality. Two independent instruments, a
rules exam and a gameplay eval, now agree.

### 20.1 The closed arm makes the model worse, not better

> **Weakened by Section 21.28.** This section rests on blunder rate, and on
> correctness as scored with the error-halving. The positive controls later
> measured the Qwen judge inventing a `common_error` against the *reference
> answer* on 40% of questions, and `base_closed` — which answers tersely from a
> supplied `legal_actions` list — is the arm that draws those most. Rescored on
> `points_hit` alone, `base_closed` and `base_open` swap order on both position
> runs. One judge and n=22, so the reversal is not settled either; what is
> settled is that the claim below is no longer supported by the metric it was
> made on.

`base_closed` blunders **77%** where `base_open` blunders 50%, and 82% vs 45%
under Llama. Handing the model an enumerated list of the legal plays makes it
play *worse*, consistently, under both judges.

This inverts the arm's design intent. The closed arm exists to separate "does
not know what is possible" from "does not know what is good" (Section 16), on
the assumption that removing the search problem could only help. It does lift
legality — 82% of named actions are legal versus 68% open — so the model is
choosing from the list. It just chooses badly, and more confidently.

The likeliest reading is that the list crowds out reasoning: given three
candidate plays the model picks one rather than working out what the board
demands. That is a hypothesis, not a finding, and it is testable — a closed arm
whose list is deliberately ordered worst-first would separate "picks from the
list" from "picks the first plausible entry".

### 20.2 What the per-category table says about the positions

| Category | n | base_open | base_closed | base_cards_open | ft_cards_open |
| --- | --- | --- | --- | --- | --- |
| blocking | 4 | 25% | 75% | 25% | 25% |
| combat math | 6 | 83% | 67% | 83% | 100% |
| land sequencing | 3 | **100%** | **100%** | **100%** | **100%** |
| mulligan | 2 | 0% | 100% | 50% | 50% |
| race vs stabilize | 2 | 0% | 50% | 0% | 100% |
| removal timing | 3 | 33% | 67% | 67% | 100% |
| trigger ordering | 2 | 50% | 100% | 100% | 100% |

**`land sequencing` is arm-invariant at 100%** — every arm blunders every one of
its three positions. Two readings were available: the models are uniformly bad
at land sequencing, or the three rubrics demand a phrasing no reasonable answer
uses. **The second was recorded here as "the likelier" and it was wrong.**

Reading the stored answers settles it. On `pos-land-sequencing-0001` the best
arm answered:

```
PLAY Swamp
PLAY Swamp
CAST Vampire Nighthawk
ATTACK Vampire Nighthawk, (opponent)
```

Two land drops in one turn, a three-mana creature cast off two lands on turn
one, and an attack with a creature that entered this turn. The other two are the
same shape — one of them emits `CAST Temple of Silence`, casting a land, and
then `KEEP BOTTOM` on a turn-one board. These are not rubrics failing to
recognize a good answer. They are catastrophically bad answers, and 100% blunder
is the correct measurement.

The lesson is the one the repo keeps relearning: a hypothesis about why a number
looks wrong is worth exactly what it costs to check, and checking cost one read
of the stored answers.

The **basic band varies across arms** (50/75/62/75), which answers the bet made
when those four were drafted: a basic position built on one rules fact is *not*
automatically arm-invariant. Basic meaning "one fact" rather than "no decision"
survives its first test.

### 20.3 The blunder call is still not a reliable per-question metric

Cohen's kappa on the blunder call between the two judges is **+0.24** (raw
agreement 65%, chance 54%), with **31 of 88 arm-position calls disputed**.
That is weak agreement — usable for large effects only.

So the honest statement of this run has two halves. The *aggregate* is sound
enough to rank arms: both judges order the four arms identically and both give
Gate 2 a comfortable pass. The *individual call* is not: on roughly a third of
arm-position pairs the two judges disagree about whether a blunder happened at
all. Section 16.12's protocol applies — a disputed position is a position to
rewrite, and the 31 are enumerated in
`eval/reports/positions_n22_AGREEMENT.md`.

Kappa did improve on the seed set's disputed-call concentration, but not enough
to call blunder rate a settled instrument. Gate 3's 47% is far enough from 25%
that judge noise does not explain it away; a Gate 3 result *near* the boundary
would not be trustworthy at this kappa.


### 20.4 The legality check could not see a second land drop

Following the land-sequencing answers down found a harness gap. `legal_actions`
is a **set** membership test — it answers "may this be done?" and carries no
notion of *how many times*. A land drop is once per turn (305.2), so naming two
is illegal, and the check could not see it. Two separate routes through:

```
PLAY Swamp / PLAY Swamp     consecutive duplicates were COLLAPSED, so the
                            harness saw one land drop and never knew
PLAY Swamp / PLAY Forest    not duplicates, both in legal_actions, so
                            all_legal came back True
```

Measured before fixing: **10 of 88 answers named more than one `PLAY`**, and
none were scored illegal for it.

The collapse itself is correct for its own purpose — a model emitting `PASS` 190
times has looped, not acted, and `repeats_collapsed` exists to say so. But that
field had quietly taken on two meanings: "degenerated into a loop" and "took a
once-per-turn action twice." Those want opposite treatment, which is the
`CROSS_REF_RE` / `JUDGE_SYSTEM_PROMPT` / `PASS` trap for the fourth time.

`ONCE_PER_TURN` verbs are now exempt from collapsing and are checked by
`rule_illegalities()`. Only `PLAY` is listed: it is unambiguous, whereas `ATTACK`
is also declared once but the grammar asks for all attackers on one line, so
flagging a split declaration would punish formatting rather than play.

`eval_positions.py` was computing `all_legal` inline instead of calling
`legality()` — a second copy of the rule, which would have missed the new check
entirely. That is the "helper duplicated with a guard in only some copies" trap,
and it is now routed through the one definition.

Recomputed over the stored n=22 answers, the correction is real but small:

| Arm | all_legal before | after |
| --- | --- | --- |
| `base_open` | 68% | 68% |
| `base_closed` | 82% | **73%** |
| `base_cards_open` | 64% | 64% |
| `ft_cards_open` | 36% | 36% |

Only `base_closed` moves, because the other arms' offending answers were already
illegal for other reasons. Gate 1's verdict is unchanged — the closed-arm bar is
≥95% and 82% already failed it — but the number it fails by is now honest, and
the check is in place before the set grows to 40.

---

## 21. Improving the metrics: four changes, in order

Section 20 left three problems, and Section 19.1 a fourth. This section is the
work on them. Nothing here is a change to how a number is *presented* to make it
read better — the point of the repo is that "fine-tuning did not beat retrieval"
is written down plainly, so the failure mode being avoided throughout is making
the numbers look better without changing anything real.

### 21.1 The models are not deliberating (new arm, not a replacement)

Measured on the n=22 run:

| Arm | prose lines / answer | chars / answer | answers with any reasoning |
| --- | --- | --- | --- |
| `base_open` | 0.0 | 39 | **0%** |
| `base_closed` | 0.2 | 109 | 18% |
| `base_cards_open` | 0.3 | 134 | 32% |
| `ft_cards_open` | 3.0 | 410 | 45% |

`base_open` — the best arm — answers every board in a mean of **39 characters**
with zero lines of reasoning on 22 of 22 positions. It is not deliberating; it
is emitting a plausible action. `GAMEPLAY_SYSTEM_PROMPT` ends with "Give your
reasoning first if you want to", which is permission, and nothing takes it.

`base_open_think` appends `GAMEPLAY_DELIBERATE_INSTRUCTION`, making it an
instruction with a shape. It is **added**, not substituted: `base_open` and
`base_open_think` differ in the system prompt and nothing else — the user
message is byte-identical — so the pair isolates one variable, and every
previously measured arm is untouched.

**Pre-registered prediction.** If deliberation is the missing thing, Gate 3's
blunder rate falls *and* Gate 2's spread narrows, because part of the current
separation between arms is verbosity rather than judgement. If blunder rate does
not move, the problem is knowledge, not deliberation.

Two harness bugs would have corrupted this before it produced a number, and both
were found by checking rather than by running:

**Prose about a play parses as that play.** `"Play Mountain first would strand
Shock in hand"` became `PLAY Mountain first would strand Shock in hand` —
silently, with no `ParseFailure`, an action with a garbage operand. Any arm
asked to reason would have had its action count inflated and its legality
destroyed by its own explanation, and the result would have read as *reasoning
makes the model play worse*. `parse_output` now honours an `ACTIONS:` marker;
outputs without one are unaffected.

**Both judge prompts hardcode "labeled A, B, C, D"** while the labels are
generated as `chr(ord("A") + i)` over however many arms exist. At four they
agree; at five the judge would see CANDIDATE A–E and be told in the same
sentence that there are four. `judge_prompt_for()` rewrites the phrase only when
the count is not four, so every four-arm run — which is all of them — reproduces
byte for byte.

### 21.2 Grounding, reported beside the score

Covered in Section 19.1: `base` scores highest while grounding its answers half
as often as `base_rag`. The report now prints grounding as a column and emits a
warning when the top-scoring arm is not the least-fabricating one, so the
misreading cannot survive a future run unnoticed. Not folded into correctness —
that would rebuild the confounded single number V3 exists to take apart.

### 21.3 The disputed calls were a missing rubric entry, not a wording defect

Section 20.3 left 31 of 88 blunder calls disputed between judges, and Section
16.12's protocol says a disputed position is a position to rewrite. The
disputes localize hard: **11 of 31 sit on three positions**, and four positions
drew none at all.

Reading the worst two settles what is actually wrong, and it is not the Section
16.12 defect. On `pos-combat-math-0005` the model answered:

```
CAST Shock TARGET Grizzly Bears
PASS
```

That is *half* the correct line — it kills the blocker and never attacks,
leaving exactly lethal on an empty board. It commits **none** of the three
enumerated errors, because none of them describes stopping early. Judge 2
correctly fired no errors, which scores a wasted win as un-blundered. Judge 1
fired all three, papering over the gap. The two judges were not disagreeing
about the answer; they were disagreeing about what to do with a rubric that had
no entry for what the answer did.

Measured, and the reason this generalizes:

- **69% of all answers (61/88) contain at most one real action.** Emitting a
  single play and passing is the dominant model behaviour on a board.
- **5 of 22 positions need two steps.** Four had no partial-execution entry.
- The fifth, `pos-removal-timing-0002`, already had one ("casts it in the window
  but declines the block") — and it is one of only **four positions in the set
  with zero disputes**.

| Multi-step position | disputes | had a partial-execution error? |
| --- | --- | --- |
| `pos-combat-math-0002` | 4/4 | no |
| `pos-combat-math-0005` | 4/4 | no |
| `pos-combat-math-0004` | 2/4 | no |
| `pos-combat-math-0001` | 1/4 | no |
| `pos-removal-timing-0002` | **0/4** | **yes** |

So the fix is one line per affected position, leading with the omission so the
16.12 lint stays quiet. This is the inverse of the 16.12 defect: not an error
whose opening clause is also true of the correct line, but a real failure mode
with no entry at all.

Because this edits rubrics that back published numbers, the n=22 results are
re-scored from the **stored answers** rather than regenerated — the judge is
deterministic and the answers do not change, so the only thing that moves is
what the rubric asked.

### 21.4 The SFT set was written by the model being trained

Section 9.4's standing instruction — fix the data, not the hyperparameters — is
now well supported: run 3's full epoch moved `finetuned_rag` by +0.04 while
control arms with byte-identical answers moved up to 0.23 on judge variance
alone. Hyperparameters are exhausted.

The constraint is that the ~2,646 SFT examples were **synthesized by
Qwen2.5-7B-Instruct — the same model the adapter is trained on top of.** A
student cannot exceed its teacher, and the distillation visibly lost information:
fine-tuned answers run 521 characters against `base_rag`'s 1104, and score
lower. What the adapter reliably learned was brevity.

`data/gold/gold_candidates.jsonl` holds 1,202 RulesGuru records with
human-verified answers. Removing the 72 already promoted into the gold eval set
leaves **1,130 uncontaminated training pairs**, and they have the property the
synthetic set structurally cannot:

- **100% cite a rule id inline**, in the exact form `score_citations` validates
- median answer 242 characters, but claim-dense rather than vague — the shape is
  `Yes. <claim>. (rule) <claim>. (rule)`, which is what a rubric judge scores
- 1,128 of 1,130 carry a resolved card list

Grounding is the largest and cleanest effect measured anywhere in this project
(86/99 vs 45/99, Section 19.1), and it is exactly what human-authored answers
carry and model-authored ones do not.

Two risks to hold, rather than discover after a training run:

- **1,130 is fewer than 2,646.** Section 8 overfit badly on 569. The mitigation
  is that these are not interchangeable examples — but the epoch count needs
  re-deriving from `iters × batch / len(train)` rather than inherited, which is
  a documented trap in this repo.
- **Contamination is the failure that would invalidate everything downstream.**
  The 72 overlapping ids must be excluded by id, and the exclusion asserted in
  the builder rather than done once by hand.

### 21.5 Adding a fifth arm moved the other four — the judge is not per-candidate

The deliberation run was designed to be additive: `base_open_think` was **added**
to the arm list, nothing else touched, so the four existing arms should have
reproduced. They did not.

| Arm | 4-arm run | 5-arm run |
| --- | --- | --- |
| `base_open` blunder | 50% | **73%** |
| `base_open` correctness | 2.62 | **1.94** |
| `ft_cards_open` actions/answer | 4.7 | **10.7** |

Checked before theorizing: **`base_open`'s answers are byte-identical across the
two runs, 22 of 22.** Generation is deterministic, and the judge is
deterministic. Nothing about that arm changed.

What changed is the judge *call*. `judge_batch_rubric` scores every candidate
for a position in ONE batched request — that is what makes the anonymized A/B/C/D
design work. So a fifth candidate is not a fifth independent grading. It is a
fifth answer in the same prompt, changing the context in which the other four
are read, plus the label-list phrase the prompt states.

**Consequence, and it is a constraint on every future experiment: runs with
different arm counts are not comparable.** An arm cannot be added to a batched
judge and compared against a run without it. Only *within-run* comparisons hold.

This does not touch any published number — every prior run used exactly four
arms, including the A3 n=99 result — but it invalidates the obvious way anyone
would try to extend one, which is why it is recorded here rather than in a
commit message.

The valid comparison for the deliberation question is therefore inside the
5-arm run, where both arms sat in the same judge call:

| | `base_open` | `base_open_think` |
| --- | --- | --- |
| blunder rate | 73% | **86%** |
| correctness | 1.94 | **2.19** |
| all legal | 68% | **23%** |
| chars / answer | 39 | **1112** |

**The pre-registered prediction was that blunder rate would fall and Gate 2's
spread would narrow. Half of it held.** The spread narrowed (32% → 23%), which
is consistent with some of the arm separation having been verbosity. Blunder
rate did not fall — it rose 13 points. Correctness rose slightly.

The instruction plainly worked as an instruction: mean answer length went from
39 characters to 1112, and the reasoning follows the requested shape ("If you do
nothing, the opponent will likely attack with Grizzly Bears..."). The model is
now deliberating. It is not thereby playing better.

Two distinct causes sit behind the legality collapse, and only one is a harness
artifact:

- **3 of 22 answers never emitted the `ACTIONS:` marker**, so their reasoning
  was parsed as plays — exactly the failure the marker was added to prevent,
  which the marker only prevents when the model complies.
- **The rest are genuinely illegal plays.** On `pos-trigger-ordering-0001`,
  where the only legal actions are the two trigger orderings, the deliberating
  model reasoned its way to `CAST Doom Blade TARGET Grizzly Bears`. On
  `pos-land-sequencing-0001` it concluded with `CAST Vampire Nighthawk` — a
  three-mana creature on turn one. Deliberation made it *more* ambitious and no
  more legal.

So the honest reading: on a 2024-era 7B, being told to think first produces
fluent reasoning that does not reach a better play, and licenses plays the
terse arm never attempted. That is a negative result for the prompt-level fix,
and it is the strongest available argument for testing a reasoning-capable base
model instead — the thing a prompt cannot install.

### 21.6 Eighteen percent of the SFT set taught the model to refuse

Section 21.4 argued from provenance that a set synthesized by the model being
trained cannot lift it. Reading the actual training targets found something more
specific and worse. The first line of `data/datasets/train.jsonl`:

```
assistant: The rules provided do not contain specific information about the card
Mishra, artificer prodigy. Therefore, I cannot answer the question based solely
on the given rules text.
```

That is a refusal, used as a training target. Measured across both splits:

| | synthetic (`data/datasets/`) | verified (`gold_candidates`) |
| --- | --- | --- |
| examples | 2,971 | 1,130 |
| **refusal-shaped targets** | **523 (18%)** | **1 (0.1%)** |
| targets citing a rule id | 87% | **100%** |
| mean target length | 618 chars | 277 chars |

**Roughly one training example in five explicitly taught the adapter to
decline.** That is a sufficient explanation on its own for the two things every
run since Section 9 has reported about the fine-tuned arm: it scores lowest on
correctness — a refusal scores 1 against any rubric — and it produces the
shortest answers of the four.

This reframes the whole fine-tuning result. Sections 8, 9, 18 and 19 all
concluded "fine-tuning did not beat retrieval", and that conclusion stands as
measured. But the cause was never established, and "the training data taught it
to refuse 18% of the time" is a much narrower and more fixable diagnosis than
"fine-tuning does not help here".

`scripts/build_sft_verified.py` builds the replacement:

```
candidates              : 1202
  excluded (in gold set): 72
  usable                : 1130
  refusal-shaped answers: 1 (0.1%)
  citing a rule inline  : 1130 (100%)
split: train=1019 valid=111
```

Three things it does deliberately:

- **Contamination is an assertion, not a comment.** 72 candidates are already in
  the n=99 eval set; training on them would make every downstream number
  meaningless *and would look like an improvement*. The builder aborts if an
  excluded id survives, and writes `excluded_ids.jsonl` so the exclusion is
  auditable rather than trusted.
- **No retrieved context is attached.** These questions carry their own card
  references and the verified answer cites its own rules; injecting retrieval
  would train the model to expect a context block the answer does not depend on.
- **Epoch count is printed, not inherited.** This repo has already shipped a
  config comment claiming ~1 epoch for a run that did 0.45. At 1,019 training
  examples the v2 recipe's 600 iterations at batch 4 is **2.36 epochs**, not the
  ~1 it was on 2,644 — and Section 8 overfit badly at 569 examples. The
  iteration count has to come down, and the builder puts that arithmetic in
  front of whoever runs it.

### 21.8 Gate 2's pass was carried by the fine-tuned arm being bad

Dropping `ft_cards_open` for the base-model comparison — it has to go, since a
LoRA trained on Qwen2.5-7B cannot be applied to a different base — produced a
result worth recording on its own.

`base_open`'s answers are **byte-identical across all three runs** (4-arm, 5-arm
and 3-arm, 22 of 22 each, verified), despite `max_tokens` moving 400 → 2000.
Generation is fully deterministic and the token budget changes nothing for an
arm that stops after 39 characters. So every difference below is the *judge*.

| Run | arms | rubrics | `base_open` blunder | Gate 2 spread |
| --- | --- | --- | --- | --- |
| `positions_n22` | 4 | old | 50% | 32% **PASS** |
| `positions_n22_think` | 5 | old | 73% | 23% **PASS** |
| `pos_qwen25_3arm` | 3 | new | 73% | **9% FAIL** |

**Gate 2 now fails.** In the 4-arm run the blunder spread was 50–82%; with the
fine-tuned arm removed the three base arms sit at 73%, 82%, 73% — a 9-point
spread, under the 15-point bar.

The honest reading is that Section 20's "Gate 2 PASSES for the first time" was
**substantially carried by `ft_cards_open` being the worst arm on every measure**,
not by the positions separating comparable systems. Hand-authored positions did
fix the seed set's zero-spread pathology — that part stands, and the seed
fixtures gave 0% spread across four arms including the fine-tuned one. But the
claim that these 22 positions discriminate among *similar* systems is not
supported. Discriminating a broken arm from three working ones is a much weaker
property than the gate was meant to test.

Two confounds sit in that table and only one is measured:

- **Arm count.** The 4→5 comparison is clean — same rubrics, byte-identical
  answers — and moved `base_open` 23 points (Section 21.5).
- **Rubrics.** The 3-arm run is the first to use the partial-execution errors
  added in Section 21.3, which can only *raise* blunder rates, and higher rates
  compress the spread against the ceiling.

So the 3-arm number cannot be attributed to arm count alone. Disentangling it
needs one more rescore: the stored 4-arm answers against the new rubrics, which
holds arm count fixed and moves only the rubric. Queued.

This is the third time in this section that a number moved for a reason that had
nothing to do with the models. Generation has been deterministic throughout;
every one of them was the instrument.

### 21.9 Only 10 of 22 positions separate the arms, and correcting a rubric cost two more

Before authoring another 18 positions, the question worth answering is which of
the existing 22 do anything. Measured on the 3-arm run, per position, asking
whether the three arms got different blunder calls:

| Category | discriminating | flat |
| --- | --- | --- |
| race vs stabilize | **2/2** | 0 |
| blocking | **3/4** | 1 |
| removal timing | **2/3** | 1 |
| trigger ordering | 1/2 | 1 |
| combat math | 2/6 | 4 |
| mulligan | 0/2 | 2 |
| land sequencing | 0/3 | 3 |

**10 of 22 discriminate. 12 do not. Zero are too easy** — every non-discriminating
position is one where *all* arms blundered.

That is directly actionable for authoring: `race vs stabilize`, `blocking` and
`removal timing` earn their slots; `mulligan` and `land sequencing` currently
contribute nothing at all, and `combat math` is the weakest of the productive
categories despite having the most positions.

**The rubric correction cost two positions their discrimination.** The four
combat-math boards that gained a partial-execution error in Section 21.3, read
across the same three arms:

| Position | old rubric | new rubric |
| --- | --- | --- |
| `pos-combat-math-0001` | **discriminating** | flat |
| `pos-combat-math-0004` | **discriminating** | flat |
| `pos-combat-math-0002` | flat | flat |
| `pos-combat-math-0005` | flat | flat |

Across all 22 the count moved 11 → 10. The added error is *correct* — abandoning
a won position is a fatal blunder, and that was the right call. But because 69%
of answers stop after one action, nearly every arm commits it, and a position
every arm fails stops separating anything.

**Rubric accuracy and discriminating power are not the same property, and they
can trade off.** That is worth stating plainly because the instinct on seeing a
flat position is to weaken the rubric, and that instinct is wrong: the rubric is
right and the metric is lossy.

### 21.10 The binary blunder call throws away half of what is there

If every arm blundered but one committed one error and another committed three,
blunder rate records them as identical. Measured on the 12 flat positions:

**6 of 12 have different error COUNTS across arms.** Half the positions Gate 3
reads as uninformative are not.

```
pos-mulligan-0001        binary [T, T, T] -> counts [1, 3, 3]
pos-combat-math-0004     binary [T, T, T] -> counts [4, 2, 5]
pos-land-sequencing-0002 binary [T, T, T] -> counts [3, 1, 2]
pos-blocking-0003        binary [T, T, T] -> counts [4, 2, 4]
```

Errors per answer is now reported beside blunder rate. It is **not** a
replacement: Gate 3 is defined on the binary and stays that way, and this does
**not** rescue Gate 2 at this sample size — the per-arm means came out 2.09 /
2.09 / 2.23, because the per-position differences point in different directions
and cancel in the mean.

So the honest statement is narrow: the information exists in half the discarded
positions, it is now visible, and a *paired* per-position analysis could use it
where a mean cannot. Reporting it is what makes that checkable rather than
assumed.

### 21.11 The rules eval discriminates; the gameplay eval is the one that does not

Section 21.9 found only 10 of 24 positions separating the arms. The same audit
had never been run on the 99 rules rubrics. It has now, over the stored `gold_n99`
answers, measuring the spread in correctness across the four arms per question:

| | rules gold (n=99) | positions (n=24) |
| --- | --- | --- |
| contribute nothing (zero spread) | **9 (9%)** | 14 (58%) |
| separate arms by ≥1.0 point | **73 (74%)** | — |
| mean spread across arms | **1.83 points** | — |

**The rules set is healthy and the gameplay set is not.** Three quarters of gold
questions separate the arms by a full point or more, and the 9 flat ones are
spread evenly across categories — 0 to 2 per category, no dead category anywhere.
Compare `land sequencing` and `mulligan` in the position set, which are
arm-invariant at 100%.

This matters for where the structural work goes. The two tracks have different
problems and the same fix would not serve both:

- **Rules eval:** the *data* is doing its job. The constraint is judge agreement
  (r = +0.49 between two judges on identical answers). Work here is judge work —
  the positive controls, V4, possibly a stronger judge model.
- **Gameplay eval:** the *positions* are the constraint. 58% contribute nothing,
  Gate 2's earlier pass turned out to be carried by the fine-tuned arm being
  broken (Section 21.8), and no amount of judge improvement fixes a position
  every arm fails identically.

> **This second bullet is wrong, and Section 21.17 corrects it.** The 58% was
> measured on the binary blunder call, which is Gate 2's metric — not on the
> positions. Scored on correctness the same positions separate the arms 82–86%
> of the time. The constraint is the metric, not the authoring.

It also puts the A3 result on firmer ground than the last few sections have
implied. The n=99 verdict — fine-tuning trails retrieval by 0.50 and 0.80 under
two judges — rests on a set where 74% of questions actively separate the arms
and the mean spread is 1.83 points, which is comfortably larger than the effect
being claimed. The instrument problems found today are real, but they are
concentrated in the gameplay track, and none of them touch that comparison.

### 21.12 The scoring arithmetic had no tests, and a shape error could kill a run

`test_actions.py` covers the action parser at 84 assertions. `eval.py` — which
holds the code that turns a judge's JSON into every number this project
publishes — had none. That is the wrong way round: a parser bug shows up as a
bad legality rate, which looks wrong, while a bug in `rubric_correctness` shows
up as a plausible score, which does not.

`scripts/test_eval.py` covers the four functions in the Section 21 audit's
priority order — `rubric_correctness`, `verify_quoted_claims`,
`judge_prompt_for`, `score_citations` — plus `stratified_sample` (which decides
which questions an eval asks), `pearson_r` (the statistic Section 14.6's
+0.30 → +0.62 result is stated in), and `_author_of`. 96 assertions, no model,
no GPU.

`judge_batch_rubric` is exercised end to end with a stub `lm_generate`, which is
what finally covers the **label mapping**. Candidates are shuffled behind
A/B/C/D and mapped back afterward; an off-by-one there would attribute every
arm's score to a different arm, and the report would look entirely normal while
being exactly wrong. The test replays the shuffle for six seeds, awards points
to whichever label carries `base_rag`, and asserts `base_rag` is the arm that
comes back scored — across those seeds the target lands on all four labels, so
the check is not passing by accident. Three degradation cases sit beside it:
unparseable judge output returns `{}` rather than a partial score, and an arm
the judge simply omitted is **absent** rather than defaulted to 1.0, which would
have read as "this arm answered badly".

**Writing them found one defect and corrected one belief.**

The defect: valid JSON of the wrong *shape* crashed the run. A judge that emits
`"points_hit": 3` where `[3]` was asked for produces JSON that parses cleanly,
so neither `json.loads` guard catches it — and then `{i for i in 3}` raises
`TypeError` inside `rubric_correctness`. There is no handler anywhere between
that line and `main()`, in any of the four callers (`eval.py` main, `rescore`,
`calibrate_judge.py`, `eval_positions.py`). A multi-hour run dies at whatever
question the judge happens to fumble.

`_as_claim_list` wraps a bare scalar rather than discarding it, because
discarding would score that answer 1.0 and write the number to the results file
as though it were measured — worse than the crash it replaces, since the crash
at least announces itself. `_claim_index` additionally rejects `bool`, which is
a subclass of `int`: `"points_hit": true` would otherwise have been read as
"hit point 1", a fabricated claim awarded silently from a response that named
no point at all.

**No published number moves.** Verified three ways rather than argued:

| Check | Result |
| --- | --- |
| `rubric_correctness`, exhaustive over well-formed inputs | 1,190 inputs, **0 differ** |
| `verify_quoted_claims`, exhaustive over claim permutations | 821 inputs, **0 differ** |
| every stored rubric score in `eval/runs/*.jsonl`, recomputed | 1,782 scores, **0 differ** |

The corrected belief: the first version of the `judge_prompt_for` test asserted
that a five-arm prompt no longer contains `"labeled A, B, C, D"`. It does — the
replacement produces `"labeled A, B, C, D and E"`, which contains the old phrase
as a prefix. The code was right and the test was wrong. The assertion that
survives is the one with teeth: a *three*-arm prompt must not mention a fourth
label, since a judge told about a candidate it was never shown has an entry to
fill and nothing to fill it from — and `judge_batch_rubric` reads scores back by
label, so the invented entry would be dropped silently rather than raising. Two
supporting assertions check that the phrase appears in each of V2/V3/V4 exactly
once, so the `str.replace` can never become a silent no-op.

### 21.13 The contamination filter missed 18 of the 90 records it was written to catch

`build_sft_verified.py` excludes gold eval records from the training set and
*asserts* the exclusion, because contamination is the one failure that
invalidates everything downstream while looking like an improvement. The
assertion passed. `scripts/audit_sft.py` — written to check exactly this, item 4
of STRUCTURAL_AUDIT.md, run before the retrain rather than after — found it was
passing on an incomplete comparison.

**16 of the 99 gold eval questions were in the run-4 training set.**

The filter compared `candidate["id"]` against `gold["id"]`. Sixteen gold records
were promoted from RulesGuru under a `qa-*` id while keeping their
`rulesguru_id`, so the candidate `rg-1156` and the gold record
`qa-amy-casts-assassin-s-trophy-targeting-ni` are the same RulesGuru question
and never compared equal:

| Filter | Excluded | What it catches |
| --- | --- | --- |
| `id` | 72 | the original comparison |
| `rulesguru_id` | **16** | promoted into gold under a renamed id |
| question text ≥0.75 overlap | **2** | duplicate RulesGuru entries, and one gold record predating the pull with no `rulesguru_id` |
| | **90** | |

The last two are worth separating from the sixteen. `rg-308` and `rg-777` are
*different* RulesGuru entries asking one question — the upstream database
contains duplicates, so no id comparison of any kind can catch them. That is why
the builder now also compares question text, at a deliberately loose threshold:
dropping a few extra records costs nothing against 1,112, while keeping one
contaminated record invalidates every number the resulting adapter produces. The
asymmetry points one way.

The text comparison alone would have caught only 5 of the 18, because the gold
set is normalized to `Player A`/`Player B` while the candidates carry raw
RulesGuru names, which drags overlap below threshold. Neither filter is
sufficient; the finding is that both were needed and one was assumed to be
enough.

**No published number is affected.** The verified set has never been trained on
— run 4 was pending, which is the whole reason the audit runs before a retrain.
Rebuilt at 1,001 train / 111 valid, and `iters` moved 1019 → 1001 to hold 2.00
epochs exactly, since a stale iteration count after a dataset change is how
v2's 0.45-epoch run happened.

#### The synthetic set, audited retrospectively

`data/datasets` is what the *published* v2 adapter was trained on, so it was
audited too. Three findings, in descending order of how much they matter:

**44% of the training lines are duplicates.** 1,478 unique lines behind 2,644 —
1,138 distinct targets repeated, up to six times each, with the *question*
identical too in 100% of cases, so this is not paraphrase augmentation. Every
epoch figure quoted for runs 1–3 overstates how much distinct text the adapter
saw. Section 18.2's "1.00 epoch with nothing seen twice" was wrong on the second
clause, which makes that run weaker evidence against overfitting than it looked.

**Three eval questions are in it verbatim** — `gloss-companion`,
`gloss-face-down`, `gloss-mutating-creature-spell`. The glossary feeds the
training set and the gold set through the same "What does X mean?" template.

This does not move the A3 verdict, and the direction is worth stating because
the naive assumption is wrong. On those three questions the fine-tuned arms
scored *worse* than their own average, not better:

| Arm | All 99 | The 96 clean | The 3 contaminated |
| --- | --- | --- | --- |
| `base_rag` | 2.38 | 2.34 | 3.67 |
| `base` | 2.46 | 2.50 | 1.33 |
| `finetuned_rag` | 1.88 | 1.89 | 1.67 |
| `finetuned` | 1.69 | 1.71 | **1.00** |

The arm that gained was `base_rag`, which was never trained at all — glossary
definitions simply retrieve well. Dropping all three moves every headline number
by at most 0.04. The adapter was trained on those exact question/answer pairs
and still scored 1.00 on them, which is a small piece of evidence about how
little the fine-tune retained, consistent with everything else in Sections 19–21.

**Validation loss is sound.** Only 4 of 327 validation lines (1%) appear in
train, so the checkpoint selection in Section 8 was not measuring memorization.
The verified set has zero overlap.

### 21.14 The V4 judge cost 60–85 points of coverage and bought no agreement

The V4 quote-verification experiment (Section 21.7) completed against the n=99
gold set under both judges. **It answered nothing, and the reason is worth
recording rather than retrying blindly.**

| Run | Judge | Prompt | Questions graded |
| --- | --- | --- | --- |
| `gold_n99` | Qwen2.5-7B | V3 | **99/99 (100%)** |
| `gold_n99_judge2` | Llama-3.1-8B | V3 | **92/99 (93%)** |
| `gold_n99_v4judge` | Qwen2.5-7B | V4 | 14/99 (14%) |
| `gold_n99_v4judge2` | Llama-3.1-8B | V4 | 32/99 (32%) |

The two V4 runs overlap on **6 questions**, so the agreement measurement the
experiment existed to produce is r = +0.46 on 24 arm-pairs, against V3's +0.49
on the full set. That is not a smaller number, it is no number: six questions
cannot separate +0.46 from +0.49, and the six are not a random six.

**The mechanism is the prompt, not the judge.** V3 asks for
`"points_hit": [1, 3]`; V4 asks for a verbatim quote behind every claim, so the
required output grows as *arms × rubric items × quote length*. At the same
`--judge-max-tokens` the judge runs out mid-JSON, and because
`judge_batch_rubric` grades every arm for a question in one call, all four arms
fail together — the survivors are whole questions rather than a scattering of
answers.

The cause is the V4 prompt itself. V3 asks for `"points_hit": [1, 3]`; V4 asks
for a verbatim quote behind every claim, so the required output grows as
*arms × rubric items × quote length*. At the same `--judge-max-tokens` the judge
runs out mid-JSON. Parse rate by rubric size:

| Rubric items | Parsed / total | Rate |
| --- | --- | --- |
| 4 | 7 / 19 | 37% |
| 5 | 1 / 14 | 7% |
| 6 | 2 / 15 | 13% |
| 7 | 4 / 50 | 8% |
| 8 | 0 / 1 | 0% |

Mean answer length is identical between the parsed and failed groups (821 vs
822 characters), so this is not about the answers. It is the rubric, which is
exactly what V4 made expensive.

**So the survivors are a subset selected by rubric size** — the questions with
the least to check. The four means each report prints over them (2.04/1.68/1.25/
1.04 under Qwen, 3.11/3.02/2.43/2.28 under Llama) describe the easiest gradable
questions in the set and must not be compared to anything. This is the same defect as the n=9 Qwen3 position run, and
it recurred within a day, in a different script, for a different underlying
reason. Both times the report presented confident-looking averages with the
sample size as a parenthetical.

`rescore()` now states the unjudged count as a blockquote above the table, and
adds an explicit "read nothing into the means below" warning past 20% — because
"(n=14)" beside a mean reads as a footnote, and it is the headline.

#### The header described a judge that had not run

The same report was titled **"the V3 rubric judge"**, and asserted:

> The judge PROMPT is identical to the first pass; only the judge MODEL differs.
> That is what Section 9.9 requires — vary the judge and nothing else.

Both false. The run was V4, so the prompt was the thing that changed.

Section 19.2 already fixed this exact defect one level up: the header used to say
"v2 judge" unconditionally, and was changed to *derive* the judge from the stored
`scored_by` field. But `scored_by` is `"rubric"` for V3 and V4 alike, so the
derivation could distinguish rubric from prose and not V3 from V4 — and the
reassuring sentence about varying one thing at a time printed underneath a run
that varied two. The header now reads `args.judge_prompt`, and the Section 9.9
sentence is replaced by a warning when the prompt changed.

The lesson is narrower than "derive, don't assume", which was already the lesson
last time: a derivation is only as good as the field it derives from, and
`scored_by` was never granular enough to answer the question being asked of it.

#### What V4 is still worth

Nothing here says the idea is wrong. `quote_drops` did what it was built to do:
26 claims discarded across 16 arm-answers under Qwen, 30 across 20 under Llama —
claims the judge asserted and could not quote from the candidate it was grading.
That is fabrication measured rather than inferred, and it is the first direct
evidence in this project that a rubric judge invents claims at a measurable rate.

What is unusable is the *run*, because the budget was set from V3's needs. The
retry is `--judge-max-tokens` at roughly 3x, and the acceptance test is coverage
before agreement: if V4 does not grade ≥90% of the set, its r is a statement
about which questions it could afford to grade.

### 21.15 The published adapter was trained under the current prompt — verified, not assumed

An adapter is only valid for the prompt format it saw. A prompt edit in
`common.py` silently invalidates every adapter on disk, and the failure presents
as a capability result: the model looks worse. That is Section 8.7, the most
expensive failure this project has had, and nothing recorded which prompt an
adapter had been trained under — so "v2-best is still valid" was an assumption
every eval since has rested on.

It is answerable from the training data, which contains the exact prompts the
adapter saw:

| Dataset | Lines | Distinct system prompts | Match |
| --- | --- | --- | --- |
| `data/datasets` | 2,644 | 2 | `SYSTEM_PROMPT` ×1,492, `RAG_SYSTEM_PROMPT` ×1,152 — both current, byte-identical |
| `data/datasets/verified` | 1,001 | 1 | `SYSTEM_PROMPT` ×1,001 — current, and `build_rag_messages` reproduces the stored prompt exactly |

**The assumption held.** Runs 1–3 are valid for the prompts `eval.py` uses today.

`scripts/stamp_adapter.py` makes it stay checkable rather than re-derivable.
`--dataset` is how a stamp is *earned*: it verifies against the training file and
refuses to stamp a dataset built under a prompt the code no longer defines,
since that is precisely the claim the stamp would otherwise assert falsely.
`eval.py` checks the stamp before generating a single answer — a mismatch is
fatal, a missing stamp is a warning, because older adapters predate the file.

#### Hashing the system prompts would not have caught Section 8.7

The fingerprint covers the assembled *message shape*, not just the three system
strings, because the original failure was a shape change with all three strings
untouched: training saw a bare question, inference saw
`Rules text: …\n\nQuestion: …`. Verified by simulating exactly that — replacing
`\n\nQuestion: ` with `\n\nQ: ` in the context branch and leaving every prompt
string alone:

    stamped : 30badae98696
    current : 428e19b67719
    changed : message_shapes

A fingerprint over the system prompts alone would have reported a match, which
is the trap: a check that passes through the one failure it was written for is
worse than no check, because it is then cited as evidence.

### 21.16 The card corpus is pinned; one of the two is read by nothing

`rules.jsonl` and `glossary.jsonl` have been content-pinned since Section 17,
because a parse that keeps the record count and changes the text is invisible to
`guard_shrink` and to git. The card corpora had a count check and nothing else,
and they are the ones fed by a **live API**: Scryfall returns errata'd oracle
text months later under the same card name and the same record count.

`CARD_PIN` closes it, verified from `CardIndex.__init__` — the chokepoint eleven
call sites reach card text through, the same role `load_rule_ids` plays for
rules. Cost is ~12ms against the 25MB read the constructor already does.

Verified by simulating the failure rather than by reading the code: one word of
one card's oracle text changed, record count untouched.

    CardIndex(<non-canonical path>)   -> allowed, as designed
    CardIndex() at the canonical path -> refused

    expected sha256 407903e2d42442db...  (34933 card chunks)
    actual   sha256 e49d178f50a323c2...  (34933 card chunks) — SAME record count,
                                          so only the TEXT changed

That last clause is in the error message on purpose. A count mismatch reads as
"wrong file"; a count match with a hash mismatch is the case the pin exists for,
and it should not require the reader to compare two numbers to notice.

`verify_cr_pin` and `verify_card_pin` share one body (`_verify_pin`). Two copies
would be the duplicated-helper trap on the guard against silent corpus drift,
and the copy that got the *only check the canonical path* rule wrong would
either nag on every deliberate `--cards somewhere_else` or check nothing at all.

#### `ruling_chunks.jsonl` is ingested and read by nothing

Found while looking for its chokepoint: there isn't one. 19,726 ruling chunks,
22.7MB, built by `ingest_rulings.py` and referenced by no other script — the
corpus was ingested and never wired into retrieval. It backs no published
number, so nothing is wrong with any result; it is simply not doing anything.

Pinned anyway, since the moment it gets wired in is the moment nobody will think
to pin it. Recorded here rather than quietly fixed, because "is this corpus
actually used" is a question worth asking of the others too.

### 21.17 Correction: the positions discriminate. Gate 2's metric does not.

Section 21.9 concluded "only 10 of 24 positions separate the arms", and 21.11
built on it to say the gameplay track is constrained by its *positions* while
the rules track is constrained by its *judge*. **The first is wrong, and it is
wrong in a way that pointed the next block of work in the wrong direction** —
toward authoring more positions, which would not have helped.

21.9 measured discrimination on the **binary blunder call** — `errors_made`
non-empty — because that is Gate 2's metric. It never measured the positions on
the 1–5 correctness scale the same positions are also scored on. Doing that, over
two independent stored runs at different arm counts:

| Run | Arms | Separate on correctness | Separate on blunder |
| --- | --- | --- | --- |
| `pos_qwen25_3arm` | 3 | **18/22 (82%)** | 10/22 (45%) |
| `positions_n22` | 4 | **19/22 (86%)** | 12/22 (55%) |

82–86% against the rules gold set's 74% (Section 21.11). **The positions are not
the weak instrument — they are marginally the stronger one.** What is weak is the
binary call layered on top of them, which discards roughly half the separation
the judge already produced.

Section 21.10 predicted exactly this in words — "the binary blunder call throws
away half of what is there" — and this measures it: 82% down to 45%, on the same
answers, from the same judge, on the same run.

The four positions that stay flat on correctness are also not "contributing
nothing" in the way 21.9 implied. Under Qwen, three are flat because **every arm
blunders** (`pos-removal-timing-0001`, `pos-combat-math-0002`,
`pos-combat-math-0005`) and one because no arm does. A position every arm fails
is a hard position, not a broken one; it stops discriminating only once the
metric collapses to a yes/no.

**What this changes:**

- Gate 2 ("the eval discriminates") should be measured on correctness spread,
  not on blunder-rate spread. It is currently failing a set that separates the
  arms 82% of the time, which makes the gate a statement about the metric.
- Blunder rate stays as a *reported* number — it is the interpretable one, and
  the user's framing (failing to take the winning line is fatal) is a real
  quality, not a proxy. It is just not sensitive enough to be a gate.
- The next block of gameplay work is not authoring. It is Gate 2's definition
  and the graded blunder severity sketched in 21.10.

The 21.11 claim that survives intact is the other half: the rules eval's
constraint is judge agreement (r = +0.49). Both tracks are judge/metric bound.
Neither is data bound.

### 21.18 Gate 2 rebuilt, and the old one reversed between judges

Section 21.17 established that Gate 2 was measuring the wrong thing. Rebuilding
it turned up a second, sharper reason the old definition had to go.

**The old gate took two independent losses.** It computed each arm's aggregate
blunder rate, then took the spread across arms. So it first collapsed a 1–5
score to a yes/no — which separates the arms on about half as many positions —
and then averaged per arm before comparing, which cancels what survives, since
per-position differences point in different directions. The comment above the
summary table already described the cancellation; it was not connected to the
gate sitting twenty lines below it.

**The new gate is per position:** what fraction separate the arms by ≥0.5 on the
correctness scale, needing ≥50%. The threshold is set on sample-size grounds
rather than to clear the current number — at n≈22, a set where only half the
positions separate anything has an effective n of 11, well under the ~40 this
project's own note requires for a proportion.

Replayed over every stored run:

| Run | Arms | New Gate 2 | Old (blunder spread) |
| --- | --- | --- | --- |
| `pos_qwen25_3arm` | 3 | PASS 18/22 (82%) | 9% → FAIL |
| `pos_qwen25_3arm_judge2` | 3 | PASS 16/19 (84%) | 26% → PASS |
| `positions_n22` | 4 | PASS 19/22 (86%) | 32% → PASS |
| `positions_n22_think` | 5 | PASS 20/22 (91%) | 23% → PASS |
| `pos_qwen3_14b_3arm` | 3 | **FAIL 2/9 (22%)** | 11% → FAIL |

It is not a gate rewritten to pass. The invalid Qwen3 run — the one whose judge
collapsed on 13 of 22 positions — still fails, and the report now says how many
positions were excluded rather than scoring an unjudged arm as a tie.

#### The finding: the old metric reversed between judges

Rows one and two are the **same answers judged twice**, varying nothing but the
judge model — the comparison Section 9.9 requires:

| Metric | Qwen judge | Llama judge | Old gate verdict |
| --- | --- | --- | --- |
| blunder-rate spread | 9% | 26% | **FAIL → PASS** |
| positions separating | 82% | 84% | PASS → PASS |

**The old Gate 2 reverses on identical answers.** Section 16.12 reported that
two judges reversed two of three gameplay gates and treated it as a fact about
judge disagreement. Part of it was the metric: a spread of two aggregate
proportions is a difference of differences, so both judges' noise lands in it
twice, and at these sample sizes that swamps the signal. The per-position
fraction moves 2 points across the same judge swap.

This does not make the judges agree — kappa on the blunder call is still +0.24,
and that is still the binding constraint on the gameplay track. It means Gate 2
is no longer the place that disagreement gets amplified into a reversed verdict.

Blunder rate stays in the report as a diagnostic. It is the interpretable number
and the one that matches how the position set was authored — failing to take the
winning line is fatal, not a rounding error. It is simply too coarse to gate on.

### 21.19 Gate 1 is judge-independent; Gate 3 agrees only because the model is bad

With Gate 2 rebuilt (21.18), the obvious question is whether the other two gates
have the same defect. Measured across three run pairs — **identical stored
answers, judge varied and nothing else**:

| Pair | Gate 1 (parse / legality) | Gate 2 old | Gate 2 new | Gate 3 |
| --- | --- | --- | --- | --- |
| `pos_qwen25_3arm` | 100/100, 64/64 | 9% → 26% **reverses** | 82% → 84% | 71% → 53% |
| `positions_n22` | 77/77, 36/36 | 32% → 36% | 86% → 82% | 47% → 47% |
| `positions_n22_think` | 77/77, 23/23 | 23% → 50% | 91% → 85% | 65% → 27% |

**Gate 1 is bit-identical under both judges**, on every pair. It should be — it
is computed by the action parser, which never sees the judge — and confirming it
is the positive control for this whole comparison. A difference there would have
meant judge output was leaking into a parser-derived metric.

**Gate 3 never reverses, and that is not reassuring.** Look at the swing: 71→53
and 65→27, against a **25%** threshold. On the think run, one judge puts the best
arm 40 points from passing and the other 2 points. The gate agrees only because
both land far from the line — the agreement is a fact about how far the model is
from passing, not about the metric. It will start reversing at exactly the moment
an arm gets good enough for the verdict to matter.

So the judge-agreement report now prints **both judges' gate verdicts side by
side**, flags a reversal, and flags this second case: agreeing while more than
half the threshold apart. Section 16.12 recorded "two of three gates reversed"
as a one-off observation; it is a standing readout now, visible in the run that
caused it rather than found by hand later.

#### Extracting Gate 3 created the duplicated-helper trap, and the test caught it

`gate3_blunder` was written for the agreement report while the original stayed
inline in `_write_report` — two implementations of one gate, exactly the trap
CLAUDE.md lists. They had already diverged on tie-breaking (`<` vs `<=`), which
silently renames the reported arm when two arms tie. The new test failed on the
first run, which is the whole argument for writing it.

Now one implementation, called from both. Verified output-preserving: all five
stored runs regenerate byte-identical reports.

One real edge case surfaced on the way. The inline version seeded the best rate
at 1.0 and used `<`, so a run where *every* arm blunders on *every* easy position
printed "Best arm `None` at 100%" — the FAIL verdict was right and the sentence
was not. `best` now starts unset. No stored run reaches it.

### 21.20 Judge disagreement is diffuse, not localized — so rubrics are not the lever

Judge agreement is the binding constraint on both tracks (kappa +0.24 on the
blunder call, r +0.49 on correctness). Section 16.12 suggested a cheap fix:
disagreement *localizes*, so find the few bad rubric items and rewrite them —
six of eight disputes sat on two of eight items.

That was n=8. Measured over both full sets, at the level of the individual
rubric item (does each judge think *this* key point was hit, *this* common error
committed), across identical stored answers:

| | positions (n=22) | rules gold (n=99) |
| --- | --- | --- |
| rubric items in play | 139 | 594 |
| items the judges ever split on | 95 (68%) | 443 (75%) |
| total item-level disputes | 178 | 929 |
| **Gini across items** | **0.48** | **0.45** |
| worst 10% of items carry | 22% | 24% |
| records with ≥1 dispute | **19/22** | **91/99** |
| median disputes per record | 9 | 9 |

**Disagreement is spread across nearly every record.** If it localized, the worst
10% of items would carry most of the disputes; they carry 23%, against 10% for a
perfectly even spread. Gini 0.45–0.48 is mild concentration — the same shape you
get from noise plus a little heterogeneity, not from a handful of broken rubrics.

The two sets agree to within 3 points on every measure, which is itself
informative: the positions were authored months apart from the rules rubrics, by
the same person, and land in the same place. This is not a property of one
authoring session.

**So rubric rewriting is not available as a lever.** You cannot hand-fix 91 of
99 records, and there is no evidence the rewrite would help even then — the
disputes are not clustered on items with an identifiable defect, which is what
Section 21.3 found when the *machine-drafted* rubrics were the problem (a missing
entry for the likeliest wrong answer, concentrated in four positions).

`eval.py --compare` said "rewrite these rubrics before adding more" in its own
output, citing 16.12. That advice is now wrong and has been replaced: the table
is worth reading to understand how the judges differ, and is not a repair list.

**What this means for the migration question.** The levers on judge agreement
were: better rubrics, a better judge prompt, or a better judge model. Rubrics are
now ruled out by measurement. The judge prompt was tried — V4 (Section 21.14) —
and cost 60–85 points of coverage without buying agreement. That leaves the judge
model, which is exactly the case STRUCTURAL_AUDIT.md names as the only genuine
reason to migrate: *"the case for more memory is about grading models, not
running them."*

This does not settle it — a bigger judge might disagree just as diffusely, and
the positive controls are still the test that decides. But it removes the cheap
alternative, and that narrows the question considerably.

One caveat worth stating: both judges here are 7B–8B models at 4-bit. "These two
judges disagree diffusely" is what was measured. Whether a stronger judge
disagrees *with itself* less is the open question, and it is not answerable from
these runs.

### 21.21 Hardening the calibration before it runs, because it is the one that buys a computer

`calibrate_judge.py` produces the three numbers STRUCTURAL_AUDIT.md's migration
trip-wires are stated against. Reviewing it against what Section 21.14 cost:

**It averaged over whatever the judge graded, and never said how much that was.**
Exactly the defect that made the V4 run unusable — four confident means over 14
of 99 questions, sample size as a parenthetical — sitting in the script whose
output is the argument for buying hardware. The judge's failures are not random;
they track how much output the prompt asks for, so the survivors are a selected
subset.

Coverage is now the first line of the report. Below 90% the report says *do not
read the numbers below*, and the trip-wire section refuses to render PASS/FAIL at
all, printing the measured values marked `(unevaluated)` for diagnosis:

    **NOT EVALUATED — coverage 20%.** These three numbers decide whether the
    instrument is sound, and they cannot be read off a subset the judge
    selected by failing on the rest.

Verified by stubbing the judge to fail four questions in five. The withheld case
matters more than the reported one: with the stub grading perfectly on the
questions it *did* answer, all three trip-wires would have printed **PASS** on
4/20 questions. A PASS is the sentence that authorizes the migration, and it must
not be obtainable from a subset.

#### One planned measurement is dropped, and said so rather than faked

The audit named a fourth check: build the `wrong` control so it asserts a
specific enumerated `common_error`, then count how often the judge reports *that*
error number — a direct test of whether `errors_made` means anything, which is
what blunder rate is defined on.

Not implemented. Constructing that candidate means turning a rubric line written
as a *description* of a mistake ("Adds Centaur Courser to the block, spending a
3/3 to save 3 life") into a first-person answer asserting it. Every one of those
is prose the harness would generate, so a low detection rate would be
unattributable between "the judge cannot spot the error" and "the sentence did
not clearly assert it" — it would measure the paraphrasing, not the judge.

`errors_made` is checked from the other side instead, which needs no
construction and is already the trip-wire: the oracle **cannot** commit a listed
error, so every one reported against it is a definitive false positive. The
true-positive side needs `common_errors` authored as assertions in the first
place, which is a data change and belongs in `SCHEMA.md`, not a harness patch.

### 21.22 The migration question has an untested middle: a 32B judge fits today

Section 21.20 ruled out rubrics as a lever on judge agreement, and 21.14 ruled
out the judge prompt. That leaves the judge *model*, which STRUCTURAL_AUDIT.md
frames as the one genuine reason to migrate — "a 70B judge at 4-bit is ~40 GB,
needing 64 GB+".

Checking that arithmetic against measured footprints turned up two things.

**First, the machine is 36 GB, not 39 GB.** Three places in STRUCTURAL_AUDIT.md
said 39; the plan has had 36 correct since Section 1.2. My error, and it matters,
because it moves a 70B judge from "tight" to "does not fit at all".

**Second, and the actual point:** measured 4-bit weights on disk are Qwen2.5-7B
4.0 GB, Llama-3.1-8B 4.2 GB, Qwen3-14B 7.8 GB — about **0.55 GB per billion
parameters**. So:

| Judge | 4-bit weights | Fits in 36 GB? |
| --- | --- | --- |
| current (7–8B) | 4.0–4.2 GB | trivially |
| 14B | 7.8 GB | yes, measured |
| 24B | ~13 GB | yes |
| **32B** | **~17.6 GB** | **yes, with room to spare** |
| 70B | ~38.6 GB | no — exceeds total RAM |

**A 32B judge is a 4× scale increase over the current one and runs on the
hardware already owned.** The audit jumped from "8B judges disagree at kappa
+0.24" straight to "a 70B judge needs 64 GB" and skipped the middle. It also had
the binding point wrong in the other direction — it claimed a base model "binds
at ~32B and above", when inference binds somewhere near 55–60B.

This makes the decision resolvable without buying anything:

- **A 32B judge materially improves agreement** → judge scale is the lever.
  Every number in the project improves at once, *and* there is measured evidence
  that a 70B judge — which genuinely needs 64 GB+ — would justify the migration.
- **A 32B judge does not improve agreement** → scale is not the lever, a 70B is
  unlikely to differ in kind, and migrating buys a more expensive version of the
  same disagreement.

Either result is worth more than migrating and finding out afterwards. The cost
is a download and one `--rescore-from` pass over stored answers, which is the
cheap half of an eval — no generation, and the comparison Section 9.9 requires
(vary the judge, nothing else) is exactly what `--rescore-from` does.

Sequenced against what is already queued: the positive controls run first,
because they say whether *any* judge difference is measurable on this scale. A
32B rescore of `gold_n99.jsonl` is the natural second run, and `eval.py --compare`
against the existing Qwen and Llama runs gives inter-judge agreement across three
model families rather than two — which also settles the self-preference question
STRUCTURAL_AUDIT.md lists as open, since a third family shares no lineage with
the `base` arm.

The specific 32B checkpoint is not named here on purpose: the exact
`mlx-community` id should be confirmed against what is actually published before
anything is downloaded, rather than guessed into a config file.

### 21.23 The 32B judge experiment, pre-registered

Downloading `mlx-community/Qwen2.5-32B-Instruct-4bit` — 18.44 GB across four
shards, within 5% of the 17.6 GB predicted in 21.22 from the measured 0.55 GB/B
scaling. It fits in 36 GB with ~17 GB to spare.

#### Why this checkpoint and not a more interesting one

The question 21.22 poses is narrow: **is judge scale the lever?** Answering it
requires varying scale and as little else as possible.

`Qwen2.5-32B-Instruct` is the same family, the same generation and the same
instruction-tuning lineage as `Qwen2.5-7B-Instruct`, which is judge 1 in every
published run. 4.5× the parameters, everything else held. A Mistral-24B or
Gemma-27B would have been a more *interesting* download, and would have
confounded scale with family — a change in agreement could then be either, and
the run would settle neither.

**The cost of that choice, stated plainly:** this does *not* address
self-preference. The judge stays in the same family as the `base` arm, which
STRUCTURAL_AUDIT.md lists as an open question and which a third family would
settle. That is a separate experiment with a separate download, and running it
second is deliberate — there is no point testing whether a third family is
kinder to its own lineage before knowing whether judge scale moves anything at
all.

#### Predictions, recorded so they can be wrong in public

1. **Coverage rises to ~100%.** The 7B judge already grades 99/99 on V3 at 500
   tokens, so there is little room, but a larger model producing better-formed
   JSON is the mechanism that would show up first. If coverage *falls*, the run
   is uninterpretable for the same reason V4 was (Section 21.14) and the token
   budget is the first thing to raise.
2. **Inter-judge agreement rises, but not to the +0.6 trip-wire.** Section 21.20
   found disagreement diffuse across 91 of 99 records rather than localized on a
   few bad rubrics, which reads more like a hard task than a fixable one. A jump
   from r +0.49 to +0.60–0.70 would be a strong result; reaching kappa ≥ 0.6 on
   the blunder call would be a surprise.
3. **The A3 ranking does not change.** base_rag > base > finetuned_rag >
   finetuned held under two judges that disagree at r +0.49. A third judge
   reversing it would be a much bigger finding than a third judge confirming it,
   and would say the n=99 verdict was never safe.
4. **Speed is the practical cost.** At ~0.55 GB/B the 32B is 4.5× the weights of
   the 7B, so expect roughly 4–5× the wall time per judge call. `--rescore-from`
   makes that affordable because generation is skipped entirely.

The one that matters is #2. If agreement barely moves, judge *scale* is not the
lever either — and with rubrics (21.20) and prompt (21.14) already ruled out,
that would say the gameplay and rules evals are near the ceiling of what a
local LLM-as-judge can measure, which is a real and reportable conclusion about
the method rather than about the hardware.

#### What runs, when the GPU frees

```bash
# rescore stored answers — no generation, judge varied and nothing else
python scripts/eval.py --rescore-from eval/runs/gold_n99.jsonl \
  --judge-model mlx-community/Qwen2.5-32B-Instruct-4bit \
  --out eval/runs/gold_n99_judge3.jsonl \
  --report-out eval/reports/gold_n99_judge3.md

# three-way agreement, one pair at a time
python scripts/eval.py --compare eval/runs/gold_n99.jsonl eval/runs/gold_n99_judge3.jsonl \
  --report-out eval/reports/gold_n99_AGREEMENT_7b_vs_32b.md
python scripts/eval.py --compare eval/runs/gold_n99_judge2.jsonl eval/runs/gold_n99_judge3.jsonl \
  --report-out eval/reports/gold_n99_AGREEMENT_8b_vs_32b.md
```

Positive controls still run first: they say whether *any* judge difference is
measurable on this scale, and a 32B result is not interpretable without them.

### 21.24 The Qwen3 rerun: the answer fix worked, and the judge is the reasoning model

The rerun completed after ~6 hours of generation. It fixed what it was meant to
fix and failed harder on something else.

| | run 1 | rerun |
| --- | --- | --- |
| positions × arms | 22 × 3 | 24 × 3 |
| **answer parse rate** | 55–82% | **92–100%** |
| visible answer, mean chars | 91 | 236 |
| **judge coverage** | 41% | **29%** |

`visible_answer()` and the raised generation budget did their job: the model now
produces a parseable answer after it finishes thinking, and Gate 1's parse rate
went from a worst arm of 55% to 92%. Judge coverage went the other way.

**The judge is not choking on its input.** Stripped of reasoning, a batched
three-arm call carries ~709 characters of candidate text — about **177 tokens**
— against a 1,200-token budget. There is no prompt-size story here. The failure
is in what the judge *produces*: `Qwen3-14B` is a reasoning model on both sides
of this run, and as judge it spends its output budget thinking before it emits
the JSON, so the JSON never closes.

That is the opposite of the Section 21.14 diagnosis, where V4 failed because it
demanded too much *output* for the budget. Same symptom — unparseable JSON,
coverage collapse, means over a selected subset — from the opposite cause. Both
are now surfaced in the report rather than left to be reconstructed: it prints
the candidate volume per judge call and says explicitly that a small number
there means the failure is on the production side.

**A reasoning model is a poor judge at any budget that suits a non-reasoning
one.** The fix is either a much larger `--judge-max-tokens` for a reasoning judge
specifically, or — better — not using one. The judge's job is extraction, not
deliberation, and the two published judges (Qwen2.5-7B, Llama-3.1-8B) grade
99/99 and 92/99 at 500 tokens.

Read the numbers with that caveat: 29% coverage, and the arms differ. Gate 2 on
the new definition reads 5/7 (71%) PASS with **17 of 24 positions excluded** for
an unjudged arm. Seven positions is not a gate verdict, and the report says so.

#### A defect this exposed: the run file discarded the reasoning

Asked whether Qwen3 had reasoned at all on the rerun, the run file could not
answer — 0 of 72 stored answers carried a `<think>` block, against 66 of 66 in
run 1. That is not a behaviour change. `_judge_all` stored `candidates[arm]`,
the text *after* stripping, so the scratchpad was discarded at write time.

The commit that introduced the stripping said, in its own message, "the full
output is still what gets STORED." It was not. The claim was written from
intent rather than from the line below it, and the cost landed exactly where it
would hurt most: the one artifact worth having about a reasoning model is the
reasoning, and it was thrown away on the first run that produced any.

Now stores the raw output, plus `reasoning_chars` so the volume is queryable
without keeping the text twice. Safe for `--rescore-from`, which re-applies
`visible_answer()` at judge time: stripping is idempotent, so a raw-stored run
strips correctly and the already-stripped runs are unchanged.

### 21.25 Qwen3-14B answered the model question, and the answer is no

The rerun's second judge finished, which turns the coverage collapse into a
controlled experiment. **Same 24 positions, same stored answers, judge varied
and nothing else:**

| Judge | Reasoning model? | Coverage |
| --- | --- | --- |
| `Qwen3-14B-4bit` | **yes** | 21/72 (**29%**) |
| `Meta-Llama-3.1-8B-4bit` | no | 66/72 (**92%**) |

A model less than half the size grades three times as much of the set, on
identical inputs. That is not a capability difference — it is the reasoning model
spending its 1,200-token output budget on `<think>` before it reaches the JSON.
**The Qwen3-as-judge column is discarded**, and Llama is the usable read.

#### The comparison the six hours were for

Both sides judged by Llama-3.1-8B, on the 22 positions the two runs share:

| Arm | Qwen2.5-7B | Qwen3-14B |
| --- | --- | --- |
| | blunder / corr / legal | blunder / corr / legal |
| `base_open` | 47% / 3.26 / 68% | 55% / 3.23 / 68% |
| `base_closed` | 68% / 2.95 / 73% | 65% / 3.20 / 59% |
| `base_cards_open` | 74% / 2.44 / 64% | **50% / 3.58 / 64%** |
| **Gate 3 (best arm)** | **53% FAIL** | **50% FAIL** |

**Doubling the parameters and adding a reasoning mode moved the best-arm blunder
rate from 53% to 50%, against a gate that needs ≤25%.** That is the answer to
"is there a better model we should be targeting": not this one, and not by
enough to matter. Both models fail Gate 3 by roughly a factor of two.

The one genuine gain is `base_cards_open` — 74% → 50% blunder, 2.44 → 3.58
correctness. Card-augmented retrieval helps the larger model substantially more
than it helps the 7B, where it was the *worst* arm. That is a real interaction
and worth a follow-up, but it moves an arm from bad to bad.

Cost, for the record: ~6 hours of generation for 24 positions × 3 arms at
`--max-tokens 4000`, against minutes for the 7B. Roughly 60× the wall time for
3 points of best-arm blunder rate.

**Caveats, stated rather than buried.** This is a *single-judge* read, because
the second judge is the one that collapsed — and Section 21.19 measured Gate 3
swinging 38 points between judges on identical answers. n=22. What survives the
caveats is the direction and the magnitude: no arm of either model comes within
25 points of the gate, and that gap is far larger than judge variance.

#### The practical rule this establishes

**A reasoning model is a poor judge at any budget that suits a non-reasoning
one.** The judge's job is extraction — which claims does this text make — not
deliberation. Both published judges grade 92–100% at 500 tokens; the reasoning
judge managed 29% at 1,200. Use a reasoning model as a *subject* if it earns its
place, never as the grader, unless its budget is raised specifically and the
coverage is checked before the numbers are read.

### 21.26 The judge fires every listed error at once, and blunder rate is defined on that

Gate 3 needs a blunder rate ≤25% and both models sit above 50% (Section 21.25).
Before accepting that as a capability finding, I read the answers on positions
where every arm blundered. Two of the first three are not capability findings at
all.

**`pos-removal-timing-0001`.** The correct line is *"Doom Blade the Nessian
Asp."* All three arms answered:

    CAST Doom Blade TARGET Nessian Asp
    PASS

— the correct play, exactly — and the judge returned `errors_made: [1, 2, 3, 4]`.
Every listed error, on an answer that made the right move. `blundered` is
`bool(errors_made)`, so all three count against the gate.

**`pos-removal-timing-0002`** is the same shape: correct line *"Flash in Ambush
Viper now, then block Centaur Courser"*, two arms answered exactly that, and got
all four errors. (`pos-combat-math-0003` is a genuine miss — the arms attacked
with both creatures when the line was Serra Angel alone — so the read is not
that everything is a false positive.)

#### It is a judge property, and it is large

`common_errors` are *alternative* wrong answers. Committing all of them is
usually not something one answer can do, so firing them all is the judge using
the error list as a "this answer is bad" flag. Counted across every stored run:

| Run | Judge | Blunder calls | All errors fired | Share |
| --- | --- | --- | --- | --- |
| `pos_qwen25_3arm` | Qwen2.5-7B | 50 | 25 | **50%** |
| `pos_qwen25_3arm_judge2` | Llama-3.1-8B | 36 | 5 | 14% |
| `positions_n22` | Qwen2.5-7B | 60 | 34 | **57%** |
| `positions_n22_judge2` | Llama-3.1-8B | 53 | 1 | 2% |
| `pos_qwen3_14b_v2_judge2` | Llama-3.1-8B | 37 | 10 | 27% |
| `gold_n99` | Qwen2.5-7B | 332 | 165 | **50%** |
| `gold_n99_judge2` | Llama-3.1-8B | 281 | 45 | 16% |

**Half of the Qwen judge's blunder calls fire every listed error; the Llama
judge does it on 2–16%.** The sharper cut is the internal contradiction — an
answer credited with half the key points *and* charged with every listed
misconception, two claims that cannot both hold. On `gold_n99` that is 49 cases
under Qwen and 25 under Llama.

This is Section 21.3 again, which saw the same behaviour on four positions and
diagnosed it as a missing rubric entry. At n=4 that was a reasonable reading.
At this scale it is plainly a judge behaviour, and a judge-specific one.

#### What it explains, and what it does not

It is a large part of why kappa is +0.24: the two judges do not merely disagree
on marginal calls, they have *different failure modes* on `errors_made`. It also
means blunder rate under the Qwen judge is materially inflated, which touches
Gate 3 directly — the metric this project reports as "how often the model throws
the game away".

It does **not** overturn Section 21.25's conclusion. That comparison used the
Llama judge on both sides, where the rate is 27%, and the models are 25 points
from the gate — far outside what this artifact can account for.

`judge_batch_rubric` now returns `all_errors_fired` and `error_contradiction`
per arm, and the report prints the rate beside the score table. **Reported, never
corrected**: sometimes an answer really is wrong on every axis, and silently
dropping errors would change a published metric on a heuristic. Same discipline
as `quote_drops` — measure the fabrication, do not paper over it.

**A prediction for the 32B judge run,** recorded before it lands: if judge scale
is the lever, the all-fired rate should fall well below the 7B's 50%. If a 4.5×
larger model in the same family still uses the error list as a bad-answer flag
at that rate, the defect is in the prompt design or the task, not in capacity —
and that is a more useful thing to know than the kappa alone.

### 21.27 The calibration lands: the score works, the error detection does not

The positive controls finished under the Qwen2.5-7B judge — the single test
STRUCTURAL_AUDIT.md names as deciding whether anything else is measurable.
**98/99 questions graded**, so the numbers are readable rather than a subset.

| Candidate | Mean | Expected |
| --- | --- | --- |
| `oracle` (the reference itself) | **3.88** | near 5 |
| `partial` (half, by sentence) | 3.63 | middle |
| `wrong` (another question's answer) | 1.10 | near 1 |
| `refusal` | 1.07 | 1 |

| Trip-wire | Required | Measured | |
| --- | --- | --- | --- |
| dynamic range | ≥ 2.5 | **+2.77** | PASS |
| ordering accuracy | ≥ 90% | **93%** | PASS |
| false errors on oracle | ≤ 5% | **40%** | **FAIL** |

**Two of three pass, and the instrument splits cleanly in half.**

#### The correctness scale is sound

A 2.77-point separation between the reference answer and a fluent answer to a
*different* question, with 93% of questions ordered correctly. `wrong` was
deliberately built as a real, well-written answer rather than gibberish, so the
judge is not being fooled by fluency — it scores a plausible off-topic answer at
1.10.

That retroactively licenses the A3 comparison. Fine-tuning trailing retrieval by
0.50 and 0.80 is 18–29% of a measured 2.77-point range, on a set where 74% of
questions separate the arms (Section 21.11). **The n=99 verdict is measurable,
and this is the first evidence that it is.**

#### The error detection is broken, at 40%

The oracle *is* the reference answer. It cannot commit a listed `common_error`.
The judge charges it with one on **39 of 98 questions**.

This is Section 21.26 arriving from the opposite direction, and the two
measurements agree: 50% of the Qwen judge's blunder calls fire every listed
error at once, and 40% of its judgements of the reference answer invent an error
outright. 11 of the 39 false positives are the fire-everything signature.

The cost is precise, because `rubric_correctness` halves credit when any error
is present:

    oracle mean when the judge invents no error : 4.86  (n=59)
    oracle mean when it invents one             : 2.39  (n=39)

**The halving takes 2.47 points off the reference answer, on 40% of the set, for
errors it definitionally did not make.** That is what drags the oracle from 4.86
to a reported 3.88.

#### What the instrument would be without it

Inverting the halving on the affected rows:

| | as scored | if `errors_made` ignored |
| --- | --- | --- |
| `oracle` | 3.88 | **4.43** |
| `wrong` | 1.10 | 1.21 |
| **dynamic range** | **+2.77** | **+3.22** |

A 0.45-point wider range, from deleting one term. The error half of the rubric is
not merely noisy — it is *subtracting* resolution from a scale that works
without it.

#### So the answer to "is hardware the constraint" is no, and now specifically no

The deciding test says the instrument is sound on `points_hit` and broken on
`errors_made`, with the mechanism identified and measured twice. That is a
prompt-and-metric defect, not a capacity one, and it is where the next work
belongs:

- **Blunder rate is defined entirely on the broken half.** Gate 3, every
  gameplay blunder number, and the kappa +0.24 all sit on a field with a 40%
  false-positive rate against a case that cannot be wrong.
- **Correctness could stop using it.** Scoring on `points_hit` alone widens the
  range to 3.22 and removes the false-positive channel from the headline metric.
  That is a change to a published number and needs a deliberate re-run, not a
  quiet edit — recorded here as the proposal, not applied.

The Llama calibration is running now, and the 32B is queued behind it. Both
speak directly to this: if the false-error rate is a Qwen property it should
drop under Llama, and if it is a capacity property it should drop at 32B. If it
does neither, the V3 error prompt is simply asking for something a local judge
cannot do, and `common_errors` needs rethinking rather than a bigger model.

### 21.28 Applied: correctness scores `points_hit` alone

`rubric_correctness` no longer halves credit when `errors_made` is non-empty.
Correctness is now `1 + 4 × (points_hit / n_points)`, full stop.

This changes published numbers, so it is done as a revision with the old rule
still reachable, not as a break in the record: `halve_on_error=True` reproduces
every figure through Section 21.27 from the same stored judge output, and every
row now carries a `scoring` field naming which rule produced it (`points_only`
or `halved_v3`).

**No re-judging was needed, and that is the point.** The judge's extraction —
`points_hit`, `errors_made` — did not change; only the arithmetic mapping did.
`scripts/rescore_stored.py` re-derives every stored run exactly, with no model
involved, so each delta below is attributable to the scoring change and nothing
else. It writes `<name>_pointsonly.jsonl` beside the original rather than over
it.

#### What moved on the rules eval

| Arm | judge 1 (Qwen) | judge 2 (Llama) |
| --- | --- | --- |
| `base_rag` | 2.38 → **2.98** | 3.16 → **3.99** |
| `base` | 2.46 → **3.17** | 3.09 → **3.96** |
| `finetuned_rag` | 1.88 → **2.52** | 2.36 → **3.36** |
| `finetuned` | 1.69 → **2.17** | 2.18 → **2.89** |

Every arm rises — the false errors were suppressing all of them — and **the A3
ranking is unchanged under both judges.** The headline gap narrows: base_rag
over finetuned_rag goes 0.50 → 0.46 (Qwen) and 0.80 → 0.63 (Llama). Fine-tuning
still loses, by less.

#### What moved on the gameplay eval: an arm ranking reversed

| Run | halved_v3 | points_only |
| --- | --- | --- |
| `pos_qwen25_3arm_judge2` | open 3.26 > closed 2.95 > cards 2.44 | **closed 4.05 > open 3.89** > cards 3.60 |
| `pos_qwen3_14b_v2_judge2` | cards 3.52 > open 3.21 > closed 3.11 | cards 4.27 > **closed 4.09 > open 3.89** |

`base_closed` and `base_open` swap in both, because the closed arm was the one
being charged with false errors most often — it answers tersely from a supplied
`legal_actions` list, which is exactly the shape the judge tends to fire the
error list at. Section 20.1 concluded "the closed arm makes the model worse, not
better". Under a scoring rule that does not depend on a field with a measured
40% false-positive rate, that conclusion reverses on this metric. It is one
judge and n=22, so it is not settled — but it is no longer supported either, and
Section 20.1 is annotated to say so.

#### Under the accurate judge, this change is roughly neutral

The Llama calibration landed after the change was requested and reframes its
value honestly:

| | Qwen (40% false errors) | Llama (4%) |
| --- | --- | --- |
| dynamic range, halved | 2.77 | 2.74 |
| dynamic range, points-only | **3.22** | 2.66 |

It buys +0.45 of range under the corrupt judge and costs −0.08 under the clean
one. That is the expected shape: when error detection is accurate the halving
carries real signal, and removing it lifts `wrong` (1.45 → 1.58) slightly more
than `oracle` (4.18 → 4.24).

So the case for it is **robustness, not raw resolution**: the headline score no
longer depends on a field whose reliability varies tenfold between two judges of
the same size. `errors_made` is still extracted, still reported, and still what
blunder rate is defined on — the broken channel was removed from the score, not
from the record.

### 21.29 The position eval judged itself, and the self-judge is the miscalibrated one

`eval_positions.py --judge-model` defaulted to `--base-model`: a bare run had the
model grading its own answers. The reports have warned about self-preference
since Section 9.9, but it was a caveat printed under the numbers rather than a
reason to change the default.

Section 21.27 turned it into a measured defect. With Qwen2.5-7B as both subject
and judge, the judge invents a `common_error` against the *reference answer* 40%
of the time — and blunder rate is defined on that field. The cost on identical
stored answers:

| Run | Gate 3 under Qwen | under Llama |
| --- | --- | --- |
| `pos_qwen25_3arm` | 71% | 53% |
| `positions_n22` | 47% | 47% |
| `positions_n22_think` | **65%** | **27%** |

| Arm (`pos_qwen25_3arm`) | Qwen | Llama | |
| --- | --- | --- | --- |
| `base_open` | 73% | 47% | **+25** |
| `base_closed` | 82% | 68% | +13 |
| `base_cards_open` | 73% | 74% | −1 |

On the think run the two judges put the same answers at 65% and 27% against a
25% gate — one of them 40 points from passing, the other 2.

**The default is now `common.CALIBRATED_JUDGE_ID`** (Llama-3.1-8B), the only
judge that passes all three trip-wires. This is not a preference between two
reasonable options: one of them fabricates the metric four times in ten.

Two things keep it from becoming the stale-default trap this repo already fell
into. It is overridable — passing the base model's id explicitly reproduces any
self-judged run — and the judge is recorded in every report, which is how the
40% was found in the first place. The constant lives in `common.py` rather than
`eval.py` because `eval_positions.py` needs it while building its argument
parser and defers `import eval` to `main()` to keep mlx out of `--help`; putting
it in `eval.py` raised a `NameError` at parser-construction time, caught by
building the parser rather than by reading it.

**What this does not fix.** Every position number published before this used the
self-judge. Sections 20 and 21.8 in particular rest on Qwen-judged blunder
rates, and those are inflated by up to 25 points. The Llama-judged columns exist
for all three runs and are the ones to read.

### 21.30 The all-errors-fired artifact does not explain the low kappa

Section 21.26 established that the Qwen judge fires every listed `common_error`
at once on half its blunder calls, and 21.27 that it invents an error against
the reference answer 40% of the time. The obvious next hypothesis: that artifact
is what makes inter-judge kappa +0.24. Removing those calls should lift it.

**It does not.** Recomputing kappa on the blunder call with every pair excluded
where *either* judge fired the whole error list:

| Run | kappa | excluding all-fired | |
| --- | --- | --- | --- |
| `positions_n22` (the +0.24 the audit cites) | +0.24 | **+0.11** | −0.13 |
| `positions_n22_think` | +0.14 | +0.31 | +0.17 |
| `pos_qwen25_3arm` | +0.34 | +0.48 | +0.14 |
| `gold_n99` (rules) | +0.41 | +0.48 | +0.07 |

Three improve, one gets notably worse, and the one that gets worse is the run
the +0.24 figure comes from. Around 44% of pairs are dropped in every case, so
this is not a sample-size accident in one direction.

The mechanism for the reversal is straightforward once stated: when *both*
judges fire the whole list on the same item, that pair is an **agreement**, and
removing agreements can lower kappa even as it removes a defect. The artifact
and the disagreement overlap; neither contains the other.

**What this rules out, and what it leaves.** The all-errors-fired behaviour is
real, measured twice, and inflates blunder rate by up to 25 points — that stands
and is why the judge default changed (21.29). But it is *not* the explanation for
judge disagreement, and fixing it should not be expected to deliver kappa ≥ 0.6.

That also means the open item in STRUCTURAL_AUDIT.md stays genuinely open rather
than pre-answered: kappa between two *calibrated* judges has to be measured, not
inferred from this. A second thing worth noting is how unstable kappa is at
these sizes — +0.14 to +0.41 across four runs of the same eval, on the same two
judges. Any kappa target should be read against that spread before it is treated
as a threshold.

### 21.31 The 32B judge agrees LESS, and prediction 2 was wrong

The 32B rescore of `gold_n99` finished: 99 questions, **396/396 arm-answers
graded**, no coverage problem at all. Section 21.23 recorded four predictions
before it ran. Scoring them honestly:

| # | Predicted | Outcome |
| --- | --- | --- |
| 1 | coverage ~100% | **held** — 396/396 |
| 2 | agreement rises, but short of +0.6 | **wrong** — it *fell*, to +0.20 |
| 3 | A3 ranking unchanged | **held** |
| 4 | ~4–5× slower | **held** — ~70 min against ~13 |

| Judge pair | Pearson r |
| --- | --- |
| Qwen2.5-7B vs Llama-3.1-8B | **+0.49** |
| Qwen2.5-7B vs Qwen2.5-32B | **+0.20** |
| Llama-3.1-8B vs Qwen2.5-32B | **+0.22** |

**The 4.5× larger judge agrees with each small judge less than they agree with
each other**, even though it shares a family and an instruction lineage with one
of them. All three files use the same scoring rule, so the comparison is
matched — the run predates Section 21.28 and carries no `scoring` field, which
is exactly the case that field was added to make recoverable.

#### Why: it credits almost nothing

| Judge | all-errors-fired | mean `points_hit` credited |
| --- | --- | --- |
| Qwen2.5-7B | 50% | 43% of rubric |
| Llama-3.1-8B | 16% | **63%** |
| Qwen2.5-32B | 40% | **8%** |

The 32B awards 8% of the available rubric points. That is five times stingier
than Llama and it is what drags every arm to 1.09–1.60 on a 1–5 scale. It also
fires the whole error list on 40% of its blunder calls, so it has the Qwen
family's error artifact and a severe points deficit on top.

One thing genuinely improves: **length neutrality**. Correlation between answer
length and score is +0.080, against +0.324 for the 7B and +0.227 for Llama. The
32B is much harder to impress with volume.

#### What this cannot yet settle

Two readings fit the same numbers, and they point opposite ways:

- **The 32B is stricter and right.** It demands the candidate actually assert a
  point rather than gesture at it, and the two smaller judges share a
  family-independent tendency to over-credit. Low agreement with them would then
  be a *feature*, and 8% would be closer to the truth than 63%.
- **The 32B is under-crediting.** It misses paraphrase, which the V3 prompt
  explicitly instructs against ("count a point as hit if the candidate states it
  in ANY wording, including paraphrase or implication").

**The positive controls discriminate, and they are running now.** The oracle IS
the reference answer, so a judge that understands the rubric should credit close
to 100% of its points. If the 32B credits the oracle at 8% too, it is
under-crediting and its dynamic range will collapse. If it credits the oracle
near 100% while crediting real answers at 8%, it is discriminating hard and the
smaller judges are the loose ones.

Nothing about judge scale should be concluded until that lands — which is the
whole reason a judge gets calibrated before its agreement number is read.

### 21.32 The position set is fully hand-adjudicated, and one board was illegal

The last four positions carried `machine-drafted (assistant); awaiting author
review`. All four lines were put to the author and confirmed:

| Position | Line | Verdict |
| --- | --- | --- |
| `pos-blocking-0004` | Block Serra Angel with Nessian Asp (reach) | confirmed |
| `pos-combat-math-0006` | Attack — the only blocker is tapped, 3 is lethal | confirmed |
| `pos-land-sequencing-0003` | Play Mountain — keeps Shock live | confirmed, with a reason the draft missed |
| `pos-removal-timing-0003` | Doom Blade the Serra Angel | confirmed |

**24/24 hand-authored.** Every card claim was resolved through `CardIndex`
first, per the standing rule — Nessian Asp's reach, Serra Angel's flying, Wall
of Omens' lack of both, and Doom Blade's *nonblack* restriction against a white
Angel and a green Courser all check out.

#### The board said something the rules forbid

`pos-blocking-0004` rendered Serra Angel as `tapped ** ATTACKING **`. Serra
Angel has **vigilance** — attacking never taps it. The play is unaffected, which
is exactly why it survived drafting and validation: `timing_problems()` checks
whether an action could legally be taken, not whether the board could legally
exist.

It is the Pacifism failure (Section 21, batch 5) in a quieter form. That one
offered a sorcery-speed play at instant speed and was caught because the *play*
was impossible. This one is a board state that cannot arise, presented to a
model being asked to reason about the rules. Now untapped.

**A validator that checks actions does not check states.** Vigilance is the
cheapest case; the general form is any keyword whose effect is a board
invariant. Worth a `board_problems()` companion to `timing_problems()` if more
positions are authored.

#### The author's reasoning added a rubric entry

On the land drop the confirmation came with a correction: Mountain is right both
because it keeps Shock castable *and* because the Forest still delivers turn-3
Centaur Courser either way — and **Giant Growth is irrelevant, because there is
no creature on the battlefield to target.**

The first two were already key points 1–3. The third was not anywhere, and it is
precisely the shape of wrong reasoning a model produces: keep double green for
the pump spell. Added as a fourth common error:

    Keeps the second Forest for Giant Growth, which has no creature on the
    battlefield to target this turn

This is the Section 21.3 lesson applied at authoring time rather than after a
run — *a rubric with no entry for the most likely wrong answer cannot catch it*,
and the lint cannot help, because it checks the lines that are there.

### 21.33 The `partial` control was 79% of the answer, and one trip-wire was vacuous

`half_answer` builds the mid-scale control in the calibration harness. It
appended a sentence and *then* tested whether half the text had been reached, so
it always overshot by a whole sentence. On these answers — short, few sentences —
the result was not half:

| | mean | median | **byte-identical to the full answer** |
| --- | --- | --- | --- |
| before | 79% | 81% | **14/99** |
| after | 56% | 51% | 0/99 |

On 14 of 99 questions `partial` *was* `oracle`, so the calibration presented the
reference answer to the judge twice and labelled one copy "partial".

Fixed by choosing the prefix whose length is closest to half and never returning
every sentence: at least one, at most n−1, so `partial` is always a proper
prefix. Pinned by 16 assertions, including the two invariants the first version
broke (never the whole answer, never empty).

#### What it cost the published numbers

Only the rows that involve `partial`:

| Trip-wire | Uses `partial`? | Status |
| --- | --- | --- |
| dynamic range (oracle − wrong) | no | **stands** — +2.77 / +2.74 |
| false errors on oracle | no | **stands** — 40% / 4% |
| ordering accuracy (oracle ≥ partial ≥ wrong) | **yes** | **needs a re-run** |

And ordering accuracy was worse than merely optimistic — it was close to
vacuous. Decomposing the three-way comparison in the stored calibration:

| Judge | `oracle ≥ partial` | `partial ≥ wrong` | `oracle > wrong` | reported |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | 94% | **100%** | 98% | 93% |
| Llama-3.1-8B | **100%** | 96% | 94% | 94% |

Both terms containing `partial` are satisfied ~100% of the time, so the
conjunction is decided almost entirely by `oracle > wrong` — which is what
dynamic range already measures. **Ordering accuracy was not an independent
check; it was the dynamic-range check restated**, and it passed for that reason
rather than because the judge can rank three distinct quality levels.

That is the specific danger of a control that does not control: it produces a
PASS which reads as corroboration and is a copy of the number beside it.

With a genuine mid-point the test becomes real, and it is worth stating in
advance that it may now fail. A judge that cleanly separates a *complete* answer
from a *half* one is a stronger claim than anything measured so far, and the
41-point spread between Llama's 63% points-credited and the 32B's 8% (Section
21.31) suggests these judges disagree considerably about what counts as stating
a point at all.

**The two trip-wires that decided "hardware is not the constraint" are the two
that stand.** Dynamic range and false-errors-on-oracle never involved `partial`,
so Section 21.27's conclusion is unaffected — but the calibration should be
re-run for a defensible ordering number.

### 21.34 The 32B judge is right and the small ones are loose

Section 21.31 left two readings open: either the 32B is stricter and correct, or
it is under-crediting by missing paraphrase. Its positive controls settle it.

All three judges, under matched `points_only` arithmetic (the two older runs
inverted exactly — the halving is invertible, so no re-run was needed):

| Judge | range | ordering | false errors | coverage | credits on **oracle** | on real answers |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | +3.22 | 93% | 40% | 98/99 | 86% | 43% |
| Llama-3.1-8B | +2.66 | 94% | 4% | 95/99 | 81% | 63% |
| **Qwen2.5-32B** | **+3.60** | **100%** | **0%** | **99/99** | **90%** | **8%** |

**The 32B wins every measure**, and the last two columns are why it is not
under-crediting: a judge that credits the *reference answer* with 90% of its own
rubric understands the rubric. Crediting real model answers at 8% is then a
statement about the answers.

Llama credits real answers at 63% against 81% on the oracle — a ratio of 1.3×.
The 32B's ratio is **11.2×**. One of these barely separates a model answer from
the reference; the other separates them by an order of magnitude.

#### Read the answers, again

Two cases where Llama credited every key point and the 32B credited none:

**`rg-7`.** The rubric's first key point is *"Player B controls both
permanents"*. The model answered *"Sir Shandlar of Eberyn would be controlled by
**Player A**"*, cited a rule number that does not exist (723.2), and introduced a
Player C who is not in the question. **Llama: 4/4 key points. 32B: 0/4.**

**`qa-armando-controls-leyline-of-the-guildpac`.** The model produced a
fabricated type line and ability text for Snow-Covered Plains, including a
power/toughness for a land. **Llama: 5/5. 32B: 0/5.**

Both were verified against the *eval* set, not the gold templates — the eval set
has **0 unresolved `[[cardN]]` slots**, so the judge saw real card names and the
Section 18.5 templating failure is not in play here.

The aggregate matches the anecdotes: **Llama awards full marks to 40% of model
answers.** The 32B awards full marks to 5% and zero to 87%.

#### Consequences

`CALIBRATED_JUDGE_ID` is now the 32B, so `eval_positions.py` defaults to it.
Cost is ~4.5× per judge call, which `--rescore-from` makes affordable because
generation is skipped.

**Two earlier conclusions need revisiting, and one gets stronger.**

- Section 21.31 called the low agreement (+0.20, +0.22) a puzzle. It is not: the
  32B disagrees with two loose judges *because* they are loose. Inter-judge
  agreement was never the right target — **agreement with a calibrated judge
  is**, and two badly-calibrated judges agreeing with each other at +0.49 is
  the number that should have looked suspicious.
- Section 21.23's prediction 2 (agreement rises) was wrong, and wrong for an
  interesting reason rather than a boring one.
- **The A3 verdict survives and sharpens.** Under the 32B the ranking is
  unchanged — base_rag 1.60 > base 1.40 > finetuned_rag 1.23 > finetuned 1.09 —
  so fine-tuning still loses under a third, better-calibrated judge.

**And the migration question is answered.** Judge quality was the binding
constraint on every number in this project, and the fix is an 18 GB model on a
36 GB machine. Not more memory — a better judge that already fits.

The honest limit on this: two answers were read by hand. The trip-wires, the
oracle/real ratio, and the full-marks rate are measured across all 99, but the
*interpretation* that the 32B is right rests partly on those two, and a wider
hand-audit would strengthen it.

### 21.35 `common_errors` are now claims, not behaviours

The positive controls showed the instrument split in half: `points_hit` works
(the judge credits the reference answer with 90% of its own key points) and
`errors_made` does not (40% false positives against an answer that cannot commit
an error). Section 21.30 ruled out the all-errors-fired artifact as the
explanation. This is the better one, and it is structural.

**The judge is asked the same question about both lists** — *which of these did
the candidate assert?* But the two lists were written in different grammars:

| | Form | Example |
| --- | --- | --- |
| `key_points` | claim | "Player B controls both permanents" |
| `common_errors` | **behaviour** | "Adds Centaur Courser to the block, spending a 3/3" |

For key points the question is answerable as asked. For common errors it becomes
*did this text describe a player doing that?* — a different and much vaguer
task. **89% of the 347-entry corpus opened with a capitalised third-person
verb**, so this was the norm and not an occasional slip.

SCHEMA.md rule 5 now states the convention, with the test: **could a wrong
answer contain this sentence?** If not, it describes a mistake rather than being
one.

    Thinks a chump block stops all the damage
      -> A chump block stops all the trample damage

    Holds Doom Blade for a better target
      -> Doom Blade should be held for a better target

**All 83 position entries rewritten**, 82 behaviour-shaped → 0. Answers, key
points and error counts untouched; `common_errors` is the only field that
changed in any of the 24 records.

#### The lint had to learn the new form, twice

`looks_like_behaviour` replaces a first-draft regex that fired on "**Triggers**
resolve in the order..." and "**This** hand should be mulliganed" — both fine
claims, matched because the opening word ends in *s*. It now also requires that
the second word is not a verb or modal, which is what separates
*subject-verb* ("Triggers resolve") from *verb-object* ("Adds Centaur Courser").

Then the *other* lint broke. `lint_common_errors`' restatement check — the one
Section 16.12 measured as closing a 28-point blunder-rate gap — fired **50 times
on 24 positions** after the rewrite. Every hit was a false positive, and the
reason is worth stating because it is a property of the new form rather than a
bug in the rewrite:

> That check was built for "Casts Lightning Strike at the opponent's face
> **instead of** the blocker", where a lead clause restates the correct play and
> the judge can match it before reaching the qualifier that makes the play
> wrong. A claim has no qualifier. "Lightning Strike should be aimed at the
> opponent" necessarily opens with the same card as the correct line — both
> sentences are *about* that card — and what differs is the predicate, which is
> always present.

So the restatement check is now gated on `looks_like_behaviour`: it still fires
on the Section 16.12 case verbatim, and is silent on claims. Two lints, each
correct for the form it was written against.

#### What this invalidates

**Every position blunder number predates this rubric.** Gate 3, the blunder
rates in Sections 20–21, and the kappa figures were computed against
behaviour-form `common_errors`; the rubric they scored against no longer exists.
Those runs are not wrong, they describe a different rubric — and the whole point
is that the new one should be easier for a judge to score. Re-running the
positions under the 32B judge is the measurement that would show whether it is.

**The 264 gold-set entries are not yet converted.** Positions came first because
blunder rate is defined there and the set is a quarter the size. The rules eval
uses `errors_made` only as a reported diagnostic now that correctness ignores it
(Section 21.28), so the cost of leaving them is lower — but the same 40%
false-positive rate applies, and the same fix should follow.

### 21.36 The claim form does not fix error detection — the hypothesis is refuted

Section 21.35 argued that `errors_made` is broken because `common_errors` were
written as behaviours while `key_points` are claims, and the judge is asked the
same *did the candidate assert this?* question about both. It is a tidy
explanation. It is wrong.

**Test 1 — the same stored answers, the same judge, only the rubric form
changed** (`--rescore-from` over `pos_qwen25_3arm`, Qwen2.5-7B):

| Rubric form | blunder calls | all-errors-fired | share |
| --- | --- | --- | --- |
| behaviour (old) | 50 | 23 | 46% |
| claim (new) | 43 | 21 | **49%** |

No improvement. Under the calibrated 32B judge the same claim-form rubric fires
the whole error list on **63%** of its blunder calls.

**Test 2 — the decisive one.** All-errors-fired on a *model* answer conflates
"the judge is spraying" with "the answer really is bad". So: feed each position
its **own correct answer** and count how often the judge charges it with an
error. The correct line definitionally commits none, so every hit is a false
positive — the calibration's oracle test, applied to positions, with rubric form
as the only variable:

| Rubric form | oracle false positives |
| --- | --- |
| behaviour (old) | 16/23 = **70%** |
| claim (new) | 18/24 = **75%** |

**+5% ± 26% at 95%.** Not distinguishable from zero, and pointing the wrong way.

#### What the real driver looks like

The same calibration data says the false-positive rate varies far more by
*question type* than anything the rubric grammar could explain:

| Category | FP rate on the oracle |
| --- | --- |
| templating/keyword meaning | **77%** |
| interaction puzzle | 46% |
| state-based actions | 46% |
| zone transition | 42% |
| turn-structure walkthrough | 38% |
| layer-system question | 23% |
| definition recall | 22% |
| priority reasoning | **17%** |

A 60-point spread across categories, against a 5-point non-effect from rubric
form. Rubric size barely differs between the clean and false-positive groups
(2.6 vs 2.8 common errors, 3.2 vs 3.6 key points), so it is not "more entries,
more chances to fire" either.

Positions are also markedly worse than rules questions for the same judge: **70%
oracle false positives against the gold set's 40%.** A board that has to be read
and simulated is a harder grading task than a rules question, and the error list
is where that difficulty surfaces.

#### What I am doing about it

**Not converting the remaining 214 gold entries.** They were justified by a
hypothesis that measurement has now refuted, and 214 hand-rewrites of
hand-labelled data on a theory the data does not support is exactly the kind of
work that should stop when the evidence arrives.

The 83 position entries stay converted. The change is defensible on its own
terms — the entries are more precise, the two lints no longer contradict each
other, and SCHEMA.md now states a testable rule — but **it should not be
described as an improvement to error detection, because it is not one.** The
cost is real and worth stating: every position blunder number now predates the
rubric it was scored against.

The honest summary is that `errors_made` remains broken, the cause is not the
one this section set out to fix, and the strongest lead is that some question
categories are simply much harder to grade for errors than others.

### 21.37 The mechanism: the judge matches the error's content and misses the negation

Section 21.36 refuted the grammar hypothesis and left the 40% false-positive
rate unexplained, with one lead — the rate varies 17%–77% by category. Reading
the worst category found the mechanism, and it is not about categories either.

`qa-aubree-has-murderous-rider`. The reference answer says:

> **No.** ... Adventurer cards only have their normal characteristics in every
> zone other than the stack (715.4), so **it's not a legal target for Flashback**.

The judge charged it with all three listed errors, of which the first two are:

> 1. An Adventure is an instant or sorcery so **it is a legal target for Flashback**
> 2. The card still has an Adventure while in the graveyard so **its still a legal target**

Same vocabulary, opposite polarity. The answer states the negation of the error
and is scored as having asserted it.

#### Measured, not just observed

Two cuts. The first fails, and is worth recording because it is the obvious one:

| | mean overlap with the oracle |
| --- | --- |
| errors the judge fired | 35% |
| errors it did not | 33% |

**+2% ± 6%** — bulk vocabulary overlap explains nothing. Which makes sense: the
error shares words with the *whole answer* either way, and the discriminating
thing is narrower than that.

The sharp cut. Take questions whose reference answer opens with a bare **Yes**
or **No** verdict, and ask whether the error's polarity *opposes* it:

| Error polarity vs the oracle's verdict | fired | rate |
| --- | --- | --- |
| **opposes** the verdict | 13/44 | **30%** |
| agrees with the verdict | 3/44 | **7%** |

**+23% ± 15% at 95%, significant.** An error that contradicts the answer's own
conclusion is more than four times as likely to be scored as committed by it.

#### This also explains Section 21.36

Rewriting `common_errors` as claims could not have helped, because it does not
touch the polarity relationship — and it plausibly explains why the claim form
came out marginally *worse* (75% against 70%): a claim states the false
conclusion more directly than a behaviour description does, which makes the
surface contradiction cleaner and the confusion easier.

#### The fix follows from the mechanism

If the negation lives in the text, then the text is the check: there is no
verbatim span in *"it's not a legal target for Flashback"* that asserts the
opposite. **V5 is V3 plus a quote requirement on `errors_made` only**, reusing
`verify_quoted_claims`, which is already built and tested.

Errors only, deliberately. V4 demanded a quote for every key point as well and
cost 60–85 points of coverage by exhausting the output budget (Section 21.14).
Errors fire on about a third of candidates and there are 2–3 of them against 3–4
key points, so V5 asks for a small fraction of V4's output. The prompt is 270
characters longer than V3.

Verified on the stubbed path before spending GPU: given the exact failure — the
judge claiming error 1 with a quote that is not in the answer — V5 drops both
claims (`quote_drops: 2`, `errors_made: []`, so `blundered` is False), while a
genuinely committed error whose quote *is* present survives with 0 drops.

The benchmark run comparing V3 and V5 on oracle false positives is in flight.
V3's number on this set is 70%; anything that does not move it substantially
means the quote requirement is not the remedy either, and the honest next step
would be to stop treating `errors_made` as recoverable and report blunder rate
with its false-positive rate attached.

### 21.38 V5 removes false positives and never adds one, but costs a third of the coverage

The V3-vs-V5 benchmark on the position oracle set, judge and rubric held fixed:

| | graded | oracle false positives | quote_drops |
| --- | --- | --- | --- |
| V3 | **24/24** | 18 (75%) | 0 |
| V5 | **16/24** | 9 (56%) | 9 |

**Those two rates are not comparable and the run was repeated to make them so.**
V5's coverage is lower, and if it fails on the questions that are hardest to
grade then the surviving 16 are easier by selection — which is exactly how
Section 21.14's V4 run produced four confident means over 14 of 99. Storing
per-question and intersecting:

| Matched on the 16 both judged | false positives | |
| --- | --- | --- |
| V3 | 12/16 (75%) | |
| V5 | 9/16 (**56%**) | **3 fixed, 0 regressions** |

So the real effect is 9 against 12, not 9 against 18. The selection concern was
warranted and the arithmetic anticipated it: a random 16 of V3's 18 would have
been ~12.

**The direction is clean.** Three false positives removed, **none introduced** —
V5 never fired an error where V3 was silent. That is what a precision fix should
look like, and it is consistent with the mechanism: nine claims were dropped for
being unquotable, which is the negation check doing its job.

**The significance is not there.** Three discordant pairs, all one way, is
McNemar exact **p ≈ 0.25**. On 16 positions this is a direction, not a result.

**And the cost is real.** Eight of 24 positions went ungraded — a third of the
set — which is the V4 failure recurring in milder form. A judge that grades 67%
of the set cannot be the default whatever its precision, because the ungraded
third is selected rather than random.

The obvious suspect is the token budget: the benchmark ran at 700, and V5 asks
the judge to emit a verbatim quote for every error it lists on rubrics carrying
three or four of them. A retry at 1,800 is running. If coverage returns to
24/24 and the three fixes hold, V5 is worth adopting and worth measuring at n=99
on the rules set where the significance question can actually be settled. If
coverage stays at two thirds, the quote requirement is too expensive for this
judge at this rubric size, and the honest position is the one Section 21.37
named: stop treating `errors_made` as recoverable and publish blunder rate with
its false-positive rate attached to it.

### 21.39 The coverage cost was a harness bug, not V5 — a single arm is graded unwrapped

Neither branch of 21.38's decision rule applied, because **the missing eight
positions were never ungraded.** The judge answered all 24. The harness threw a
third of the answers away.

Isolated in two steps, budget first because it was the stated hypothesis:

| | coverage |
| --- | --- |
| V5 at 700 tokens | 16/24 |
| V5 at 1,800 tokens (2.6×) | 16/24 — **identical**, so not the budget |
| V5 at 1,800 with the fix below | **24/24** |

Identical at 2.6× the budget is not what truncation looks like, so the raw text
came out of the model and got read. It parsed:

```json
{ "points_hit": [1, 2], "errors_made": [{"n": 1, "quote": "…"}], "citation": 5 }
```

That is a complete, valid entry — **without the `{"A": …}` wrapper.** Asked to
grade one candidate labeled A, the judge reasonably emits the entry itself.
`judge_batch_rubric` did `scored.get("A")`, got `None`, and recorded *the judge
said nothing about this position*. Absence and unexpected-shape are the same
value, and only one of them is true.

`judge_batch_rubric` now accepts the unwrapped form, but **only when there is
exactly one label and the object looks like an entry rather than a label map.**
With several arms an unwrapped object cannot be attributed to any of them and
must still fail — that is the case worth keeping a test on, along with a wrapped
single-arm result not being double-wrapped.

#### Why it hit V5 and not V3 — the bug was correlated with the treatment

Both versions are single-arm here, so both ran through the same broken line. V3
lost nothing. Counting the shape the judge actually emitted, n = 24 each:

| | wrapped `{"A": …}` | unwrapped | unparseable |
| --- | --- | --- | --- |
| V3 | **24** | 0 | 0 |
| V5 | 16 | **8** | 0 |

Eight unwrapped, eight "ungraded" — the same eight. **The V5 prompt changed the
judge's serialization on a third of its calls**, and V5's added text says nothing
about output format; it adds a verbatim-quote requirement to `errors_made` and a
sentence about polarity. Naming a new per-entry field appears to pull the model
toward emitting an entry rather than a map of them.

This is the part worth carrying forward. **A harness bug that fires uniformly
gets noticed as a bug. One whose trigger rate is a function of the condition
under test gets read as a finding about that condition** — here, "the quote
requirement costs a third of the coverage," which is a coherent, plausible,
entirely fictional result that survived one round of hypothesis-testing. The
control is what exposed it: V3 at 24/24 through the same code path meant the
difference could not be in the shared harness *unless* the harness was
prompt-sensitive, which is the possibility that took longest to reach.

**No published number can move.** The unwrapping fires only at one arm, and
every published run used three to five; the run that checked this reports *none*.
The bug was reachable only from the single-arm benchmark harness — which is to
say it was invisible everywhere except the one place it was measuring.

#### The corrected result

Same judge, same rubrics, same 1,800-token budget, both versions now at full
coverage:

| Matched on all **24** positions | oracle false positives | |
| --- | --- | --- |
| V3 | 18/24 (75%) | |
| V5 | 14/24 (**58%**) | **4 fixed, 0 regressions** |

The direction from 21.38 holds and strengthens — one more fix, still nothing
introduced. Significance is still short: four discordant pairs all one way is
McNemar exact **p = 0.125**. Six one-way pairs would reach 0.031, which is
within reach of the n=99 rules set and not of 24 positions.

> **"0 regressions" here is measured on oracle answers only, where a regression
> cannot occur** — an answer committing no error cannot lose a true positive.
> Section 21.40 adds the negative control and finds V5 loses two on this judge.
> The same caveat applies to 21.38's "3 fixed, 0 regressions" above.

**Two of the four fixes are not the quote check.** `pos-blocking-0005` and
`pos-mulligan-0002` flipped with **zero** quote drops — nothing was verified
away, the judge simply did not fire the error once the prompt told it that an
answer stating the OPPOSITE of an error has not committed it. Only
`pos-removal-timing-0003` and `pos-trigger-ordering-0002` were fixed by the
verification itself. So V5 is two changes wearing one name: a polarity
instruction and a receipt check, and on this set they contributed equally.

**The quote check prunes far more than it clears.** Thirteen claims were dropped
across the matched set, and exactly two of them changed a verdict. Blunder rate
is defined on `errors_made` being *non-empty*, so dropping three of four
fabricated errors leaves the position still counted as blundered. A precision
fix on a field consumed as a boolean has to clear the last one to show up at all.

#### What this does not fix

**58% is still a false-positive rate on answers that definitionally cannot
commit an error.** 75% → 58% is real, and it is a rate that makes the metric
unusable moving to a rate that makes the metric unusable. Section 21.37's
position stands unamended: `errors_made` is not yet a field a gate can be
defined on, and blunder rate is published with its false-positive rate attached.

What has changed is that the ceiling is no longer known to be a third of the
set, so the n=99 measurement that can settle significance is worth running.

**Every number in 21.38 is superseded by this section**, and 21.38 is kept
because the reasoning that caught it — store per-question and intersect, never
compare aggregate rates over different subsets — is what made the discrepancy
visible in the first place. It was right about the selection risk and wrong
about the cause, and it named the budget as a hypothesis to test rather than a
finding, which is the only reason testing it was the first thing done.

#### The trap, which has now fired twice

Section 21.12 was a judge emitting `"points_hit": 3` where a list was expected.
This is the same failure one level up: **valid JSON in an unexpected shape,
silently read as absence.** A missing key is indistinguishable from a key the
model chose to spell differently, and both arrive as `None`.

The tell both times was a number that was *plausible* — 16/24 reads as a hard
judge, 33% coverage reads as an expensive prompt — and both times the diagnosis
came from printing the bytes rather than reasoning about them. Two hypotheses
were formed and tested before that (token budget, JSON validity) and both were
wrong. **When a parse produces a surprising count, read the raw text first;
reasoning about what the model probably emitted has a worse track record here
than looking.**

### 21.40 `errors_made` is not broken — the 7B judge is. On the 32B it is a working instrument

Sections 21.36 through 21.39 measured one judge, `Qwen2.5-7B-Instruct-4bit`, and
wrote their conclusions about **the field**. Running the same benchmark on
`CALIBRATED_JUDGE_ID` — the 32B, the judge every published comparison actually
uses — changes the answer completely. Same 24 positions, same claim-form
rubrics, same prompts, same 1,800-token budget; the judge is the only variable.

| Oracle false positives | V3 | V5 |
| --- | --- | --- |
| Qwen2.5-**7B** | 18/24 (**75%**) | 14/24 (58%) |
| Qwen2.5-**32B** | **1/24 (4%)** | **0/24 (0%)** |

**The 75% was never a fact about `errors_made`; it was a fact about a 7B model.**

**What this benchmark is and is not.** The judge is the only variable across
those four cells — same positions, same rubrics, same prompt, same budget, same
single-arm harness — so the 75%-versus-4% comparison is sound. It is *not* the
same measurement as `calibrate_judge.py`'s `false errors` trip-wire, which runs
**four** arms on the **rules** gold set; arm count changes every arm's score by
as much as 23 points (21.5), so 4% here should not be read directly against the
≤5% bar. The trip-wire's own verdict for this judge is the separate 4-arm rules
run in 21.31, which measured **0%**. Two independent constructions, both clean,
neither substituting for the other.

#### Specificity alone proves nothing, so here is the other half

A judge that fires no errors at all scores a perfect 0% on the test above and is
worthless. The oracle test measures only specificity, and nothing in 21.36–21.39
measured sensitivity — which is the control that should have been there from the
start, and is the same omission CLAUDE.md already warns about under *"when adding
an arm, add the subset where it should have no effect."*

Ground truth is definitional here, which is the payoff of the 21.35 rewrite:
`common_errors` are now claims a wrong answer could contain **verbatim**. So an
answer asserting one has committed it by construction. Each position was fed its
own error *i* (rotating *i* by position, so this is not a statement about first
entries), and the judge must fire exactly that one:

| n=24, one planted error each | fired any | fired **the planted one** | mean errors fired | quote drops |
| --- | --- | --- | --- | --- |
| **32B** V3 | 24/24 | **24/24** | **1.12** | 0 |
| **32B** V5 | 24/24 | **24/24** | **1.00** | 0 |
| 7B V3 | 24/24 | 24/24 | **2.75** | 0 |
| 7B V5 | 24/24 | 22/24 | 1.38 | 14 |

**Perfect sensitivity on the 32B, and precise with it.** On an answer carrying
exactly one listed error it fires 1.12 under V3 and exactly 1.00 under V5 — it
finds the one that is there and does not spray. Set against 1/24 and 0/24 on
answers carrying none, `errors_made` on this judge separates blundered from
clean essentially cleanly.

**The 7B fires 2.75 of them.** Given one error out of three or four, it charges
the answer with most of the list. That is Section 21.26's "every error at once"
signature — the error list used as a *this answer is bad* flag — and it is the
same behaviour as the 75% oracle rate seen from the other side. It does not
appear on the 32B at all.

#### The control also corrects 21.38 and 21.39: V5's regressions were invisible

Both sections reported V5 as **"fixed N, 0 regressions."** That was measured on
oracle answers only, where a regression is *by definition* unobservable — an
answer that commits no error cannot lose a true positive. With the sensitivity
half in place, V5 on the 7B is:

| Qwen2.5-7B | false positives removed | **true positives lost** |
| --- | --- | --- |
| V3 → V5 | 4 (18 → 14) | **2 (24 → 22)** |

`pos-land-sequencing-0001` and `pos-trigger-ordering-0002` assert their planted
error verbatim and V5 discards it — 14 quote drops on answers where the quote is
present *exactly*, which is the check's easiest possible case. So V5 buys
precision with recall rather than for free, and the earlier claim was an artifact
of measuring only the arm where the cost cannot show up. **"No regressions" from
a positive-control-only design means "no regressions were measurable."**

#### So V5 is unnecessary, and 21.37's conclusion is withdrawn

V5 has nothing left to do on the 32B. **Zero quote drops** across both the oracle
and sensitivity runs, on all 96 gradings — the 32B never fabricated an
unquotable error claim, so the receipt check never fired once. Its one apparent
gain (1 → 0 false positives) is a single discordant pair, McNemar p = 1.0, and
on the 7B the same change costs two true positives.

V3 stays the default. V5 remains in the code, tested, as the instrument to reach
for if a future judge shows the 7B's failure — it is 270 characters and one
`verify_quoted_claims` call, and keeping it costs nothing.

**Section 21.37's closing sentence — "stop treating `errors_made` as
recoverable" — is withdrawn.** It was true of the 7B and stated of the field.

#### The mechanism was right, and the 32B's single miss confirms it

21.37 identified polarity: an error that contradicts the answer's own conclusion
fires 30% of the time against 7% for one that agrees (+23% ± 15%). That holds,
and the 32B's *only* false positive is a textbook instance of it.
`pos-combat-math-0002`:

| | |
| --- | --- |
| reference answer | "**Cast Lightning Strike** at the opponent **first, then attack** with both creatures." |
| error 1, fired | "**Attack first and cast Lightning Strike afterwards**" |

Every content word shared; only the order inverted. The answer states the
error's negation and is charged with it. And V5's polarity instruction — *an
answer that states the OPPOSITE of an error has not committed it* — is exactly
what cleared it. n=1, so this illustrates the mechanism rather than measuring it,
but the mechanism now has independent confirmation at two scales: at rate on the
7B, and on the single residual failure of the 32B.

#### What this costs, and what it was worth

The honest accounting: **the answer was already in the calibration table.**
Section 21.31 measured the 32B at **0% false errors** on the rules gold set and
recorded it as a win over the 7B's 40%. Four sections then investigated the 40%
as a property of `errors_made` — a grammar hypothesis (21.36, refuted), a
polarity mechanism (21.37, correct), a V5 remedy (21.38), and a harness bug
(21.39) — without re-running the one comparison that was already known to
matter.

It was not unreasonable to check positions separately; they are a different
rubric shape and 21.35 had just rewritten every entry. What was unreasonable was
writing "the judge" and "`errors_made` is broken" when the measurement said
"this 7B". **A rate measured on one judge is a statement about that judge** —
which is this project's oldest rule, applied to arms and gates since 21.19 and
not, until now, to the diagnosis of a metric.

#### What this unblocks

Blunder rate is a metric a gate can be defined on, so the stored 32B run on
claim-form positions can be read as evidence rather than as suspect:

| `pos_claimform_32b.jsonl`, n=22 | blunder rate | mean correctness |
| --- | --- | --- |
| `base_closed` | 41% | 3.14 |
| `base_open` | 59% | 2.09 |
| `base_cards_open` | 73% | 1.95 |

Against a 25% bar, **Gate 3 fails on every arm, and the failure is now
attributable to the answers** rather than to the instrument: on this same
position set the judge charges a clean answer 1 time in 24 and catches a
planted error 24 times out of 24. The 7B judging the same run puts `base_closed`
at 77%, worst of the three rather than best — a fourth gate reversal between
judges, and it needs no further comment.

Three things this does **not** establish, in decreasing order of how much they
should hold back a published gate number:

1. **Specificity was measured on the reference answers, which are one or two
   sentences.** Real arm answers are far longer and carry reasoning. A judge
   asked *"did this text assert claim X"* has more surface to match against in
   400 words than in 20, and the 1/24 gives no evidence about that regime. This
   is the gap that matters, and the arms above are exactly where it bites.
2. **The controls ran single-arm; the gate runs three.** Arm count moves scores
   (21.5), so the control rates were measured under a different prompt shape
   than the numbers they are being used to license.
3. n=22 is a proportion at n=22, and the current set is 24.

A re-run at n=24 under the 32B with the current rubrics addresses (3) and
supplies the answers needed to attack (1) directly — the arms it stores can be
fed back through the sensitivity construction to measure specificity on
answer-length text.

### 21.41 The n=24 gate run holds; the long-answer question does not close

The re-run, three arms, 32B judge, current rubrics. Against the stored n=22 it
is almost a replication:

| Arm | blunder n=22 | **blunder n=24** | errors/answer | correctness | all legal |
| --- | --- | --- | --- | --- | --- |
| `base_closed` | 41% | **46%** | 1.17 | 3.04 | 75% |
| `base_open` | 59% | **58%** | 1.58 | 2.08 | 71% |
| `base_cards_open` | 73% | **71%** | 1.83 | 2.04 | 62% |

Gate 1 **FAIL** (legality 75%, needs 95% — parsing is 100%, so this is the model
choosing unavailable plays, not the grammar). Gate 2 **PASS** (67% of positions
separate the arms, bar is 50%). Gate 3 **FAIL**, best arm 42% against a 25% bar.

`errors/answer` is the useful new column. The 32B fires 1.17–1.83 errors per
answer here against **1.12 on an answer carrying exactly one** and 2.75 for the
7B (21.40). These arms are not being sprayed. `error_contradiction` is **0 on
all three** — the judge never fires the whole list while also crediting half the
key points, so where it does fire everything, it also credits nothing, which is
at least coherent.

#### Specificity on position answers: confirmed

21.40 flagged that specificity was measured only on short reference answers.
For **positions that concern is now closed, because the answers are also short**
— median 41–59 characters, since an arm emits an action list rather than prose.
Taking every arm answer credited with **all** its key points and asking how
often it was nonetheless charged with an error: **1 of 22 (5%)**, against 4% for
the oracle. The instrument behaves the same on real answers as on the reference,
so the 46% gate failure is a statement about the model.

#### On the rules set it does not close, and the attempt is worth recording

Rules answers are a different regime: median **428–1,243** characters against
41–59. The same test on `gold_n99_judge3.jsonl` (the 32B — 8% key-point credit
identifies it against 21.31's table):

| answers crediting… | n | charged with an error | mean length |
| --- | --- | --- | --- |
| **all** key points | 21 | 3 (**14%**) | 968 chars |
| at least half | 31 | 11 (35%) | 954 chars |

14% looks like specificity degrading with length. **It is not evidence of that**,
for three reasons, and the third is the one that matters:

1. **The denominator is not a clean control.** The oracle *cannot* commit an
   error; a model answer that states every key point can still say something
   wrong alongside them. 14% is an upper bound on false positives, not a
   measurement of them.
2. **There is no length trend inside it.** Binned: 0% (n=2) under 400 chars, 40%
   (n=5) at 400–900, 7% (n=14) at 900+. If length were the driver the longest
   band would be the worst; it is the best. At these counts none of it resolves.
3. **Reading the three cases finds three different causes.**
   `gloss-summoning-sickness-rule` is charged with "extends the restriction to
   blocking" and "blocks every activated ability" while its text says the
   opposite of both — a **polarity false positive**, 21.37's mechanism intact on
   the 32B at 752 characters. `qa-alex-casts-solitude` genuinely never gives
   priority back to Player A — a **true positive**. `rg-22` is ambiguous.

So the honest position: **21.40's first caveat stands for the rules set.** The
32B's clean rates are established at 122 characters (positions) and 236 (rules
references, 0% at n=99 in 21.31), and untested at 1,000. Polarity is visible at
that length at low rate. Settling it needs a control that is definitionally
clean *and* answer-length — the reference answer expanded rather than a model
answer selected — which is a construction that does not exist yet.

**Recorded because the attempt failed, not despite it.** An n=21 upper bound
with a confound and no trend is the kind of number that becomes "false positives
rise to 14% on long answers" if only the headline survives.

### 21.42 Of four judges, only the 32B separates blundered from clean

This section was published twice with the wrong headline before the table below
was complete, and the drafts are described in 21.43 because the pattern that
produced them matters more than either wrong answer.

The measurement: same 24 positions, same claim-form rubrics, **single arm**, v3,
1,800 tokens. Two halves — an answer that **cannot** commit a listed error (the
reference), and one that commits **exactly one**, planted verbatim.

| Judge | fires at a **clean** answer | fires at a **1-error** answer | **separation** |
| --- | --- | --- | --- |
| Qwen2.5-7B | 18/24 (75%) | 24/24, mean 2.75 | 25 pts |
| Llama-3.1-8B | 14/24 (58%), mean 0.58 | 24/24, mean 1.00 | 42 pts |
| Qwen3-14B | *not measured* | 24/24, mean 1.00 | **unknown** |
| **Qwen2.5-32B** | **1/24 (4%)** | 24/24, mean 1.12 | **96 pts** |

Separation is `P(fire | error) − P(fire | clean)`, and it is the only number here
that means anything on its own. **Blunder rate is defined on `errors_made` being
non-empty**, so a judge firing at 58% of clean answers cannot support the metric
however well it catches real errors.

#### Sensitivity alone is worthless, and three judges prove it

All four judges score **24/24** on catching the planted error. Three of them
score a mean of 1.00–1.12, which reads as precision. Llama-3.1-8B's 1.00 is the
best number in that column — **and it fires at 14 of 24 answers that contain no
error at all.**

Ranked by sensitivity the four are indistinguishable. Ranked by separation the
32B beats the next best by 54 points. **The half of the control that discriminates
is the half that was missing**, and it was missing for every judge but the 7B
until this section was on its third draft.

#### What this does and does not establish

**Established:** every small judge tested fails, and they fail *differently* —
the 7B by spraying (2.75 errors at an answer with one, and it fires at 75% of
clean ones too), Llama by firing at more than half of clean answers while being
otherwise precise. The 32B is the only one that works.

**Not established:** that this is about size. Two models within a billion
parameters of each other sit 17 points apart, and Qwen3-14B's clean-answer rate
is **not measured** — its 1.00 sensitivity says nothing, as Llama demonstrates.
Three points, two sizes, and one of the three incomplete does not identify a
capacity threshold. What can be said is the weaker, true thing: **no judge under
32B has yet been shown to work, and the one that works fits in 18 GB.**

#### The 32B's other advantage is independent of all this

Discrimination on the **correctness** scale, from the four-arm rules calibration
(21.31) — a different measurement, not to be read against the table above:

| | credits the oracle | credits real answers | ratio |
| --- | --- | --- | --- |
| Llama-3.1-8B | 81% | 63% | 1.3× |
| **Qwen2.5-32B** | 90% | 8% | **11.2×** |

A judge crediting real model answers with 63% of a rubric it credits the
reference with 81% of is barely separating them, and every arm comparison in this
project is read off that separation. So the 32B wins both halves of the
instrument, for unrelated reasons.

#### One number that does not fit, and is now the priority

Llama scored **4%** false errors on the rules set at four arms (21.31) and
**58%** on positions at one arm. **Two variables moved at once** — arm count and
rubric type — and 21.5 forbids reading four-arm and single-arm numbers against
each other, which is exactly what makes this unresolved rather than a
contradiction.

It has to be settled before anything here is built on, because one of the two
possibilities is corrosive: if **arm count** drives it, then every single-arm
number in the table above is suspect *including the 32B's 4%*, and the judge
matrix is measuring the harness. If **rubric type** drives it, positions are
simply harder to judge than rules questions and the position gates need a wider
margin. Running Llama on the rules set single-arm isolates it.

### 21.43 Three wrong headlines from the same defect: half a control pair

Sections 21.40 and 21.42 were both published with conclusions that the next
measurement overturned, and STRUCTURAL_AUDIT.md was edited twice to match. The
three claims, in order:

| # | Claim | Written from | Overturned by |
| --- | --- | --- | --- |
| 1 | "`errors_made` is broken" (21.37) | the 7B alone | the 32B at 4% |
| 2 | "error detection needs a bigger judge" (21.40) | 7B vs 32B specificity | Llama-8B at 4% on rules |
| 3 | "an 8B does it better than the 32B" (21.42) | Llama **sensitivity** alone | Llama at 58% specificity |

Each was a defensible reading of what was on the table at the time. Each was
wrong. And the shape is identical every time: **a judge quality number reported
from one side of a control that only means something as a pair.**

Claim 3 is the sharpest case because the tooling actively invited it.
`calibrate_judge.py --sensitivity` prints a clean, confident report — 100% hit
rate, mean 1.00, zero quote drops — with no indication that the number is
uninterpretable without its other half. The report was accurate. The conclusion
drawn from it was not, and nothing in the output pushed back.

**So the fix is not discipline, it is the tool.** A judge quality report that can
be obtained one-sided will be read one-sided, by me and by anyone else. The
paired report becomes the only way to get either number, and the headline it
prints is the separation, which cannot be computed from one half at all.

This is the same lesson as the trap in CLAUDE.md about a rate measured on one
judge being a statement about that judge — applied one level up. **A rate
measured on one half of a control is a statement about that half.**

### 21.44 The matrix, completed by the paired tool — and a 14B ties the 32B

`calibrate_judge.py --judge-report` (21.43) run over every cached judge:
positions, n=24, one arm, v3, 1,800 tokens, both halves in one pass.

| Judge | GB | fires at **clean** | mean | fires at **1-error** | mean | **separation** |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | 4.0 | 18/24 (75%) | **1.96** | 24/24 | 2.75 | +25% |
| Llama-3.1-8B | 4.2 | 14/24 (58%) | 0.58 | 24/24 | 1.00 | +42% |
| Qwen3-14B | 7.8 | **0/22 (0%)** | 0.00 | 23/23 | 1.00 | **+100%** |
| Qwen2.5-32B | 17.6 | 1/24 (4%) | 0.04 | 24/24 | 1.12 | +96% |

The 7B and 32B reproduced their earlier numbers exactly, which is the check that
the promoted tool is the same instrument. And the recovered column is worth its
own sentence: **the 7B fires 1.96 errors at an answer that contains none.** That
was stored as a boolean before and could not be recovered without re-running.

#### Qwen3-14B does not beat the 32B — it ties, and it ties by not grading

Read at full coverage the 14B wins, +100% against +96%. But it graded 22 of 24
clean answers and the 32B graded all 24, so those are rates over different sets —
the exact error 21.38 was written about. Intersecting:

| Matched on the 22 every judge graded on **both** halves | clean | planted | separation |
| --- | --- | --- | --- |
| Qwen2.5-7B | 16/22 | 22/22 | +27% |
| Llama-3.1-8B | 12/22 | 22/22 | +45% |
| **Qwen3-14B** | **0/22** | 22/22 | **+100%** |
| **Qwen2.5-32B** | **0/22** | 22/22 | **+100%** |

**Identical.** Not close — the same numbers. The 32B's entire false-positive rate
is the two positions excluded from this set, and one of them is
`pos-combat-math-0002`: the polarity case from 21.40, where the answer says
"cast first, then attack" and the error reads "Attack first and cast Lightning
Strike afterwards". **The single hardest position in the set is one the 14B
failed to grade.**

So the +100% is not evidence the 14B is more precise. It is 0/22 with the one
case that defeats the 32B removed, and nothing says it would have survived it.

#### What this actually buys: a second judge that works

The practical result is large anyway. **A 7.8 GB judge matches a 17.6 GB judge on
error detection** across every position both can grade, which is the first time
any judge but the 32B has cleared this control. Until now the honest position was
"one working judge, so no agreement number is available" — Section 21.42's open
item and the last untested trip-wire in STRUCTURAL_AUDIT.md.

That unblocks kappa. Two judges that both pass the paired control now exist, so
the +0.24 measured with a judge since shown to invent blunders 40% of the time
can finally be re-measured between calibrated instruments.

**The 32B stays the default**, on coverage. It grades 24 of 24; the 14B grades
22, it is a reasoning model whose `<think>` block competes with the JSON for the
token budget, and §21.25 measured it at roughly **60× the wall time**. A judge
that skips 8% of a set — and skips the hard end of it — cannot be the primary,
whatever it scores on what it does grade.

**Caveat on the family question.** Qwen3-14B is a different *generation*, not a
different vendor. Every judge that passes this control is a Qwen, and every
`base` arm is a Qwen. Self-preference is still untested and still needs
Mistral-24B or Gemma-27B.

#### And Qwen3-14B cannot sustain a three-arm grading

Tying the 32B at one arm did not make it usable as the second judge. Rescoring
the n=24 gate run — three arms, the configuration every published gate number
uses:

| `--judge-max-tokens` | arms graded |
| --- | --- |
| 600 (the default) | **0 / 72** |
| 4,000 | **45 / 72 (62%)** |

The failure is per *position*, not per arm: exactly 15 of 24 positions graded on
all three arms and 9 failed on all three, so it is the call that fails, not
individual candidates within it.

The mechanism is arithmetic. A reasoning model pays a large **fixed** cost in
`<think>` before emitting anything, and the JSON it must then produce scales
with arm count. At one arm and 1,800 tokens it grades 22–23 of 24; at three arms
even 4,000 leaves it short. Buying more budget would work eventually and costs
~60× the wall time of a non-reasoning judge (21.25) for a number two other
models produce directly.

**So the practical rule is narrower than 21.44's table suggests: Qwen3-14B is a
usable single-arm control judge and not a usable gate judge.** Both halves of
that sentence come from the same model on the same positions; only the arm count
differs, which is 21.45's effect appearing as a coverage limit rather than as a
rate.

Twice now the guard is what surfaced this rather than a wrong number: the
first run printed three bold gate FAILs computed from zero gradings before
`_write_report` learned to withhold below 90% coverage.

### 21.47 Kappa is +0.47, and the two judges reverse an arm

**Self-preference, closed.** `Mistral-Small-24B` — a different vendor, not a
different generation — passes the paired control at full coverage:

| positions, n=24, one arm | clean | 1-error | mean | caught the planted one | **sep** |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | 1/24 | 24/24 | 0.04 / 1.12 | 100% | +96% |
| **Mistral-24B** | **0/24** | 24/24 | 0.00 / 1.08 | 96% | **+100%** |

So the 32B's numbers are not a Qwen judging Qwen flattering itself. *(Gemma-2-27B
could not run at all: its chat template raises `System role not supported` and
all three judge prompts open with a system message. `eval.apply_judge_template`
now folds `system` into the first user turn on the **judge** paths only —
`build_prompt` is left alone, because reshaping what the model under test sees
invalidates the adapter, which is Section 8.7.)*

**Kappa, finally measurable.** Two judges that both pass the control, rescoring
byte-identical stored answers — the last open trip-wire in STRUCTURAL_AUDIT.md:

| | |
| --- | --- |
| Cohen's kappa on the blunder call | **+0.47** (raw 72%, chance 48%) |
| target | ≥ 0.60 |
| previous measurement | +0.24, with a judge since shown to invent blunders 40% of the time |
| correctness correlation | r = +0.71 |

**Roughly doubled, and still short.** Replacing a broken judge moved it from
+0.24 to +0.47; it did not reach the bar.

#### Passing the control does not make two judges agree

This is the result, and it is uncomfortable:

| Arm | 32B blunder | Mistral blunder |
| --- | --- | --- |
| `base_closed` | **46%** ← best | 38% |
| `base_open` | 58% | 38% |
| `base_cards_open` | **71%** ← *worst* | **33%** ← *best* |

**`base_cards_open` is the worst arm under one calibrated judge and the best
under the other, on identical answers.** Gate 3 returns FAIL under both, but the
best-arm *identity* flips and the rates sit 16 points apart against a 25%
threshold — so the agreement is about how far the model is from passing, not
about the metric. The report's own 21.19 stability check fires on exactly this.

The paired control validates a judge's error detection **in isolation, against
constructed extremes**. It does not make two such judges agree about real
answers, and 21.44's +96 / +100 should not have been read as implying it would.

#### Where the disagreement lives, and why

The 20 disputed calls are not spread evenly. **Ten of the fifteen listed are
`blocking` positions**, the rest combat math and mulligan.

That has a mechanism, and the first human verdicts found it before this run did.
`pos-blocking-0002`, arm `base_open`, answer **`PASS`**:

| | |
| --- | --- |
| 32B | fired errors **1, 2, 3** — every listed error |
| human | fired **none** |
| probably true | **1 only** |

Errors 2 and 3 name specific mis-blocks ("Llanowar Elves should block Grizzly
Bears"). `PASS` blocks nothing, so it commits neither — the judge is wrong twice,
the 21.26 signature caught in the open. But error 1 is "Take 2 from Elite
Vanguard rather than trading Llanowar Elves", and passing does exactly that, so
the human is wrong once.

**Both errors have the same cause.** Section 21.35 rewrote `common_errors` as
*claims* so a judge could ask "did the answer assert this?" — which works for
rules questions, whose answers are prose. **Position answers are action lists.**
`PASS` asserts nothing, so a claim-matching judge has nothing to match and falls
back to flagging everything, while a careful reader correctly reports that no
claim was asserted. The mismatch bites hardest on `blocking`, where the errors
name specific creature pairings, which is where the disputes are.

#### This implicates the control itself

The sensitivity half plants the claim as **literal prose** — *"Llanowar Elves
should block Grizzly Bears. That is the play here."* — which resembles no arm
answer on this set. So **+96 and +100 were measured on prose candidates while
the gate scores action lists.** The control is sound for what it tests and does
not transfer to the gate as directly as 21.44 implied. Same shape as 21.41's
open caveat: a control measured in one regime, quoted in another.

#### What changed as a result

The adjudication form asks the right question now. It said *"which of these does
it commit?"*, which invites reading the claim as text; it now says **"which of
these mistakes does the answer make?"** with *"judge the play, not the wording —
an action list like `PASS` still takes 2 from the Vanguard even though it never
says so."*

And it gained a third verdict beyond hit / not-hit: **"bad, but not for any
reason above"**, stored as `not_covered`. `PASS` and `PLAY Plains ×6` are
useless answers that commit no listed claim; recording them as simply "no error"
would score a judge as correct for missing them. That flag measures **rubric
coverage**, which no judge number can, and the first eight human verdicts are
the reason it exists.

### 21.45 Arm count and rubric type both inflate false errors, unequally

Section 21.42 left one number unexplained and flagged it as corrosive: Llama
scored **4%** false errors on the rules set at four arms and **58%** on
positions at one arm. Arm count and rubric type moved together, and if arm count
were the cause then every number in 21.44's matrix — all single-arm — would be
measuring the harness.

Holding the **record set fixed** separates them. The stored four-arm calibration
already contains the answer for one cell: restricted to the same 42 claim-form
records the single-arm report covers, Llama fires on **0/41**, mean 0.00. Then
running the single-arm report on those same records:

| Llama-3.1-8B, fires at a clean answer | rules (same 42 records) | positions (24) |
| --- | --- | --- |
| **4 arms** | **0/41 (0%)**, mean 0.00 | *not measured* |
| **1 arm** | **7/42 (17%)**, mean 0.17 | **14/24 (58%)**, mean 0.58 |

**Both effects are real and they are not the same size.**

- **Arm count: +17 points.** Same records, same judge, same rubric; only the
  number of candidates in the prompt changed. Dropping from four arms to one
  takes a perfectly clean judge to 17%.
- **Rubric type: +41 points on top.** Same judge, same single arm; rules
  questions to board positions. This is the larger effect by more than double.

So **positions are genuinely harder to judge than rules questions**, and that —
not the harness — is most of the 58%.

#### What this does and does not invalidate

**The judge matrix stands.** Every judge in 21.44 was measured the same way —
one arm, positions, v3, 1,800 tokens — so the comparison holds and the ranking
is unaffected. What is *not* transportable is any absolute rate: a single-arm
false-positive number is roughly 17 points pessimistic against a four-arm one.

**And that direction favours the published gate numbers.** The controls run at
one arm; `pos_n24_32b.jsonl` runs at **three**. Fewer arms means more false
positives, so the 32B's true rate in the setting the gates were actually
measured in is *lower* than the 4% the control reported, not higher. Section
21.41 licensed the 46% Gate 3 failure on a 4% control; that license is stronger
than it looked, not weaker.

**Two limits, stated.** The four-arm calibration predates the provenance fix so
its token budget is not recorded; the default is 900 (≈225 per arm) against the
single-arm run's 1,800. Budget therefore differs alongside arm count — but it
points the wrong way to be the explanation, since *more* room per arm produced
*worse* specificity. And the fourth cell, positions at four arms, is unmeasured:
`calibrate_judge.py`'s four-arm mode reads gold-shaped records only, so
completing the square symmetrically would need work the three cells already make
unnecessary.

### 21.46 Run 4 trained; validation plateaus at 0.9 epochs and the final weights are not the best

The retrain that has been ready since 21.13 and never launched.
`data/datasets/verified` — 1,001 train / 111 valid — under
`configs/phase1_lora_v4_verified.yaml`, `iters: 1001` × `batch_size: 2` =
**2.00 epochs**. Contamination audited **before** the run, as the rule requires:
no exact matches across 1,112 training lines × 568 eval questions, nothing above
75% token overlap, exit 0.

| iter | val loss | |
| --- | --- | --- |
| 1 | 4.003 | |
| 150 | 1.776 | |
| 300 | 1.632 | |
| **450** | **1.534** | plateau begins (**0.9 epochs**) |
| 600 | 1.546 | |
| 750 | 1.553 | |
| **900** | **1.527** | best |
| 1001 | 1.542 | final weights |

1h05m wall, peak memory **7.4 GB of 36** — the guardrail was never close to
binding, as in every run here.

**Two things worth carrying into the review.**

The curve is **flat after iter 450**: from there to the end val loss moves
**+0.008**, wandering inside a ±0.015 band. The whole second epoch bought
nothing measurable. That is the same arithmetic error `phase1_lora_v2.yaml`
carried in comment form — a config asserting an epoch count nobody had checked
against the loss — arriving from the other direction: the epochs ran, and were
not needed.

And **`adapters.safetensors` is not the best checkpoint.** The final weights sit
+0.015 above iter 900. `mlx_lm.lora` writes the last iteration, not the best, so
the file the eval default would pick is the worse one. Checkpoints every 100
iterations are on disk; `0000900_adapters.safetensors` is the candidate.

**None of this is a capability result.** Validation loss is not the deliverable
and has already been shown not to predict the thing that is: run 3 removed the
under-training confound, moved `finetuned_rag` by +0.04 under one judge and
+0.02 under another, and was **not promoted** because control arms moved up to
0.23 on judge variance alone (18.3). Run 4 gets the same treatment — spot-checks
and a scored eval decide it, not the curve above.

Stamped immediately (`prompt_fingerprint 30badae98696`, verified against the
training file rather than asserted), so `eval.py` will refuse it if a prompt
changes underneath it — Section 8.7's failure, made an error message.

### 21.48 Length is ruled out: the 32B holds +100% at 3.6× the answer length

21.41 left one caveat open and named the construction that would close it — a
control that is *definitionally* clean **and** answer-length. `--pad-clean`
builds it: take the reference answer and append the **verbatim Comprehensive
Rules text of the rules the record already cites**. That adds several hundred
characters which cannot introduce an error, because it is the rulebook quoted on
the rule the answer is about. The planted-error half is left untouched, so
length is the only variable that moves.

Rules gold set, 42 usable records, one arm, v3, `Qwen2.5-32B`. Every one of the
42 clean answers padded — a record whose citation does not resolve returns
unchanged and would have been counted as padded otherwise, which is why the
count is printed.

| clean half | median chars | max | fires at clean | fires at 1-error | **separation** |
| --- | --- | --- | --- | --- | --- |
| unpadded | 230 | 837 | 0/42 | 42/42, mean 1.02 | **+100%** |
| **padded** | **838** | **4,484** | **0/42** | 42/42, mean 1.02 | **+100%** |

**Identical, in every cell.** A 3.6× increase in median length — and a longest
answer of 4,484 characters, well past the ~1,000 real rules answers run to —
moves nothing. The judge fires at a clean answer zero times either way.

So 21.41's 14% was never a length effect. That number came from *model* answers
that credited every key point, which can state everything right and still say
something wrong alongside it; 21.41 said so and declined to publish it as
specificity. This is the measurement it deferred to, and it comes back flat.
**21.40's remaining caveat is closed for the rules set.** Polarity false
positives (21.37) are a separate mechanism and are not addressed here — one of
21.41's three cases was one, at 752 characters.

Worth noting what the test could **not** cover. 57 of 99 records are unusable
because their `common_errors` are authored in behaviour form rather than as
claims, so this is a selected subset, not a sample. The gold-set conversion was
resolved as "none" on the ground that `common_errors` reaches no training
script — correct as a training decision, and this is its cost: the sensitivity
control is permanently capped at 42 of 99 records.

### 21.49 Human adjudication found a harness bug at n=12, and Gate 3 is measuring the wrong thing

B1 exists to answer a question nothing else could: judge-vs-judge agreement says
nothing about correctness, and the paired control tests only the two extremes —
an answer that cannot err and one carrying exactly one planted error. Real
answers live in between. Twelve verdicts in, the set has already returned two
things, and neither is the accuracy number it was built for.

#### A reviewer's note found a legality gap the whole harness had

Against `pos-blocking-0003::base_closed` the note reads *"it tries to use fog
bank to block twice."* The harness had scored that answer `all_legal=True`:

```
BLOCK Fog Bank -> Serra Angel
BLOCK Fog Bank -> Grizzly Bears
PASS
```

Both actions appear in the position's `legal_actions`, so the enumerated-set
check matched each and passed the sequence. But `legal_actions` lists
**alternatives** — every block legal *on its own* — and one creature blocks at
most one attacker (509.1a).

This is worse than the class `rule_illegalities` was written for. A second land
drop fails because it **is not** in the list; a second block passes because it
**is**. The check that should catch it is the one that was fooled.

**39 of 1,516** stored position answers contain a double block, concentrated in
`base_closed` on blocking boards — which follows, since the closed arm picks
from the enumerated list. Re-derived on the n=24 run, no model needed:

| Arm | legality was | **now** | flipped |
| --- | --- | --- | --- |
| `base_closed` | 18/24 (75%) | **15/24 (62%)** | 3 |
| `base_open` | 17/24 (71%) | 17/24 (71%) | 0 |
| `base_cards_open` | 15/24 (62%) | 15/24 (62%) | 0 |

Gate 1 needs 95% and already failed at 75%. It fails harder at 62%: no published
verdict flips, and the error ran in the safe direction.

**The commit that fixed this reported the sweep as "59 of 1,292" and that does
not reproduce.** Re-running it: 1,516 stored position answers, of which 962 were
written `all_legal=True`, of which **43** fail today — 39 for the double block
and **4** for a *second land drop*, a check `rule_illegalities` already had.
Those four are answers stored before it existed (`PLAY Swamp / PLAY Swamp /
PASS`, `PLAY Swamp / PLAY Temple of Silence / PASS`), which the same reviewer
independently flagged in two other notes. The per-arm table above reproduces
exactly, so nothing published moves; the denominator in the commit message is
simply wrong, and a count nobody re-ran is the shape of "a number that never
was" (21.13) arriving again.

The rule is deliberately **not** a rules engine — no menace, banding, or "may
block an additional creature", because no position in the set grants one; adding
such a card means this check needs the exception. `test_actions.py` gains 5
assertions in **both** directions (89 total), since a check that also fires on
two *different* blockers would hide every real result.

The general shape recurs and belongs with the traps: **a validity check that
consumes a list of alternatives as though it were a list of permissions.** It
cannot see a constraint that only exists *between* two individually-legal
choices.

#### 8 of 9 verdicts say the rubric does not cover the answer

Under the v2 wording, `not_covered` is checked on **8 of 9**. Reading the notes,
they split into two kinds and neither is a strategy error:

- **Illegal**: two lands in one turn (×2), six Plains with an empty hand, a
  redundant second Doom Blade, the double block above.
- **Legal and useless**: `PASS` alone (×3), half of a two-step line.

`common_errors` enumerate *strategy* blunders — playing the wrong card, blocking
the wrong creature. An answer that passes its turn commits none of them and is
not a good answer. So on this model **blunder rate is measuring something the
answers mostly do not do**, and Gate 3's 42–46% is diluted by answers that never
reached the point of having a strategy to get wrong.

The consequence is an ordering, not a threshold: **Gate 1 has to pass before
Gate 3 means anything.** That reordering came out of nine hand-written notes, not
out of any judge number — which is the argument for B1 continuing, independent
of the accuracy estimate it was launched to produce.

One note is worth flagging back: on `pos-mulligan-0001` the reviewer wrote
*"there isn't an explanation exactly why it mulliganed so I chose the most likely
reason."* That is exactly what the **genuinely ambiguous** box is for — used
there, the guess stays out of the precision numbers instead of counting as a firm
verdict.

### 21.50 Run 4 trained on one prompt shape and would be evaluated on two

Before spending the GPU-hours to evaluate run 4, one thing in its stamp is
worth reading. `stamp_adapter.py --dataset` records not just the fingerprint but
**which system prompts the training file actually contained**:

| dataset | adapter | `SYSTEM_PROMPT` | `RAG_SYSTEM_PROMPT` |
| --- | --- | --- | --- |
| `data/datasets` | v2 (published) | 1,679 | **1,292** |
| `data/datasets/verified` | **v4** | 1,112 | **0** |

Run 4's training set contains **zero** RAG-shaped examples. `eval.py` generates
four arms, and `finetuned_rag` builds its prompt with `RAG_SYSTEM_PROMPT` and a
`Rules text:` block in the user turn — a shape the v4 weights have never seen.
Constructed rather than reasoned about:

```
TRAIN  system = SYSTEM_PROMPT       user = 'Does X have haste?'
INFER  system = RAG_SYSTEM_PROMPT   user = 'Rules text:\n702.10b...\n\nQuestion: Does X have haste?'
```

**This is Section 8.7 exactly, and every fingerprint matches.** 8.7 is usually
retold as "someone edited a prompt and invalidated an adapter", so the guard
built for it hashes the prompt *definitions* and compares them at eval time.
That guard passes here — nothing was edited. Its mechanism was never the edit;
it was a model trained on bare question→answer while inference wrapped the
question in retrieved rules text. A training set that only ever uses one of the
two shapes reaches the identical state with the prompts untouched.

**The omission itself is deliberate and correct.**
`build_sft_verified.build_examples` attaches no retrieved context on purpose:
these questions carry their own card references and the verified answers cite
their own rules, so injecting retrieval would train the model to expect a context
block the answer does not depend on. The bug is not in the dataset. It is that
nothing connected that decision to the arm list downstream of it.

**What it would have cost.** `finetuned_rag` is the arm Section 6.1 names as the
intended final architecture, so it is the row that gets quoted. Had this run
blind and that row come back low, the available reading — "the decontaminated
verified set did not help" — is about the training *data*, plausible, and wrong.
The true statement is about the training *shape*. Same failure family as 21.5's
arm count and 21.44's token budget: a harness property that moves the headline
number while wearing the shape of a result.

`stamp_adapter.unseen_arms` now answers the question the fingerprint cannot —
*did the weights ever meet this shape* — and `eval.py` calls it in two places:
before generation, beside the fingerprint line, and **again in the report body**,
because a warning printed at the start of a run is a warning nobody reads next
to the number it qualifies at the end of one. Verified by running: on v4 it prints with
the fingerprint matching; on v2-best it stays silent.

Two details it gets right on purpose, both tested. A count of **0** is an
absence, so `in counts` would pass straight through the case this exists for.
And `base`/`base_rag` are never flagged — they never load the adapter, and a
warning that fires on every run is one the reader learns to skip.

It is silent when the stamp carries no `dataset_prompt_counts`, which a stamp
written without `--dataset` does not. That is the same distinction the stamp
itself is built on: `--dataset` is what makes a stamp evidence rather than an
assertion, and a check with no evidence should say nothing rather than guess.

**Run 4 is still worth evaluating** — three of four arms are unaffected, and
`finetuned` (the matched-shape arm) is the honest test of whether decontaminated
data helped. `finetuned_rag` gets measured too, and its row now carries the
reason it is not a data result.

### 21.51 The one path where the judge is guaranteed to differ was the one that did not record it

Preparing the second judge for run 4. `--rescore-from` is what the two-judge
rule is built on — it re-judges stored answers so the judge varies and *nothing
else* does — so it was worth reading before trusting its output.

`rescore()` assigns nothing at row level. Checked by AST rather than by eye,
because "does this function write that key" is exactly the question reading
answers wrongly:

```
subscript assignments inside rescore():
   data['citation_score']   data['correctness']    data['errors_made']
   data['judge_note_v2']    data['points_hit']     data['points_total']
   data['scored_by']
row-level ('r[...]') assignments: NONE
```

And `carry_diagnostics` copies `RUBRIC_DIAGNOSTICS` — `scoring`, `quote_drops`,
`all_errors_fired`, `error_contradiction` — which does not include
`judge_model`. So rescoring a 32B-judged run with Mistral produced a file whose
every row still read `"judge_model": "mlx-community/Qwen2.5-32B-Instruct-4bit"`.

**The report header was right the whole time.** It derives the judge from
`args`, and 21.20 already hardened it twice — it stopped claiming "the v2 judge"
when V3 had run, then stopped claiming "V3" when `--judge-prompt v4` had. Both
fixes went to the *report*. The data file was never revisited, and the data file
is what `--compare` reads, what a later rescore reads, and what outlives every
report.

Three things make this worse than the absence 21.40 fixed:

1. **It is a wrong value, not a missing one.** A missing key reads as unknown
   and prompts a question. `judge_model: Qwen2.5-32B` on a Mistral-judged row
   reads as a fact, and it is exactly the fact the two-judge protocol turns on.
2. **It fires precisely where it matters.** A rescore whose judge equals the
   source's judge is pointless; the whole reason to run one is that the judge
   differs. So the field is wrong on 100% of the runs anyone would care about,
   and correct only on the ones nobody makes.
3. **It survives a `--compare`.** Two files, both labelled with the same judge,
   compared to see whether the judges agree.

`rescore()` now stamps `judge_model`, `judge_prompt`, and `rescored_from` on
every row it rewrites. The test is an AST check, not an execution — `rescore`
loads a judge model and the suite must run without a GPU — asserting the
property that broke: which row-level keys the function assigns.

**And the test for 21.50 had the same class of bug, found by running it from a
different directory.** It passed `models/mtg-rules-adapter-v4` as a cwd-relative
path, so from `/tmp` the stamp did not resolve and `unseen_arms` returned `[]` —
*"nothing to warn about"*. A check whose failure mode is silence must be
anchored to `REPO_ROOT`, which is what `REPO_ROOT` is for, and running the suite
from somewhere else is how that gets caught. Both tests now use it.

### 21.52 What B3 has to decide against: the judge moves Gate 3 further than the bar is wide

B3 is "with the instrument's error rate now known, is 25% still the right blunder
bar?" This is the evidence, and it is stored data — no model was run.

Every position run on disk turns out to be the **same generated text**: six
judgings of one set of 72 answers (24 positions × 3 arms), byte-identical across
all six. So there is one experiment here, not six, and the judge is the only
thing that varies. That makes it the cleanest available measurement of what the
judge alone does to the gate metric.

#### Blunder rate on identical answers, by judge

Restricted to judges that pass the paired control (21.44), graded answers only:

| Arm | Qwen2.5-32B | Mistral-24B | **spread** |
| --- | --- | --- | --- |
| `base_closed` | 46% | 38% | 8 |
| `base_open` | 58% | 38% | 21 |
| `base_cards_open` | **71%** | **33%** | **38** |

**Gate 3's bar is 25 points wide. On `base_cards_open`, swapping one calibrated
judge for another moves the metric 38.** Not the model, not the prompt, not the
arm count — the same bytes, read by two judges that both separate blundered from
clean at ≥ +96%. This is 9.9 and 16.12 again, now measured against the threshold
it has to be compared with rather than against another arm.

Qwen3-14B is **excluded, not quoted**: it graded 45/72 (62%), below the coverage
guard, exactly as 21.44 predicted for a reasoning judge at three arms.

#### Gate 3 partly re-measures Gate 1

Legality comes from the parser and is judge-independent, so it can be crossed
with the blunder call:

| judging | blunder \| legal | blunder \| illegal | difference |
| --- | --- | --- | --- |
| Qwen2.5-32B | 25/47 (53%) | 17/25 (68%) | **+15** |
| Mistral-24B | 12/47 (26%) | 14/25 (56%) | **+30** |

An illegal answer is substantially more likely to be charged with a blunder.
That is 21.49's reviewer notes as a rate: the judge is charging *unplayable*
answers with strategy errors. Since Gate 1 already gates on legality, Gate 3
applied to all answers counts part of Gate 1 a second time — which is the
mechanical form of "Gate 1 has to pass before Gate 3 means anything."

#### The analysis walked into this repo's own trap first

The first version of both tables scored `bool(errors_made)` — and `errors_made`
is `None` when the judge did not grade an answer, which `bool()` renders as
**False, i.e. clean**. With 27 of Qwen3-14B's 72 ungraded, its blunder rates came
out 33/42/38% instead of the true **53/67/60%**: a 20-point deflation that made
the harshest judge on what it actually graded look like the most lenient of the
three, and would have been quoted as a judge-leniency finding.

`eval_positions.py` does **not** have this bug — `blundered` is explicitly `None`
when ungraded and every denominator filters on it, with a comment saying why. The
production path was hardened; a fresh analysis over the same files was not,
because reading a field directly bypasses the place the care lives. That is the
"valid JSON in an unexpected shape, read as absence" trap arriving for a third
time, and the first two were also *plausible* numbers rather than obvious
failures.

#### What this leaves B3

Setting a number is not the useful move here. Three findings constrain it:

1. **A bar narrower than 38 points cannot be defended on one judge.** Either
   Gate 3 reports two judges and passes only if both agree — which 21.19's
   report already flags for — or the bar has to sit outside the judge spread,
   which at 38 points is most of the scale.
2. **The metric should condition on legality**, or it re-measures Gate 1.
3. **The two calibrated judges reverse the ranking outright**, which is what the
   38-point spread actually is:

   | judge | best → worst |
   | --- | --- |
   | Qwen2.5-32B | `closed` 46% < `open` 58% < **`cards_open` 71%** |
   | Mistral-24B | **`cards_open` 33%** < `closed` 38% < `open` 38% |

   `base_cards_open` is the **worst** arm under one and the **best** under the
   other, on identical bytes. So this is not a bar that can be tightened or
   loosened into usefulness — at the moment Gate 3 does not agree with itself
   about which arm is winning. That is 9.9 and 16.12 a third time, and 21.47's
   "two calibrated judges reverse an arm" showing up in the gate metric rather
   than in kappa.

### 21.53 The coverage guard fired on half the runs, and the half it missed is the common one

Auditing the report path of the run that was generating, rather than waiting to
read its output. Both writers assemble a table of per-arm means; only one warns
when the judge failed to grade the set.

```
rescore  calls coverage_lines: True
main     calls coverage_lines: False      <- the first-pass writer
```

21.14's V4 run scored **14 of 99** and printed four confident-looking averages
over the 14. The guard written in response — report the unjudged count, and above
20% tell the reader to read nothing into the means, because a judge's parse
failures track rubric size and the survivors are a subset *selected* by rubric
size — went into `rescore()`. The first-pass writer never got it.

**That is backwards from where it is needed.** A rescore re-reads answers that
were already graded once, so its coverage is usually inherited and fine. The
first pass is where a judge meets a rubric for the first time and runs out of
tokens — it is where 21.14 happened, where 21.39's unwrapped JSON happened, and
where 21.44's 0/72 happened. The guard was absent from every one of those paths
and present on the one that mostly does not need it.

Third instance of one shape: **a hardening applied to one of two writers.**
`carry_diagnostics` (both writers enumerated diagnostics by hand, neither listed
`scoring`), 21.51 (`judge_model` stamped by the first-pass writer, not by the
rescore), and now this — and note the direction flips each time, so "check the
other writer" is the rule rather than "check `rescore`". It is now one function,
`coverage_lines`, called by both.

**The test asserts the structure, not just the arithmetic.** The arithmetic was
never wrong; the second caller was missing, and no test over inputs and outputs
can see that. So the test parses `eval.py` and asserts both `main()` and
`rescore()` call it — verified by deleting a caller and watching it fail, since a
check that only ever passes is one this repo has shipped before.

**Not a correction to any published number.** Every archived rules run has
uniform per-arm coverage — 0 or 1 answers apart — because `judge_batch_rubric`
grades all arms in one call, so a question is either graded for every arm or for
none. The means were over the same questions in every case. What was missing was
the sentence saying *how many* questions that was.

### 21.54 The judge-agreement report identified its judges by filename

Continuing the audit of report writers rather than waiting on the run. Three
provenance guards should hold everywhere — name the judge (21.40), state
coverage (21.14), state the arm count (21.5). Checked across all six writers:

| writer | names the judge | arm count | coverage |
| --- | --- | --- | --- |
| `eval.main` / `eval.rescore` | yes | yes | yes |
| `eval_positions._write_report` | yes | yes | yes |
| `calibrate_judge.run_judge_report` | yes | yes | yes |
| **`eval.compare_judges`** | **no** | yes | no |
| **`eval_positions.compare_judges`** | **no** | yes | no |

Both judge-agreement writers — and the judge is the *entire subject* of those
two reports. They headed their output ``` `pos_n24_32b.jsonl` vs
`pos_n24_mistral24b.jsonl` ```, which is precisely the record 21.40 found
insufficient when `cards_n100_judge2.md` carried its most important variable in
its filename. Every row has carried `judge_model` since; the report that most
needed it was reading the name off the path.

**And nothing stopped comparing a file with itself.** The shared-id check
passes, the arm-overlap check passes, and the output is:

```
- **Cohen's kappa on the blunder call: +1.00** (raw agreement 100%, chance 51%)
- correctness correlation: r = +1.00
```

under a heading that says *Position Judge Agreement*. The judge is deterministic
(that is a documented property relied on elsewhere), so this measures
determinism and reports it as agreement. A perfect kappa is the most quotable
number this report can emit and it was the one case where it means nothing.

`judges_of` and `assert_two_judges` now sit in `eval.py` — one definition,
called by both comparators, because this exact hardening has now landed on one
of two writers four times. An unrecorded judge is *stated* rather than omitted:
archived runs predate the field, and "unrecorded" is a fact about the evidence
where a blank looks like nothing was wrong.

#### A third failure the gate comparison could not see

The verdict column compares PASS/FAIL. On the real 32B-vs-Mistral comparison:

```
| 3 — blunder ≤25% | 42% (base_closed) FAIL | 26% (base_cards_open) FAIL | agree |

*(That row is how the gate read when this section was written. Blunder rate is
measured and no longer gated — Section 21.122 — so a current report prints the
two rates and their spread instead of two verdicts.)*
```

It says **agree** — both FAIL — while the two judges name *different best arms*.
21.19's guard catches "agrees but far apart" and fires here too (16 points
against a 25-point bar). Neither catches "agrees about the verdict, disagrees
about which arm is winning", which is 21.52's reversal in the place it would
actually be read. If this gate ever passes, the two judges would be passing
different arms. Both warnings now print.

**Nothing published moves.** Every archived comparison did vary the judge; the
self-comparison is a case that was reachable, not one that was taken. What
changes is that the report now says which two models it compared, in the report
whose only job is to compare two models.

### 21.55 The "genuinely ambiguous" checkbox was collected, stored, and read by nothing

Auditing the code that will consume B1's 60 verdicts, since that is where eight
hours of human time turns into a number.

The adjudication form asks, on every task: *"Genuinely ambiguous — I could argue
it either way."* `rubric_server.py` stores it as `unsure` on the submission,
`--ingest-submissions` writes it to `judge_adjudications.jsonl`, and
`score_run()` reads `errors_present` **and nothing else**.

So a reviewer who used the box changed nothing. Their explicit "this is a coin
flip" was scored as a firm verdict, identical to one they were certain of.

**That inverts the point of the whole exercise.** Human adjudication exists here
because judge-vs-judge agreement says nothing about correctness — it is meant to
be the tiebreaker (Track B1). A tiebreaker that silently includes the cases the
tiebreaker called a coin flip is not one, and it is worse than a smaller sample
because the contamination is invisible in the output.

It has not corrupted anything yet — all 9 v2 verdicts have `unsure: false`. It
was about to: the box is exactly what `pos-mulligan-0001` needed, where the
reviewer wrote *"there isn't an explanation exactly why it mulliganed so I chose
the most likely reason."* Telling someone to use a control that does nothing is
worse than not having it.

`score_run` now excludes `unsure` and reports three separate reasons the
denominator is smaller than the queue — ambiguous, ungraded by this judge, and
`not_covered` — beside the numbers rather than nowhere.

#### `not_covered` is counted, deliberately, and the reason is worth stating

8 of 9 verdicts carry it, so how it is scored decides the result. It is **not**
excluded: a reviewer saying *"the rubric has no entry for what this answer did"*
while listing no error is agreeing that no **listed** error occurred, and a
listed error is exactly what blunder rate is defined on. Counting it as "human
says clean" is correct *for this metric*.

What it is not is "the answer was good" — `pos-blocking-0003::base_closed` is
flagged `not_covered`, scored clean, and blocks with the same creature twice.
The output now says so next to the number, because the gap between "committed no
listed error" and "was a good answer" is 21.49's whole finding and it is
invisible in a precision score.

#### First human ground truth, n=12 — direction only

| judge | precision | recall | F1 | blunder accuracy |
| --- | --- | --- | --- | --- |
| Qwen2.5-32B | 28% | 88% | 0.42 | 75% |
| Mistral-24B | 44% | 50% | 0.47 | 67% |

**Do not read these as settled.** n=12 against a 60-verdict target, 8 of them
`not_covered`, and the two judges trade precision for recall rather than one
dominating. What they do show is that the paired control and human ground truth
disagree: the 32B scores +96% separation against *constructed* extremes and 28%
precision against *real* answers. That is the gap B1 was built to measure, and
it is the first evidence it exists.

### 21.56 The judge credits plays the answer never made, and that is checkable without a judge

From a reviewer's note: creature spells do not target the way removal does, and
the model treats them as if they did. Both halves of that check out, and chasing
the second half found something larger.

#### The targeting error is real, and the harness already catches it

Across 251 distinct stored position answers, **3** cast a spell with a target
where the position lists it without one — and all three are the same position
under all three arms:

```
pos-removal-timing-0002   legal_actions: ['CAST Ambush Viper']
base_open / base_closed / base_cards_open:
    CAST Ambush Viper TARGET Centaur Courser
```

Ambush Viper is a creature; casting it targets nothing. All three arms make the
error, and Gate 1 marks all three illegal — correctly, and for the right reason:
adding a target makes it a *different string* from the one enumerated, so the
list-of-alternatives check works here where 21.49 showed it failing.

**Tapping lands does not occur at all**: 0 of 251 answers contain a tap line.
The grammar has no `TAP` verb — mana is implicit — so there was a real question
whether such lines were being silently swallowed as prose. They are not being
written in the first place.

#### The rubric has no entry for the error all three arms made

`pos-removal-timing-0002`'s four `common_errors` are all about *timing* — cast at
end of turn, hold for the main phase, decline the block, summoning sickness.
None describes treating a creature spell as targeting.

This is the documented trap — *a rubric with no entry for the most likely wrong
answer* — with the most likely wrong answer now being **3 of 3 arms**. The
consequence runs straight into 21.55: the judge cannot charge an error that is
not listed, so it either fires nothing (the answer scores clean) or fires a
neighbouring error (a false positive against human ground truth). Both happened
here, and they are why precision is 28%.

#### The larger finding: credited key points for plays never made

`base_closed` on that position is:

```
CAST Ambush Viper TARGET Centaur Courser
PASS
```

Illegal, and it never blocks. The 32B credits it **`points_hit: [1,2,3,4]`** —
all four key points, including *"Block Centaur Courser: deathtouch means any
damage the Viper deals destroys it"* — with `errors_made: []`. Full marks, no
blunder, on an answer whose only other action is `PASS`.

It generalises. Key points on a position are instructions, and whether the
answer carried one out is **not a judgement call** — the action list comes from
the parser, which never sees the judge:

| judge | points naming an action | never taken | as a share of ALL credited |
| --- | --- | --- | --- |
| Qwen2.5-32B | 18 | **9 (50%)** | 9/99 (9%) |
| Mistral-24B | 17 | **5 (29%)** | 5/85 (6%) |
| Qwen3-14B | 16 | **6 (38%)** | 6/61 (10%) |
| **pooled** | **51** | **20 (39%)** | 20/245 (8%) |

Every judge does it, including the two that pass the paired control and the one
that ties them. Mistral is the least prone at 29% and that is still more than
one credited instruction in four.

And **none of the 20** contains the word anywhere in its text. The judge is not
crediting stated intent over executed action — it is crediting nothing at all.

`unearned_action_points()` computes this, and it is the positions analogue of
`verify_quoted_claims` (V4, 21.7) with two advantages: it costs **no judge
tokens** and carries **no judge noise**, because both sides are already stored.

**Both denominators are printed, always.** 20 of 51 checkable points (39%) and
20 of 245 credited points (8%) are the *same twenty events*; quoting either
alone misstates it. That is 21.43's rule — a rate is a statement about its
denominator — applied before publishing rather than after.

**Correctness is unchanged.** The count sits beside it, exactly as fabricated
citations sit beside the score in the rules report (19.1), because folding it in
would rebuild the confounded single number the rubric judge exists to take
apart.

Two narrowings, both tested, because a diagnostic that over-fires gets ignored:
only key points that **open** with an action verb (one mentioning blocking
mid-sentence is explaining a rule), and **`KEEP` is never flagged**, since an
answer that does not mulligan has kept. An answer that parsed no actions is also
exempt — that is Gate 1's problem, and blaming the judge for it would move a
known failure into a new column.

### 21.57 Judge-vs-human kappa is +0.20. Judge-vs-judge is +0.47

Seventeen verdicts in, and the number B1 exists to produce is measurable. It
required one fix first: `adjudicate.score_run` reported **raw agreement** on the
blunder call. `eval_positions.compare_judges` explicitly refuses to do that —
*"raw percent agreement flatters a skewed one: if both judges say 'blundered'
80% of the time, they agree ~68% by chance alone"* — and reports Cohen's kappa
first. That reasoning applies harder to the judge-versus-**human** comparison,
because the human's calls are the skewed ones, and that is the one place it was
not applied. Kappa is now one definition in `common.py`, beside `pearson_r`, for
the reason `pearson_r` is one. The shared version reproduces 21.47's **+0.47**
exactly.

| judge | n | precision | recall | F1 | blunder acc | **kappa vs human** |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | 17 | 19% | 88% | 0.32 | 59% | **+0.23** |
| Mistral-24B | 17 | 25% | 50% | 0.33 | 59% | **+0.17** |
| Qwen3-14B | 12 | 27% | 67% | 0.38 | 58% | **+0.21** |

Against **+0.47** between two of those same judges on the same answers.

**The judges agree with each other about twice as well as any of them agrees
with the human.** That is the entire premise of B1 confirmed the hard way:
inter-judge agreement is not a proxy for correctness, and a project that had
only ever measured judge-vs-judge would have read +0.47 as the instrument
working. All three judges sit in the band `eval_positions`' own verdict text
calls *"barely agreeing beyond chance… not yet a usable gate metric"*.

Note the numbers **got worse** as verdicts arrived — the 32B was 28% precision
and 75% blunder accuracy at n=12, and is 19% and 59% at n=17. An n=12 reading
would have been the optimistic one.

#### Why the paired control could not have found this

The 32B separates blundered from clean at **+96%** on the paired control and
sits at **+0.23** against human ground truth. Both are correct measurements of
different things, and the gap has a mechanism:

**12 of 17 adjudicated answers are flagged `not_covered`** — bad for a reason no
listed error describes. The paired control constructs its sensitivity half by
asserting a **listed** error verbatim. So it measures the judge on exactly the
case that mostly does not occur, and cannot measure it on the case that does.

That is not a flaw in the control; it is the limit of what a constructed control
can do, and it is the same lesson as 21.43 one level up. **A control built from
the rubric cannot detect that the rubric is missing entries.** Only a human
looking at the answer can, which is what B1 is.

#### What the reviewer's notes name, and what the harness does with it

The new notes describe two model behaviours precisely:

- *"Casts 4 Forests (assuming it meant to Tap them) then tries to cast Grizzly
  Bears as a non-creature spell"*
- *"It didnt declare any tapping of lands to cast Ambush Viper"*
- *"Temple of Silence is casted so its being assumed to be a spell"*

Measured across 251 distinct stored answers: **5 (2.0%)** cast a land rather
than playing it, concentrated in `pos-land-sequencing-0002`. My first pass
looked for `TAP` lines and found zero — the pattern is `CAST <land>`, not a tap
line, and the check missed it by looking for the wrong token.

**All five score `all_legal=False`.** Gate 1 catches every one, because
`CAST Forest` is a different string from `PLAY Forest`. The same is true of the
targeting error (21.56) and of the double block once 509.1a was added.

So the consistent picture across 17 verdicts: **the legality checker sees these;
the rubric cannot describe them; the judge therefore fires the wrong error or
none.** Gate 1 is measuring real failures, and Gate 3 is measuring the judge's
guesses about failures the rubric never enumerated. That is why Gate 1 has to
pass first, now with a number attached rather than an argument.

### 21.58 The best-scoring thing a model can do here is decline to play

Ten of the seventeen adjudicated answers are flagged `not_covered`, and reading
the reviewer's notes against the rubrics they were scored under shows why. Every
one of the ten positions enumerates **alternative strategies** in
`common_errors` — *"Serra Angel should stay home as a blocker"*, *"Ambush Viper
should be cast at end of turn instead"*. Every one of the notes describes
something else entirely:

| reviewer's note | class |
| --- | --- |
| *"It only passes and makes no moves"* (×3) | **did nothing** |
| *"It only lightning strikes… and does no other attacks"* | did half the line |
| *"It played two swamps before passing"*, *"6 Plains"* | illegal repetition |
| *"Casts 4 Forests… tries to cast Grizzly Bears as a non-creature spell"* | wrong action type |
| *"tries to use fog bank to block twice"* | illegal assignment |

None is a strategy error. They are **protocol** failures, and they are
position-independent — which is why adding them to 24 hand-written rubrics would
be both enormous and wrong. Three of the five classes are already caught by
`rule_illegalities`. The first two were caught by nothing.

#### Doing nothing scores better than playing

`PASS` always matches `legal_actions`, because it is the protocol terminator
every answer must end with (16.13). So an answer whose only action is `PASS`:

- scores **`all_legal=True` on 69 of 69** such answers across the three n=24
  judgings, and
- is called **clean on 37 of 59 graded (63%)**, because doing nothing commits
  none of the enumerated strategies.

**Gate 1 passes it and Gate 3 calls it clean.** And it is not rare — `base_open`
declines to play on **42%** of positions, `base_cards_open` on **46%**, against
`base_closed`'s 8%.

#### This is the whole of 21.52's arm reversal

Splitting blunder rate by whether the answer did anything at all, on identical
text, graded answers only:

| judge | did nothing | made a play |
| --- | --- | --- |
| Qwen2.5-32B | 13/23 (**57%**) | 29/49 (59%) |
| Mistral-24B | 2/23 (**9%**) | 24/49 (49%) |
| Qwen3-14B | 7/13 (54%) | 20/32 (62%) |

On answers that make a play the two calibrated judges are ten points apart. On
answers that do nothing they are **forty-eight** apart. The 32B treats declining
to play as roughly as bad as playing badly; Mistral treats it as almost always
clean.

`base_cards_open` is 46% do-nothing answers and `base_closed` is 8%. That is the
entire mechanism of 21.52's reversal — the arm that plays least looks **best**
under Mistral (33%) and **worst** under the 32B (71%). Not judge quality, and
not a 38-point spread needing a wider bar: a specific disagreement about one
behaviour the rubric never names, amplified by how much of each arm consists of
it.

`only_pass` is now measured, reported as its own column, and called out above
20%. It is deliberately **not** folded into a gate — that is a threshold
decision, and B3 should make it with this number in hand.

#### A loop the loop-detector could not see

`repeats_collapsed` skips `ONCE_PER_TURN` verbs on purpose, so that
`PLAY Swamp / PLAY Swamp` stays visible to `rule_illegalities` as a second land
drop. `degenerate` was derived from that field alone and therefore inherited
half its meaning: an answer emitting **`PLAY Plains` 133 times** reported
`degenerate=False`, and the report's Degenerate column read **0%**.

The comment above `ONCE_PER_TURN` already names this — *"`repeats_collapsed` was
carrying two meanings at once"* — and split them one level up. The split needed
to go one further: `repeats_collapsed` is about the action **list**, `degenerate`
is about the **output**. A loop is a loop whichever verb it loops on.

**Measurement changed, stated rather than silent:** per-arm degenerate on the
published n=24 run moves `base_closed` 0% → 4% and `base_open` 0% → 4%;
`base_cards_open` stays 0%. No gate reads this column, so no verdict moves.

Adding the field also broke `test_eval_positions.py` immediately — `rate()`
indexed `r["arms"][arm][key]` directly and raised `KeyError` on every archived
run. It now uses `.get()`, so a diagnostic that did not exist when a run was
written reads as *not measured*, the same treatment an ungraded answer gets.

### 21.59 Phase and mana: three of the four checks already existed, and the fourth did not

A reviewer asked for phase-timing and mana-pool understanding to be built into
position judging at a mechanical level, so correctness does not wait on the LLM
judge. Measuring what each would add, before building any of it.

#### The model does violate phases — and it is already caught

Card-independent phase rules (a land drop only in a main phase, `BLOCK` only in
declare blockers, `MULLIGAN`/`KEEP` only from an opening hand) applied to 251
distinct stored answers:

| violation | n |
| --- | --- |
| `CAST` / `PLAY` during **opening hand** | 22 |
| `BLOCK` during **declare attackers** | 5 |
| `PLAY` during **upkeep** (305.1) | 4 |
| other | 3 |
| **total** | **24 (9.6%)** |

A higher rate than any error class found so far. And the overlap with the
existing check is total:

```
caught by BOTH                 : 24
caught ONLY by the phase check : 0     <- what it would add
caught ONLY by legal_actions   : 88
```

**Zero.** The reason is structural: `legal_actions` is hand-enumerated, so an
author simply never writes down a phase-illegal action, and anything the model
invents fails to match. A phase check on model answers would have been a new
column, new tests, and a second name for a signal already reported — visible as
an improvement while measuring nothing.

#### Positions are already validated for timing, and rendered with the phase

`positions.timing_problems` catches a `legal_action` that is sorcery-speed in a
step that does not allow it — written after a board offered `CAST Pacifism`
during declare attackers. Its `_COMBAT_STEPS` is misnamed: it lists every
*non-main* step, upkeep and end step included, so the coverage is not just
combat.

And `render_position` already puts the phase in the header the model reads
(`=== Turn 7 — opponent's declare attackers ===`), with per-permanent
tapped/untapped state. The model is told when it is.

#### What did not exist: affordability

Nothing asked whether an enumerated play could be **paid for**. A board offering
`CAST Doom Blade` with `{G}{G}` available passes every check there is — the card
resolves against Oracle, the action parses, the step allows an instant.

`mana_problems()` closes it, parsing `mana_available` against the card's
`mana_cost`. It finds **0 problems in the 24 gold positions**, and fires
correctly on a deliberately broken control — three flagged actions when
`{W}{W}{W}{R}{R}` is replaced with `{G}`. Both directions are tested, because a
check that only ever passes hides every real result.

`parse_mana` returns **None** on hybrid, Phyrexian, X and snow symbols rather
than guessing. "Not modelled" rendered as a cost of zero would read as
*affordable*, which is the direction that looks like a pass.

#### Two silences, now reported as coverage

Both `timing_problems` and `mana_problems` return `[]` when their input is
missing — no card index, or no stated pool. "No problems found" over a set that
was never examined is the same shape as a gate verdict with no coverage (21.14)
and a one-sided control (21.43). The validator now prints:

```
  8/24 state mana_available (the affordability check sees only these)
  ! no card index loaded — the timing and affordability checks are SKIPPED, not passed
```

**16 of 24 positions state no mana pool.** The model sees `Swamp — untapped x5`
and must infer `{B}{B}{B}{B}{B}` itself. Making that explicit everywhere is a
`render_position` change, and `render_position` is what the model reads — so it
would invalidate all 251 stored position answers for comparison. That is a
decision with a real cost attached, not a cleanup, and it is left for the
gameplay track to take deliberately rather than made here as a side effect.

### 21.60 A verbose grammar: TAP declarations, and the guard that had to exist first

A reviewer's decision, and the reasoning is right: real play *is* verbose. A
player states the phase and says which land taps for which mana, because some
lands add more than one colour and some add a colour only under a condition.
Enforcing that gives a mechanical check where terse answers give none.

Building it needed a guard first.

#### `GAMEPLAY_SYSTEM_PROMPT` was not fingerprinted at all

`prompt_fingerprint` guards the rules track — three system prompts and the
assembled message shape — and does not cover the gameplay prompt. So an edit to
the **grammar block the model is told to answer in** changed the output contract
with no trace in any run file. Section 8.7's mechanism, unguarded, on the one
prompt whose *text is the output format*.

`gameplay_fingerprint()` is deliberately **separate** rather than a fifth part of
the existing digest. The two tracks share no prompt, and folding them together
would make a gameplay-grammar edit invalidate a rules adapter that never saw it —
trading a missing guard for a false one. Confirmed: adding it left the rules
fingerprint at `30badae98696`, so **no adapter needed re-stamping**.

Every position run now records it, and `compare_judges` refuses to interpret an
agreement number across two different values — answers produced under two
grammars are not the same answers, whatever the judge did with them.

#### The grammar

```
TAP <permanent> FOR <mana>        tap for mana, one permanent per line
```

`FOR` is **required**. `TAP Forest` says a land was tapped and not what it
produced, and on a land with two mana abilities those are different statements —
an optional operand would read as fine on basics and be ambiguous on exactly the
cards the verbosity exists for. A `TAP` without it is a **`ParseFailure`**, not
prose: the line opened with a known verb, so the model meant to act and
malformed it.

The prompt now also says, in words: *"Be explicit rather than brief… a land that
can add more than one colour produces only what you name… Do not leave mana
implicit."*

#### `ONCE_PER_TURN` could not absorb TAP, and the split says why

The collapser skips repeated lines for `ONCE_PER_TURN` verbs so a second land
drop stays visible. `TAP` needs the same treatment for a different reason:
repeating the line means tapping *another copy*. But it is not once per turn —
you may tap many lands — and folding it in would have made
`rule_illegalities` assert a land-drop restriction on tapping.

So `REPEAT_IS_MEANINGFUL = ONCE_PER_TURN + ("TAP",)`, a separate name, because
`ONCE_PER_TURN` is a claim about the **rules** (305.2) and this is a claim about
the **parser**. Caught by a failing test rather than by reasoning: with `TAP`
collapsed, an answer tapping three copies of a land it had two of reported *ok*.

#### What it checks, without a judge

`tap_problems()` reads the battlefield and the oracle text — `mana_abilities()`
parses `{T}: Add {G}.` and `{T}: Add {W} or {B}.` out of the card text, since
this corpus has no `produced_mana` field. Three distinct claims are checked:
tapping something you do not control, tapping more copies than are untapped, and
naming mana the permanent cannot add. `mana_abilities` returns **None** when it
finds no ability, never `[]` — "cannot parse this" rendered as "produces
nothing" would flag every correct tap of that land.

#### Retroactive safety, verified rather than assumed

Adding a verb changes how *existing* text parses. Re-parsing all 251 stored
answers: **0 TAP actions produced, 0 parse-failure counts moved.** Eight action
counts do differ from what was stored — all in `positions_n22.jsonl`, all
repeated `PLAY`, which is the earlier `ONCE_PER_TURN` change and predates this
one. No `pos_n24_*` row moves.

**The new prompt is not yet run.** Every stored position answer was generated
under `5c196f40afd8`; the current prompt is `a4218f5de4e2`. Numbers from the two
are not comparable, which is now enforced rather than remembered, and getting
comparable ones means re-running the arms.

#### What it cannot exercise yet

Every battlefield in the 24 positions holds **only basic lands** — Swamp,
Forest, Plains, Mountain, Island. The dual and conditional cases the verbosity is
*for* appear in hand (`Temple of Silence`, `Dismal Backwater`) but never in play,
so on this set every untapped land taps for exactly one colour and the check can
only ever confirm the obvious. `mana_abilities` is verified against
`Temple of Silence` and `Ancient Tomb` in tests instead. The machinery is right;
the positions have not caught up to it, and that is an authoring gap rather than
a code one.

### 21.61 Declaring the phase, and the trap that mandating verbosity walks into

21.60 built the mana half of the verbose grammar. This is the other half — *"the
player in control declaring what phase they're in"* — and building it exposed a
way the whole idea could have backfired.

#### `PHASE <step>` is the one thing `legal_actions` cannot see

21.59 measured 24 of 251 answers taking an action impossible in the stated phase,
and found **all 24 already caught** — a phase-illegal action is simply not in the
enumerated list. That result has a hole in it: a model can take a **perfectly
legal action while believing it is a different step**, and nothing about the
action reveals that. `CAST Ambush Viper` is right in declare attackers and right
in the main phase; only the belief differs.

A declaration makes the belief checkable, against `pos["phase"]`, with no judge
and no card data. Matching is loose on purpose — the board says *"opponent's
declare attackers"* and an answer saying *"declare attackers"* means the same
step. A false mismatch would manufacture a finding out of phrasing, which is
`_PLAYER_PREP`'s failure exactly.

#### A closed vocabulary, because the whole-word test cannot help here

`parse_line` separates verbs from narration by requiring a whole word, which
stops `Attacking with Swiftspear…` becoming an `ATTACK`. That defence does not
work for `PHASE`: *"Phase two of my plan is to attack"* opens with the exact
word, no suffix to notice. So the body must name a real step — MTG's are
enumerated in the CR — and anything else parses as prose rather than as a
malformed declaration.

#### The trap: requiring verbosity destroys Gate 1

`legality()` scored every parsed action against `legal_actions`. That list
enumerates **plays**, so a `PHASE` or `TAP` line can never appear in it.
Measured, before shipping:

```
terse    CAST Ambush Viper / PASS
         all_legal=True
verbose  PHASE declare attackers / TAP Forest FOR {G} / TAP Forest FOR {G} /
         CAST Ambush Viper / PASS
         all_legal=False   illegal=['PHASE declare attackers', 'TAP Forest, {G}', ...]
```

The same play, told the way the prompt now demands, scores **illegal on every
line it was asked to add**. Every arm's legality would have collapsed the moment
verbosity was required, and it would have arrived in the shape of a result:
*"asking the model to show its working makes it play worse."*

This repo has had that exact shape twice — a grammar whose optional-operand
brackets the model copied, and prose about a play parsing AS that play. Both
times the harness punished the model for doing what it was told. The general
form is worth naming: **when a prompt starts asking for new output, every check
that consumes the output is a candidate for punishing it.**

`ParsedOutput.plays` is the fix — actions minus `DECLARATIONS = ("PHASE", "TAP")`
— and `legality`, `only_pass` and the action count all read it. Terse and verbose
now score identically, and a wrong play is still caught through the declarations.

Two consequences worth keeping:

- **`actions/answer` is a count of plays**, so a model cannot look busier by
  being more verbose.
- **Verbosity cannot escape 21.58's do-nothing detector.**
  `PHASE upkeep / TAP Forest FOR {G} / PASS` still reports `only_pass=True`,
  because none of that is a play. Asked directly, since the obvious way to game
  a "did nothing" check is to say more while doing nothing.

#### Status

Retroactively inert, verified: 0 of 251 stored answers contain a line beginning
`PHASE`, and no `pos_n24_*` action count moves. The gameplay fingerprint is now
`d094e3934d2d` — a third distinct value, with stored runs still at
`5c196f40afd8` and `compare_judges` refusing to read agreement across them.
Nothing has been run under the new grammar yet.

### 21.62 A human verdict is about a text, not about a key

Regenerating the position arms under the verbose grammar means the same board is
answered differently. The adjudication queue is keyed `record_id::arm`, and
**that key is stable while the text behind it is not**.

Nothing recorded which text a verdict was made against. So rebuilding the queue
from a new run would have left 22 human verdicts pointing at answers their author
never saw, and `--score` would have reported precision, recall and kappa against
them without a word. Nothing fails: the key matches, the arm matches, the
position matches.

This is 21.13's identifier problem one level down — *anything joining two files
on an id must first ask whether the id survived the trip* — with the trip made
worse by surviving. There the id changed and the join silently missed; here the
id is preserved and its **meaning** changes, so the join silently succeeds.

`answer_sha` is a 12-character digest of the exact answer text, recorded on every
verdict and compared at score time. A verdict whose digest disagrees with the run
being scored is excluded and **reported** as stale, next to the numbers, rather
than dropped.

Three behaviours, each tested:

- a matching digest scores normally;
- a verdict on different text does not score, and is counted;
- **a verdict with no digest still scores** — refusing those would discard human
  work to enforce a field that did not exist when the work was done.

The 22 existing verdicts were backfilled from `adjudication_queue.json`, which
still held the exact text each was shown, and each carries
`answer_sha_source: "backfilled…"` so a recovered value never reads as one
recorded at the time. Verified two ways: scoring against `pos_n24_32b` reproduces
n=17, precision 19%, kappa +0.23 unchanged — so the digests match the run they
were made against — and scoring the same verdicts against a synthetically
regenerated run reports **22 stale, n=0**.

**What this costs, stated plainly.** The 17 covered answers were adjudicated
under the old grammar. They remain valid evidence about *that* run and stop being
evidence about the new one. Judge accuracy on the verbose arms needs fresh
verdicts; the earlier work is not lost, it is simply about a different set of
answers.

### 21.63 The verbose grammar, measured — and a conjunctive metric that punishes it

First run under `d094e3934d2d`: 24 positions × 3 arms, 32B judge, everything else
held. Stored answers are `5c196f40afd8`, so judge-scored columns are not
comparable across the two; **parser-derived ones are**, because the parser never
sees the judge.

#### The closed arm complies; the open arms loop

```
pos-combat-math-0005 / base_closed
    PHASE Pre-combat main phase
    TAP Forest FOR {G}
    TAP Forest FOR {G}
    TAP Mountain FOR {R}
    CAST Shock TARGET Grizzly Bears
    PASS
```

Exactly the format asked for. `base_open` on the same board emits **73 actions**,
cycling `PHASE Main phase (continuing) / PASS` and re-casting a spell it has one
copy of. Degeneracy went from **0% to 17–33%** across the arms: giving a weak
model a new line to emit gave it a new thing to loop on.

#### Per-answer legality fell; per-play legality did not

| Arm | `all_legal` terse → verbose | **legal plays** terse → verbose | plays/answer |
| --- | --- | --- | --- |
| `base_closed` | 75% → **67%** | 84% → **90%** | 2.5 → 2.8 |
| `base_open` | 71% → **46%** | 65% → 63% | 2.3 → 4.4 |
| `base_cards_open` | 62% → **33%** | 68% → 62% | 1.7 → 6.5 |

**`all_legal` is an AND over an answer's plays.** Verbose answers carry 1.1–3.8×
more plays, so the same per-play quality yields a lower per-answer rate
mechanically. On the arm Gate 1 actually reads, the model got **better** at legal
play — 84% to 90% — while the number the gate consumes went **down**.

This is 21.61's trap one level up. There it was the parser punishing lines the
prompt demanded, fixed by scoring `plays`. Here it is the *metric*: a conjunction
over a longer list, punishing length rather than quality. Same shape, different
layer, and it would have read as *"the verbose grammar made the model worse at
legal play"* — a conclusion the per-play column refutes.

Both columns are printed now. **The gate still reads `all_legal`** — changing
what Gate 1 consumes is a threshold decision and belongs with B3, not with the
change that exposed it.

#### What verbosity actually bought

`only_pass` fell where the model was declining most: `base_open` **42% → 21%**,
`base_cards_open` **46% → 12%**, `base_closed` 8% → 8%. Asking for the phase and
the mana gave the model something to do before deciding to do nothing, and it
did substantially less nothing. That is the one clear win, and it is on the
behaviour 21.58 identified as the largest single failure class.

#### The declarations are wrong often enough to be worth having

**20 of 72 answers (28%) declare a phase that disagrees with the board**, and 21
of 72 (29%) declare a tap the board could not produce. The most common single
error is *"declare attackers"* stated on a **declare blockers** board, ten times.

That is exactly the class 21.59 proved invisible: `pos-removal-timing-0002 /
base_closed` casts Ambush Viper — the **correct** play — while declaring the
wrong step. Every legality check passes it. Only the declaration reveals it.

#### A denominator error, mine, in the column added to fix a denominator error

The per-play column first printed **48%** for `base_closed` where the true figure
is **90%**. `n_legal` counts matched *plays*; the row's `n_actions` counts every
parsed line, declarations included. Dividing one by the other measured "legal
plays per line emitted", which is not a quantity anyone wants.

The row now stores `n_plays` beside `n_actions` with a note on which is which.
Worth recording because it is the third denominator mistake in this session
(21.52's `bool(errors_made)`, 21.56's two rates, this) and because the first
number it produced was *plausible* — 48% is a believable legality rate, and
nothing about it looks wrong.

### 21.65 The regrade nearly discarded itself, and judge-vs-human kappa is now +0.05

Six verdicts arrived against the regenerated arms. Ingesting them found the same
identifier assumption in a third place — and this time it destroyed work rather
than mis-scoring it.

#### The dedup key was `(key, author)`

`--ingest-submissions` skipped anything whose `(key, author)` was already on
file. After a regrade, every fresh verdict on an already-adjudicated key
collides with the old one. Dry-run, before the fix:

```
DROPPED  pos-land-sequencing-0002::base_closed
DROPPED  pos-race-vs-stabilize-0001::base_open
DROPPED  pos-combat-math-0004::base_open
DROPPED  pos-trigger-ordering-0002::base_closed
DROPPED  pos-land-sequencing-0001::base_closed
DROPPED  pos-race-vs-stabilize-0001::base_cards_open
```

**Six of six**, silently, while eleven unrelated older rows appended in their
place. 21.62 fixed scoring and the done-set and left the ingest — the one path
that *deletes* the evidence rather than misreading it.

#### And the first fix for it committed the error it was written to prevent

The repaired ingest stamped every unstamped submission from
`adjudication_queue.json` — which now holds the **regenerated** answers. It
reported "stamped 35", cheerfully attributing thirty-five pre-redeploy verdicts
to text their author never saw. The fix for 21.62 committing 21.62.

Queue-stamping is gone. The form stamps at submission time, so a missing digest
now means *collected before that existed*, and dedup has two rules:

- **with** a digest → identified by `(key, author, digest)`, so a regrade is a
  distinct verdict;
- **without** one → falls back to `(key, author)`, or every legacy row
  re-appends forever, since the copy on file was backfilled with a digest and no
  longer matches.

Eleven pre-redeploy verdicts that had never been pulled were recovered by
stamping them from the **archived** queue — 11/11 resolvable, and **0** of their
answers survive unchanged into the new run, which is the check that says the
archive was the right source.

#### The numbers, and the direction they keep moving

| scored against | n | precision | recall | blunder acc | **kappa** | excluded as stale |
| --- | --- | --- | --- | --- | --- | --- |
| `pos_n24_verbose_32b` | 7 | 5% | 100% | 29% | **+0.05** | 20 |
| `pos_n24_32b` | 21 | 7% | 75% | 43% | **+0.05** | 6 |

Each run is scored only by the verdicts made against its own answers; the guard
excludes the rest and says how many.

**Judge-vs-human kappa is +0.05 — chance.** It was +0.23 at n=17 (21.57) and
+0.23→+0.05 as verdicts accumulated, with precision tracking it down: 28% at
n=12, 19% at n=17, **7%** now. Three successive samples, each larger, each worse.
An early reading of this measurement was the optimistic one every time, which is
the argument for finishing the 60 rather than stopping at a number that looks
tolerable.

Against **+0.47** judge-vs-judge on the same answers. The judges agree with each
other roughly ten times better than either agrees with a person.

#### Two new checkable classes, from the notes

The reviewer's notes on the verbose answers name two failures nothing catches:

- *"It taps excess mana as lightning strike costs 1 generic mana and 1 red
  mana"* — **over-tapping**. `tap_problems` validates each tap against the board
  and never sums them against the spell's cost, so a correct-looking three-land
  payment for a two-mana spell passes.
- *"It is not seeing Serra Angel as already on the field so it is attempting to
  cast it"* — **casting a permanent already on the battlefield**. Caught today
  only because the string is absent from `legal_actions`, which gives no reason.

Both are mechanical, both need only the board, and neither is built.

### 21.66 Two checks the reviewer's notes asked for, and what they found

Both come from notes on the verbose answers, and both are mechanical — board and
oracle text, no judge.

#### Taps that do not pay for the casts

*"It taps excess mana as lightning strike costs 1 generic mana and 1 red mana."*

`tap_problems` (21.60) checks each tap on its own — is it yours, is it untapped,
can it add that colour — and never adds them up. So three lands tapped for a
two-mana spell passes with every individual tap correct. `payment_problems`
compares the declared pool against the summed cost of the `CAST` actions beside
it, reporting **short** and **floated** separately, since they are different
mistakes.

| verbose run, 72 answers | |
| --- | --- |
| declared taps that do not pay | **24 (33%)** |
| — of which floated mana for nothing | 16 |
| — of which short | 8 |

**Zero on the terse run**, which is the check working rather than a gap: it is
silent unless an answer declares at least one tap *and* casts at least one
spell. An answer that declares nothing is not over-tapping — that is
`only_pass`'s finding, and firing here would manufacture a result on every run
made before the grammar existed.

#### Casting a permanent already on the battlefield

*"It is not seeing Serra Angel as already on the field so it is attempting to
cast it."*

Caught today only because the string is absent from `legal_actions`, which
reports "not a legal action" and gives no reason — and would not catch it at all
on a position that enumerated that card for some other purpose.

Needs no card data at all: the board says what is in play and what is in hand.
**8 of 72 (11%)** on the verbose run and **6 of 72 (8%)** on the terse one — so
this one predates the grammar change and was simply never named.

A card in **both** hand and play is fine, because a second copy is castable, so
the check is "on your battlefield and *not* in hand" — the only unambiguous
case. Asserted, along with the opponent's permanents being none of your
business.

#### Both were invisible for the same reason

`legal_actions` enumerates plays that are individually legal. An over-tapped
payment consists **entirely** of legal taps; a spell already in play is missing
from the list, and a missing entry cannot explain itself. That is the same shape
as 21.49's double block — a constraint that exists *between* individually legal
choices, or *outside* what the list can express — and it is now the fourth
instance. The enumerated set is a good check for "is this play available" and a
poor one for anything relational.

### 21.67 A second rules corpus: the wiki gloss, pinned and unwired

`rules.jsonl` is the Comprehensive Rules — authoritative, complete, written for
judges. It says what a rule **is** and never what it **means**. A player asking
"can a creature with summoning sickness block?" is answered by 302.6, but only
by someone who already knows to look there.

The 51 pages behind `Portal:Rules` are the other half: Object, Zone, Timing and
priority, one page per phase. **207,574 characters, 176 chunks.**

#### Fetching

Plain scraping is Cloudflare-challenged — even `robots.txt` returns a JS
challenge — so this uses the MediaWiki API the operator publishes for the
purpose, one request at a time with a delay. Two things went wrong and both were
found by running it:

**`exlimit` documents a maximum of 20, and using it returned 3 pages of 60.**
TextExtracts honours `exlimit > 1` only when `exintro` is set; ask for full text
and it serves one page per request and drops the rest — no error, no `continue`,
57 pages simply absent. The comment above `BATCH` warned about this exact
failure and the code did it anyway. The fetcher now refuses to write a snapshot
missing more than a quarter of what was asked for, because a corpus quietly
holding a third of its content is worse than a failed fetch.

**Nine portal links are redirects to section anchors** — `Upkeep step` →
`Beginning phase#Upkeep step` — and looked like missing pages. They are aliases
for text already captured. Attaching them had to happen *after* the fetch loop:
at one title per request a redirect resolves after its target is already
written, so filling it inline left the alias empty and the redirect still
counted as missing.

Accounting reconciles: **51 captured + 6 reached by alias + 3 template-only
pages with no extractable prose = the 60 links.**

#### What a plain scrape drags in, and what it does not

TextExtracts strips Fandom's page furniture entirely — no ad markup, no cookie
banner, no navigation. The snapshot's two hits for *"advertisement"* and
*"subscribe"* are article prose about token cards and judge fees.

What it does **not** strip is the article's own apparatus:

| dropped section | pages |
| --- | --- |
| References — and **empty** after extraction, since citations are markup | 41 of 51 |
| External links | 16 |
| Trivia | 9 |
| See also | 8 |
| Gallery, Notes | 9 |

Dropped **by heading**, never by pattern-matching the body: the wiki labels these
itself, so no guess is required and no real content can be caught by accident.
`History` is kept — rule changes are rules content — and an unlisted heading is
included by default rather than silently dropped. 234,670 → 207,574 chars, with
all **40** non-empty `Rules` sections verified present afterwards.

#### Three sizing failures, each visible only in the output

| symptom | cause |
| --- | --- |
| a 50-char chunk | page **leads** have no previous chunk to merge into |
| a **7,235**-char chunk | TextExtracts renders lists with *single* newlines, so a long list section contained no paragraph break for the splitter to use |
| a **4,687**-char chunk | merging a short tail into a full-size chunk undid the split — the merge ran after the split and never re-checked size |

Now 361–3,397, bounded at `TARGET + MIN_CHUNK` so "soft target" is a stated
number rather than unbounded drift.

#### It is a gloss, and the corpus says so at every layer

Community-edited, so it is **not a citation source**. This project's entire
citation discipline rests on a rule id resolving against the pinned CR, and a
corpus that reads like rules text but is not the rules text is the fastest way
to break it. So: a separate file, `authority: "unofficial"` on every record, and
the **page revision** on every record — *"the wiki says X"* is not a citation,
*"revision 564208 says X"* is.

`WIKI_PIN` goes through the shared `_verify_pin` body. Proved it fires on
tampered text **at an unchanged record count**, which is the case a pin exists
for, and that it ignores a non-canonical path.

#### Licence, and the three decisions

Fandom serves this **CC BY-NC-SA 2.5**, stored on every record. Noncommercial is
satisfied — this is a hobby project — and attribution travels with the data.

- **Retrieval: not wired.** Whether wiki prose competes with CR text for slots
  is unmeasured. Cards already needed a separate budget because they outnumber
  rules chunks 78:1; wiki prose resembles a player's *question* more closely
  than the rule that answers it, so it would win slots on phrasing. That is a
  measurement, not a default.
- **Training: opt-in, `--with-wiki`.** Wiki lines go to **train only** — mixing
  a second distribution into valid would move the loss curve for a reason
  unrelated to the gold set being measured. Each answer carries its source and
  revision **in the trained text**, because a comment cannot survive into
  weights.
- **Pinning: yes**, and more urgently than elsewhere. A wiki page is edited far
  more often than the CR is published, so a re-fetch silently changing text
  under a published number is likelier here than anywhere else in this repo.

### 21.68 The wiki should annotate the CR, not compete with it

The retrieval question from 21.67, measured — and the answer changed the design.

#### Competing costs cited-rule retrieval

Embedding the 99 gold questions against a merged index and counting which corpus
wins each slot:

| top-k | wiki share of slots |
| --- | --- |
| 1 | 29% |
| 3 | 26% |
| 5 | 25% |
| 10 | 24% |

Against a **28% chance baseline** (176 of 624 chunks). So wiki text takes almost
exactly its proportional share and slightly *less* than chance at k=10.
**21.67's stated prediction — that wiki prose would win slots on phrasing — is
wrong**, and the crowding number alone would have said "merge freely".

It would have been the wrong conclusion. Share is not harm. Asking instead
whether the rule the gold record *cites* still gets retrieved:

| k | rules only | rules + wiki | delta |
| --- | --- | --- | --- |
| 3 | 17/99 (17%) | 15/99 (15%) | **−2** |
| 5 | 23/99 (23%) | 21/99 (21%) | **−2** |
| 10 | 34/99 (34%) | 31/99 (31%) | **−3** |

Small, and consistently negative at every budget. Merging strictly loses the
thing every citation check depends on.

#### The wiki carries its own CR references — in markup `explaintext` deletes

Only **4 of 176** chunks showed a rule id, which looked like the wiki simply not
citing rules. It cites them constantly; TextExtracts strips the markup, which is
the same reason References sections came out empty (21.67):

```
Timing and priority   {{CR|Timing and Priority}}  {{CR|glossary|Pass}}
                      {{CR|glossary|Priority}}    {{CR|glossary|In Response To}}
Exile                 {{CR|701.11}}               {{CR|glossary|Set Aside}}
```

72 templates across 29 of 51 pages — **51 glossary references, 14 rule numbers,
7 section names**. Thin, and mostly not numbered rules.

#### The dense join is the page title itself

**42 of 51 page titles match a CR glossary term exactly (82%).** A wiki page
*is* an expanded gloss of a glossary entry, and the term is the join key. That
is structural rather than a similarity heuristic, and an order of magnitude
denser than the templates.

The nine that do not match are informative rather than a gap:

| | |
| --- | --- |
| **not rules content** — Commander series, DCI, Judge, Magic tournament, Set | dropped |
| CR *sections* rather than glossary terms — Timing and priority, Turn structure | kept |
| design vocabulary — Evergreen, Comprehensive Rules | kept |

Those five are precisely the "unrelated content" a portal scrape drags in, and
they were identified **structurally** — no glossary term and no rules citation —
rather than by taste.

#### Chaining through the glossary, and the granularity trap

The join carries `cr_rule_ids` and `cr_sections` from the matched entry, so
*wiki page → glossary term → CR rule* is followable. The first attempt reached
**5%** of chunks, because `RULE_ID_RE` matches `117.1a` and a glossary entry
mostly says *"See rule 117"* — a **section**, three digits, no decimal. Measured:
**89%** of the 739 entries cite a section and only 65% a numbered rule.

`_CR_SECTION_RE` is deliberately separate from `common.RULE_ID_RE` rather than a
loosened version of it. `RULE_ID_RE` validates citations; making it also match
bare section numbers would mean one name for two granularities, silently
disagreeing about whether "117" is a rule id — the trap `CROSS_REF_RE` and
`PASS` already cost this project.

**5% → 90% of chunks chain to a rule or section**, reaching 38 distinct CR
sections.

#### Where this leaves retrieval

Final corpus: **46 pages, 141 chunks**, 91% joined to a glossary term, 90%
chaining to a rule or section. Still wired into no pipeline — what the
measurement establishes is that the *merge* design was wrong, not that the
aligned one is right. Serving a glossary definition and its wiki expansion
together, under one retrieval slot rather than two competing ones, is the design
the evidence points at and is not yet built or measured.

### 21.69 One page, read closely: the wiki's Rules sections are an index, not prose

Preparing the corpus for training use meant reading a single page against the CR
rather than counting things across all of them. `Target` — CR §115, 26 rules
chunks against 4 wiki chunks.

#### What each side actually provides

| | CR §115 | wiki `Target` |
| --- | --- | --- |
| size | 11,371 chars | 7,402 chars |
| content | normative definition, when targets are declared, the "target [something]" test, worked examples | plain definition, **what does not target**, hexproof/shroud/fizzling gathered in one place, **common misconceptions** |

The wiki's unique contribution there is **negative space** — *"a card does not
target a creature just because it damages or destroys one"* — which the CR
establishes only by omission, and **cross-concept gathering** of material the CR
scatters across sections.

**And `Target` is not representative.** *"Common misconceptions"*, *"Key ideas"*
and *"Related mechanics"* appear on **exactly one page each**. Reading one page
and generalising from its structure was the obvious mistake available here, and
the headings that actually recur are `Rules` (40 pages) and `Description` (25).

#### The Rules sections extract to zero characters — correctly

35 of 40 `Rules` sections are under 200 characters, and on inspection **empty**.
The wikitext says why:

```
Activated ability:  {{CR|glossary|Activated Ability}}
                    {{CR|Activating Activated Abilities}}
                    {{CR|glossary|Activation Cost}}
Artifact:           {{CR+G|Artifact|s}}
```

They **transclude the official rules text**. The rendered page shows CR text;
`explaintext` strips the template and leaves nothing.

That is the right outcome, not a loss. The CR is already held verbatim and
pinned, and a paraphrase of it competing with the original is the single thing
that would actively hurt — 21.68 already measured merged retrieval costing
cited-rule recall.

**The templates are the valuable part.** Each `Rules` section is a hand-curated
index from a concept to the CR passages that govern it, written by someone who
knows the game. That is a better artifact than the prose around it.

#### Resolving the index

The first survey found 72 templates by matching `{{CR|`. The real pattern is a
family — `CR` (69) and `CR+G` (27) — so **96**, and a narrow regex had quietly
dropped a quarter of them.

Resolution needs three tables, and the third was missing:

| reference kind | resolved against | n |
| --- | --- | --- |
| `{{CR\|glossary\|Priority}}` | `glossary.jsonl` terms | 39 |
| `{{CR\|Targets}}` | **CR rule-group headings** | 33 |
| `{{CR\|701.11}}` | rule ids | 13 |

`rules.jsonl` carries only the nine top-level part titles ("Game Concepts",
"Zones"), so matching a template against those resolved **nothing**. The 294
rule-group headings — `602. Activating Activated Abilities` — exist only in the
raw CR text, and parsing them there resolves every probe exactly: *Targets* →
115, *Timing and Priority* → 117, *Artifacts* → 301.

**85 of 98 references resolve (87%)**, across 38 of 46 pages.

#### The 12 that do not are a finding, not a gap

`Mono Artifact`, `Poly Artifact`, `In Play`, `Global Enchantment`,
`Remove from the Game`, `At End of Turn`. These are **obsolete terms** the modern
CR no longer defines and the wiki documents as historical. They are kept and
labelled `unresolved` rather than dropped: a model that treats them as current
vocabulary is exactly the failure this corpus should help avoid, and deleting the
evidence would remove the only signal that they are dead terms.

#### Where this leaves training

The split is cleaner than "train on the wiki":

- **`Description` and the lead are trainable** — explanatory prose with no CR
  equivalent, which is what the model lacks.
- **`Rules` sections are links, not text.** They contribute a concept→CR index
  and no prose at all, so there is nothing there to train on and nothing lost by
  it being empty.

Counted per page after a first attempt reported **265 resolved references where
there are 85** — every chunk of a page repeats that page's citations, so summing
over chunks multiplied a 13-chunk page by thirteen. The same denominator error
as 21.63 and 21.56, and the inflated number was again perfectly plausible.

### 21.70 Protocol errors: the gameplay layer supervises the judge

21.65 measured judge-vs-human kappa at **+0.06** — chance — with precision 3%,
recall 100% and zero misses. The judge fires at nearly everything and misses
nothing, which is not a judge that works. 21.49 already named why: **81% of
adjudicated answers were bad for a reason no listed error describes**, and a
judge cannot charge an error that is not enumerated, so it charges a neighbour.

#### The gameplay layer already holds the truth the judge layer lacks

Crossing the judge-free checks against the judge on the verbose run:

| | judge fired | judge silent |
| --- | --- | --- |
| a mechanical check fired | 42 | **20** |
| nothing mechanical fired | 5 | 5 |

**86% of answers (62/72) can be convicted with no judge at all.** The judge is
silent on 20 that are provably wrong and fires on 5 of the 10 that are provably
clean.

The silence is arguably *obedient*, not broken: a do-nothing answer commits no
listed **strategy** error. The gap between "mechanically convictable" and "the
rubric can describe it" **is** the 81%.

This inverts the layering the project assumed. A judge cannot validate itself
and judge-vs-judge is worth +0.47 against a truth of +0.06 — but a board state
is machine-checkable in a way a rules question never is. **The gameplay layer
supervises the judge layer, not the reverse.** The corollary is worth knowing
before it bites: deckbuilding has no parser and no legal-action list, so it
inherits the judge's weaknesses with none of these correctives.

#### Seven protocol errors, appended to every rubric

Position-independent, so they need no authoring per board. Order is
**append-only**: the judge returns error NUMBERS, so strategy errors keep
`1..n` and every verdict already collected still means what it meant.

| # | class | invalidates the turn? |
| --- | --- | --- |
| 1 | takes no action at all | yes |
| 2 | repeats an action instead of playing a line | yes |
| 3 | names a play that is not available | yes |
| 4 | states the wrong phase | yes |
| 5 | **does not tap enough** to pay | yes |
| 6 | **taps more than required** | no — legal, and still discouraged |
| 7 | casts a permanent already on the battlefield | yes |

5 and 6 began as one entry. A reviewer split them, and the distinction is real:
under-tapping means the spell cannot be cast at all, while over-tapping is a
legal play a real player makes. One entry asked the judge to charge two
different mistakes with one number, and asked the parser to confirm a charge
that could be true for either reason.

#### What makes these different from every other rubric entry

Each is **decidable by a parser**. When the judge charges "the answer takes no
action", a machine confirms or refutes it — **per-error precision with no human
and no second judge**, which this project has never been able to compute.
`protocol_findings()` returns the machine verdict per class and the report turns
it into a precision line.

Two restraints, both deliberate. `None` means *not decidable here* — class 5/6
need the card index, class 4 needs the position to state a phase — and those are
**excluded** rather than counted as the judge being wrong, which would be
21.43's one-sided control in a new place. And **blunder rate is unchanged**: it
stays defined on `errors_made` exactly as 21.28 set it, with the
strategy/protocol split stored beside it, so no published number moves.

#### The target is a valid turn, not an optimal one

The reviewer's framing, and it is sharper than either gate: *"aiming not for a
whole game correct at this point but simply a full turn completely valid."*
`valid_turn` is exactly that — no invalidating class fired — and it is
judge-free:

| arm | terse grammar | verbose grammar |
| --- | --- | --- |
| `base_closed` | **54%** | 46% |
| `base_open` | 29% | 25% |
| `base_cards_open` | 17% | 17% |

The best arm plays a completely valid turn about half the time. And the reasons
differ by grammar — terse fails on *unavailable play* (25) and *does nothing*
(23); verbose on *unavailable play* (37), *wrong phase* (20) and *repeats* (18),
having traded doing nothing for saying the wrong thing, exactly as 21.63
measured.

**Gate 3 stays on strategy errors.** Counting protocol errors toward it would
make a do-nothing answer a "blunder" and stop the metric meaning "picked the
wrong play". Validity is Gate 1's business, and `valid_turn` is the number it
should be read against.

### 21.71 Stepwise scenarios: a turn is a sequence, not a harder position

A reviewer set out the gameplay curriculum as seven stages — mulligan, land
order, payment, combat, closing the turn, full-turn validity, full-game
validity — and mapping it against the 24 positions showed where the shape runs
out.

| stage | current coverage |
| --- | --- |
| 1 mulligan — curve, mana dorks | **both exist** (`mulligan-0002`, `mulligan-0001`) |
| 2 land order — dual colour, tapland first | **both exist** (`-0003`, `-0001`/`-0002`) |
| 3 payment in the main phase | **gap** — nothing tests it, though the `TAP` grammar was built for it |
| 4 combat — lethal, trick-then-attack, blocking | 11 positions, all **one** decision |
| 5 closing the turn post-combat | **zero** — the set has no `postcombat main` position at all |
| 6 full-turn validity | metric exists; positions cannot express it |
| 7 full-game validity | out of reach |

Stages 4, 6 and 7 are not harder positions. Every position is **one board, one
decision** — 1 to 4 legal actions, answered once — and *"cast the trick, then
attack"* is two decisions with the board changing between them.

#### Teacher-forced, and that is the design decision

A scenario is a base position plus ordered **steps**, each carrying its own
phase, `legal_actions` and rubric. The board advances on the **reference** line,
never on what the model actually did. Two reasons, and the second decides it:

- Applying the model's own actions needs a **rules engine** — resolve the spell,
  update the battlefield, recompute legality. This repo is deliberately not one,
  and a half-built engine produces wrong board states that read as model errors.
- **Errors compound.** If step 2 inherits step 1's mistake, every later step
  measures step 1 again, and a model that misplays the first decision scores
  zero on four independent skills it might have.

So each step is scored independently and the **sequence** is the unit: a turn is
valid when every step was. What that does not test is recovery from one's own
mistake, which needs the engine and is a later problem — stated rather than
quietly absent.

Each expanded step is a **complete position**, so `legality`,
`protocol_findings` and `judge_batch_rubric` all work on it unchanged. Nothing
downstream needs to know a scenario existed. The sequence reaches the model as
`Already done this turn: TAP Mountain FOR {R}; CAST Shock TARGET Wall of Omens`
— what a real player knows, and no board derived from another board.

#### The validator earns its keep immediately

The first authored scenario had step 2 still showing **Shock in hand** after step
1 cast it. `players` is inherited unless a step overrides it, and forgetting the
override shows the model a card it cannot have — which reads as *the model
ignoring an obvious play* when it is **the scenario lying to it**. That is the
worst class of harness bug: it manufactures a model error out of an authoring
slip, and it would have been invisible in the scores.

`validate_scenario` now tracks what the reference line spends and refuses a
scenario that leaves it in hand. It caught the bug on the first run.

**And the check had the falsy-value bug on its first draft.** `step_hand or
base_hand` fell through to the scenario's hand whenever a step declared an
**empty** hand — which is the commonest state on a turn's final step — so every
correctly-authored ending would have been reported as an error. Presence of the
override decides now, never its truthiness. Same shape as `entry.get(k)` dropping
a legitimate `False`, and it was caught by a test asserting a clean scenario
validates clean, not by one asserting the failure.

All new positions are authored for the **verbose grammar**, so the format that
scores them and the format they are written for are the same from the start.

### 21.72 The first parser-checked judge number, and what it was actually measuring

The protocol rubric ran: 24 positions + 2 scenario steps × 3 arms, 32B judge.
**Precision 36% (69/190), recall 57% (69/122)**, with 62 undecidable checks
excluded rather than counted against the judge.

**The prediction written before the run was wrong.** It said recall would be
high and precision the informative number, on the theory that a judge firing at
everything (recall 100%, precision 3% against human verdicts, 21.65) would
simply spread that firing across more classes. It is wrong in *both* directions
— inaccurate rather than indiscriminate. Recording it because a prediction that
survives contact is worth less than one that does not.

#### Per class, and the miss that explained itself

| # | class | prec | recall |
| --- | --- | --- | --- |
| 1 | takes no action | 20% | **42%** |
| 2 | repeats an action | 45% | 50% |
| 3 | unavailable play | 74% | 51% |
| 4 | wrong phase | 41% | 90% |
| 5 | under-taps | 27% | 75% |
| 6 | over-taps | 13% | **18%** |
| 7 | casts what is in play | 28% | 100% |

No positional decay — class 7 is last and has 100% recall, class 1 is first and
has 42% — so it is not attention drift down a list.

Class 1 is the anomaly: *"the answer takes no action at all"* should be the
easiest thing in the list to see. Reading the misses:

```
PHASE Declare Blockers
TAP Forest FOR {G}   TAP Forest FOR {G}
TAP Forest FOR {G}   TAP Forest FOR {G}
PASS
```

**The judge was right and the rubric was wrong.** Told "takes no action at all",
it looked at four visible actions and correctly declined to charge. The checker
meant *no PLAY* — declarations excluded — and the wording said *no action*.

A correct judge and a correct checker disagreed because the entry did not
describe what the entry decides. And the cause is an interaction: **the verbose
grammar (21.60) is what made a do-nothing answer look busy.** Before it, doing
nothing was literally `PASS`.

Every entry is reworded to name the grammar it talks about — plays versus `TAP`
and `PHASE` lines — including "charge this if ANY line is unavailable" for class
3, whose misses were answers with one good play and one bad one.

#### And a real bug in the checker, found the same way

`degenerate` fired on **13 of 18** answers whose longest run was a `TAP`.
Tapping five Forests is what paying five mana looks like; it is not a loop.
21.58 widened `degenerate` from collapsed repeats to *all* repetition — correct
for `PLAY Plains` ×133 — and the verbose grammar then made that fire on correct
play. Repetition is now counted over **plays**: `max_play_repeat`.

21.58's case survives (a repeated `PLAY` is still a loop), and the genuinely
abusive `TAP Island` ×56 is still convicted — by `tap_problems`, which reports
*"only 3 untapped copies, tapped 4 times"*, and the turn is still invalid on
class 3.

**Measurement changed, stated:**

| arm | `valid_turn` before | after | degenerate |
| --- | --- | --- | --- |
| `base_closed` | 46% | **58%** | 15% → **0%** |
| `base_open` | 23% | 25% | 23% → 8% |
| `base_cards_open` | 19% | 17% | 31% → 17% |

The best arm plays a completely valid turn **58%** of the time, not 46%. The
earlier figure was a checker artifact.

#### Two things the run confirmed in passing

The protocol run and the verbose run produce **byte-identical parser numbers**:
only the rubric changed, generation is deterministic, and it reproduced exactly.
That is the positive control for the whole comparison.

And blunder rate moved 75→96%, 50→54%, 71→85% on those same answers, purely
because the rubric gained seven chargeable entries. **Blunder rate is
rubric-dependent**, so a fixed Gate 3 threshold is meaningless unless the rubric
is frozen — a direct input to B3 that was not obvious before.

#### The scenario table rendered nothing

Both steps ran and the Turn scenarios section was empty, because the result rows
never carried `scenario_id` — the report was built and the field it groups by
was not. Output identical to *"no scenarios were run"*. Fixed, and backfilled
from the ids:

```
turn-payment-combat-0001    base_open 0/2   base_closed 1/2   base_cards_open 1/2
```

No arm completed the turn. At one scenario that is plumbing confirmation, not a
measurement.

#### What this does not yet answer

Whether the rewording recovers the missed recall. The 36%/57% was measured
against wording that misdescribed two of seven classes and a checker that was
wrong about a third, so it is a **lower bound on a superseded instrument**, not a
verdict on the judge. Re-running is the next measurement, and until then nothing
here says the judge cannot do this job — only that it was not asked properly.

### 21.73 The recall regression was mostly carpet-bombing, and wording moves classes not aggregates

21.72 reworded all seven protocol entries after diagnosing one, and the re-run
reported recall falling **57% → 43%**. Investigating before acting on it found
the number was measuring something else.

#### Class 7's "100% recall" was an accident

Of its 8 true cases in the first run, **4 were answers charged with every one of
the seven classes at once**. The judge was not detecting "casts a permanent
already on the battlefield"; it was carpet-bombing, and class 7 happened to be
in the barrage. That is 21.26's *every error at once* pathology, which
`all_errors_fired` already exists to flag — and which nothing applied to the
protocol block, so one act of blanket firing counted as seven verdicts.

| | first run | reworded |
| --- | --- | --- |
| mean protocol charges per answer | 2.6 | **1.7** |
| fired **all seven** at once | 14 (18%) | **2 (3%)** |
| charged **exactly one** class | 10 | **24** |

Blanket firing fell six-fold. The rewording did not make the judge cautious; it
made it **discriminate**.

#### With blanket fires excluded, the two runs are the same

| | precision | recall |
| --- | --- | --- |
| first run, as first reported | 36% | 57% |
| first run, blanket excluded | **38%** | **42%** |
| reworded, blanket excluded | **36%** | **39%** |

The 14-point regression was almost entirely artifact. **Rubric wording moved the
aggregate by about two points.**

#### But per class it moved a lot, and the direction is predictable

| # | class | recall before → after | edit made |
| --- | --- | --- | --- |
| 1 | makes no play | 36% → **67%** | *"Declaring a phase and tapping lands are not plays."* |
| 2 | repeats a PLAY | 18% → **40%** | *"Repeated TAP lines are not this error."* |
| 3 | unavailable play | 32% → **41%** | *"Charge this if ANY line is unavailable."* |
| 4 | wrong phase | 88% → 56% | narrowed to *"The answer's PHASE line…"* |
| 5 | under-taps | 60% → 29% | added *"— it is short."* |
| 7 | casts what is in play | 100% → 25% | added *"rather than one in hand."* |

The three that improved were told **what not to count**. The three that
regressed had a **condition to verify** appended — and a judge in doubt resolves
by not charging. (Class 7's before-figure is also the one most inflated by
blanket fires, so its true fall is smaller than 75 points.)

That is a usable rule for writing these: *scope-narrowing clarifies, condition-
adding suppresses.*

#### The durable number

On classes a **parser decides with certainty**, the 32B judge sits at roughly
**37% precision and 40% recall** — right about a third of the time it speaks,
finding two fifths of what is there — and rubric wording moves individual
classes by 30 points while leaving the aggregate flat.

That is the answer to *"can this judge do this job"* for the protocol classes:
**not at this accuracy, and not via wording.** `valid_turn` (58% on the best
arm) remains the number to trust, because no judge touches it.

#### Two instrument gaps this exposed

`judge_note` was **not stored on the position path** while the rules path keeps
it, so diagnosing class 7 had to be done by inferring from firing patterns when
the judge had presumably said why. Now stored.

And blanket fires are now **excluded from the parser-checked figures and
reported separately**, because counting one act of carpet-bombing as seven
verdicts inflates recall with accidental hits — which is exactly how a 100%
appeared and then vanished.

**A method failure of mine, recorded.** 21.72 changed the wording *and* the
`degenerate` checker in one commit, after a day spent documenting that mistake.
It is only interpretable because five of six affected classes turned out to have
unchanged truth sets — luck, not design.

### 21.74 The rescore path never expanded scenarios — and 37% precision is the task, not the judge

Two findings, and the first is why the second was nearly unobtainable.

**A run containing scenario steps could be produced and never re-judged.**
`--scenarios` expands a turn scenario into position-shaped steps and appends
them. The steps live in `turn_scenarios.jsonl`; `positions.jsonl` never sees
them. `--rescore-from` loads answers from a stored run and looks each `id` up in
`positions.jsonl`, so it died on:

```
2 stored ids are not in positions.jsonl:
  turn-payment-combat-0001::step1, turn-payment-combat-0001::step2
```

The failure is loud, which is the only good thing about it. What it cost is not:
`--rescore-from` is the **only** way a second judge ever sees an answer, and
9.9's rule is that a number from one judge is a statement about the judge. So
21.71's stepwise harness shipped able to produce a number and structurally
unable to validate it. The gates would have gone on reading as single-judge
verdicts for exactly as long as nobody tried.

**Fifth one-of-two-paths bug**, after `carry_diagnostics` (neither writer listed
`scoring`), 21.51 (`judge_model` on rescored rows), 21.53 (the coverage guard in
`rescore()` and not the first-pass writer), and 21.54 (the judge's identity in
both `compare_judges` writers). The direction flips every time, which is why the
rule in CLAUDE.md is *check every other writer*, never "check `rescore`".

Fixed the way the previous four should have been: `turns.load_steps()` is the
one definition, both call sites reach scenarios through it, and
`test_eval_positions.test_scenario_paths` asserts **both callers exist** by
reading the source. That is not a stylistic preference. No test over inputs and
outputs can see a missing caller — the generation path was correct, its tests
passed, and the bug lived entirely in a path those tests never entered.
Confirmed by mutation: deleting the rescore caller fails two assertions.

#### Protocol precision under a second judge

With the path fixed, `pos_n24_protocol2` rescored under
`Mistral-Small-24B-Instruct-2501-4bit` against the stored 32B run — **the same
78 answers, byte for byte**, judge as the only variable.

| | Qwen2.5-32B | Mistral-24B |
| --- | --- | --- |
| charges made | 126 | 171 |
| **precision** | **37%** (47/126) | **34%** (58/171) |
| **recall** | **43%** (47/110) | **53%** (58/110) |
| undecidable, excluded | 62 | 62 |

**37% was the task, not the 32B.** A judge from a different vendor, given
identical answers and an identical rubric, lands within three points. Mistral
fires 36% more charges and converts the extra volume into +10 recall at −3
precision — the ordinary sensitivity/specificity trade, not a different
instrument. The identical 62 undecidable checks are the positive control: the
parser never sees the judge, so that column *must* match, and it does.

Cohen's kappa on the blunder call is **+0.67** here against +0.47 in 21.57. Not
an improvement in judges — a different rubric on a different set. What it does
say is that the protocol entries are more agreeable *between judges* than the
strategy entries were, which is what a rubric decidable from the board should
do, and which says nothing about whether either judge is right. Both gate
verdicts also agree (Gate 2 PASS 65% vs 85%; Gate 3 FAIL 57% vs 67%), and Gate 3
is again the 21.19 pattern: agreement bought by distance from the bar, not by
the judges being close.

#### The judge is accurate exactly where it is least needed

The aggregate hides the finding. Per class, under both judges:

| # | protocol error | 32B prec | Mistral prec | 32B rec | Mistral rec |
| --- | --- | --- | --- | --- | --- |
| 3 | names an unavailable play | **85%** | **80%** | 44% | 41% |
| 4 | PHASE names the wrong step | 43% | 35% | 60% | 75% |
| 2 | repeats one PLAY | 38% | 33% | 50% | 50% |
| 7 | casts a permanent already in play | 25% | 31% | 25% | 50% |
| 1 | only passes | 22% | 22% | 67% | 83% |
| 5 | TAP lines underpay | 21% | 29% | 38% | 75% |
| 6 | TAP lines overpay | 18% | 21% | 12% | 24% |

Class 3 is 80–85% under both judges. Every other class is 18–43% under both.
That ordering is stable across two vendors, so it is a property of the classes.

And class 3 is **the one class `legal_actions` already checks mechanically** —
it is `all_legal` restated as a rubric entry. So the judge is reliable precisely
where a parser makes it redundant, and unreliable on all six entries that exist
*because* the parser could not see them (21.66, 21.70). The clean reading is not
"the judge is 37% accurate"; it is that a judge asked to check arithmetic
(classes 5 and 6, precision 18–29% and recall as low as 12%) is being asked for
something it does not do, while the same judge asked "is this play on the list"
does it well.

This is the strongest available argument for 21.70's direction. The gameplay
layer should supervise the judge layer rather than the reverse, and the split
above says *which* entries to take from the parser: everything the board decides
mechanically. The judge's remaining job is the strategy entries — which is where
81% of adjudicated answers came back `not_covered`, and where no mechanical
check exists at all.

**Open, and not answered by this run.** Class 6 (over-tapping) has 12–24% recall
under both judges — the judges are missing most real over-taps, and over-tapping
is the entry the user deliberately separated from under-tapping because it is
legal-but-wasteful rather than turn-invalidating. Being legal may be exactly why
it is hard to notice. `PROTOCOL_INVALIDATING` excludes it from `valid_turn`, so
the recall gap costs nothing on Gate 1 today; it would cost something the moment
over-tapping became a scored blunder.

### 21.75 The adjudication form was one release behind the judge, and it reversed the comparison

`PROTOCOL_ERRORS` was added to the judge's rubric in 21.70 and to the form in
neither place it needed to go. The consequence has two halves, and each half
reads as a finding about something other than a form.

#### Half one: the reviewer had no box, so they used the note

`eval_positions` grades against `common_errors + PROTOCOL_ERRORS` — eleven
entries on a typical position. `adjudicate.task_for` sent `common_errors` alone
— four. The reviewer saw the strategy entries and nothing else.

So when an answer tapped six lands it did not control, or cast a creature
already on the battlefield, there was no checkbox for it. The reviewer did the
only thing available: ticked **"Bad, but not for any reason above"** and wrote
what happened in the note. Reading all 33 not-covered notes, **30 describe an
entry that was already in `PROTOCOL_ERRORS`**. One of them names four at once:

> *"Tapping all their forests to cast no spell, declaring that they are playing
> 4 forests when they are already on the battlefield, casting a Grizzly Bears
> that is not in their hand, Casting Grizzly Bears as a non-creature spell
> targeting Centaur Courser twice."*

That is entries 6, 7, 3 and 2, written out by hand, on a form that offered none
of them.

The 69% not-covered rate was read as **the rubric missing entries** (21.47). It
is mostly the *form* missing entries the rubric already had. Only two of 33 —
"there isn't an explanation why it mulliganed" and "doesn't go for lethal,
holds up attacking instead" — are genuinely outside both lists, and those two
are the real coverage gap.

This is the 21.55 shape rotated one turn. There, a field the human filled in was
stored and read by nothing. Here, a field the *judge* fills in was never offered
to the human — so the reviewer's careful prose became unscoreable free text, and
their effort was spent restating a list that existed.

#### Half two: unofferable charges scored as false positives

`score_run` computes `fp = len(judge - human)`. A protocol charge is never in
`human`, because `human` can only contain numbers the form displayed. So every
protocol charge counted as a judge error by construction.

The trigger rate is the problem. On `pos_n24_verbose_32b`, produced before
21.70, **0 of 129** fired numbers exceed the strategy list. On
`pos_n24_protocol2_32b`, **129 of 190 (68%)** do. Same code, same verdicts:

| scored against | unfixed precision | fixed |
| --- | --- | --- |
| `pos_n24_verbose_32b` (pre-protocol) | 2.8% | 2.8% |
| `pos_n24_protocol2_32b` (protocol) | **2.2%** | **5.6%** |

Unfixed, the protocol rubric looks like it made the judge *worse*. Fixed, it
doubles precision. **The bug reversed the sign of the comparison**, and it did so
because its rate is a function of the condition under test — the exact family as
the unwrapped-JSON bug that hit V5 on 8 of 24 calls and V3 on 0 (21.39), where a
harness failure arrived wearing the shape of a result about the treatment.

A bug that fires uniformly is visible as a bug. One correlated with the
treatment is indistinguishable from a finding, and the direction it points is
arbitrary.

#### Fixed on both sides, and neither fix alone is enough

`task_for` now sends `common_errors + PROTOCOL_ERRORS` in the **judge's order**,
so strategy entries keep 1..n and protocol entries take n+1..n+7 — the same
concatenation `eval_positions` performs, because any other order renumbers every
verdict silently. `n_strategy` marks the split so the form can group the two
("Mistakes specific to this position" / "Mistakes any answer can make"), and
form_version is 3.

`score_run` restricts each judge charge to `n <= n_shown`, recorded by the form
from v3 on. Verdicts collected earlier carry no `n_shown` and fall back to the
rubric's strategy count — which is exactly what that form displayed, so the
fallback is *exact* rather than a guess. The fallback derives the record id from
the key rather than trusting a separate field, because a missing field would
silently mean "no restriction", which is the unsafe direction: it is what the
bug already did.

Restricted charges are counted in `charges_not_shown` and printed as a loud
separate line, not folded into the exclusions list. The others there
(`unsure`, `stale`, `unmatched`) are properties of the sample. This one is a
defect in the form, and its remedy is to re-export and redeploy, so it says so.

#### What this does NOT do

It does not produce a judge precision number. Both figures above rest on
`tp = 1`, and the reason is now understood rather than mysterious: the reviewer
could tick only strategy entries and the answers were mostly failing on
protocol. **The 27 covered verdicts cannot be rescued by re-scoring** — the
information was never collected. They have to be re-adjudicated against the
eleven-entry form, and the 21.62 digest machinery makes that safe: the answers
have not changed, so a fresh verdict on the same key is correctly a second
verdict rather than a collision.

The honest statement of judge-vs-human precision today is **not measured**, and
the reason is a form field, not a judge.

#### And the done-set had to change with it, or nobody would redo them

One consequence nearly cancelled the fix. A task is marked done when this
author has a verdict carrying the current answer's digest — 21.62's machinery,
which exists because regenerating the arms changes the text under a stable key.

Here the text did **not** change. The rubric did. So all 16 covered tasks would
have shown **done** on the redeployed form, and the reviewer would have skipped
precisely the ones needing re-reading — 21.62's failure one level up, with the
identifier surviving while what it identifies changes underneath.

`common.verdict_is_current` now checks both, and it compares the **entry count
offered** rather than `form_version`, so a version bump for an unrelated reason
does not throw away human work while a rubric that grows always does. A verdict
with no `n_shown` is treated as not current, matching the asymmetry already
applied to a missing digest: asking for one duplicate verdict is visible and
cheap, silently skipping one is neither.

Two callers — `rubric_server.api_tasks` and `adjudicate --status` — so it is one
function with a test asserting both exist, per 21.74. Coverage restated
honestly:

```
covered    : 0/60 of the queue
             (11 carry a verdict on an EARLIER answer — the arms were regenerated)
             (16 were adjudicated against a SHORTER rubric — same answer, fewer
              entries offered, so they need re-reading)
```

Those two lines were one line reading *"27 more keys carry a verdict on an
EARLIER answer"*, which is wrong for 16 of them and points at the wrong remedy:
a regenerated arm means the old verdict describes text nobody will see again,
while a grown rubric means the same answer needs re-reading. Splitting them also
made the arithmetic check out — 11 and 16 are exactly the pre-change stale count
and the pre-change covered count.

### 21.76 Blunder rate quietly changed meaning, and the parser can arbitrate

Two findings from the same decomposition, both about the mixed number Gate 3
reads.

#### Adding a rubric moved a metric that was documented as unchanged

21.70 gave the judge `common_errors + PROTOCOL_ERRORS`. `errors_made` is the
whole judge verdict, and blunder rate is `bool(errors_made)`. CLAUDE.md recorded
that blunder rate was **unchanged** because it still reads `errors_made` — true
to the letter, and false in substance. On the same three arms:

| arm | blunder, pre-protocol | blunder, protocol |
| --- | --- | --- |
| `base_open` | 75% | **96%** |
| `base_cards_open` | 71% | **92%** |
| `base_closed` | 50% | 58% |

That is not a model that got worse by 21 points. **A run made before 21.70 is
not comparable to one made after it on this number**, and nothing said so —
`gameplay_fingerprint` catches a prompt edit, not a rubric that grew.

This is "one name, two meanings" again, and the way it arrived is worth naming:
21.28's rule ("blunder rate is `errors_made` and nothing else") was followed
exactly, and following it is what broke the metric, because the rule constrains
the *expression* and the meaning lives in what feeds it.

#### And the two halves rank the arms differently

Decomposed, under the 32B:

| arm | strategy (judge) | protocol (judge) | protocol (**parser**) | headline |
| --- | --- | --- | --- | --- |
| `base_open` | 35% | 92% | **96%** | 96% |
| `base_closed` | **46%** | 35% | **62%** | 58% |
| `base_cards_open` | 38% | 88% | **88%** | 92% |

`base_closed` is the **best** arm on the headline (58%) and the **worst** on
strategy alone (46% against 35% and 38%). The mechanism is 21.58's exactly: an
answer that declines to play commits none of the enumerated *strategies*, and
`base_open` passes on 42% of positions. Protocol entry 1 exists to close that
blindness, so the mixed number is not "strategy plus a correction" — it is two
metrics that disagree, averaged.

Both readings are defensible. A single number that silently switched between
them is not. All three columns are printed now, and **Gate 3 is deliberately not
redefined** — which one it should read is B3's call, the same disposition
`only_pass` got.

#### The parser can settle a disagreement between two judges

`protocol_truth` is computed from the answer text and the board. It never sees
the judge, so across the 32B and Mistral runs it is **identical on 78 of 78**
arm-positions — the positive control for the whole comparison, the same role
Gate 1 plays for the gates.

Which means it can do something nothing else in this project can. On the 58
answers where the two judges fired *different* protocol sets:

| | |
| --- | --- |
| parser sides with **Mistral-24B** | **19** |
| parser sides with **Qwen2.5-32B** | 10 |
| neither closer | 29 |

21.57's difficulty was that judge-vs-judge kappa is +0.47 while judge-vs-human
is +0.06 — two judges agreeing says nothing about either being right, and a
judge cannot validate itself. On the protocol half that is no longer true. This
is the first judge disagreement in the project resolved **without a person**,
and it says Mistral is closer to the board, consistent with its higher recall
(53% vs 43%, 21.74).

Read the parser column and the same conclusion arrives from the other side: it
reports `base_closed` at 62% under both judges, while the 32B's *judge* column
says 35% and Mistral's says 62%. The 32B undercounts `base_closed`'s protocol
errors by 27 points, and the mixed headline (58% vs 65%) hides it.

Three limits, all real. It covers the **protocol entries only** — the strategy
entries have no mechanical check and are exactly where 81% of adjudicated
answers came back `not_covered`. Classes 5 and 6 are decidable on only 60% of
answers (the payment check is silent when no taps are declared); the other five
are 100%. And a judge that loses here is worse at **the half a parser could have
done anyway** (21.74) — this ranks judges on the part of the job that least
needs one.

`compare_judges` also now refuses to interpret a run pair whose `protocol_truth`
differs: the parser is deterministic over (answer, board), so a mismatch means
the two files are not over identical answers and every agreement number above it
is comparing two different things.

### 21.77 The note becomes the reasoning, and two stylesheets pointed at the wrong pages

A reviewer request, mid-session: the note should describe **why** a play is
wrong on every faulted verdict, rather than serving as a fallback when no
checkbox applied. The boxes are the verdict; the note is the argument.

That is a change of *meaning*, not of wording, so it is `form_version` **4**.
Under v3 an absent note means "a box covered it"; under v4 it means "the
reviewer did not explain". Pooling the two would read the older convention as a
coverage failure of the newer one, which is the same mistake 21.75 made in the
other direction.

Four changes follow from the new job:

- `<input type="text">` → `<textarea>`. A rationale is prose; one v3 note
  already reached **461** characters against a 500 cap, so the cap is 2,000.
- A soft confirm when something is marked wrong with no explanation. It
  **confirms, never blocks** — a missing rationale is usually an oversight, but
  the reviewer is the authority on their own verdict and a hard gate would trap
  them mid-queue.
- The `not_covered` box no longer advertises the note as its companion; it asks
  what the missing entry should *be*.
- **`form_version` is deliberately not read by `verdict_is_current`.** That
  compares the entry count shown (21.75), so a v3 checkbox verdict stays valid
  under the v4 form. Only a rubric that GREW invalidates work. A reviewer
  adjudicating while this shipped loses nothing.

#### The field was collected, stored, and read by nothing

Exactly 21.55's shape, one field over. The 30-of-33 finding in 21.75 came from
reading these notes **by hand**; no code in the repo could have produced it.
`adjudicate.py --notes` now reads them, split by what each is evidence of:
notes on `not_covered` verdicts name entries the rubric should have — the only
evidence-driven route by which it grows rather than being guessed at — and notes
on boxed verdicts show whether the reasoning matches the box it was filed under,
which is how 21.73's ±30-point wording sensitivity would surface next time. v3
and v4 are counted separately and never pooled.

#### And two stylesheets were on the wrong pages

`rubric_server.py` holds **four complete pages** as separate Python strings.
Twice now a CSS rule has been added to one page while the markup it styles lives
in another: `h3.grp` went into `INDEX_HTML` while the group headings 21.75
introduced are in `ADJUDICATE_HTML`, so those headings shipped **unstyled on a
deployed form**; the same nearly happened to `#note` an hour later.

It cannot fail loudly. The page renders, the JavaScript parses, every endpoint
returns 200 — the rule is simply dead in one page and absent from the other.
Same family as the `#` comments that blanked four views: a string in one
language inside a file of another gets no checking from either, and here even
the JavaScript parse-check added for that cannot see it, because CSS is a third
language inside the second.

`test_webui.py` now asserts that every `#id` and `.class` rule in a page's
`<style>` names something that page's markup actually uses. Checked in the
dead-rule direction only — markup without CSS is ordinary, CSS without markup is
a mistake. Matching is word-boundary and never after a dot, because a bare
substring reported nine hits of which six were noise (`t.done` is a property
read, `common_errors` is not `.err`). The surviving three were all genuinely
dead: `.rules` and `.cancelled` were removed, and `.done` — which existed to
green the "saved" marker and was applied to nothing — is now applied.

Confirmed by mutation: putting `h3.grp` back on the wrong page fails the check.

### 21.78 Twelve verdicts the ingest would have thrown away, and a kappa that cannot exist

The first batch collected on the v4 form, and three defects between it and a
number — each in the layer below the last one fixed.

#### The ingest dropped every re-adjudication it was collected to receive

`--ingest-submissions --dry-run` reported **63 submitted, 0 new**. All twelve
were real work.

The identity was `(key, author, answer_sha)`. 21.75 established that a verdict
given four rubric entries is not a verdict on the eleven-entry form, and
`verdict_is_current` acts on exactly that — `--status` correctly listed those
tasks as needing redoing. **The ingest disagreed and won.** The answers had not
changed, so the digest matched, so every fresh verdict collided with the v2 row
it was collected to replace. One half of the system said *redo this* and the
other said *already on file*, about the same verdict, and the work vanished
between them.

Fourth appearance of one assumption, and each time the key survived while what
it identified changed underneath:

| | what changed under a stable key |
| --- | --- |
| 21.13 | promotion renamed the id (`rg-1156` → `qa-…`) |
| 21.62 | the arms were regenerated — same key, different text |
| 21.65 | the same, in the ingest rather than in scoring |
| **21.78** | **the rubric grew under fixed text** |

This one the digest structurally *cannot* see, because nothing about the answer
changed. `n_shown` is the fourth component now.

And keeping both rows created a second problem immediately: `score_run` iterates
verdicts, so the same answer would have been scored twice — once against four
entries and once against eleven. It now keeps one verdict per
`(key, author, answer_sha)`, preferring the one offered more rubric. Grouped on
all three deliberately: an earlier attempt grouped on `(key, author)` and
collapsed verdicts about *different* text, which are different verdicts (21.62)
— it silently cut n from 15 to 11.

#### The measurement it unblocked cannot be made yet

With the form finally showing what the judge sees, `charges_not_shown` is **0**
on the v4 subset — the fix worked. The numbers:

| | n | precision | recall | blunder acc | kappa |
| --- | --- | --- | --- | --- | --- |
| 32B, v4 only | 9 | 18% | 22% | **100%** | **undefined** |
| Mistral, v4 only | 9 | 24% | 56% | 89% | undefined |
| pooled across v2/v3/v4 | 15 | 14% | 24% | 87% | *+0.67* |

**That +0.67 is not a result.** Kappa needs variance in both raters, and the v4
sample has none: human 9/9 blundered, judge 9/9 blundered. Perfect agreement,
zero information. The pooled figure only becomes computable by mixing two form
regimes that showed the reviewer different rubrics — the variance is
manufactured by the regime difference, not by the judge agreeing better. Had
this been quoted as *"judge-vs-human kappa rose from +0.06 to +0.67"* it would
have been the most encouraging wrong number the project has produced.

Both are reported now rather than trusted: a degenerate sample prints `n/a` with
the reason instead of `nan`, and a sample spanning form versions says so.

#### It is the default outcome, not bad luck

These arms blunder on **82%** of answers as the judge sees them, so nine
blundered draws is unremarkable (p ≈ 0.17). Adjudicating more of the same queue
changes nothing.

This is 21.43 one level over. A control measured on one half of a distribution
is uninterpretable; so is an agreement statistic measured only on answers that
blundered. `build_queue` now interleaves by the judge's blunder call — sampling
on judge output is allowed and *showing* it is not, the line `disputed` already
walks, so the reviewer sees a shuffled queue and is never told which side
anything sits on. The balancing field is deleted before the queue is written and
`export_tasks`' blind-task assertion covers it.

**The ceiling is 14.** That is every clean answer these three arms produced
across 26 positions, so 23% is the best a 60-task queue can be. A properly
powered blunder-call kappa needs either more positions or arms that fail less
often — it is not reachable by adjudicating harder.

One more reason the sample is one-sided, and it is not a defect: the single
human-clean verdict is also the only one marked *genuinely ambiguous*, so 21.55
excludes it, correctly. The reviewer used the box for "clean, but the line could
be shown more fully" — a fair reading of the words, and a different claim from
"I could argue it either way". Worth distinguishing, because that box is the one
thing that removes a verdict from a sample already short of them.

### 21.79 A kappa that rests on one cell, and the eleven answers that would fix it

Four more verdicts, sixteen total, and the blunder-call kappa becomes computable
for the first time. It should not be believed.

| | n | precision | recall | blunder acc | **kappa** |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B, v4 only | 13 | 24% | 35% | 92% | **+0.63** |
| Mistral-24B, v4 only | 13 | 27% | 61% | 77% | **−0.11** |

A 0.74 spread between two judges on identical answers looks like a decisive
result. The confusion matrices say otherwise:

```
32B                judge:BLUNDER  judge:clean       Mistral        BLUNDER  clean
  human BLUNDER         11              1             human BLUNDER    10      2
  human clean            0              1             human clean       1      0
```

**Both rest on a single human-clean answer.** The 32B "wins" because that one
answer is also the one it called clean. Flip any single judge call and the 32B's
kappa lands anywhere in **[−0.08, +1.00]**; Mistral's in [−0.13, +0.43]. That
range is the resolution of the sample, and it spans essentially the whole
statistic.

So `score_run` reports it. A kappa whose single-flip range exceeds 0.4 prints
the range and says the point estimate is not the number. This is the same
discipline as refusing a one-sided control (21.43) or a single-judge gate
verdict (9.9), applied to sample size rather than to design: **the instrument
must state its own resolution**, because +0.63 and −0.11 are both perfectly
plausible-looking and neither is measuring the judge.

It also catches the near-miss in the other direction. 21.78's pooled +0.67 was
manufactured by mixing form regimes; this +0.63 is a clean v4-only sample and is
*still* uninterpretable, for an unrelated reason. Two different ways to get an
encouraging wrong number out of the same fifteen rows.

#### The fix is eleven specific answers, and they are already queued

The reviewer has adjudicated 14 answers of which **three** were ones the judge
called clean — because they were working the pre-rebalance queue, which is 82%
blundered. 21.78's interleave put the clean answers on odd positions, and every
blundered answer they have done occupies an even one.

The result is an accident worth naming: **queue positions 7, 9, 11 … 27 are all
undone and all judge-clean.** After a redeploy the next eleven tasks are exactly
the eleven the measurement is short of — no instruction needed, and the reviewer
is still never told which side of the call anything sits on.

That takes the human-clean column from 1 to a plausible 8–10, which is where a
kappa stops being one cell. It does not solve the ceiling: 14 clean answers is
all three arms produced across 26 positions, so this measurement tops out at
n≈25 with ~11 clean. Enough to separate a working judge from a broken one; not
enough to rank two working ones. Ranking needs more positions, which is the
authoring work stages 3 and 5 already represent.

### 21.80 A creature spell handed a target: the rubric's first evidence-driven gap

The seventeenth verdict, and the first one where the reviewer's `not_covered`
box did the job 21.77 rebuilt it for. The answer:

```
PHASE Declare Attackers Step
TAP Forest FOR {G}
TAP Forest FOR {G}
CAST Ambush Viper TARGET Centaur Courser
PASS
```

The reviewer ticked **no boxes**, marked `not_covered`, and wrote why: the cast
was right, but `TARGET Centaur Courser` is invalid because Ambush Viper is a
creature spell — the correct line was CAST, PASS, then BLOCK in the declare
blockers step.

Three graders, three different answers about the same four lines:

| | verdict |
| --- | --- |
| **parser** | entry 3 true — *names a play not available in this position* |
| **judge** (32B) | entry 4 — *PHASE names the wrong step*. A false positive, and it missed entry 3 |
| **human** | nothing on the list describes this |

The parser is right by technicality and wrong in substance. `legal_actions`
holds `CAST Ambush Viper`; the answer said `CAST Ambush Viper TARGET Centaur
Courser`, so the string does not match and entry 3 fires. But the play *was*
available — the operand was invented. The reviewer read entry 3 the way it is
written and correctly declined it.

That matters because entry 3 is the **one** entry the judge grades well (80–85%
precision, 21.74) and the only one a parser confirms cleanly. Quietly using it
as the bucket for a second, unrelated error is how a good class becomes a muddy
one.

#### Measured before deciding anything

`positions.targeting_problems` checks it against oracle text, no judge involved:

| run | answers | fires |
| --- | --- | --- |
| `pos_n24_protocol2_32b` | 78 | **6 (8%)** |
| `pos_n24_verbose_32b` | 72 | 5 (7%) |

Every hit is a creature spell — Fog Bank, Serra Angel, Ambush Viper ×2, Grizzly
Bears ×2. The model is treating creature spells like removal. And it is not a
new observation: **four** reviewer notes name it independently, across three
different cards, going back to the v2 form. It was visible the whole time and
unaddressable, because free text in a note reaches no number.

The check **misses rather than guesses**, the rule `find_players` follows. It
fires only on a Creature that is not an Aura whose oracle text contains no
"target" at all. Auras genuinely do target on cast (303.4c) while their text
says "Enchant", which is the one case the heuristic would get backwards, and a
creature whose *ETB trigger* targets is skipped even though the spell still does
not target. Both are deliberate misses: a missed case costs a diagnostic line, a
false one convicts a correct play.

#### Why it is a diagnostic and not entry 8

`PROTOCOL_ERRORS` is append-only, so an eighth entry renumbers nothing. What it
does do is move `n_shown` from 11 to 12, and `verdict_is_current` then marks
**every collected v4 verdict** as needing re-reading (21.78). That is 15
verdicts of human time, spent to gain coverage of an 8% error class.

So it is stored per-answer and reported per-run, and the promotion is left as a
decision with its cost stated rather than taken silently — the same disposition
`only_pass` (21.58) and the Gate 3 decomposition (21.76) got. Three sections now
end this way, which is the correct shape: **the instrument measures, the person
whose time it costs decides.**

Worth noting what made this reachable. 21.77 changed the note from a fallback
into the reasoning and 21.75 put the protocol entries on the form; without
either, this verdict would have been another empty `not_covered` indistinguishable
from the other 33.

### 21.81 The judge's *clean* verdicts are the unreliable half, and Gate 3 is optimistic

Seven verdicts on the rebalanced queue — all of them answers the judge called
clean — and the blunder-call kappa becomes a measurement rather than a cell.

| | n | κ | one-flip range | precision | recall |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | 21 | **+0.27** | [+0.11, +0.41] | 23% | 29% |
| Mistral-24B | 21 | −0.04 | [−0.27, +0.16] | 25% | 54% |

21.79's range was [−0.08, +1.00] on the same statistic. Eleven judge-clean
answers collapsed it to a third of a point, and the 32B's **+0.63 became
+0.27** — the earlier figure was the one-cell artifact, exactly as the range
said it might be. Modest positive agreement, robust to any single call. Mistral
is at chance.

#### Where the disagreement actually lives

```
32B                judge:BLUNDER  judge:clean
  human BLUNDER         11              6
  human clean            1              3
```

The dominant error cell is **human-blunder / judge-clean: 6**. The judge's
false *positives* were never the main problem — its false *negatives* are. On
`base_closed`, the arm Gate 3 reads, the human found a blunder in **5 of the 8**
answers the judge called clean.

This was structurally invisible until now. The unstratified queue was 82%
blundered, so it estimated *P(human blunder | judge blunder)* well and
*P(human blunder | judge clean)* not at all — and every judge miss lives in the
second. The instrument could not see its own dominant failure mode.

#### Gate 3 is optimistic, and by how much is a reweighting question

Correcting the judge's rate with the two conditional rates:

```
corrected = P(judge blunder)·P(human blunder | judge blunder)
          + P(judge clean)  ·P(human blunder | judge clean)
```

| arm | judged | corrected | n clean / n blunder |
| --- | --- | --- | --- |
| **`base_closed`** | **58%** | **70%** | 8 / 4 |
| `base_open` | 96% | 100% | 1 / 8 |
| `base_cards_open` | 92% | — | 0 / 2 |

`base_closed` = 0.577 × 0.750 + 0.423 × 0.625 = **0.697**. Gate 3's best arm is
twelve points worse than reported, and the gate still fails — the bar is 25% —
so no verdict moves. What moves is the margin, and *which* arm looks best: the
correction is largest exactly where the judge calls answers clean most often,
which is the arm the gate selects.

**The raw comparison would have said +24 points.** In-sample the human calls 81%
blundered against the judge's 57%. That gap is mostly selection: the queue is now
**43%** judge-clean against **18%** in the run. Reweighting to each arm's own mix
gives +12 for `base_closed`. Quoting the raw gap would have been a selection
effect wearing the shape of a result — the same shape as 21.39's harness bug and
21.75's unofferable charges, arrived at by a third route.

An arm with **no** adjudicated judge-clean answer gets **no** estimate rather
than one that silently assumes the judge was right, and `n_clean` prints beside
every row, because a correction resting on two verdicts is not a correction.

And the correction is judge-specific in *direction*: the 32B goes 58% → 70%,
Mistral 65% → **62%**. They call different answers clean, so the human corrects
them opposite ways. There is no single "true" blunder rate to be recovered here,
only a per-judge one — which is 9.9's rule surviving one level deeper than it
has been tested before.

#### Two bugs found writing this

The first draft pooled every form version into the conditional rates. A verdict
shown four entries ticked fewer boxes *because it had fewer*, so counting it as
"the human found no error" biases the correction toward the judge being right —
the exact quantity under measurement. Filtered to the current rubric: 70%, not
the 62% the pooled version reported.

The second: the reporting loop reached for `r`, the variable of a *finished*
loop, so every row re-opened the **last** run. Both judges printed identical
corrected tables under different names — and identical tables are precisely what
one expects when a parser is involved (21.76), so it read as a positive control
rather than as a bug. The run path rides on the result now.

### 21.82 At n=33 the two judges are indistinguishable, and the clean-verdict failure holds

Twelve more verdicts, 36 of 60 covered, and the numbers have stopped moving.

| | n | κ | one-flip range | precision | recall | P(human blunder \| judge **clean**) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | 33 | **+0.29** | [+0.18, +0.39] | 30% | 41% | **9/13 = 69%** |
| Mistral-24B | 33 | +0.11 | [−0.04, +0.25] | 30% | 53% | **7/9 = 78%** |

The 32B's kappa has gone **+0.63 → +0.27 → +0.29** as the sample went 13 → 21 →
33, and its fragility range has narrowed from [−0.08, +1.00] to a fifth of a
point. The first figure was one cell; this one is a measurement.

#### The headline replicates and strengthens

**When either judge calls a position answer clean, it is wrong about 70–78% of
the time.** That is now 13 and 9 judge-clean answers respectively, up from 8 and
6, and both rates rose. The judge's *false negatives* remain the dominant error
cell — 9 human-blunder/judge-clean against 1 the other way for the 32B.

Gate 3's correction holds: `base_closed` reads **58%** and corrects to **72%**
(n_clean = 10, up from 8). `base_cards_open` now has an estimate too, 92% → 88%,
on n_clean = 2 — which is why `n_clean` prints beside every row and should be
read before the number is.

#### And the two judges do not separate

Two kappas printed side by side invite ranking on the point estimates. +0.29
against +0.11 looks decisive. It is not:

```
paired on 33 answers, same human reference
  difference           +0.18
  bootstrap 95% CI     [-0.12, +0.53]      includes zero
  matched the human alone   3 vs 3
  exact McNemar        p = 1.000
```

Three answers only the 32B got right, three only Mistral did — a perfectly
balanced disagreement. **The two judges are not distinguishable on this sample**,
and a default-judge decision taken from +0.29 versus +0.11 would have been a
decision taken from noise.

`adjudicate --score` runs this whenever two runs are given and prints the
refusal in the same block as the kappas, because a caveat that arrives after the
exclusions list arrives too late. The bootstrap is seeded and reproduces
exactly — the same discipline the judge itself is held to (9.9), applied to the
statistics rather than to the model. Paired, because both judges graded
byte-identical text and an unpaired interval discards the pairing that makes the
comparison sharp; and McNemar alongside it, because when the discordant count is
six, an exact test with no distributional assumption is the honest one.

This is the third consecutive way the same sample has offered an encouraging
wrong number. 21.78: a kappa manufactured by pooling form regimes. 21.79: a
kappa resting on one cell. 21.81: a raw sample gap that was mostly selection.
Now a judge ranking that is within noise. Each was plausible, each pointed
somewhere different, and none survived being checked — which is the argument for
building the check into the tool rather than remembering to do it.

#### What the sample can and cannot now support

Supported: the 32B agrees with a human at κ ≈ +0.29 on the blunder call; its
clean verdicts are wrong roughly 70% of the time; Gate 3's best arm is ~12
points worse than reported and still fails by a wide margin either way.

Not supported: **which judge is better.** That needs either a much larger
sample or a larger true difference, and 21.79's ceiling still binds — 14 clean
answers is all three arms produced across 26 positions. More positions, not more
adjudication.

### 21.83 The notes report earns itself: two more classes the rubric has no entry for

21.77 rebuilt the note as the reviewer's reasoning and added `--notes` so
something would read it. With 34 v4 verdicts on file it has produced its first
result, and a prediction check on the way.

#### The coverage gap was mostly the form, and now it is measured

21.75 argued that most `not_covered` ticks were the *form* lacking protocol
entries rather than the *rubric* lacking them, on the strength of 30 of 33 notes
naming an entry `PROTOCOL_ERRORS` already had. The prediction: put those entries
on the form and `not_covered` should collapse.

| form | `not_covered` |
| --- | --- |
| v2 (strategy entries only) | **33/48 = 69%** |
| v4 (strategy + protocol) | **8/34 = 24%** |

It collapsed, and not all the way — which is the useful part. The residual 24%
is the genuinely uncovered material, and eight notes is few enough to read.

#### Three themes, two of them parser-decidable

Reading all eight, three recur:

1. **mana tapped and never spent** — *"Taps lands for no reason"*, *"it
   pointlessly taps the forests before passing wasting the mana"*, *"Taps mana
   unnecessarily"*
2. **a play made with no PHASE declared** — *"doesnt declare the change to the
   declare attackers phase"* (three notes)
3. **reasoning that misstates a card** — Doom Blade described as dealing "3
   damage to it" when it destroys

| class | fires | pre-protocol run | already convicted? |
| --- | --- | --- | --- |
| wasted taps | **11/78 (14%)** | 15% | entry 6 fires on **0 of 11** |
| missing PHASE | **10/78 (13%)** | 14% | nothing covers it |
| creature TARGET (21.80) | 6/78 (8%) | 7% | entry 3, for the wrong reason |

Both new classes are more frequent than the targeting one, both are stable
across two runs, and both fail for a reason worth stating.

**Entry 6 does not cover wasted taps.** Its wording — *"add up to MORE than the
spells it casts require"* — presumes spells were cast, and `payment_problems` is
silent when none were, so it fires on none of the eleven. Three of the eleven
have no entry firing at all. And the case is unambiguous rather than a judgement
call: mana empties at end of step, so tapping without spending never holds up a
trick — the way to represent holding removal is to leave the lands untapped.

**`phase_problems` deliberately does not cover a missing declaration.** Its
docstring says so: *"Returns [] when nothing was declared — silence is 'not
stated', not 'agreed'."* That was correct while the grammar merely *allowed* the
line. 21.60 made it required, so silence became a contract violation and the
check was never revisited. Third instance of 21.61's shape — *when a prompt
starts asking for new output, audit every consumer of that output* — and this
time the consumer that needed changing was one that had been deliberately
written to abstain.

A PASS-only answer is exempt from the phase check: it makes no play, entry 1
already describes it, and charging it here would double-charge 21.58's
do-nothing case, which is already the most over-charged answer in the set.

#### Still diagnostics, and the cost of promoting has gone up

All three are stored per-answer and printed per-run. None is a `PROTOCOL_ERRORS`
entry, for the reason 21.80 gave and more so: appending now costs **34** v4
verdicts rather than 15, since `verdict_is_current` invalidates on `n_shown`.

The stored runs were backfilled with all three fields, which is safe precisely
because they are pure functions of (answer, board) — the same property that lets
`protocol_truth` arbitrate between judges (21.76). The backfill asserts that no
pre-existing field changed on any of 228 arm-answers, and none did.

Fourth section ending in a stated decision rather than a taken one. That is the
shape: **the instrument measures and reports; the person whose time it costs
decides.** What is now measured is that ~35% of answers commit at least one
error the rubric cannot name — which is a better argument for a rubric revision
than any of the individual classes.

### 21.84 Entries 8-10 promoted, and regenerating the arms would produce identical bytes

The reviewer authorised both halves of 21.83's stated decision: promote the
three measured classes, and regenerate the arms. The first is done. The second
turns out to buy nothing, and finding that out cost one GPU minute rather than
a run.

#### The promotion

`PROTOCOL_ERRORS` goes 7 → 10, appended, so 1-7 keep their meaning and every
verdict ever collected still says what it said:

| # | entry | invalidating? |
| --- | --- | --- |
| 8 | a CREATURE spell cast with a TARGET | **yes** |
| 9 | TAP lines declared and nothing cast | no |
| 10 | a play made with no PHASE line | no |

`PROTOCOL_INVALIDATING` follows the line the reviewer drew for 5 versus 6 —
*"one is a truly invalid play while over tapping is a player blunder"*:

- **8 invalidates.** Casting a creature with a target is not a legal action.
- **9 does not.** Wasting mana is legal and merely bad, exactly like 6.
- **10 does not**, and this is the interesting one. Entry 4 (a *wrong* phase)
  invalidates because the play provably happened at the wrong time. Silence
  proves nothing: a turn cannot be called invalid because the player failed to
  narrate it. An unverifiable claim is not a false one.

**Gate 1 did not move.** Adding an invalidating entry should be expected to cost
some arm its `valid_turn` score, and it cost none — `base_open` 23%,
`base_closed` 58%, `base_cards_open` 19%, with **0 of 78** answers flipping.
Every answer entry 8 convicts was already convicted by entry 3, which also
invalidates. So the promotion buys a *reason* rather than a *conviction*: the
answer the reviewer marked `not_covered` now fires `[3, 8]` where it fired `[3]`,
and 8 is the one that says what actually went wrong.

`protocol_findings` decides all ten. 8 needs the card index and is `None`
without one, like 5 and 6; 9 and 10 read only the answer's own lines and are
always decidable. `test_eval` now asserts **every entry has a checker**, because
an entry a parser cannot decide is one the judge can be charged against with
nothing to confirm or refute it — and that loss is indistinguishable from the
class never firing. Confirmed by mutation: an eleventh entry with no checker
fails two assertions.

Cost, as stated and accepted: `n_shown` 11 → 14, so **36 verdicts** need
re-reading and coverage returns to 0/60. The answers themselves are untouched,
so this is a re-read with three more boxes, not fresh work.

#### Regenerating the arms would produce byte-identical answers

`eval_positions.generate` passes no sampler to `mlx_lm.generate`, which means
greedy decoding. Verified by running rather than by reading the signature: two
successive calls on `pos-trigger-ordering-0001` were identical to each other
**and identical to the stored answer**.

So "regenerate the arms to get a new set" is a no-op with the same model, prompt
and adapter. It would invalidate 36 verdicts a second time and hand back the
same 78 answers. Something has to actually change:

| change | gives | costs |
| --- | --- | --- |
| a fourth `ft_cards_open` arm | 26 genuinely new answers, and the first test of whether fine-tuning helps *gameplay* | not comparable to any 3-arm run on blunder rate (21.5) |
| a stronger base model | a different question, not a new sample of this one | |
| temperature > 0 | variance | forfeits "the harness is deterministic, so a number that moved means something changed" |

Worth recording that the determinism check took under a minute and removed a
several-hour run from the plan. The stored `gameplay_fingerprint` is
`d094e3934d2d` on all three current runs and matches the live prompt, so the
warning in CLAUDE.md about stored positions being stale refers to the older
`positions_n22` family and not to these.

### 21.85 A reference line the parser has to agree with, and the one board where passing is right

The reviewer asked for a place to record the 100%-correct answer for a board.
Built as `reference_actions`, and it found a bug in the gold set on the way in.

#### The field, and why it is not 21.55 again

`positions.jsonl` carried `answer` (prose) and `key_points` (bullets) and no
canonical line in the grammar the model is actually required to emit. That gap
matters: a prose answer cannot be an oracle control, because the parser scores
prose as illegal — *"prose about a play parsing AS that play"* is already a
documented trap here.

The name is taken from scenario steps, which have carried `reference_actions`
since 21.71. One name, one meaning.

The reason this is not another field collected and read by nothing (21.55) is
that a reference line is the one claim about gold data a machine can settle. It
asserts *this is the perfect output*, so it must survive exactly what every arm
answer survives. `check_reference` requires it to parse, requires every play to
be in `legal_actions`, and requires no decidable `PROTOCOL_ERRORS` entry to
fire. A submission that fails is **refused at the ingest**, not stored with a
warning. `validate_position` re-checks stored lines too, so a reference that was
clean when written and stops being clean when the rubric grows — the list is
append-only — fails loudly rather than quietly becoming wrong.

It works immediately, and on the entries promoted one section earlier:

```
refused pos-blocking-0001:  not a legal play here: 'CAST Black Lotus'
                            commits PROTOCOL_ERRORS entry 3
                            commits PROTOCOL_ERRORS entry 10
refused pos-blocking-0002:  commits PROTOCOL_ERRORS entry 1
                            commits PROTOCOL_ERRORS entry 9
```

Entries 9 and 10 existed for a day and are already catching bad **gold**, not
just bad model output.

#### Where it sits, and why that is deliberate

The block renders **below** the grading panels, with the legal plays behind a
collapsed `<details>`. `legal_actions` is not judge output — it is the same list
the closed arm is handed — so showing it breaks no blind rule. But it *is* the
answer key for entry 3, and a reviewer who reads it before ticking boxes stops
being independent evidence on the one entry where the judge and the parser
already agree (21.74). Grade first, author second.

The server validates shape and size only. It has no card index, no positions
file and no parser — `common.py` must stay pure stdlib and the grammar lives in
`gameplay/actions.py` — so the real check runs locally at ingest. That is the
invariant the rubric form has always kept: **the server never writes the gold
set.** Re-submitting appends and the newest row per (record, author) wins, so a
correction needs no row deleted.

#### The bug it found: passing is correct on exactly one board

Stage 3 introduced `sample-stage3-payment-0005` — one Island, Chart a Course at
`{1}{U}`, Shock at `{R}`. Nothing is castable, so `legal_actions` is empty and
the correct answer is to do nothing. Writing the test for "PASS alone is valid
where nothing is legal" failed, and the test was right:

```
'PASS'   fires [1]   valid_turn=False        <- the CORRECT answer
```

`PROTOCOL_ERRORS` entry 1 is *"the answer makes no play: it only passes"*, and
it is invalidating. So on the one board where doing nothing is the whole point,
the right answer scored as an invalid turn.

This is the "one name, two meanings" trap in its exact documented form. *Only
passes* means **declined to play when a play was available** (a blunder, which
is what 21.58 built the entry for) and **correctly did nothing when nothing was
legal** (right). The two readings agree on all 31 boards that enumerate a play
and come apart on the first one that does not — and stage 3's whole subject is
recognising that nothing is affordable, so this position type will only become
more common.

Entry 1 is now conditioned on the board offering a play. Conditioned on the
*position's* list rather than on what the arm was shown, because the open arm
never sees `legal_actions` while the board still has them; and an **absent**
list is left alone, since that is "not stated" rather than "nothing is legal" —
the same distinction `phase_problems` draws.

Impact, measured rather than assumed: **one** stored answer flips, and it is the
arm that answered correctly.

| arm | valid_turn before | after |
| --- | --- | --- |
| `base_cards_open` | 24% | **26%** |
| every other arm | — | unchanged |

The stored `positions_n32_scenarios` runs were corrected rather than left to
disagree with the checker, with an assertion that no field other than
`valid_turn` and `protocol_truth` moved on any of 170 arm-answers. A stored run
whose numbers disagree with the code that produced them is a trap for whoever
reads it next.

### 21.86 The first reference line, and three things it found

One reference submitted through the deployed form, for `pos-combat-math-0005`:

```
PHASE Pre-Combat Main Phase
CAST Shock TARGET Grizzly Bears
PASS
END PHASE Pre-Combat Main Phase
PHASE Declare Attackers
ATTACK Centaur Courser -> Opponent
PASS
END PHASE Declare Attackers
```

Every one of the three problems it exposed is about the *instrument*, not the
reviewer, which is what asking a person for the perfect answer is for.

#### 1. A line outside the grammar was accepted in silence

`END PHASE <step>` is not a verb. `parse_output` did not fail on it — it routed
both lines to `ParsedOutput.ignored`, and **nothing outside `test_actions.py`
reads `ignored`**. So `check_reference` accepted two invented lines without
comment and complained only about the ATTACK.

That bucket is deliberate and right for **arm answers**: a model that reasons
aloud must not have its explanation parsed as plays, which is 21.61's whole
subject. It is wrong for a **reference**, where there is no prose — every line
is a claim about the correct play, so a line the grammar never saw is a line the
reviewer believed they had written. `check_reference` now refuses them by name.

The general shape is worth naming, because it is the inverse of a trap already
in this file. 21.61 was *prose parsing AS a play*; this is *a play-shaped line
parsing as prose*. The same `ignored` bucket produces both, and which one is the
bug depends entirely on whether the text was supposed to contain prose.

#### 2. The form asked for a grammar it never showed

The **model** is given `ACTION_GRAMMAR` in every gameplay prompt. The reviewer
authoring the correct line was not. Both mistakes in the submission are what a
careful person guesses without it:

- `END PHASE <step>` — reasonable, since `PHASE <step>` exists and a turn has to
  advance somehow;
- `ATTACK Centaur Courser -> Opponent` — reasonable, since `BLOCK <blocker> ->
  <attacker>` uses exactly that arrow. The grammar is `ATTACK <creature>,
  <creature>`.

**Models make the same arrow guess**: 6 of 418 stored answers write `ATTACK … ->
…`. So this is a real confusion in the grammar's design rather than one
person's slip, and the arrow means "assign X to Y" in one verb and nothing in
the other.

The form now serves `common.ACTION_GRAMMAR` from `/api/grammar` and shows it
collapsed beside the legal plays. Served from the constant rather than copied,
so the form cannot drift from what the model is told — the same one-definition
rule `build_rag_messages` follows for prompts.

#### 3. A position whose correct line spans two phases

The remaining charge is entry 4, *the PHASE line names a step other than the one
the position is in*. It is correct on its own terms: `pos-combat-math-0005` has
`phase: "precombat main"`, and the answer declares `Declare Attackers`.

But the position's own `legal_actions` are:

```
CAST Shock TARGET Grizzly Bears     <- main phase
CAST Shock TARGET opponent          <- main phase
ATTACK Centaur Courser              <- combat
```

So the board enumerates plays from two steps while recording one, and **any
correct line that both casts and attacks must trip entry 4**. The reviewer's
answer is right about the game and wrong about the rubric.

Not resolved here, because the three available fixes are different claims about
what a position *is*:

- drop `ATTACK` from the legal list, making it a single-decision position;
- let entry 4 accept a *later* step in the same turn, so a line may advance;
- treat it as a turn scenario — which is what `turns.py` exists for, and what
  stage 6 (full-turn validity) is about.

The third is the principled one: a position whose correct answer spans phases is
a sequence, and 21.71 built the harness for sequences precisely so that a
multi-step line is scored step by step rather than crammed into one board. But
it is a decision about the gold set's shape, so it is stated with its options
rather than taken.

### 21.87 `END PHASE` and a direction for `ATTACK`: the grammar could not express a turn

Two grammar changes, both from the reviewer, both correcting places where the
notation could not say something the game requires.

#### `PASS` was doing two jobs

*"`PASS` means for whatever action a player takes they are passing priority to
make a reaction to that play to each opponent. `END PHASE` is moving out of the
declared phase and into the next phase in the turn."*

Those are different acts. Passing priority opens a window for each opponent to
respond to the play just made (117.3); ending a phase leaves the step. The
grammar had only `PASS`, so **there was no way to write "I am done here, move
to combat"** — which is most of what a turn is.

That matters more than a missing convenience. Stage 6 is *full-turn validity*
and stage 7 is *full-game validity*; both are sequences of steps, and neither
was expressible in the notation they were to be scored in. The harness for
sequences has existed since 21.71 and the vocabulary for them did not.

`END PHASE [<step>]` is a **declaration**, like `PHASE` and `TAP`: it changes
nothing `legal_actions` enumerates, so adding it cannot move Gate 1. The step is
optional but validated against `PHASE_NAMES` when given, the same closed
vocabulary `PHASE` uses — otherwise "End phase two of my plan" becomes a
declaration.

The reviewer also supplied the exception worth recording: the active player ends
each phase **except declare blockers**, which the defending player ends.

#### The arrow meant something in one verb and nothing in the other

`BLOCK <blocker> -> <attacker>` used an arrow to mean *assign the left to the
right*. `ATTACK <creature>, <creature>` had no direction at all — but a creature
attacks a player or a planeswalker that player controls (506.2), and those are
the only legal directions, so the direction is part of the play.

The near-miss was measured on both sides. **6 of 418** stored answers wrote
`ATTACK … -> …` before the grammar allowed it, and the first human-authored
reference line did too. When a model and a careful person independently reach
for the same syntax, the grammar is what is wrong.

Now `ATTACK <c>, <c> -> <defender>`, one line per defender, so an attack can be
split across a player and a planeswalker.

**Backwards compatibility was the risk, and it is handled explicitly.** 32
stored positions enumerate `ATTACK <creature>` with no direction. Requiring one
would have scored every correct attack illegal and arrived as *"the model got
worse at combat"* — 21.61's shape exactly. `match_to_legal` treats an
undirected `legal_action` as not constraining the direction, while a position
that **does** name a defender still requires the answer to match it.

Verified rather than assumed: re-scoring **248 stored answers** under the new
grammar produced **0** changes to `all_legal` or `n_plays`. The six arrow
answers do now parse differently — attackers split correctly and the defender
extracted, where before the whole tail was one attacker name — and stay illegal
for the reasons they already were, so the check is not vacuous.

#### And entry 4 had to follow

The reviewer's reference line then failed on one charge only: entry 4, *the
PHASE line names a step other than the one the position is in*. Correct on its
own terms — `pos-combat-math-0005` records `precombat main` while enumerating
both a main-phase cast and a combat attack, so **any** correct line that does
both declares a second step.

21.86 left that open with three options. `END PHASE` answers it: only the FIRST
declaration is a claim about where the position is, and a later one that follows
an `END PHASE` is the answer *advancing* rather than misreporting. A step
declared **without** ending the previous one is still a jump and still charged,
and the opening step is still checked. Zero of 248 stored verdicts changed,
because nothing written before this used a verb that did not exist.

The reviewer's line is now accepted and is the first `reference_actions` in the
gold set.

**The fingerprint moved** — `4a0fee3c83ee` → `f050d4aed707`. Every stored
position run is now incomparable to a new one, which is what
`gameplay_fingerprint` exists to make loud. The arms have to be regenerated
before any new position number means anything, and unlike 21.84's no-op that
regeneration will produce genuinely different answers, because the prompt is
genuinely different.

### 21.89 Eight reference lines, an em dash, and a grammar with only one player in it

The reviewer authored ten more lines. **Seven accepted, one refused**, and both
of the initial failures were the instrument's rather than theirs.

#### A bug in the validator, found by the first hard parse failure

`check_reference` crashed: `ParseFailure` carries `raw`, not `line`. The branch
had never run — the first submitted reference produced *ignored* lines and
*illegal* plays, neither of which is a `ParseFailure`, so the one path that
formats a parse error was written and never executed until a reviewer typed
something the parser genuinely could not read. A test that exercises three of
four branches looks the same as one that exercises four.

#### An em dash is the same arrow

```
BLOCK Fog Bank -> Serra Angel          parsed
BLOCK Centaur Courser —> Grizzly Bears refused
```

Two consecutive lines of one submission, same intent, different bytes: macOS
autocorrect turns `->` into `—>` after a space. The parser saw a `BLOCK` with no
arrow at all and refused a correct block.

`—`, `–`, `−` before `>` and a bare `→` now normalise to `->`, beside the
existing bracket stripping. Costless in the other direction: **0 of 418** stored
model answers use any of them, so this only ever rescues input. With it, the
line is accepted.

Same family as the `[TARGET <x>]` brackets the grammar once invited models to
copy — a correct play scored wrong on a character.

#### The step vocabulary was invisible, exactly as the grammar had been

The remaining refusal named `PHASE Declare Damage` and `PHASE End`. Both are
reasonable — *declare damage* follows *declare attackers* / *declare blockers*,
and *end* is what the step is usually called out loud. Neither is in
`PHASE_NAMES`, which holds `combat damage` and `end step`.

That vocabulary is deliberately closed: it is what stops "Phase two of my plan"
becoming a declaration. But it was never shown to the person being asked to
write in it — the same defect 21.86 fixed for the action grammar, one field
over. `/api/grammar` now serves the accepted step names alongside the verbs.

The list is mirrored in `common.PHASE_VOCABULARY` rather than imported, because
`common.py` must stay pure stdlib for the deployed server and `PHASE_NAMES`
lives in `gameplay.actions`. Two copies of a list is the shape this repo has
been bitten by, so `test_eval` asserts they are equal — the only thing keeping
them honest.

#### The finding: the grammar has one player in it

The refused line was not a near-miss. It walks a whole turn — every phase in
order, `PASS` in each — and then crosses into the opponent's turn:

```
PHASE Opponent Declare Attackers
PASS
ATTACK Goblin Guide -> Player          <- Goblin Guide is the OPPONENT's creature
TAP Mountain FOR {R}
CAST Shock TARGET Goblin Guide
```

`ATTACK` is refused because Goblin Guide is not the answering player's creature,
and the check is right. But what the line is *trying* to say — *the opponent
attacks, and I respond* — has no expression at all. **Every verb in the grammar
is an action by the answering player.** There is no way to write what the
opponent does, so there is no way to write a response to it.

That is not a gap in the notation so much as a boundary in the design, and
21.71 already drew it: a turn scenario advances the board on the **reference**
line, and each step is a complete position. What the opponent does belongs to
the *board* between steps, not to the answer. Written as a scenario, the
opponent's attack is step 2's battlefield and the answer to step 2 is
`CAST Shock TARGET Goblin Guide` — one decision, checkable, with no second
player in the grammar.

So the honest answer is that stage 6 (a full turn) is now expressible and stage
7 (a full game) is not, and the missing piece is scenario authoring rather than
more verbs. Adding an `OPPONENT` prefix would make every existing check ask
"whose action is this?" — including `legal_actions`, which enumerates the
answering player's plays and nothing else.

Also worth separating: `sample-stage3-payment-0003` is a **payment** position.
Its key points are which land pays which half of `{1}{R}`. The correct reference
for it is five lines, and it validates clean. The full-turn line is a different
and larger claim about a board that was not asking for one.

Eight positions of 32 now carry a reference, and every one passes
`validate_position`.

### 21.90 A view for authoring, and the reference field that was never checked

Two requests: separate authoring from adjudication, and make scenario steps
authorable so a full game can be written down. Building the second found that
the first was hiding a defect.

#### The two jobs have different shapes

The reference box lived inside the adjudication form, so editing a line meant
finding its record in a queue ordered by *grading* status. Grading walks a
sample once; authoring revisits a board until the line is right. `/reference`
is its own page: every board in a sidebar, a tick against the ones that have a
line, free navigation, and the board, key points, legal plays, action grammar
and step vocabulary all in view while typing.

The deploy boundary did not grow. `tasks.json` is keyed `(record, arm)` because
grading is per answer; `/api/reference-boards` collapses the arms server-side,
so one file still ships and the `.dockerignore` allowlist is untouched.

#### Scenario steps are boards, and their reference was invisible

`expand_steps` already returns a step position-shaped (21.71), so a step needs
no separate authoring path — it needs to be *in* the list. Adding it exposed
this, in `expand_steps`:

```python
pos["legal_actions"] = step.get("legal_actions") or []
pos["key_points"]    = step.get("key_points") or []
pos["common_errors"] = step.get("common_errors") or []
# ... and never reference_actions
...
prior.extend(step.get("reference_actions") or [])     # read, only for this
```

The step's own correct line was **read for teacher forcing and never set on the
position**. So it reached no consumer: the authoring form showed both steps as
unwritten while the scenario file held lines for them, and — worse —
`check_reference` had never seen them. The one field in this repo that claims to
be parser-verified was the one field nothing verified.

Carried through under the same name a position uses, and both existing lines
were checked for the first time. **Both failed:**

```
turn-payment-combat-0001::step1  ->  commits PROTOCOL_ERRORS entry 10
turn-payment-combat-0001::step2  ->  commits PROTOCOL_ERRORS entry 10
```

Entry 10 is *a play made with no PHASE line*. They were authored before 21.60
made the line mandatory and nothing re-read them since, which is exactly what
`validate_position` re-checking stored references exists to prevent — the check
was written in 21.85 and these were out of its reach. Fixed by prepending the
step's own declared phase; both clean.

#### And that fix needed a second one

Adding `PHASE precombat main` to step 1's reference would have written
*"Already done this turn: PHASE precombat main; TAP Mountain…"* into step 2's
board, because `prior` takes the reference line verbatim. A phase declaration
says **where** the turn is, not what was done in it. `TAP` does belong — a land
tapped in step 1 is still tapped in step 2, and that is the state the next step
needs. `prior` now filters `PHASE`/`END PHASE` and keeps everything else.

Third time a change to the grammar has required auditing what consumes its
output (21.61, 21.87, here). The pattern is stable enough to state as a rule:
**a new mandatory line has to be checked against everything that reads a line,
including the things that read it for a purpose other than scoring.**

#### Consistency, as a command

`positions.py --check-references` runs every stored line — positions and
scenario steps — through one `check_reference`, because a step *is* a position
once expanded and a second checker would be a second place for them to disagree.
It prints whether the card index loaded, since without it entries 5, 6 and 8 are
undecidable and the check is weaker than it looks.

```
10/34 boards carry a reference line
  positions      : 8/32
  scenario steps : 2/2
  card index     : loaded
all reference lines are consistent with their boards
```

The ingest writes a step's line back to `turn_scenarios.jsonl` and a position's
to `positions.jsonl`, keyed off the `::step` in the id — one command, two
destinations, no second form.

### 21.91 One identity under three keys, and what "unified" turned out to mean

The reviewer edited nine reference lines and asked whether they were consistent.
Ingesting them found a defect first, and answering the question needed a
different check than the one that existed.

#### Every new submission had an empty author

Three pages stored the reviewer's name under three different `localStorage`
keys — `adjWho`, `author`, `mlr_author` — so a name typed in one view was
invisible in the next, and the new authoring page started blank. Nine
submissions arrived unattributed.

Attribution riding on each submission is the invariant that lets one file hold
several authors, so an empty one is a lost verdict rather than a cosmetic gap.
One definition now, `AUTHOR_JS`, injected into all three pages, reading the
legacy keys once and migrating them.

`--ingest-references --author` attributes rows the form did not stamp. It fills
only an EMPTY field and never overwrites a name, because it acts on the
operator's word: 21.65 records the harm of stamping old submissions from a
source that could not know who wrote them, and the difference here is that the
person told me.

**And the fix shipped broken for one iteration.** `AUTHOR_JS` was defined below
two of the pages that reference it, and my "already injected" guard matched the
*call* rather than the *definition* — so two deployed pages called a function
they did not define. That parses perfectly and throws at runtime, blanking the
view. The JavaScript parse-check added after the `#`-comment incident cannot see
it, and neither can the dead-CSS check.

`test_webui` now collects every helper this repo defines anywhere and asserts
that a page calling one also defines it. Third language, third instance of the
same shape: **a page is only whole if what it references lives on it.**
Confirmed by mutation.

#### "Consistent" is two questions, and only one had a check

`check_reference` asks whether a line is legal on its board. Ten of ten pass.
That is not what the reviewer was asking: they had been editing for *"better
unification of expectations"*, which is whether the lines agree with **each
other**. A set of gold answers that each open differently teaches the opening
rather than the play.

`reference_consistency` reports the shape of the set:

```
10 reference lines
  open with PHASE: 10/10
  use END PHASE at all: 7/10
      differs: pos-mulligan-0002, pos-trigger-ordering-0002,
               turn-payment-combat-0001::step1
  contain a PASS: 9/10
      differs: turn-payment-combat-0001::step1
  final action: END x6, PASS x2, PHASE x1, CAST x1
  step names spelled one way: 5/6
      'declare attackers' appears as ['Declare Attackers', 'declare attackers']
  ASCII arrows only: 9/10
      non-ASCII: pos-blocking-0003
```

It **reports and does not enforce**, because some of the differences are
correct. A mulligan happens before any phase can be ended, so no `END PHASE`
there is right; the same absence on a combat board probably is not. The tool
names the minority and stops.

Two entries are worth separating from the rest:

- `turn-payment-combat-0001::step1` contains **no `PASS` at all**, and the
  gameplay prompt's own instruction is *"End with a single PASS."* A reference
  that does not follow the instruction the model is given is a standard
  disagreeing with its own brief.
- `pos-blocking-0003` keeps an em-dash arrow on one `BLOCK` line and an ASCII
  one on the other. It parses, because 21.89 normalises it — which is exactly
  why it survives in stored gold that nobody re-reads. The check that never
  fails is the one that lets a difference persist.

The reported distribution `END x6, PASS x2, PHASE x1, CAST x1` is the real
answer to the question asked: **the set is unified on how it opens and not yet
on how it closes.**

### 21.92 Where a reference line ends, and why a PASS is the boundary

The reviewer, on why the closing action was hard to settle:

> *"Deciding on an ending action is difficult as each PASS comes with possible
> Opponent interaction that would change whatever plan goes beyond that."*

That is right, and it decides the question rather than complicating it. `PASS`
offers each opponent a window (117.3). Anything a reference line says after one
is conditional on that window being declined — and the board cannot support
that claim, because nothing in the position says what the opponent will do.

So a reference that continues past a `PASS` is asserting something it does not
know. But not every continuation asserts the same amount, and the distinction
is what makes the rule usable.

#### The measurement

| | |
| --- | --- |
| lines continuing past a `PASS` | **7 of 10** |
| lines taking a **play** after one | **1 of 10** |

The other six add only `END PHASE` or `PHASE`. Ending a phase once nobody
responded is how a turn proceeds; it asserts almost nothing. Taking another
*play* asserts the opponent declined a window in which they could have acted.

Only `pos-combat-math-0005` does it:

```
PHASE precombat main
CAST Shock TARGET Grizzly Bears
PASS                                 <- the opponent may respond here
END PHASE precombat main
PHASE declare attackers
ATTACK Centaur Courser -> Opponent   <- assumes they did not
PASS
END PHASE declare attackers
```

If the opponent has a trick, the attack is a different decision. The line is not
wrong about Magic; it is over-claiming for a single board.

#### The rule this yields

**A reference line ends where the answer stops being determined by the board.**
Declarations may follow a `PASS`. A play may not — and a line that needs one is
a **scenario**, where the next step's board *states* what the opponent did
rather than the answer assuming it. That is the structure 21.71 already built,
and it is the same answer 21.89 reached from the other direction, where the
grammar could not express an opponent's action inside one answer.

So the three ways a line can close are not a style inconsistency to be
normalised away:

- **`PASS`** — the board's decision is made and the opponent acts next
  (`pos-mulligan-0002`, `pos-trigger-ordering-0002`);
- **`END PHASE`** — the window passed unused and the turn moves on, asserting
  nothing about a later decision (six lines);
- **a further play** — only valid when a board states the intervening window's
  outcome, which a single position never does.

`reference_consistency` reports the split and names the one line that crosses
it. It still does not enforce: the fix for `pos-combat-math-0005` is to shorten
it or to promote it to a two-step scenario, and which is right depends on what
the position is meant to teach — combat math, in that case, which argues for
the scenario.

The general shape is worth keeping. The reviewer's difficulty was not a gap in
the tooling but a fact about the game that the tooling had no way to express,
and the fix was to make the distinction measurable rather than to pick a
convention and enforce it. A standard that flattens a real rules distinction
would have produced consistent gold data teaching something false.

### 21.93 Shorter answers, and the case for widening categories that the data does not yet make

The reviewer, after 21.92:

> *"This feels like a case that can be made for shortening expected actions and
> instead widening the category types for each move."*

Half of that is strongly supported and half is not, and the half that holds is
the one that matters.

#### Shortening: supported

Across the ten stored reference lines:

| | median | range |
| --- | --- | --- |
| total actions | **4.5** | 3–8 |
| actual **plays** | **1.5** | 1–3 |

Six of ten encode exactly one play. The three-play cases are `TAP`, `TAP`,
`CAST` — one decision plus the payment the grammar requires for it. **Most of a
reference line is protocol scaffolding, and the decision itself is usually a
single move.**

That is what makes the position the right unit: it asks one question, and the
answer to one question is one play plus whatever the notation demands around it.

The check follows directly. Grouping a reference's plays by the phase they
happen in, a line with plays in two phases is two positions written as one —
and it is the same line the play-after-a-`PASS` check flags, arrived at from the
other side. One of ten crosses it:

```
pos-combat-math-0005 plays in 2 phases:
    precombat main:    CAST Shock TARGET Grizzly Bears
    declare attackers: ATTACK Centaur Courser -> Opponent
```

Two independent checks, built for different reasons, naming the same line. That
is the useful kind of agreement: 21.92's rule is about what a board can
*assert*, this one is about what a position *asks*, and a line that violates one
violates the other because they are the same constraint seen from two sides.

#### Widening: not yet

The proposal was that narrower answers should be paid for with more category
types. The evidence does not support it — and my first pass at gathering that
evidence over-claimed.

A crude keyword sweep suggested **7 of 32** positions turn on targeting with no
category for it. Requiring instead that a `common_error` actually name choosing
the *wrong* target — "should target the player instead of Goblin Guide" — gives
**3 of 32**, and all three are `payment` positions where targeting is a
*distractor* inside the question rather than the question. That is not a missing
category; that is a well-constructed distractor doing its job.

Recorded because the first number was mine and was wrong in the encouraging
direction: a regex over prose measures which words appear, and a category is
about what the position *turns on*. The same distinction 21.80 required before
promoting an error class — measure the thing itself, not a word that co-occurs
with it.

#### The asymmetry worth keeping

Categories are strings used by a dropdown and a membership check, so widening
them changes nothing already recorded. `PROTOCOL_ERRORS` is append-only because
the judge returns error NUMBERS, so growing it costs every collected verdict
(21.78, 21.84 — 36 of them). **Widening the taxonomy is the cheap direction and
lengthening the rubric is the expensive one**, which is what makes the
reviewer's instinct right even where this particular evidence does not carry it.

So the path is: shorten the answers, split the positions that were doing two
jobs, and widen the categories **when a split produces a position that does not
fit one**. That gives the same taxonomy growth as an evidence trail rather than
as a guess — the disposition 21.80 and 21.84 already established for rubric
entries, applied one level up.

### 21.95 The done-arm lock needed a way back

`0d0e1bb` hardened the grouped adjudication save: a saved arm's controls are
disabled, and the submit loop skips it. Both are right — the page posts every
arm in a record, so revisiting a partly-graded group re-posted the ones already
saved. The ingest dedup would have absorbed those, but `submissions.jsonl` is
the file every analysis is pulled from, and duplicate rows in it are noise in
the source of truth rather than in a derived number.

What it did not have was an unlock. Checked rather than assumed: no re-open or
re-adjudicate affordance existed anywhere in the page.

That collides with a workflow this repo deliberately supports. A verdict is
identified by `(key, author, answer_sha, n_shown)` **because** re-adjudication
happens — 21.78 added the fourth component after twelve re-adjudications were
silently dropped, and the reviewer has since redone verdicts more than once. A
lock with no way back means a reviewer who changes their mind cannot act on it
in the form; the only routes back were a rubric change (which bumps `n_shown`)
or an arm regeneration, both of which invalidate far more than the one verdict
being corrected.

The two problems are different and want different answers:

| | |
| --- | --- |
| a grouped save re-posting arms nobody touched | the lock |
| a reviewer correcting a verdict | the unlock |

`bindReopen` clears the lock on **one panel, in place** — it does not call
`render()`. Re-rendering would rebuild the whole record and discard whatever is
typed into the record's other panels, which is the entire reason the page groups
them (`0684055`). So it removes `disabled` from that panel's inputs, drops the
`saved` tag, clears `arm.done` so the submit loop stops skipping it, and
recomputes the record's own `done` and the progress counter.

Nothing changes on the server: a re-opened arm posts a normal verdict, and the
ingest keeps it beside the earlier one exactly as it keeps any other regrade.

Three checks, all mutation-confirmed: the control exists, re-opening clears the
flag the submit loop reads and re-enables the inputs, and it does **not**
re-render. The last one is the interesting assertion — it is not about
correctness in isolation but about not destroying work in the panels next to
it, which no test of the endpoint could see.

### 21.96 Authoring a scenario, and the reference line making entry 1 decidable

21.89 established that the grammar has one player in it: an opponent's action
cannot be written inside an answer, so a line that crosses an opponent's window
belongs in a **scenario**, where the next step's board states what happened.
The harness for scenarios has existed since 21.71 and the way to author one had
not.

#### Steps are authored whole and stored as a diff

A step is authored as a **complete board**, because that is what a person can
read and check. A scenario **stores** only what changes between steps
(`STEP_OVERRIDES`), because that is what keeps one board from being restated
five times and drifting on the fourth.

Both are true at once, so the diff happens in `turns.scenario_from_submission`:
step 1's board becomes the scenario's base, and each later step keeps only the
fields that actually differ. Server-side for the reason `position_from_form` is
— the browser never constructs the stored schema, so the form can change shape
without the records changing shape.

Measured on the turn cycle 21.89 could not express — hold the burn, then kill
the attacker:

```
step 2 overrides stored: ['active_player', 'battlefield', 'known_information']
```

Three fields, from a board with nineteen. The hands, lands, life totals and
libraries are inherited because they did not change, and nothing had to be
retyped to say so. In the form, **adding a step clones the previous one** and
blanks only the three fields that are per-step by definition — the rubric and
the reference line.

A scenario is **refused whole**. A half-written turn is worse than none:
`expand_steps` teacher-forces the board along the reference line, so one bad
step silently changes every board after it.

#### The reference line settled entry 1

Authoring that cycle exposed something the earlier fix could not reach. 21.85
stopped entry 1 (*"makes no play: it only passes"*) firing on a board whose
`legal_actions` is empty. But there is a second way passing is correct, and it
is the harder one: **declining an available play**. Holding removal for the
attack is a real skill, `legal_actions` is not empty there, and the correct
answer fired entry 1 as a blunder.

The board's own reference line settles it. If the 100%-correct answer only
passes, an answer that only passes cannot be committing *makes no play* — the
gold answer makes none either.

That check did not exist until references did, which is what the authoring work
buys. Entry 1 was previously decidable only from the **shape of the board**; it
is now decidable from **the answer the board is known to have**. Re-scoring 248
stored answers changed 0 verdicts, because no stored board yet has a
passing reference — purely additive, and waiting for the boards that need it.

#### A third mirrored constant

`POSITION_CATEGORIES` now exists in `common.py` as well as `gameplay.positions`,
because the deployed form needs the dropdown and `common.py` must stay pure
stdlib. That is the same arrangement `PHASE_VOCABULARY` has, and the same risk:
two copies of a list, kept honest only by `test_eval` asserting they are equal.
Three such pairs now exist. Worth watching — the fourth is the point at which
the boundary needs a better answer than "mirror it and assert".

### 21.97 One home per vocabulary, and a check that keeps it that way

Four closed vocabularies were each defined in more than one module. All four
agreed when anyone last looked, which is the only reason none had bitten.

| constant | homes | shape |
| --- | --- | --- |
| `DIFFICULTIES` | **3** — `positions`, `label_store`, `validate_gold` | list, list, set |
| `CATEGORIES` | 2 — `label_store`, `validate_gold` | list and set |
| `POSITION_CATEGORIES` | 2 — mine (21.96) | both tuples |
| `PHASE_NAMES` / `PHASE_VOCABULARY` | 2 — mine (21.89) | **different names** |

The last is the worst shape: a mirror under a different name is invisible to a
scan that looks for a repeated identifier, so it can only be found by knowing it
is there.

All four now live in one section of `common.py` and every other module
re-exports. That is where CLAUDE.md already said to put shared values, and the
deploy boundary forces it regardless — `rubric_server.py` ships with only
`common.py`, so anything the public form validates against had to be there. They
are plain strings, so the pure-stdlib rule costs nothing.

They are **tuples**, so no consumer can mutate the shared vocabulary for
everyone. That surfaced two `[""] + CATEGORIES` sites in `webui.py`, now explicit
`list()` calls.

`_STOP` is deliberately **not** merged. `audit_sft._STOP` is contamination-overlap
tokens and `common._STOP` is rubric-lint stopwords: two different lists doing two
different jobs that happen to share a private name. Merging them would have been
a real bug — the right fix there is distinct names, and the duplicate check
allowlists it for exactly that reason.

#### The check matters more than the cleanup

Consolidating today does not stop a fifth appearing, and a duplicate is invisible
until the copies drift — at which point the symptom is a value that is valid in
one half of the program and rejected by the other.

`test_imports.py` now fails on any module-level collection constant defined in
two production modules. Re-exports correctly do not count, since only assignments
are inspected: **one home, any number of doors.** Test files are excluded — a
fixture named after the thing it stands in for is a fixture, not a second home,
which `test_eval_positions`' own `ARMS` demonstrated immediately.

Mutation-confirmed by re-adding `DIFFICULTIES` to `validate_gold`.

The equality assertions 21.89 and 21.96 added are replaced by identity checks
(`PHASE_NAMES is common.PHASE_NAMES`) — a stronger guarantee than any test,
because there is now only one object to be wrong about.

### 21.98 Every position is generated, and that qualifies the whole gameplay track

Stated by the reviewer, and it is the most important fact in this file about how
to read the sections above it: **all 32 positions are machine-drafted**, with
minor human corrections in places. Not the seed file — the set.

14.6 measured hand-authored rubrics beating machine drafts by a wide margin,
**r +0.30 → +0.62**, and CLAUDE.md carries that caveat scoped to
`positions_seed.jsonl` as "plumbing verification, not gate evidence." That
scoping was too narrow. The caveat covers the gameplay track.

#### What it qualifies

Every gameplay number was measured on generated boards **and generated rubrics**:

- Gates 1, 2 and 3, including "the eval discriminates" at 79%
- blunder rates, the arm ranking, and `valid_turn`
- 37% parser-checked protocol precision (21.74), and its per-class bimodality
- the judge-vs-human kappa work (21.79–21.82), and "the judge's clean verdicts
  are 70–78% wrong"

The sharpest case is the rubric-coverage thread. 21.47 through 21.83 read
`not_covered` as evidence that **the rubric was missing entries** — the 69% → 24%
collapse, and the three classes promoted in 21.84. If `common_errors` is
generated, then "no entry describes this" partly means *the generator did not
think of it*, which is a fact about the generator and not about the domain. That
reading was available throughout and was not taken. The promoted entries are
still real error classes — they were measured against parser truth, not against
the rubric — but the *coverage* framing was weaker than it read.

#### What it does not qualify

The **instrument**. `PROTOCOL_ERRORS`, `protocol_truth`, `legality`,
`check_reference`, the reference validator and the judge-comparison machinery are
all about the harness, and a generated board is still a board a parser can check.
Their results are claims about the tooling and survive.

The **rules track**. `gold_questions.jsonl` is human-labelled through
`label_store` and the `#/label` workflow, a different corpus and a different
provenance. *Fine-tuning does not beat retrieval*, under four judges and
confirmed on run 4 by a second vendor, is untouched.

And the **reference lines**. Those are hand-authored — the first genuinely human
artifact in the position set, and the reason 21.96 could make entry 1 decidable
from the answer rather than from the shape of the board.

#### The consequence for what comes next

This inverts the priority. Adjudicating and correcting generated positions
produces a better-measured synthetic set; it does not produce a real one. So the
next step is not more adjudication but **positions taken from played games** —
boards that are structurally natural because a person actually reached them.

That also re-ranks the platforms, and on a criterion this file had not been
using. Arena requires *owning the cards*, so an arbitrary Standard board cannot
be constructed at all — a hard ceiling that no log quality compensates for.
Cockatrice, EDHPlay and XMage all have the full card pool freely available and
allow one person to drive both sides, which turns recording from *mining* into
*authoring by playing*: both decks chosen, the board played toward deliberately,
and none of the opponent-variance or selection-bias problems that passive log
mining has. XMage additionally enforces the rules, so it is the initial platform.

### 21.99 XMage can be asked, and the export found the phase vocabulary was never one

Two results from starting on XMage as the platform, and the second arrived
before any Java was compiled.

#### The blocker is answered: the engine can be queried

`PLAN_NEXT` recorded the open question as whether XMage can be *asked* what is
legal, or only asserted against outcomes — the difference between a two-sided
check and a one-sided one that cannot find a MISSING action (21.43). From
`Mage/src/main/java/mage/players/Player.java`:

```java
List<ActivatedAbility> getPlayable(Game game, boolean hidden);
PlayableObjectsList    getPlayableObjects(Game game, Zone zone);
Map<UUID, ActivatedAbility> getPlayableActivatedAbilities(MageObject, Zone, Game);
```

So it can be asked. `scripts/gameplay/xmage_export.py` emits one
`CardTestPlayerBase` test per position — life, battlefield grouped by card,
hand, a library of the stated size, `setStopAt(turn, PhaseStep.X)` — and prints
`getPlayable(...)` beside the position's own `legal_actions`.

Two mappings this repo cannot verify are isolated at the top of that file rather
than spread through the templates: `PHASE_STEP` and `TAPPED_CALL`. They are the
parts most likely to be wrong, and a silently wrong tap state makes a mana check
pass that should fail. One fix each against a real checkout and everything
re-exports.

**The emitted Java has never been compiled** — there is no JVM here, and a
generated test that has never run is a draft, not a result. The tool says so in
its own output.

#### The phase vocabulary was never a vocabulary

Exporting all 32 positions failed to map a phase on **10 of them**, and the
reason is the finding:

| written as | count | intended |
| --- | --- | --- |
| `precombat main` | 13 | — |
| `pre-combat main phase` | **4** | `precombat main` |
| `post-combat main phase` | **2** | `postcombat main` |
| `opening hand, on the play` | **2** | `opening hand` |
| `opponent's declare attackers step` | 1 | `declare attackers` |
| `opponent's declare blockers step` | 1 | `declare blockers` |

Nine spellings for what `PHASE_NAMES` defines as fourteen closed steps.

It passed every existing check because `phase_problems` matches by SUBSTRING, on
purpose — 21.61 made it loose so that "opponent's declare attackers" and
"declare attackers" would agree, which is right. But `'pre-combat main phase'`
satisfies it via `'main'` and **never via `'precombat main'`**, because of the
hyphen. The check has been passing by accident on 4 boards, and would keep
passing if the phase were changed to something else containing "main".

This is 21.98 arriving as a concrete artifact rather than a caveat. A person
authoring fourteen boards does not spell one step four ways; a generator does.
And the looseness that makes the check humane is exactly what stopped it being
noticed — the same shape as every check in this file that is right about its
own question and silent about a neighbouring one.

Not enforced here. Making `validate_position` reject a non-canonical phase
would fail 10 of 32 positions immediately, which is a decision about the gold
set rather than about the checker, and the set is about to be replaced from
played games anyway. The export reports it, which is enough to act on.

### 21.100 A rules engine checked the positions, and 10 of 32 are wrong

> **PARTLY RETRACTED by 21.101.** The headline is wrong: **30 of 32 agree**, not
> 22. Class 1 below — eight positions — was my query being step-scoped while
> `legal_actions` is turn-scoped, not a defect in the data. Class 2 stands, and
> everything under *"corrections to my own method"* stands. Read 21.101 first.

The XMage track, end to end: JDK and Maven installed, `magefree/mage` cloned
(291 MB shallow), built in 2:45, 32 generated tests compiled and run against a
real rules engine. `xmage_export.py` emits them; `xmage_diff.py` compares the
engine's answer to each position's `legal_actions`.

**22 of 32 agree. 10 do not**, in exactly two classes. ← wrong; see 21.101.

#### Class 1 — an attack listed in a main phase (8 positions) — RETRACTED

```
pos-combat-math-0001 …-0006, pos-race-vs-stabilize-0001, -0002
    claims 'Serra Angel' may attack; the engine does not — at phase 'precombat main'
```

Every one records `phase: "precombat main"` and lists an `ATTACK` in
`legal_actions`. Attacking is declared in the declare attackers step, so the
enumeration is impossible as written.

21.86 found this by hand on **one** board and left it as a question about that
position's design. It is not one board; it is a quarter of the set, and a
generator putting combat actions on main-phase boards is the obvious
explanation (21.98).

The engine settles the interpretation the hand analysis could not. A control at
`DECLARE_ATTACKERS` returns `ATTACKER: Centaur Courser` where `PRECOMBAT_MAIN`
returns none — so the creature *can* attack, just not then. Summoning sickness
is ruled out, and the defect is the phase, not the creature.

**That last paragraph is the error.** The control is right and the conclusion
does not follow from it: *"the creature can attack, just not then"* is also what
a correct position looks like, because `legal_actions` is turn-scoped. 21.101
has the evidence.

#### Class 2 — a castable spell omitted (2 positions)

```
pos-trigger-ordering-0001, -0002
    the engine offers 'Doom Blade'; the position does not list it
```

`pos-trigger-ordering-0001` is at upkeep with Doom Blade in hand and lists only
its two `ORDER TRIGGERS` options. Doom Blade is an instant and genuinely
castable; the closed arm is told *"these are the only legal actions available to
you"*, which is false about the board.

This one has teeth. `PROTOCOL_ERRORS` entry 3 charges *names a play that is not
available*, and `legality` matches against `legal_actions` — so **a model that
correctly casts Doom Blade is scored illegal for a correct play.** Entry 3 is
also the entry the judge grades best (80–85%, 21.74), so the error is confident
and consistent.

And this class is only findable by a two-sided query. Asserting that each
enumerated action is legal can never surface an action that is missing —
21.43's rule, and the reason the export queries the engine rather than testing
assertions.

#### Three corrections to my own method, each caught by a control

The first clean run reported one line, `Cast Shock`, for a board that also lists
an attack. That read as the position being wrong. It was the query:
`getPlayable` returns `List<ActivatedAbility>` and **declaring an attack is a
turn-based action, not an activated ability**, so attacks and blocks cannot
appear there at all. `getAvailableAttackers` and `getAvailableBlockers` are
separate calls. A one-sided query reading as a finding about the data — 21.43's
shape, a third time.

Before that, three artifacts of my own setup polluted the first diff:

- every test player silently loads `"RB Aggro.dck"`
  (`CardTestPlayerAPIImpl:215`), so both had a real hand and library that were
  not the position's — the engine offered `Play Mountain` three times from a
  hand no position specified. `removeAllCardsFromHand` / `FromLibrary` first;
- `setStopAt(7, …)` **simulates** seven turns rather than jumping to one, so the
  board under test was the position plus six turns of draws. Turn 1 always: a
  position's `turn` says what the board represents, not how long to play first;
- mana abilities (`{T}: Add {G}.`) flooded the output, and `legal_actions`
  enumerates plays while the grammar handles mana through TAP lines.

And once, surefire's `-Dtest` was given `A+B` and then a package glob; neither
matched anything, and the stale reports from the previous run were read as
current results. A test runner that matches nothing exits differently from one
that runs and passes, and only the exit code says which.

#### What the comparison does not claim

Matching is on the **card**, not the line: the engine names a spell once
(`Cast Shock`) where a position names each targeting of it. So this asks whether
a play is available at all, never whether its target is legal.

Blocks are excluded. `getAvailableBlockers` returned the same creature at a main
phase and at declare blockers, so it is not phase-sensitive — a `BLOCKER` line
means *could block something*, not *blocking is legal now*. Comparing it would
have marked every position as able to block and read as a finding.

### 21.101 `legal_actions` is turn-scoped; the query was step-scoped. 30 of 32 agree

21.100 reported ten defective positions. **Eight of them were the measurement.**
The honest number is **30 of 32**, and the two survivors are the Doom Blade
class, which stands unchanged.

#### What the reference lines settled

Two of the eight carry a *hand-authored* `reference_actions` — the 100%-correct
line, written by a person in `#/reference` and refused unless the parser agrees
(21.85). Both walk out of the main phase before attacking:

```
pos-combat-math-0003   PHASE precombat main / END PHASE precombat main /
                       PHASE declare attackers / ATTACK Serra Angel -> Opponent / ...
pos-combat-math-0005   PHASE precombat main / CAST Shock TARGET Grizzly Bears / PASS /
                       END PHASE precombat main / PHASE declare attackers /
                       ATTACK Centaur Courser -> Opponent / ...
```

`check_reference` accepts a line only if **every play in it is in
`legal_actions`**. So the repo's own validator already required
`ATTACK Serra Angel` to be listed on a board whose `phase` is `precombat main`.
Three independent things say the same thing:

1. the human reference walks the turn and then attacks;
2. `check_reference` demands the attack be listed to accept that line;
3. `PHASE` and `END PHASE` exist in the grammar *for* this — an answer is
   expected to move between steps.

`legal_actions` therefore enumerates the plays available **over the turn** from
this board, not the plays available in the step it states. Asking
`getAvailableAttackers` at the stated step returns nothing on every main-phase
board — which is what "8 of 8 disagree" actually measured.

#### The same shape a fourth time

21.100 itself corrected a one-sided query (`getPlayable` cannot report attacks)
and named it as 21.43's shape a third time. This is the fourth, and it survived
that correction because fixing *which* engine call to make left *at which step*
untouched. The control that exposed the third — attackers at `DECLARE_ATTACKERS`
versus `PRECOMBAT_MAIN` — is the same control that would have exposed this one,
and I read its output as evidence about the data instead of about the query.

**A measurement that disagrees with the data on a whole class is a claim about
the instrument until the instrument is checked at that class.** Eight of eight
is not a defect rate; it is a constant, and a constant is the signature of a
harness bug (21.5's family: a harness metric moving with the condition).

#### The fix

`getPlayable` at the stated step, `getAvailableAttackers` at
`DECLARE_ATTACKERS`, as two `@Test` methods — `execute()` runs once per method
and JUnit gives each a fresh game, so the board is emitted twice rather than
hoisted. `combat_reachable()` suppresses the attacker query when the stated
phase is past declare attackers, where an empty answer *is* the honest one; no
position in the set is in that case, but a future one can be.

The `getPlayable` half stays step-scoped and is therefore under-inclusive for a
sorcery-speed play on a pre-main board. **No position does that** — checked, no
non-main board lists a land drop — so it is a stated gap in the docstring, not a
built one.

`common.PHASE_ORDER` is new, and exists because `PHASE_NAMES` is a membership
list whose first four entries are in turn order and whose fifth is
`postcombat main`. It reads as ordered and is not; asking it "is combat still
ahead?" answers that combat follows the postcombat main phase.

```
32 positions checked against the engine, 30 agree

  pos-trigger-ordering-0001   the engine offers 'Doom Blade'; the position does not list it
  pos-trigger-ordering-0002   the engine offers 'Doom Blade'; the position does not list it
```

#### What survives, and what it costs

Class 2 stands: both boards are at upkeep with Doom Blade in
`players.you.hand`, five untapped Swamps, and `legal_actions` listing only the
two `ORDER TRIGGERS` orderings. Entry 3 charges *names a play that is not
available* and `legality` matches against `legal_actions`, so a model that
correctly casts Doom Blade is scored illegal for a correct play — on the entry
the judge grades best (80–85%, 21.74).

So the engine found a **6%** defect rate, not 31%. That is a much better result
for the position set and a much worse one for the argument that it needs
replacing wholesale — the generated boards are mechanically sound far more often
than 21.100 claimed. 21.98's caveat is untouched: mechanically legal is not the
same as *worth asking*, and every board and rubric is still machine-drafted.

#### Turn-scoped `legal_actions` is a design cost, not a defect

Worth separating from the retraction. A board that states `precombat main` and
offers both a cast and an attack is asking the model to plan a turn, and 21.93
measured that a position asks ONE question — median 4.5 actions but **1.5
plays**. Six of the eight are `combat math`, where the decision is the attack and
the main phase is scaffolding. That is exactly what `POSITION_REVIEW_KINDS`
`shorten` is for (21.94), and it is B3's call, not a correctness fix.

The distinction matters because the two look identical in a diff and are not:
one is *this position is impossible*, the other is *this position is longer than
its question*.

### 21.102 The phase vocabulary was ours; it is XMage's now

Direction from the user, and the right one: *"if the format is different from
what we have vs what XMage does I would like to prefer the XMage usages as what
we have was simply created by us and has no strong basis for patterns."*

Applied to the one place the two formats actually collide. `PHASE_NAMES` was
invented here, and every property it had was an accident of that:

| | ours (before) | `mage.constants.PhaseStep` |
| --- | --- | --- |
| entries | 14 | 13 |
| ordered | **no** — and the first four *were*, so it read as ordered | yes, by `index` |
| `main` | present, ambiguous between two steps | absent |
| first strike damage | **missing** | `FIRST_COMBAT_DAMAGE` |
| spellings in 32 positions | **9**, needing a 5-entry alias table to export | n/a |

`common.PHASE_STEPS` now mirrors the enum row for row, with XMage's own four
renderings of each step, and `PHASE_NAMES` / `PHASE_ORDER` / `phase_step` /
`phase_index` / `canonical_phase` all derive from it. Two hand-written tables in
`xmage_export.py` — one mapping our words onto XMage's, one aliasing our nine
spellings onto our own fourteen — were **deleted**, not extended: both described
a problem that stopped existing.

#### What the migration cost, and why it was free

`--canonicalize-phases` rewrote **49 of 51** stored phases (the two left are
`opening hand`, already canonical). Nine spellings became six values. Nothing
was unresolvable; `main` would have been, and no stored board used it.

Editing `phase` changes `render_position`, which invalidates every stored
position answer for comparison (21.59). **That cost was already paid.** No
stored position run is at the current `gameplay_fingerprint` — the newest are
`4a0fee3c83ee` and `d094e3934d2d` against a current `f050d4aed707` — so the
answers this could invalidate were already incomparable. Checking that first is
what made this a cleanup rather than a decision.

Only the `phase` FIELD moved. Hand-authored `PHASE` lines inside
`reference_actions` are untouched: the parser accepts XMage's spellings and the
CR's both, and rewriting a person's submitted text would leave
`check_reference` validating a line nobody wrote. All 10 reference lines still
pass unchanged.

#### Two vocabularies, because there are two questions

Removing `main` broke `test_actions` immediately, and the failure was right.
`PHASE Main` is a fine thing for a MODEL to write — it names a kind of step —
and refusing it turns a correct declaration into narration, which is 21.61's
failure in the same file that documents 21.61. But a POSITION may not *store*
`main`, because a board sits in one specific step.

So the two questions are separated rather than merged:

```
PHASE_NAMES   is this a step name at all, or narration?   "main" -> accepted
phase_step    WHICH step is it?                           "main" -> None
```

`AMBIGUOUS_PHASE_WORDS` holds the difference. This is the "one name, two
meanings" trap caught *before* it shipped rather than after — the two readings
agree on all twelve unambiguous steps and come apart on exactly one word, which
is the shape every previous instance had (`PASS`, `CROSS_REF_RE`,
`JUDGE_SYSTEM_PROMPT`).

`LEGACY_PHASE_ALIASES` is the other half: seven spellings XMage does not use,
four of them ours and **three of them the Comprehensive Rules' own** — "end
step" (513), "beginning of combat step" (507), "end of combat step" (511),
where XMage says end turn / begin combat / end combat. A model trained on the CR
writes the CR's words. XMage wins on what is stored; the CR stays readable. The
two do not conflict, because one is an identity and the other is an input.

#### A negation replaced a list

`timing_problems` selected on `_NON_MAIN_STEPS`, a hand-written list of the
eleven steps that are *not* main phases. It is now `_MAIN_STEPS` — two entries
and a negation. The old form had to be edited every time the vocabulary gained a
step, and it silently stopped covering one the moment `FIRST_COMBAT_DAMAGE`
arrived. A complement enumerated by hand is a list that decays.

One behaviour needed care: `phase_step` returns `None` for two different things,
and they want opposite answers. A mulligan is a step known to cast nothing, so
the check runs; an *unrecognised* phase is a step nobody identified, and
charging it would be a finding about the vocabulary rather than the position.

#### Result

```
32 tests -> …/magicllm
2 could not be mapped to an XMage PhaseStep.   (was 10)
```

The two are the mulligan boards, which correctly have no step: XMage has no
`PhaseStep` for the opening hand because the CR does not either. The engine diff
is byte-identical before and after — **30 of 32** — so the migration changed the
vocabulary and not a single measurement.

### 21.103 Recording real games: XMage's own hook, and why the AI path is not it

21.98 says every board in the set is machine-drafted, and 21.101 established the
boards are mechanically sound far more often than feared — 30 of 32. That
settles legality and leaves the actual objection: *a generated board has no
reason to be a board anyone would ever face.*

Boards from a played game do. Built this session:

- **`MagicLlmPositionDataCollector`** (in the XMage checkout) — one board
  snapshot per step, JSONL. It is XMage's **own** documented extension point
  (`mage.collectors.DataCollector`: *"create new class and extends
  EmptyDataCollector … modify DataCollectorServices.init"*), so it needs no fork
  and costs nothing unless `-Dxmage.dataCollectors.magicllmPositions=true` is
  passed. The whole change to the checkout is one file plus one `add()` line.
- **`scripts/gameplay/import_game.py`** — snapshots to position DRAFTS.

#### It records the board and nothing else, on purpose

The collector could compute `legal_actions` — the engine is right there, and it
is the field most likely to be wrong. It does not, because `xmage_export.py` +
`xmage_diff.py` already ask that question and are validated at 30 of 32. Two
implementations of one field is how two copies drift, which this repo has
already paid for five times (`carry_diagnostics`, `coverage_lines`,
`turns.load_steps`, and twice more).

Drafts also carry no `key_points`, no `common_errors`, and no `category`. A
generated rubric is *half* of what 21.98 is a caveat about, so filling them here
would put the caveat straight back under a new name. An empty rubric is honest.

#### The AI-vs-AI path does not work, and the reason is one line

The plan was to validate at volume with AI-vs-AI games in the test framework —
no server, no client, one maven command. It produced 24 valid snapshots and a
game with nothing in it: **both hands empty for all 12 turns**, both players on
20 life throughout, one Mountain per turn as the only permanent ever played.

`GameImpl.init`:

```java
if (!gameOptions.testMode) { mulligan.drawHand(startingHandSize, player, this); }
```

The harness sets `testMode = true`, so **no opening hand is ever dealt**. That is
correct for its purpose — every card test places an exact board — and fatal for
this one. Not a fixable detail: it is the assumption the whole test API rests on.

Worth recording as a near miss. The snapshots were structurally perfect —
valid JSONL, turns advancing, permanents accumulating, life and library counts
moving — and a filter that only checked *shape* would have passed all 24. The
`--interesting` shortlist rejected **0 of 24**, which is the honest answer, and
the thing that made the emptiness visible was reading a rendered board rather
than a count.

So the recorder test keeps its place as a smoke test for the collector and is
labelled as one. Real games come from the server path, where `testMode` is
false. The collector is unchanged between them: it hangs off `GameImpl`'s own
log hook, which both paths call.

#### Deck choice is the sampling frame

The recording contains whatever the decks contain, so **the decks decide what
categories of position exist**. No constraint comes from this side: the card
index holds 34,881 Oracle cards and every modern card tested resolves `exact`,
so any real deck works. That is the point the user made about virtual clients
generally — no collection to buy, unlike Arena.

### 21.104 A double-faced card was not a card, and nine gates agreed

The user supplied five 40-card welcome decks to record games with. Checking
whether the cards resolve found something better than an answer about decks.

**All 92 distinct cards resolve.** My first check said 7 did not, and that was
my filter, not the data: I gated on `how == "exact"` and the seven are
double-faced cards, indexed under `"Front // Back"` and resolving as
`normalized`.

```
Gollum, Silent Slinker    -> Gollum, Silent Slinker // Meager Meal
Smaug, the Great Calamity -> Smaug, the Great Calamity // Spew Flame
Bofur, Reliable Guardian  -> Bofur, Reliable Guardian // Concerted Care
```

#### The bug under it

`validate_position` does not skip a non-exact match, it **rejects** one:

```python
elif how != "exact":
    problems.append(f"card {name!r} only resolves by {how} — use the exact name …")
```

So a board naming `Gollum, Silent Slinker` fails validation, and the fix it
demands is `Gollum, Silent Slinker // Meager Meal` — a string **no game client,
no deck list and no player ever writes**, which would then be rendered into the
board the model reads. The suggested correction is worse than the error.

884 of 34,881 cards are double-faced (2.5%), so this was a live trap for any
position drawn from real play, and 7 of the user's 92 cards are in it.

#### One name, two meanings, again

Faces were already indexed — folded into `by_norm` beside the case- and
apostrophe-normalized whole names. So `resolve` could not tell *you wrote a real
name sloppily* from *you wrote one printed side of a two-sided card*, and both
came back `normalized`. The two readings agree everywhere except on DFCs, which
is this repo's most repeated failure shape (`PASS`, `CROSS_REF_RE`,
`JUDGE_SYSTEM_PROMPT`, and now this).

They are now separate indexes and separate answers, and the distinction is not
strictness: a `normalized` hit *should* be corrected, because the exact string
exists and costs nothing to write. A `face` hit should not, because the string
already is a real printed card name.

#### Nine gates, one predicate

`how != "exact"` appeared in **six files and nine places** — `validate_position`,
`timing_problems`, `mana_problems`, two more in `positions.py`,
`eval_positions`, `label_store`, `validate_gold`, `ingest_qa_pastes`. Fixing the
resolver alone would have left every one of them rejecting faces.

They now call `card_lookup.names_a_card(how)`. A hardening reaching a subset of
its callers is the bug this project has shipped five times (21.53, 21.54, 21.74),
and the countermeasure is the same one: one function, and the direction of the
next fix no longer matters.

Note the two halves failed in *opposite* directions, which is why both were
needed. `validate_position` rejected loudly. `timing_problems` and
`mana_problems` returned `[]` and skipped — so a DFC silently disabled the
timing and affordability checks for its position, and "no problems found" over
a card nothing examined is 21.49's shape: it fails as a pass.

#### Ambiguity is refused, not guessed

Measured over the corpus: 1,771 face names, of which **2** are claimed by two
different cards (`Fire` is a face of both `Fire // Ice` and `Start // Fire`;
likewise `Start`) and **25** are also real single-faced cards (`Brainstorm`,
`Ancestral Recall`, `Bind`). The first class resolves to **nothing** —
`ambiguous-face` — and the second is handled by check order, whole names before
faces, so `Brainstorm` is Brainstorm. 1,744 usable aliases.

Same rule as ambiguous-prefix and as `find_players`: miss rather than guess,
because a wrong card puts confidently incorrect text in front of the model.

### 21.105 "30 of 32 agree" counted 8 boards it never compared. 22 of 24 do

Building the emit half of the pipeline — `import_game.py` leaves `legal_actions`
empty and 21.103 says the existing path supplies it — found that the existing
path could not, and then found why the number it reports was too kind.

#### The denominator

`claimed()` files a position's actions under PLAYABLE / ATTACKER / BLOCKER, and
blocks are deliberately not compared. So a board whose `legal_actions` are
**all blocks** contributes to neither column, has no problems, and counts as
*agreeing*. Same for a mulligan, and for a board with no list at all.

```
24 of 32 positions were actually COMPARED, 22 of those agree
2 disagree.
8 contributed NOTHING to either column, and were previously counted as agreeing
```

Five blocking boards, two mulligans, one with no `legal_actions`. This is
21.49's shape and `validate_position`'s own rule — *"no problems found" over a
set nothing examined is the same defect as a gate verdict with no coverage* —
applied everywhere in this repo except the newest instrument. The diff now
prints coverage **before** the caveats, and names each uncompared board and why.

The two real disagreements are unchanged: `pos-trigger-ordering-000{1,2}` and
the castable Doom Blade.

#### The blocking boards were exported as a different board

Worse than uncounted. **Nine of thirty-two positions carry `attacking`
permanents and `xmage_export` dropped the flag entirely** — it grouped
permanents by `(card, tapped)` and nothing else. A board at declare blockers
was handed to the engine with nobody attacking.

So even the casts compared on those boards were compared against the wrong
state, and the diff structurally could not notice, because the field that would
have shown it is the one field it does not compare.

Fixed by declaring the attacks for real — `attack(turn, playerB, "Grizzly
Bears", playerA)` — which needs the turn to be the ACTIVE player's:
`playerA` starts, so a board on the opponent's turn runs to turn 2. All nine
blocking boards have `active_player: opp`, as they must.

That immediately caught a position defect: **`sample-stage3-payment-0004` has
YOUR creature attacking on the OPPONENT'S turn**, which only the active player
may do (508.1). Reported, not emitted — `attack()` would fail at runtime, and
silently dropping it would export a board missing the creature the position is
about.

#### Targets, because the grammar has them

`legal_actions` names each targeting separately (`CAST Shock TARGET Grizzly
Bears`), and `match_to_legal` has **no untargeted-CAST fallback** the way it has
for ATTACK. So emitting `CAST Shock` would score every correct targeted cast
illegal — 21.61's shape a fourth time, arriving as *"the model got worse at
removal"*.

The engine knows: `Target.possibleTargets(controllerId, ability, game)`. Asked
per playable ability, it does real rules work — `Cast Doom Blade` offers Wall of
Omens, Grizzly Bears and Nessian Asp and **not** the black creature.

#### And where it must refuse

Reproducing the 31 stored lists from the engine: 6 identical, 11 where the
engine is a strict superset, 14 genuinely different. The 14 are almost all the
emitter being blind, not the positions being wrong:

| what | why the engine cannot say it |
| --- | --- |
| `BLOCK x -> y` | `getAvailableBlockers` is not phase-sensitive |
| `ORDER TRIGGERS a, b` | not an `ActivatedAbility`, so `getPlayable` never sees it |
| `KEEP` / `MULLIGAN` | before any step |
| `ATTACK a, b` (a subset) | the engine lists attackers individually, not power sets |

On those boards the engine's answer is not incomplete, it is **empty** — and an
empty `legal_actions` makes the closed prompt say *"no legal plays are
available; PASS is the only response"*. That is false about the board, and every
correct block would score illegal against it. So `emitter_blind_spot` refuses by
name rather than writing a confident wrong answer, and `emit_legal_actions`
refuses a board that already has a list, so it can never quietly overwrite a
curated field with a raw dump.

The eleven supersets are the emitter working correctly and being *untrimmed*:
it offers `CAST Shock TARGET you`, which is legal and which no one would do.
`legal_actions` is a curated list — POSITIONS.md: *list only the plays that are
genuinely available* — so the engine's answer is raw material, and the closed
arm measures something different when handed thirty options.

### 21.106 The first real game recorded, and two bugs the test game could not find

A human-vs-AI game on the modified server: **66 snapshots, turns 1–15, zero
errors**, life from 20 down to −3, hands populated on 65 of 66 boards, up to 24
permanents. A mid-game board renders correctly through `render_position`. The
pipeline works end to end.

It took two fixes, and the interesting thing about both is that the AI-vs-AI
test game (21.103) could not have found either.

#### One: `getCards(null)` — hidden by the empty-hand defect

The first real game wrote **nothing**, and the server log said why on every
step:

```
magic-llm positions: snapshot failed - NullPointerException:
Cannot invoke "mage.game.Game.getPermanent(UUID)" because "game" is null
```

`player.getHand().getCards(null)` — written assuming the argument was a filter.
It is the **game**: the one-arg form is `getCards(Game)` and the filter form is
`getCards(FilterCard, Game)`. `getPermanentOrCard(cardId, game)` then
dereferenced the null on the first card any player held.

The test game wrote 24 clean snapshots against this exact code, because
`testMode` deals no opening hand — so both hands were empty for twelve turns,
`stream()` mapped nothing, and the null was never touched. **The defect that
made the test game useless is precisely what hid this one.** 21.5's family: a
harness bug whose trigger rate depends on the condition under test.

The error handling did its job — caught, logged, game undisturbed — which is the
only reason this was diagnosable from one log line instead of a repro.

#### Two: a token is not a card

With the NPE fixed, 3 of 16 card names in the recording did not resolve:
`Treasure Token`, `Hero Token`, and **`Everywhere`** — the land token *Overlord
of the Hauntwoods* creates. `validate_position` rejects a name that does not
resolve against Oracle, and an Oracle dump contains no tokens, so **every board
a modern game produces would have been refused.**

`Everywhere` is why the flag comes from the engine's own `Permanent.isToken()`
rather than from the string: nothing in that name says "token", and a
name-based heuristic would have passed it straight into the Oracle check.

Tokens are still recorded, still rendered, still on the board — they just are
not looked up. `position_card_names` skips a permanent marked `token`, and a
permanent with **no** flag is still checked, since absent means "a card" rather
than "unknown".

#### The client's own log is not a substitute

Worth recording, because it looks like one. The XMage client saves every match
to `gamelogsJson/game-<uuid>.json` — 2.5 MB for a single game. It is a **UI
message trace**: 1,025 messages, mostly `GAME_UPDATE` / `GAME_SELECT` /
`SEND_PLAYER_BOOLEAN`, whose `gameView` carries only `combat`, `players`,
`priorityTime`, `stack`.

**No phase, no step, no turn number, no hand, no library count** — four fields a
position needs, and phase is the one that makes a board a decision point at all.
It records what the UI had to draw, not what the board was.

`gamelogs/*.html` is a readable play-by-play and is genuinely useful for
reviewing a game before picking boards from it. XMage's own `saveGameHistory`
collector is now enabled alongside ours for the same reason: an independent
account of the same game, to check a suspicious snapshot against.

#### Setup notes that cost real time

- **XMage 1.4.61 targets Java 8 and JBoss Remoting reflects into `java.io`.**
  On JDK 17+ the login dies with `InaccessibleObjectException`, and the failure
  is invisible from the server side — nothing reaches it, so its log stays
  empty and the problem reads as "the server isn't listening". Both launchers
  pass `--add-opens` now. The distribution's own `installed.properties` still
  sets `-XX:MaxPermSize`, a flag **removed in Java 8**, which is the clearest
  statement that it expects an ancient JRE.
- **A stock install cannot be patched with this collector.** Compiling against
  the installed jars showed the `DataCollector` interface had drifted
  (`onTestsStackResolve(Game)` vs `onTestsStackResolveStart/End`) in the 13 days
  between builds. A `git log --since` check said "unchanged" and meant nothing —
  the clone is shallow, depth 1. Compiling against the actual artifact is what
  settled it; reading history could not.

### 21.107 What the generated set is actually worth, and a loop the detector could not see

An audit of the 32 positions against everything now available — five stored
arms, the rules engine, and 84 human verdicts — to answer whether the set is
worth repairing or replacing (21.98).

#### It is in better shape than "every board is generated" suggests

**28 of 34 boards discriminate** (82%): the arms differ on correctness, on the
blunder call, or both. Ranking each board by discrimination, human verdicts
collected, and whether it carries an authored reference line:

| | boards |
| --- | --- |
| strong — discriminates AND carries human investment | **16** |
| usable — one of the two | **13** |
| weak — neither, or engine-defective | **3** |

The three weak ones are named, not estimated: `pos-trigger-ordering-0001` and
`sample-stage3-payment-0004` are the engine defects from 21.101/21.105, and
`pos-blocking-0005` discriminates on nothing with one verdict against it.

So the answer to *fix or replace* is neither: **29 of 32 boards are carrying
evidential weight**, and 84 human verdicts are attached to 23 of them. Replacing
the set wholesale discards that. 21.98's caveat still stands on every number
they produce — generated board, generated rubric — but "generated" turned out
not to mean "worthless".

#### Where the human time went is not where it paid

`pos-race-vs-stabilize-0001` has **7 verdicts and does not discriminate**; so
does `pos-combat-math-0004` with 4. Meanwhile every `sample-stage3-payment-*`
board has **zero** verdicts, and four of them discriminate. Adjudication has
been going to the boards that were queued, not the boards that separate arms —
which is worth fixing in `build_queue` before the next batch.

#### 32% of "illegal" was one runaway answer

Cross-checking every action the harness scored illegal against the engine's own
answer: **393 illegal actions, of which the engine says 126 (32%) were legal.**

That number is a lie of aggregation. **107 of the 126 come from a single arm on
a single board** — `base_open` on `pos-combat-math-0005` emitted 755 actions
with `CAST Shock` repeated 107 times. Excluding it, the data-defect share is
**19 of 286 (6.6%)**, spread over 11 boards, and those 19 are real: the Doom
Blade cases and a handful of correct plays the enumeration omits.

Fourth time a rate in this project has rested on one cell (21.78, 21.79, 21.81,
21.82). The decomposition is not optional.

#### And that answer was not flagged degenerate

`degenerate` reads `repeats_collapsed >= 5 or max_play_repeat >= 5`, and
`max_play_repeat` compares each action **to the one before it**. The runaway
answer interleaved:

```
PHASE ... / PASS / CAST Shock / PASS / PHASE ... / PASS / CAST Shock ...
```

217 `PASS`es and 107 `CAST Shock`, never two identical plays adjacent, so
`max_play_repeat = 3` and the answer scored **`degenerate = False`** at 755
actions. `PLAY Plains` x133 — the case 21.58 fixed — happened to be
consecutive, and the fix generalised from that one shape. **A decode loop cycles
through a pattern; it does not usually stutter.**

`max_play_total` counts the most-repeated play anywhere in the output, excluding
`PASS` — the prompt mandates one after every play, so a correct multi-play turn
repeats it legitimately and counting it would flag good answers.

Measured across **2,312 stored answers**: the new check reclassifies **20
(0.87%)**. A further 48 differ from their stored value for an unrelated reason —
they predate 21.58 — and separating the two mattered, because the first
measurement showed 68 flips and reported them all as this fix's doing.

### 21.108 The priority flow is 10x our notation, and 80% of it carries nothing

The user's observation on reading the recorded game: XMage models priority as
real clicks — cast a spell, take priority back to respond to your own spell,
pass, opponent passes, it resolves — for **both** players, while our grammar has
one player in it (21.89) and one `PASS` standing for the whole thing (16.13).
The concern was bloat: a notation that under-models the real protocol on our
side and does not model the opponent's at all.

Measured against the first real recording (n=**1 game**, so directional):

| | count |
| --- | --- |
| priority prompts to ONE player (`GAME_SELECT`) | **186** |
| plays in the whole game, BOTH players | **37** |
| priority windows declined | **≥ 80%** |
| our reference lines (21.93) | ~3 protocol actions per play |
| the engine | ~10 priority windows per play |

#### The simplification is right, and this is the argument for it

Four fifths of the priority windows in a real game are declined. They are not
compressible detail that we happen to be dropping — they carry no information at
all. Representing them would multiply every reference line by ten with content
that is almost always *"nothing happened"*, and 21.61 already measured what
happens when the prompt demands more protocol output: the same correct answer
starts failing the checks that consume it.

#### But it flips 21.92's caution

21.92 reads a `PASS` as an opponent window and warns that **a play after one
asserts the opponent did not act** — treated as a risky claim that belongs in a
scenario. At an 80% decline rate that assertion is usually *true*. The rule
still holds for the cases that matter (a held removal spell, an untapped blue
opponent), but the default is safer than it was written as. Worth re-measuring
across several games before changing the guidance.

#### Per-step snapshotting absorbs the click noise, and that is the design

The collector snapshots on a **(turn, step, active player)** change, not per
priority window. Consequence, measured: 66 snapshots over 15 turns — 4.4 per
turn against the engine's 12.4 prompts per turn *per player* — and **0 duplicate
board states among the 66**.

That is what makes the user's other point harmless. Real games contain
misplays, cancelled casts and re-taken priority; none of them change the board,
so none of them produce a snapshot. The click-level record is where the noise
lives, and we deliberately do not read it.

#### The real gap is the opposite of bloat

Not that our notation says too little about our own passes — it is that it
cannot say anything about the **opponent's** action (21.89). When a recorded
board arises *because* the opponent responded, that fact belongs in the board of
the next step, which is exactly what a scenario is for (21.71). Recording from
real games makes those boards available for the first time; the grammar does not
need a new verb to use them.

### 21.109 Split balance as a position selector, borrowed from a 20-questions game

The user's side project (`jslinker/TwentyQuestionsMagicTheGathering`) builds a
decision tree over Scryfall Oracle cards and picks each question by:

```python
balance = min(yes_count, card_count - yes_count)
if balance > best_balance:
    best_id, best_balance = question_id, balance
```

**That heuristic is not an approximation of information gain — for a binary
split with a uniform prior it is ordinally equivalent to it.** Checked across
splits from 500/500 to 1/999: `min(yes,no)` and
`H(p) = -p log p - (1-p) log(1-p)` induce the same ordering, so `argmax` picks
the same question. It is the same choice at the cost of an integer `min` rather
than two logarithms.

Applied to this project's own problem — *which positions are worth having?* —
it says something 21.107 missed.

#### 14 of 34 positions carry no information about the blunder gate

Ranking each board by how evenly the five arms split on `blundered`:

| balance | boards | meaning |
| --- | --- | --- |
| 2 (of a possible 2) | 9 | maximally informative |
| 1 | 11 | informative |
| **0** | **14** | every arm agrees — the board separates nothing |

21.107 called a board discriminating if the arms differed on correctness **or**
on the blunder call, and **8 of the 14** passed that test on correctness alone.
They still cannot move Gate 3, which reads `errors_made`.

#### And every one of the 14 is saturated, not clean

All fourteen are **5/5 blundered**. Not one board has all five arms clean. That
is the same fact CLAUDE.md records from the other direction — these arms blunder
on 82% of answers — but stated per board it is sharper: **Gate 3 is being
measured on a set where 41% of the boards are pinned at the ceiling.** A gate
cannot be moved by a board that is already maximally failed, so the effective n
for Gate 3 is 20, not 34.

That is a stronger reason to source new positions than "the boards are
generated" (21.98). A generated board can still discriminate; a saturated one
cannot, whatever its provenance.

#### Where else the same selector applies

**Adjudication order.** 21.107 found human verdicts had gone to boards that
separate nothing — 7 on `pos-race-vs-stabilize-0001`, which is balance 0, and
0 on four payment boards that are balance 1–2. `build_queue` interleaves on the
judge's call; balance over the two judges' verdicts is the selector that would
put the time where it moves a number.

**Deck choice for recording.** The side project's other asset is
`config/semantic-questions.json` — **56 curated Scryfall Tagger tags**
(`removal`, `spot-removal`, `sweeper`, `counterspell`, `ramp`, `tutor`,
`cantrip`, …) mapped to player-facing questions. Our card chunks carry
`oracle_id`, so those tags join to our corpus directly with no matching
heuristics. That turns "pick decks with more decision density" (21.108) from
taste into a query: a deck built from `spot-removal` + `counterspell` cards
produces `removal timing` boards by construction.

#### What it is not useful for

Card identification. `card_lookup` resolves a named card by dictionary lookup at
99%, and this project never has to infer a card from its properties.

### 21.110 The pin says the corpus has not changed; it could not say what it is

Asked whether the Scryfall API has anything we should adopt. It does not — we
already use both endpoints and comply with the policy — but the audit found the
provenance gap.

`data/cards/raw/` holds `oracle_cards.jsonl` and `rulings.jsonl`, the two files
`card_chunks.jsonl` is built from, and **neither had a manifest**. The only one
there was `standard_MANIFEST.md`, for a `/cards/search` pull that is not what
the corpus is built from — and `fetch_cards.py`'s own header said the project
uses search, which is how that impression survived.

So `CARD_PIN` proved the bytes had not drifted and could not say which snapshot
they were. `download_bulk` already RETURNS Scryfall's upstream `updated_at`;
nothing recorded it.

Recovered what the files still allow:

```
oracle_cards.jsonl   downloaded 2026-08-11T16:06Z   38,626 raw -> 34,933 chunks
rulings.jsonl        downloaded 2026-08-12T08:23Z
newest released_at   2026-11-13 (Star Trek, spoiled ahead of release)
```

Those dates are local mtimes, not the upstream timestamp, and the upstream value
is unrecoverable — Scryfall rebuilds bulk data daily and serves only the current
file. `CARD_PIN` now carries them, marked for what they are, with
`oracle_bulk_updated_at: None` to be filled on the next deliberate re-pin.

`oracle_cards` remains the right bulk type: one row per distinct game object.
`default_cards` is one row per printing and `all_cards` one per printing per
language, both repeating identical Oracle text across reprints.

### 21.111 Six ways to put two spells on one card, and they do not agree

Prompted by the user: *"there are more than a few different types of two spells
on one card."* Correct, and the audit found the shape of the problem.

885 cards in the corpus carry a `//` name, across six layouts, and 21.104's face
index treats them identically. For LOOKUP that is right — every one of those is
a real printed name. For a CAST action it is not:

| layout | n | is the back face a play? |
| --- | --- | --- |
| `transform` | 401 | **no** — reached by transforming |
| `adventure` | 170 | yes |
| `split` | 137 | yes |
| `modal_dfc` | 100 | yes — often a land, so played rather than cast |
| `prepare` | 55 | yes — cast as a copy while prepared |
| `flip` | 22 | **no** |

**529 back-face names are things you can never play**, and nothing distinguished
them from the 356 you can.

#### `mana_cost` looks like the discriminator and is not

`'{2}{B} // {B}'` for adventure, **`None`** for both `modal_dfc` and
`transform`. Per-face costs live in `card_faces[]`, which `chunk_cards.py`
flattens away — so the field that would settle it is the field we dropped. The
joined `type_line` survives, which is what makes the fallback below possible at
all.

#### The layout list is open, which is the real constraint

`prepare` is from Secrets of Strixhaven, two sets ago, and is already 55 cards —
newer than the assistant's knowledge, and it would have been silently
mishandled. A hardcoded layout table goes stale within a set or two, so
`back_face_is_a_play` falls back to the TYPE SHAPE when the layout is unknown:
a creature front with an instant/sorcery back is playable across 210 cards with
no `transform` or `flip` among them.

Two corrections to that, both from the user:

- **The shape identifies the VERDICT, not the layout.** `Creature -> Sorcery` is
  `adventure` (82), `prepare` (45) *and* `modal_dfc` (7). All three agree the
  back is playable, which is why the rule holds — not because the shape is
  diagnostic.
- **A creature can have a LAND on the back**, and that shape is conflicted:
  `modal_dfc` 13 against `transform` 6. Deliberately left undecided. A creature
  that transforms into a land and a creature you may play as a land are the same
  two type lines.

Measured overall: **shape settles the verdict for 56% of multi-face cards** and
cannot for the other 44%, the largest conflicted class being
`Creature -> Creature` at 223 (`transform` 191, `modal_dfc` 17, `flip` 15).
`None` is returned there, because guessing either way is wrong somewhere.

#### Latent, not live — and pointed straight at us

Zero of these names appear in the 32 stored positions or in the recorded
candidates. But the user's own Hobbit decks contain seven, recordings use modern
cards, and `Everywhere` already cost 35 of 52 exported boards (21.106). This is
the same class: a modern card whose shape our corpus flattened.

### 21.112 One recorded game yields five boards with a real decision

The pipeline ran end to end for the first time, on a human-vs-AI game (MAD bot,
skill 10) with two 60-card constructed decks rather than the 40-card welcome
decks of 21.106. **Zero collector errors, tokens flagged correctly.**

```
64 snapshots                      one board per step, 15 turns
31 pass the shortlist filter      hand >= 2 cards AND >= 2 untapped permanents
21 comparable                     8 lost to tokens, 2 to blind spots
 5 offer >= 2 distinct plays      the boards that ask something
```

| plays offered | boards |
| --- | --- |
| 0 | 5 |
| 1 | 11 |
| **2+** | **5** |

The five:

```
turn 1  PLAY Ba Sing Se / PLAY Forest
turn 3  CAST Llanowar Elves / PLAY Ba Sing Se / PLAY Forest
turn 5  ATTACK Llanowar Elves / CAST Llanowar Elves / CAST Lumbering Worldwagon / PLAY Forest
turn 7  ATTACK Llanowar Elves / CAST Glimpse the Core / CAST Lumbering Worldwagon / CAST Sapling Nursery
turn 9  ATTACK Llanowar Elves / CAST Lumbering Worldwagon
```

And **five overstates it**: turn 1 is a choice between two untapped lands, which
is a decision the way a coin flip is a decision. Three of the five (turns 5, 7,
9) are boards worth asking about.

#### What that costs

At ~3-5 usable boards per game, reaching the ~40 positions blunder rate needs
(README sizing) is **8-13 recorded games**, before any rubric is written. A game
is roughly half an hour, so that is 4-6 hours of play plus authoring — against
32 existing positions that already exist and, per 21.107, mostly still work.

That reframes the plan. Recording is **not** a fast way to replace the set. It
is a way to add boards whose provenance is unimpeachable, at roughly one
position per ten minutes of play, and the honest comparison is against fixing
what 21.109 identified: 12 of 28 boards pinned at the Gate 3 ceiling.

#### The biggest recoverable loss is tokens

**8 of 31 candidates (26%)** were excluded because a token was on the board and
`CardTestPlayerAPIImpl` cannot place one (21.106). Those boards are fine — the
recording is accurate — but the engine cannot be asked about them, so their
`legal_actions` cannot be generated or checked. One deck making one Treefolk
Token cost a quarter of the harvest, and modern decks make tokens constantly.

That is the highest-value fix available: it is worth more than a better deck,
and unlike deck choice it is a one-time cost.

#### The decision window moved but did not widen

21.108 measured turns 9-11 on welcome decks. With 60-card constructed decks it
was **turns 3-7** — earlier, because the curve is real and Llanowar Elves
accelerates — but the same size, 17 dense boards against 20. A faster deck moved
the window rather than widening it; from turn 8 on, the hand was empty.

So the density lever is **card advantage**, not curve. A deck that refills —
draw spells, recursion — should hold the window open past the point where these
two both ran out.

### 21.113 Tokens rebuilt from characteristics, and the ones that must not be

21.112 measured the biggest recoverable loss in the recording pipeline: **8 of
31 boards (26%)** excluded because a token was on them. `addCard` takes a card
NAME and a token is not a card, so those boards could not be handed to the
engine at all.

The name cannot fix it either. XMage has **796 token classes** and five are
Treefolk, so `"Treefolk Token"` does not say which one — and `TokenImpl` is
abstract, so there is nothing generic to instantiate.

#### Record the shape, rebuild the token

The collector now records, per token: `token_types`, `token_subtypes`,
`token_colors`, `token_rules`. Types and subtypes are stored as the enum
**`name()`**, so the export emits `SubType.TREEFOLK` with no spelling table
between them — the same reason the phase vocabulary mirrors XMage's enum rather
than paraphrasing it (21.102).

The export emits one nested `TokenImpl` subclass per distinct token and places
it with `putOntoBattlefield(1, game, null, controllerId, tapped, attacking)`,
which carries the recorded tapped and attacking state. Verified by running:

```
=== pos-tokentest-0001 @DECLARE_ATTACKERS ===
ATTACKER: Treefolk Token
BLOCKER: Treefolk Token
PLAYABLE: Cast Llanowar Elves
```

The rebuilt token is a real permanent the engine will attack and block with.

#### Abilities are not rebuilt, and that stays a refusal

Nothing here can turn `"{T}: Add one mana of any color."` back into Java. A land
token that taps for mana is exactly the case where a missing ability changes
which spells are castable — `Everywhere` again (21.106).

So a token carrying rules text is still **reported as a problem** and still not
rebuilt. Rebuilding it would re-create the bug this fixes, with a wrong token
instead of an absent one, and the wrong version is worse: an absent token is
visibly missing and a vanilla stand-in looks correct.

A token recorded before this change has no characteristics at all and is refused
for a third, separate reason. The existing two recordings are in that state.

#### An operational trap that cost two restarts

`startServer.sh` ships **inside** `mage-server.zip`. Deploying a new build
unzips over it and restores the stock version — relative jar path, no
`--add-opens`, no collector flags — so the server fails with
`Unable to access jarfile ./lib/...` and the fix looks like a path bug rather
than a clobbered file. The launcher is now `magicllm-server.sh`, a name the
assembly does not contain and a redeploy therefore cannot overwrite.

### 21.114 A client NPE that was an operational mistake, not a bug

> **WRONG, and corrected by 21.115.** The stale session was a plausible
> coincidence, not the cause: the same NPE recurred on a FRESH client. The cause
> is the client/server build mismatch this section dismissed at the bottom as
> "the mismatch this did NOT turn out to be" — it did. Read 21.115.

Reported mid-session: the client died with

```
NullPointerException: Cannot invoke "mage.view.GameClientMessage.getGameView()"
because "message" is null
    at mage.client.remote.CallbackClientImpl.lambda$onCallback$3
```

The stack is entirely client-side, which makes the collector the obvious
suspect and the wrong one. Timeline from the two logs:

| time | event |
| --- | --- |
| 14:24 | client started |
| **16:06** | **server restarted** to deploy the token fix (21.113) |
| 16:25 | client reconnected *"with restored session"* |
| 16:30:41 | game started; one snapshot written — turn 1, hands 7/7 |
| 16:30:50 | client NPE |

The server dealt a correct game and the collector recorded it: **zero collector
errors that day**, and the one snapshot is well-formed. The only
`magic-llm positions: snapshot failed` lines in the whole server log are from
2026-08-26 and are the `getCards(null)` bug 21.106 already fixed.

XMage restores a session **by id**, so a client holding an id from a dead server
process reconnects to a new one that has never heard of it. The game then runs
server-side while the client's first callback arrives null.

Two things worth keeping. **A client-side stack trace is not evidence of a
client-side cause** — here the cause was neither side's code but the order the
two processes were restarted in. And the failure is silent on the half that
matters: the server logged nothing wrong, because from its point of view nothing
was.

The operational rule is now in the launcher: restarting the server requires
restarting the client.

#### The mismatch this did NOT turn out to be

Worth recording since it was the first hypothesis and remains a live risk. The
server is built from the 2026-08-25 checkout and the client is the installed
2026-08-12 build. The version handshake passes — both are `1.4.61` / `"V1"`, and
`MageVersion` compares major/minor/release/releaseInfo — but it does **not**
compare card pools, and thirteen days of set implementations separate them.
Nothing has been attributed to that yet; building the client from the same
checkout is the way to remove the variable if an unexplained client failure
recurs.

### 21.115 Three creature types apart: the version check that cannot see a card pool

21.114 blamed a stale session and was wrong. The same NPE recurred on a
**freshly started client**, and the user supplied the pattern that solved it:
both failures came from a *"choose a creature type"* effect — Lorwyn Eclipsed
(a tribal set) and Secluded Courtyard (*"as this enters, choose a creature
type"*).

Measured directly, comparing `mage/constants/SubType.class` in the two jars:

```
client (2026-08-12 build):  545 SubType constants
server (2026-08-25 build):  548
in server but not client:   FEROZ, GREENSLEEVES, WORZEL
```

"Choose a creature type" is the one interaction that serialises the **entire
SubType space** to the client. Three constants it cannot resolve, and the
callback arrives `null` — hence

```
NullPointerException: Cannot invoke "GameClientMessage.getGameView()"
because "message" is null
```

#### Why the version check let it through

`MageVersion` compares major, minor, release and `releaseInfo`, and
`MAGE_VERSION_RELEASE_INFO_MUST_BE_SAME` is true — so the handshake is strict.
Both builds report `1.4.61` / `"V1"` and pass it.

**A version string is not a content hash.** The number changes on release and
the card pool changes every day, so two builds thirteen days apart are
`compareTo() == 0` and disagree about what exists. Exactly the gap `CARD_PIN`
exists to close on our side (21.110): a sha proves the corpus is the one the
numbers were computed against, and a version string proves nothing of the kind.

#### The diagnostic lesson, which is the one worth keeping

21.114 had the right hypothesis written down and dismissed it — the section ends
with *"the mismatch this did NOT turn out to be"* — because a simpler story fit
the timeline. It fit because it was **constructed from the timeline**: the
server had been restarted, so a session explanation was available, and the
evidence for it was entirely circumstantial.

What broke it was a *reproduction under a changed condition* (fresh client) and
a **pattern across two instances** (both choice effects), neither of which came
from the logs. The logs could not have settled it: the server logged nothing
wrong, because from its side nothing was.

Fixed by building the client from the same checkout, so both sides carry the
same 548 — verified by comparing the enum in the two jars after deploying, and
visible in the client's own startup: the rebuilt client imports **586 sets and
92,134 cards** where the 2026-08-12 one had 584 and 92,030. A hundred-odd cards
is what "the same version" covered.

The launchers are `~/xmage-magicllm/magicllm-server.sh` and
`magicllm-client.sh`, both outside any distribution zip so a redeploy cannot
restore a stock one over them (21.113).

### 21.116 A permanent is not a card name plus tapped

Prompted by the user after 21.115: *"choosing a creature type is a very common
variable for kindred decks and there are other similar style value choices we
need to assure are handled correctly."* An audit of what the engine tracks on a
permanent against what the collector records and the export can rebuild.

The collector recorded: name, controller, tapped, attacking, sick, P/T, token.
Everything below changes what the engine answers and none of it survived
`addCard(name)`.

| unrecorded state | cards | why it matters |
| --- | --- | --- |
| +1/+1 or -1/-1 counters | **10.6%** | P/T; `addCard` gives the printed body |
| Auras | 3.7% | grant and remove abilities |
| other counters (loyalty, charge…) | 2.0% | loyalty gates planeswalker abilities |
| Equipment | 1.9% | grants abilities, changes P/T |
| **chosen value** (type/colour/name) | **0.9%** | decides what mana a land makes |
| face-down (morph/manifest/disguise) | 0.9% | it is a 2/2 with no abilities |
| phased out | 0.2% | the permanent is not there at all |

#### The empirical number is much worse than the corpus rate

Across the four recordings so far: **150 of 538 creature permanents (28%) have a
P/T that differs from the printed card.** Five cards account for all of it —
`Marauding Mako` 47 times, `Textbook Tabulator` 43 (base 0/3, recorded 1/4).

So more than a quarter of recorded creatures would be rebuilt wrong by name
alone, and nothing said so. The corpus rate says a mechanic is *rare*; the
recordings say the cards that have it are *played repeatedly*.

#### The chosen value is the one that already broke a game

`ChooseCreatureTypeEffect` stores its result in the **game state**, not on the
permanent:

```java
game.getState().setValue(source.getSourceId() + "_type", SubType.byDescription(...));
```

So two Secluded Courtyards with identical names, types and P/T make different
mana, and nothing the collector read could tell them apart. That is 21.115's
failure from the other direction: there it broke the client, here it would have
silently produced a board where the engine reports the wrong castable spells.

Recorded now by probing the known suffixes (`_type`, `_color`, `_name`,
`_cardName`, `_subtype`). There is **no registry of suffixes**, so an effect
using another one stays invisible — which is exactly why the export refuses on
what it sees rather than assuming absence means nothing was chosen.

#### Rebuild what is exact, refuse the rest

The rule the token work settled (21.113) generalises: a wrong permanent looks
correct and an absent one does not, so the only safe options are reproduce it
exactly or refuse the board. Tokens are rebuilt because their characteristics
are fully recorded; none of the state above is, so `permanent_state_problems`
reports and the board is marked as one the engine cannot be asked about.

That trades yield for honesty, and the yield cost is real — on a counters-heavy
deck it could exceed the 26% tokens were costing. The alternative is a board
that answers a different question, which this project has now paid for four
times (21.105 combat state, 21.106 tokens, 21.111 faces, 21.115 subtypes).

### 21.117 Ask the engine what it built, instead of predicting what it cannot

21.116 traded yield for honesty: every permanent carrying counters, an
attachment, a chosen value, a face-down state or phasing was refused. Two
follow-ups, and the second supersedes the approach.

#### Counters are rebuilt, not refused

`CardTestPlayerAPIImpl.addCounters(turn, step, player, cardName, CounterType,
count)` exists, and `CounterType.findByName` resolves the recorded name inside
the engine — so there is no counter-name table on this side to drift, the same
reason `SubType.name()` is recorded rather than paraphrased (21.102, 21.113).

That matters more than the other classes combined: counters are **10.6% of
cards and 28% of recorded permanents**. Refusing them would have cost more yield
than tokens did.

Whether `findByName` knows a given name is not predictable from Python without
copying the enum, so the emitted Java calls it directly: an unknown name returns
null, `addCounters` throws, the test fails, and the diff already excludes a
failed board. Loud and self-correcting rather than a second list to maintain.

Verified by running — base 2/2, recorded 4/4, engine built:

```
PERM: Grizzly Bears 4/4
```

#### The readback, which is the actual fix

That `PERM:` line is new, and it changes the shape of the problem. Every bug in
this thread has been the same one:

| | the engine was handed |
| --- | --- |
| 21.105 | a board with nobody attacking |
| 21.106 | a board missing a token |
| 21.111 | a card face that is never cast |
| 21.116 | a permanent without its counters or chosen type |

Four instances, each found only after it had cost something, and each fixed by
adding a class to a list of things to check for. **That list grows every set.**
`prepare` was two sets old and unknown to this project; the next mechanic will
be too.

So the emitted test now prints the board **as the engine actually built it**,
and `board_fidelity` compares it to the recording — names, counts and P/T. A
difference means the engine's answer is about a board the position does not
describe, *whatever the reason*, including reasons nobody has thought of.

The diff reports those separately and excludes them from both columns, because a
setup mismatch is not a legality disagreement and counting it as one would put a
harness bug in the findings — 21.5's shape, which this project has been caught by
before.

A report with no `PERM:` lines predates the readback and is **not** treated as a
mismatch: absence of evidence must not read as evidence of difference. That is
the same rule as `protocol_findings` returning None for an undecidable entry.

### 21.118 A clock loss is not a game loss, and it left a duplicate board

The user, after a game: *"another thing to look out for in the log analysis is
losses based on timeout."* Two defects, one cause.

#### Nothing recorded how a game ended

`onGameEnd` cleared a map and wrote nothing, so a game decided on the clock was
indistinguishable from one played to a win. XMage distinguishes them precisely —
`hasTimerTimeout()`, `hasIdleTimeout()`, `hasQuit()`, `hasLeft()`, `hasWon()`,
`hasLost()`, `getWinner()` — and we simply were not asking.

The collector now writes one `record: game_end` line per game carrying all of
it. `split_records` keeps it out of the board stream, because it has no
battlefield and `to_position` would emit an empty one.

**The boards from a timeout game are real and usable; the OUTCOME is not.** A
clock loss says nothing about the line that was played, and if game outcome ever
becomes a signal — "this line won" — a timeout would poison it silently.

#### The duplicate board, which is the same bug from the other end

A timeout game ends mid-turn, and the recording showed it: turn 12
`PRECOMBAT_MAIN` twice, byte-identical, both players alive at 11 and 7.

`snapshotIfNewStep` skips when the (turn, step, active player) key matches the
previous one — and `onGameEnd` **removed** that key. So the next log message
found no previous key, treated the final board as a new step, and wrote it
again. One duplicate in 51 snapshots, in the one place a duplicate is least
visible: the end, where the game has stopped changing anyway.

A `finished` set now suppresses snapshots after the end, and `onGameStart`
clears it so a reused game id still records.

Worth noting the earlier measurement that missed this. 21.108 reported **0
duplicate boards among 66 snapshots** and was right — that game played to a
conclusion, so no post-end log messages arrived. The check was sound and the
condition that triggers the bug was simply absent, which is 21.5's shape once
more: a defect whose visibility depends on how the run happened to end.

### 21.119 A cold image cache reads as a failing network

Reported: the server dropped the client *"almost once per move"*, with
`The connection to the server was lost. Reconnect?` — reconnecting worked every
time. Nothing to do with the collector: **zero collector errors**, and the
server log was 85 lines with no stack traces.

The server log said `LOST CONNECTION (bad network) - steve`, from
`Main$MageServerConnectionListener.handleConnectionException` on `[Timer-0]` —
the lease timer. And the timing was the tell:

```
01:11:36 connected -> 01:11:56 LOST   (20s)
01:11:58 connected -> 01:12:18 LOST   (20s)
```

Exactly twenty seconds, every time. A flaky network is irregular; a fixed
interval is a timer firing.

#### The cause

| | `plugins/images` |
| --- | --- |
| the installed client, used for weeks | **4.1 GB** |
| the client built in 21.115 | **13 MB** |

A cold client downloads card art **on demand, on the Event Dispatch Thread** —
the same thread that answers the server's keepalive. Every new card stalls the
EDT, the lease period elapses with no response, and the server drops the
session. Hence *once per move*: a move reveals a card, the card triggers a
download.

So building a matching client to fix 21.115 introduced this, and the symptom
named the wrong subsystem. `(bad network)` is XMage's own wording for "no
response in time", which is true and points away from the cause.

#### Preserved across rebuilds

The user's question — can the symbols and images survive a rebuild — is the
durable half. Deploying is `unzip -o` over the install directory, which wipes:

- `plugins/images` — 4.2 GB of card art, **and the symbols**, which are a
  subdirectory of it rather than a separate cache;
- `db/` — 294 MB card database;
- `config/` — preferences, including the server address.

`~/xmage-magicllm/redeploy.sh` now moves those three aside, unzips, and moves
them back. A move rather than a copy: 4.2 GB copied twice per deploy is a minute
of disk for nothing, and a move is atomic on one volume.

Client start-up: **12 seconds cold, 5 seconds warm**.

#### Two diagnostic notes

The machine's LAN address had changed under DHCP (192.168.1.4 to .67) between
sessions, which briefly looked like the client was on another device. Worth
checking before reading anything into an address.

And this is the second failure in two days where a client-side symptom named the
wrong subsystem — 21.115's NPE was a card-pool mismatch, this one a cold cache,
and both presented as connection faults. When the client reports a connection
problem, the connection is the last thing to suspect.

### 21.120 The end record's first use found one of its own fields unreliable

The first fully-instrumented recording — token characteristics, permanent state,
end records, all live at once. It worked:

```
33 boards + 1 end record
counters captured on 24 permanents
chosen   captured on 14
duplicate boards: 0          (the 21.118 fix holding)
```

And the end record immediately earned its place by contradicting itself:

```
END RECORD: turn 10, winner 'Game is a draw'
   Computer 2   life  -1   ['lost', 'left']
   steve        life  25   [(none)]
```

A draw, with one player dead at **-1 life** and flagged `lost`. The user won
that game.

#### Why `winner` is wrong at that moment

`GameImpl.findWinnersAndLosers()` both sets `winnerId` **and** calls
`player.won(this)` on the survivor. Our record shows `steve` with `won: false`,
so that method had not run when `onGameEnd` fired — `winnerId` was still null,
and `getWinner()` returns the literal string `"Game is a draw"` for a null
winner.

So the field is not merely stale, it is **confidently wrong in a specific
direction**: every game we record will claim a draw unless the winner happens to
be decided first.

#### Derive from what is set, not from what is present

`lost`, `left`, `quit`, the timeout flags and life totals were all correct in
the same record — they are set as the game plays, not at the end. `end_summary`
now ignores `winner` entirely and derives the outcome from those:

```
game f8300987: played out to turn 10 — steve won, Computer 2 lost at -1 life
game 53737496: ended by concession at turn 2 (steve quit)
game 46d26625: played out to turn 4 — Computer 2 won, steve lost at 19 life
```

The raw field is still written to the JSONL — removing it would hide the
evidence for this section — but nothing reads it.

Worth naming the shape, because it is the fourth version of it here: a field
that EXISTS and is populated is not thereby true. `getWinner()` never returns
null and never errors; it returns a plausible sentence. The same trap as a
version string that compares equal across different card pools (21.115) and a
`mana_cost` that is `None` for two layouts meaning different things (21.111) —
the value is present, well-formed, and answering a question you did not ask.

An unresolved game now says *"outcome unclear"* rather than being assigned a
winner, on the same principle that a check which cannot run must not report an
answer.

### 21.121 Life, not `left`, separates a real loss from an abandoned game

21.120 replaced the unreliable `winner` field with a derivation from `lost`,
`left` and `quit` — and the derivation was wrong in the same direction. It
reported

```
game 46d26625: played out to turn 4 — Computer 2 won, steve lost at 19 life
```

for a game the user identified as **abandoned**: it was the session interrupted
by the cold-cache disconnects of 21.119. Nothing in the data said so; the user
did.

#### Three games, and the flag that discriminates nothing

| game | loser | life | quit | left | actually |
| --- | --- | --- | --- | --- | --- |
| f8300987 | Computer 2 | **-1** | no | yes | died |
| 53737496 | steve | 20 | yes | yes | conceded |
| 46d26625 | steve | **19** | no | yes | **abandoned** |

`left` is true on all three, **including the genuine loss**, because a player
leaves the table after losing normally. It looks like the abandonment flag and
carries no information at all.

**Life is the discriminator.** A player flagged `lost` at positive life did not
die; something removed them from the game. The rule is now: `life <= 0` is a
real loss, `quit` is a concession, and `lost` at positive life without `quit` is
an abandoned game whose outcome is not evidence.

#### The limit, stated rather than papered over

A loss by **decking or poison** also leaves the loser at positive life and will
be reported as abandoned. Nothing in the end record separates them. The last
board's `library_count` would settle decking and is deliberately not read —
adding a dependency from the end record to the board stream for a case that has
not occurred is a coupling with no evidence behind it. Written down so the wrong
answer is a known one.

#### The lesson, which is about evidence rather than code

Twice now a plausible outcome has been derived from fields that were present and
insufficient — `getWinner()` in 21.120, `left` here — and both times the error
was invisible in the data. What settled it was **the user naming a game they
remembered**. Three recordings with known ground truth was enough to find a
discriminator that four fields of flags could not.

That is worth keeping as a method: when a derived field cannot be checked against
anything, the cheapest validation is a small number of cases someone can
independently identify. The same argument as adjudication for judge calls
(21.57), one level down.

### 21.122 B3 resolved: blunder rate is measured, not gated

The open question since the judge work: the 25% Gate 3 bar was set when blunder
rate was read by a judge that fired at 75% of clean answers, and the instrument
has changed underneath it. Resolved by the user (B3): **no threshold, at any
value.** The rate is reported; nothing passes or fails on it.

Three findings retired the bar rather than moved it.

#### The metric selects the arm handed the answer

Split three ways on the n=32 run, basic+intermediate, 32B judge:

| arm | headline | strategy | protocol | only-PASS |
| --- | --- | --- | --- | --- |
| **base_closed** | **54%** | 39% | **36%** | 4% |
| base_cards_open | 82% | 39% | 79% | 11% |
| base_open | 86% | **32%** | 75% | 25% |
| base_open_think | 89% | 46% | 75% | 14% |
| ft_cards_open | 96% | **32%** | 89% | 4% |

`base_closed` wins the headline by 28 points and **ties or loses on strategy**
(39% against 32%). Its entire advantage is protocol: 36% against 75–89%. It is
the arm *given* its legal actions, so it has fewer opportunities to name a play
that is not available — and `PROTOCOL_ERRORS` entered `errors_made` in 21.76.

So the gate was substantially a protocol-compliance measure wearing the name of
a play-quality one, and it would have crowned the arm that was handed the answer
key. Note this is **not** 21.58's do-nothing confound: `only_pass` is 4% for
`base_closed`, the lowest of the five.

#### The number is wrong in a known direction

Reweighting for the judge's unreliable clean verdicts (21.81) moves the best arm
from **54% to ~70%**. A bar the arms are 29 points from becomes one they are 45
points from — and the correction is largest exactly where the judge calls
answers clean most often, which is the arm the gate selects.

#### And 41% of the boards cannot move it

12 of 28 pinned, every one at all-blundered, so the effective n is 16 (21.109).

#### What changed

`gate3_blunder` is unchanged and still computes the rate. What went is the
verdict: the single-judge report prints the rate beside the strategy column and
the headroom, and the two-judge comparison prints **both rates and their
spread** rather than two PASS/FAILs and a reversal flag. The spread is what the
reversal check was proxying for, and it stays informative at every distance from
the retired bar rather than only near it.

`GATE3_MAX_BLUNDER` is now `GATE3_REFERENCE_BLUNDER` — the same 0.25, kept so
the reports can say how far the arms sit from the bar that used to exist. The
rename is the point: a constant named `MAX` invites a comparison, and there is
no longer a comparison to make.

Gates 1 and 2 are untouched. Gate 1 comes from the action parser and never sees
the judge; Gate 2 measures discrimination, which is exactly what a set with 12
pinned boards needs reported.

#### Why not the other options

Redefining on strategy alone was the near miss. It measures the right thing —
but the strategy rates are 32–46%, so a 25% bar still fails everything, and
picking a new number would be arbitrary. Raising the bar to current performance
makes the gate a moving target that certifies whatever the model does. Keeping
25% as an aspiration leaves a FAIL that carries no information about progress.

Measuring without gating keeps every number and drops only the claim the
evidence does not support. It also matches how `only_pass` is already handled:
reported, not gated, pending a reason to gate.

### 21.123 The fidelity check's first run: 17 false positives, then 16 usable boards

The first game recorded with everything instrumented — 63 boards over 16 turns,
a real loss at −21 life, **counters on 165 permanents**, zero duplicates.
Counters at that density is exactly the case 21.116 would have refused and
21.117 rebuilt.

The new `board_fidelity` check then excluded **17 of 26** boards it looked at.
Which was itself a bug.

#### The check read a key the writer never sets

A position stores P/T as the **string** `pt` — `"1/1"` — and `import_game` omits
it entirely when zero, so a rendered board does not claim a Mountain is a 0/0.
`board_fidelity` read `b.get("power", 0)`, a key candidates never carry, so
**every permanent compared as 0/0**:

```
recorded 1x 'Llanowar Elves' 0/0 that the engine did not build
the engine built 1x 'Llanowar Elves' 1/1 that was not recorded
```

21.49's shape — a check reading a default as a value — with one difference that
mattered: it failed **loudly**, as 17 impossible mismatches, rather than as a
pass. That is the only reason it was caught in the same hour it shipped, and it
is the argument for checks that fail toward noise rather than toward silence.

#### The yield, before and after

| | before the fix | after |
| --- | --- | --- |
| comparable boards | 9 | **20** |
| **≥2 distinct plays** | 6 | **16** |
| board mismatches | 17 | 6 |

**Sixteen boards with a real decision from one game**, against the 5 measured in
21.112. That changes the arithmetic behind the plan: ~40 positions is **2-3
games**, not 8-13.

The remaining exclusions are honest ones: 15 tokens refused (the Treefolk Token
has `reach`, and only keyword-free tokens rebuild exactly), 8 test failures, 6
real mismatches, 2 blind spots.

#### The six real mismatches are animated lands

```
recorded 'Ba Sing Se' 2/2  — the engine built a land
recorded 'Forest'    9/9   — the engine built a land
recorded 'Lumbering Worldwagon' 4/4 vs engine 5/4
```

A land that is currently a creature cannot be rebuilt by `addCard`, which places
the printed card. Nothing in the pipeline knew that until the readback said so —
and it is precisely the class of state 21.116 enumerated and could not have
finished enumerating. The readback found it without being told to look.

### 21.124 Keyword tokens rebuilt, one P/T parser, and the deck that animates lands

Three findings from squeezing the counters-heavy game for validation.

#### Keyword abilities rebuild; real text still refuses

15 boards were excluded because one Treefolk Token has `reach`. Keyword
abilities are singletons — `ReachAbility.getInstance()` — so a token whose
abilities are ALL keywords rebuilds exactly. Recorded rule texts are plain
lowercase words: `flying` (43), `reach` (12), `indestructible` (3).

`_KEYWORD_ABILITY` is **a table on our side**, which this file otherwise avoids
on principle (`CounterType.findByName`, `SubType.name()`, `PHASE_STEPS`). It is
here because the engine offers no rule-text lookup — no Keyword enum, no
`Ability.byRule` — and every entry was checked against the checkout rather than
assumed. **`MenaceAbility` is not a singleton**, so it takes its constructor;
guessing would have compiled to nothing.

A token whose ability is real text still refuses. `{T}: Add one mana of any
color` cannot be rebuilt, and a vanilla stand-in for it is 21.113's mistake with
a keyword instead of a token.

#### The same P/T bug, in a second place

21.123 fixed `board_fidelity` reading `power` from records that store the string
`pt`. The token rebuild had **the same bug**: a 1/1 Faerie was emitted as

```java
super("Faerie Token", "0/0 token");
power = new MageInt(0);
```

A 0/0 creature dies to state-based actions the moment it enters, so the rebuilt
board would have been missing the token entirely — while looking correct in the
generated source.

Second occurrence is why `permanent_pt` now lives in `common.py` and all three
callers use it. Fixing 21.123 in place and moving on would have left this one,
which is this project's most repeated bug in its purest form: the same defect,
two callers, one fixed.

#### 18 usable boards, and 17 lost to the deck

```
51 candidates -> 22 comparable -> 18 with >=2 distinct plays
excluded: 17 board mismatch, 8 test failure, 4 blind spot
```

**Every one of the 17 mismatches is an animated land** — `Ba Sing Se 2/2`,
`Forest 9/9`, `Forest 15/15` — recorded as creatures and rebuilt as lands,
because `addCard` places the printed card and the animation came from a resolved
ability. Not fixable without replaying the game.

That is deck guidance, not a tooling gap: **a deck that animates lands makes a
third of its boards unusable for engine validation.** The boards are still real
and still renderable — they just cannot have their `legal_actions` generated or
checked.

Yield across the three measurements: **5** (21.112, welcome decks) -> **16**
(21.123, after the P/T fix) -> **18**. Roughly 2-3 games for ~40 positions,
against the 8-13 estimated before any of this.

### 21.125 The authoring path, built out of the form that already existed

The recording pipeline ended in a dead end: 18 usable boards a game, and
`validate_position` correctly refusing every one of them for having no rubric
(21.103 leaves it empty on purpose). Nothing could attach one — `#/position`
authors a board and rubric together from scratch, `#/reference` authors a LINE
for an existing board, and neither takes "here is a real board, write its
rubric".

#### No fourth view

`rubric_server.py` already has `/rubric`: it collects `key_points` and
`common_errors` against a `question`, and its CSS is already `pre-wrap`. So a
recorded board becomes a task by **rendering it into `question`** with the same
`render_position` the model reads.

That last point is the reason to do it this way rather than build a board-aware
view. The author and the model then see the *same string*, so a rubric cannot be
written against a different board than the one evaluated — which is the failure
21.105, 21.106 and 21.116 all were, in the other half of the pipeline.

Verified end to end against a running server: `/rubric` 200, `/api/tasks` lists
5, `/api/task/<id>` returns the rendered board with `draft: []`, a POST to
`/api/submit` stores the rubric, and the ingest merges it.

`draft` ships **empty**. The rules flow sends a machine sentence-split of the
answer to argue with; a recorded board has no answer yet, and 14.6 measured
machine drafts at r = +0.30 against hand-written +0.62. A generated draft here
would put 21.98's caveat straight back into the one part of the pipeline meant
to escape it.

#### What the ingest may and may not touch

`--ingest-rubrics` merges `key_points`, `common_errors` and attribution. **It
does not rewrite the board.** The board came from a played game, and editing it
would destroy the only property recording exists to provide; a wrong board is
dropped, not fixed.

`category` and `difficulty` come from the command line, not the form. They are a
claim about what the board *asks*, and a wrong category silently changes what
`label_store` stratified sampling draws.

#### Splitting, because `--ingest` refuses a file whole

That rule is right for a hand-written drafts file: a malformed record means the
batch was not reviewed. It is wrong for incremental authoring, where one
finished rubric would be blocked by four unstarted ones. So complete boards are
split into `<candidates>_ready.jsonl` and the rest stay put. Both behaviours
keep their meaning.

Measured on the first pass: 1 rubric authored, 1 board split out, **0 problems**
from `validate_position`, and `--ingest --dry-run` parsed it. The gold set is
untouched — promotion is still a separate, deliberate act.

#### What this unblocks

B5's missing number. Sourcing is now ~18 boards a game; authoring cost has never
been measured, and it is the number the gameplay track's budget actually turns
on. Two or three rubrics through this path give it.

### 21.126 The form the boards were sent to was the wrong form, and it cost three things

21.125 argued that reusing the deployed rules form beat writing a fourth
authoring view: it is deployed, it has been used, and its submission path,
attribution and duplicate handling are all tested. That argument was about the
*server*, and it was correct. It said nothing about the *page*, and the page is
where the author is.

The first real authoring session found it immediately. Against a recorded board
the form displayed:

> **Verified answer — this is correct, do not re-adjudicate**
>
> *(nothing)*

and the reviewer's note read: *"This is not a Rules question but a gameplay
question."*

#### Three costs, in increasing order of seriousness

**A verified answer that does not exist.** The rules flow's premise is that a
correct answer is on file and the rubric is written against it. A recorded board
has none — `export_rubric_tasks` sets `answer: ""` deliberately — so the card
asserted an empty string was verified correct. Cosmetic, but it is the first
thing on the page.

**Hints for the wrong artefact.** *"The claims a correct answer must make…
never a bare Yes or No"* is guidance for a rules question. A position's
`key_points` is the correct **line** and its `common_errors` is the blunder
list. Nothing on the page said so.

**A rubric the form needed and did not have.** This is the one that moves a
number. A position's `common_errors` is unioned with `PROTOCOL_ERRORS` before
anything reads it — `eval_positions` line 552 for the judge, `adjudicate.task_for`
line 211 for the human form. The judge returns error **numbers**, and `score_run`
splits strategy from protocol **by index**. So an author who cannot see those ten
entries writes an eleventh that restates one, and the copy is scored as a
*strategy* charge that no parser confirms or refutes — moving a mistake the
parser already decides at 80–85% into the column measured at 18–43%, and
inflating it (21.74).

21.75 was the judge holding a rubric the human form lacked, and it reversed the
sign of a precision comparison. This is the same join in the same place, one
step earlier: the form needs a rubric **the judge already has**, so the author
can avoid it rather than duplicate it.

#### The defect underneath: `legal_actions` was empty on all 71

Worth more than the form bug. The recorded drafts carry no `legal_actions` —
correctly, since the collector deliberately does not compute them and the
validated engine path fills them in later. But nothing said so, and the
consequence is not a missing field:

```
legality(parsed, [])  -> {'n_legal': 1, 'all_legal': False,
                          'illegal': ['PLAY Starting Town']}
protocol_findings     -> {3: True}
```

A **correct** play scores `all_legal=False` and fires protocol entry 3 — the one
entry the judge gets right, and the one `legal_actions` exists to decide. It is
`emitter_blind_spot` from the other side: there an empty list made the closed
prompt claim no play was available; here it makes every play unavailable.

The rubric work is **not** wasted, and that is worth stating precisely rather
than assuming either way. `render_position` ignores `legal_actions` entirely —
verified, the rendered string is byte-identical with and without it — so a rubric
authored now is authored against exactly the board the model reads. The field is
scoring machinery, not question text. Fill it later, and nothing authored moves.

#### What changed

- Each task declares `kind: "position"` and carries `protocol_errors` and
  `legal_actions`. They ride **on the task**, so the file stays the
  self-contained thing `rubric_server` reads, and a task exported today keeps
  showing the ten entries it was authored against after `PROTOCOL_ERRORS` grows.
- The **file** kind stays `"rubric"`. The two answer different questions: the
  file's picks the form and the submission shape, the task's picks the framing
  inside it. Setting the file kind made `load_tasks` refuse the file outright —
  caught by starting the server, not by `--help`.
- `common.protocol_restatements` warns when an authored blunder restates one of
  the ten, live in the form and again at ingest. Signatures are **derived from
  the entries' own words**, so they cannot drift when the list is appended to.
- `import_game.py --export-from` re-exports `tasks.json` from an existing
  candidates file. Re-deriving from snapshots cannot promise stable ids — `--as`
  and `--interesting` both change which board gets which number — and the ids
  are what already-collected submissions key on.

#### The two-gate warning, and why one gate was not enough

The first version fired only on a shared *distinctive* word — one occurring in
exactly one entry. It missed both halves of the obvious case. *"Only passes
instead of making a play"* shares **every** content word it has with entry 1 and
none of them are distinctive; and entry 4 (a wrong `PHASE` step) has **no**
distinctive word at all, so it could never fire on any wording, including its
own verbatim text.

Coverage catches both: an author who has said nothing the entry does not say is
restating it whether or not a rare word is involved. Measured after the fix,
both directions, because `lint_common_errors` already records that a warning
wrong in both directions gets ignored:

| | fires | silent |
| --- | --- | --- |
| each of the ten on its own text | **10/10** | — |
| paraphrases of an entry | 7/8 | — |
| real strategy errors from a real board | — | **8/8** |

The remaining miss (*"states the wrong phase for the board"*) shares no
vocabulary with entry 4 and is left as a miss: this is a warning, not a gate,
and loosening it further buys the miss back at the price of the silent column.
The false-positive side stays quiet for a structural reason worth keeping — a
strategy error names **cards**, and card names never appear in the ten entries.

#### Status

All seven suites pass. The gold set is untouched. 71 tasks re-exported with
byte-identical ids. `legal_actions` is present on **0 of 71**, so these boards
are authorable now and not scorable until the engine path runs.

### 21.127 Moved to a new machine (2026-08-29): what was re-verified and what was not

The repository moved to a new Mac: **Apple M3 Max, 16 cores (12P+4E), 40-core
GPU, 64GB unified memory**, macOS 26.6.2. Previously built and measured on an
M3 Pro / 36GB (README's own "Requirements" line). Every timing figure in this
document (`~2.5 hours` evaluation, `~6 hours` training, per-run wall clocks
throughout) was measured on the old hardware and is now **stale as a number,
not as a ranking** — more cores and more memory should only move these figures
down, but none has been re-timed on the new machine, so quote the old numbers
as "measured on M3 Pro" until they are.

The repo directory also moved: `mlx_env/pyvenv.cfg` still records its venv as
created at `/Users/codyclark/Documents/code/magic-llm/mlx_env` (the old path),
while the working copy is now at
`/Users/codyclark/Documents/personal_code/magic-llm`. This is the exact shape
of trap this file already warns about — a stale absolute path silently
resolving to the wrong thing — so it is called out here rather than assumed
harmless. It has not caused a failure (imports and every test suite below ran
clean against it), but a `pip install` or interpreter-path issue that shows up
later on this venv specifically should be diagnosed as this, not as a new bug.
If anything gets weird here, rebuild it (`python3 -m venv mlx_env` from the new
path) rather than debugging the old one.

**Re-verified on the new machine, all clean:**

- `scripts/test_imports.py`, `scripts/test_eval.py` (507 assertions),
  `scripts/test_docs.py`, `scripts/test_webui.py` (168), `scripts/test_deploy.py`
  (118), `scripts/gameplay/test_actions.py` (130), and
  `scripts/gameplay/test_eval_positions.py` (61) — all seven suites pass with
  no code changes.
- `mlx.core.default_device()` reports `Device(gpu, 0)` — Metal is visible to
  MLX on this chip, unmodified.
- `data/gold/`, `eval/`, and `models/` show no uncommitted diff — the machine
  move touched no published measurement.
- `models/` carries all nine adapters (`mtg-rules-adapter` through `-v4`) and
  `data/processed/chunk_embeddings.npz` is present, so both the index and the
  fine-tuned weights survived the move rather than needing a rebuild.
- An actual end-to-end query — `python scripts/rag.py query "when are
  state-based actions checked?" --k 3` — ran for real against the existing
  index (not just imported): it fetched the embedding model fresh from Hugging
  Face (confirming network access, and that model downloads are not yet
  cached on this machine) and returned the correct rule (704.1) top-ranked.

**Also fixed:** every console-script shebang under `mlx_env/bin/` (and
`activate`/`activate.csh`/`activate.fish`) still hardcoded the old venv's
absolute path (`.../Documents/code/magic-llm/mlx_env`), which no longer exists.
`mlx_env/bin/mlx_lm.lora --help` failed with `bad interpreter: ... no such file
or directory`; `python -m mlx_lm lora` (the invocation this repo's configs
document) was unaffected, since it goes through the `python3` symlink chain,
which resolves via absolute homebrew paths independent of the venv's own
location. Rewrote the stale path across every affected file with a plain
string substitution — no reinstall needed — and re-ran `test_imports.py` clean
afterward.

**`mlx_lm.lora` fine-tuning re-verified with a real smoke-test run** (10
iterations, `configs/phase1_lora_v3.yaml`, scratch adapter path, not
`models/`): base model downloaded fresh, LoRA trained, checkpoint and val loss
both wrote correctly, peak memory 9.207 GB — identical to the 9.2GB the v3
config's own header records for this exact model/batch on the M3 Pro, so the
memory envelope did not change with the new GPU. Throughput did: measured
0.138–0.215 it/s across the four report points (mean 0.176 it/s) against the
v3 header's documented **0.066 it/s** on the M3 Pro — a **~2.7x** speedup. That
projects the full 1,322-iteration run at **~2.1 hours** versus the documented
~6.2, but this is a 10-iteration smoke test, not a full run — the mean glosses
over warm-up and any later thermal throttling a real multi-hour run could show.
Treat ~2.1h as a planning estimate, not a quotable number, until an actual full
run is timed. One cosmetic, non-fatal warning appeared during the run
(`shmem: mmap: an error occurred while determining whether or not
.../sm_segment... could be created`, from an OpenMPI shared-memory backend some
dependency initializes) — it did not stop or affect training and is noted here
so it isn't mistaken for a new failure on a future run.

**`eval.py` and `eval_positions.py` re-verified against the live calibrated
judge** (`mlx-community/Qwen2.5-32B-Instruct-4bit`, loaded alongside the 7B
base model — confirms the 64GB machine holds both at once with room to spare).
Two smoke runs, n=8 rules questions and n=4 positions x 5 arms — sizes chosen
to be fast, not to measure anything (this project's own bar is ~100-150
questions / ~40 positions before a number means something):

- Rules smoke run reproduced the project's own headline negative result on
  fresh output: `base_rag` (2.08) beat `finetuned_rag` (1.25), matching
  "fine-tuning did not beat retrieval." The finetuned_rag consistency check
  (8/8 identical on rerun) confirms judge determinism holds on this machine's
  MLX build, the same property Section 9's methodology depends on.
- Positions smoke run's report reproduced every documented diagnostic by
  shape: the parser-vs-judge protocol precision split, the phase/tap/mana
  blind-spot notes (21.61, 21.66, 21.83), the n=4 ceiling-effect warning ("3 of
  3 boards cannot move this rate"), and Gate 1/2 verdicts computed the same
  way the plan describes them. That is the report-writer and rubric judge
  executing the documented logic, not new or broken behavior — the 100%
  blunder rate across every arm is an n=4 artifact the report itself flags,
  not a finding.

Both runs wrote to untracked scratch files (`eval/runs/_smoke_*.jsonl`,
`eval/reports/_smoke_*.md`) rather than `latest.*` or any archived stem, so
nothing published or reusable was touched or overwritten.

**Web console re-verified in an actual browser** — all four views load and
function, which is the check `test_webui.py`'s JS/CSS parse-checks cannot
themselves provide (per this file's own traps: a page can parse clean and
still blank or throw at runtime).

**A new machine-specific trap, found running the full timed training run:**
sustained training throughput collapsed to **0.032–0.049 it/s** — *slower*
than the M3 Pro's documented 0.066 it/s, and ~4-5x slower than the
10-iteration smoke test's 0.176 it/s — while the machine sat at **7% battery,
charging, with High Power Mode off**. Not thermal creep (the rate held flat
across 60 iterations rather than trending down) and not resource contention
(no competing CPU/GPU process; `ps -eo pid,pcpu,pmem,comm -r` showed nothing
above 19% CPU and nothing GPU-heavy). Apple Silicon MacBook Pros throttle
sustained compute when the battery is critically low even on AC power — the
charger has to power the system and refill the battery simultaneously — and
the M3 Max specifically ships a **High Power Mode** toggle (System Settings →
Battery → Power Adapter Options) meant for exactly this workload shape, off by
default. Killing the run, letting the battery charge, and enabling High Power
Mode fixed it: throughput returned to **0.138–0.198 it/s** (mean ~0.160),
matching the smoke test. **A slow, flat-rate training run on this machine
should be read as a power-state check first, not a regression** — the same
class of misdiagnosis this file already warns against for `pgrep -f`
self-matching and buffered background logs: a monitoring signal (it/sec) that
reads as one thing (a hardware or software regression) and means another (a
setting).

Also a correctness signal, not just a speed one: at the recovered rate, val
loss at iteration 1 (1.986) and iteration 150 (1.231) matched the v3 config's
own documented values to three decimal places — same seed, same data, same
config reproducing the same loss curve **across machines**, the cross-machine
form of the "d == 0.0 exactly" determinism check the v3 header already ran
within one machine.

**Full run completed and timed.** All 1,322 iterations, adapter path
`models/_timing_test_v3_m3max` (deleted after; scratch only, never promoted).
**Wall-clock 2h 15m** (checkpoint timestamps 11:11:27 → 13:26:35), against the
v3 header's documented ~6.2 hours on the M3 Pro — a **2.75x** speedup, matching
the smoke test's 2.7x estimate almost exactly. Sustained rate held
0.14–0.24 it/s throughout the second half with no further throttling once
High Power Mode was on.

**Final val loss 0.657** — an exact match to the v3 config header's own
recorded value ("the run ended at 0.657"), on top of the iteration-1 (1.986)
and iteration-150 (1.231) matches already noted above. Same seed, same data,
same config, same loss curve start to finish, on different silicon: this is
the v3 header's own "d == 0.0 exactly" determinism check reproduced across
machines rather than across runs on one machine, and it is the strongest
evidence in this section that the training pipeline is not just *running* on
the new hardware but computing the identical thing it computed on the old one.

**Revised M3 Max planning numbers** (supersede the smoke-test estimate;
originals are the M3 Pro figures already in this document and in README's
pipeline table, both correctly labeled by machine and left as-is):
training run (v3 config, 1,322 iters): **~2.25 hours** on M3 Max vs ~6.2h on
M3 Pro.

**Not yet re-verified — do before trusting any new number from this
machine:**

- No wall-clock timing has been recaptured for a *full* rules-eval or
  positions-eval run (only the n=8 / n=4 smoke tests above) — training is now
  the one path with a completed, confirmed timing on this machine.

Nothing above changed any measurement, adapter, or gold record. This section
exists so that a future run's numbers are read against "same code, new
hardware, retrieval path re-verified, training and judged-eval paths not yet
re-verified" rather than assumed identical to the M3 Pro runs by default.

### 21.128 Closing the legal_actions gap on the 71 recorded boards, and a stale-report trap

Continuing 21.126: the recorded boards were authorable but not scorable
because `legal_actions` was empty on all 71. The path to fill it already
existed and was already validated at 22/24 on the original 32-position set
(21.105) — it had simply never been pointed at these boards.

**Setup, on the new machine.** No JDK was registered (`java_home -V` failed;
only the macOS stub `/usr/bin/java` existed), though `openjdk` was already
installed via Homebrew from an earlier `brew` run and just needed linking
(`sudo ln -sfn .../openjdk.jdk /Library/Java/JavaVirtualMachines/`). The
collector-patched `magefree/mage` checkout at
`~/Documents/personal_code/mage` survived the machine move intact —
`MagicLlmPositionDataCollector.java` was already there, so no re-clone or
re-patch was needed, only the JDK.

**A stale-compiled-class trap, caught before it produced a wrong number.**
The first `mvn test -Dtest='PosRecorded*Test'` run against the 71 boards
reported 45 of 71 classes with a failure or error, almost all
`[TEST] Couldn't find a card: Everywhere` — which reads exactly like a data
defect (a token misrecorded as a real card). It wasn't: `surefire-reports`
still held XML reports from an **earlier, unrelated session's** test batch
(different id-hash prefixes, e.g. `PosRecorded35b93b21...`, matching no file
in the current `Mage.Tests` source tree at all), and `mvn test` without
`clean` had run stale compiled `.class` files left over from that batch
alongside the new ones — 154 `PosRecorded*` reports for only 71 current
source files. `mvn clean test` reduced this to exactly 71 reports, and the
"Everywhere" errors vanished completely: **zero** real boards had it. Same
family as this file's own "harness bug whose trigger rate depends on the
condition under test" — the failure rate here depended on which OTHER
session had last touched this checkout, not on anything about these 71
boards. **A test-count discrepancy (154 reports, 71 source files) is the
tell** — check it before trusting a scary failure rate on a shared, unclean
checkout.

**The clean run's real numbers**: 136 tests (71 boards, most with both
`listPlayableActions` and `listAvailableAttackers`), 12 failures + 6 errors —
6 `getActivePlayerId() is null` (mulligan-phase boards, where no `PhaseStep`
exists yet — the same documented gap `xmage_export.py` already reports as
unmappable) and 12 "Player X must have 0 actions but found N" (a
`CardTestPlayerBase` leftover-scripted-action check, on boards recorded from
a real game rather than hand-authored — a board shape the original 32 never
exercised). Neither prevented `xmage_diff.py` from reading the printed
`PLAYABLE:`/`ATTACKER:` lines: those failures fire in an `@After`-style check
that runs after the test body's output is already captured.

**`xmage_diff.py --emit-legal-actions` result**: **65 of 71** filled (11 of
those with zero actions — a board where nothing is legal, not a failure to
answer), **6 refused** as `emitter_blind_spot` — every one a blocking board,
correctly left alone rather than handed a misleading empty list (21.85's
distinction, applied here). Re-exporting `position_tasks.json` via
`import_game.py --export-from` brought it to **54/71 scorable** (the 11
zero-action boards don't count as scorable, matching this repo's own
"absent/empty legal_actions convicts every correct play" rule) with **ids
unchanged**, so the collaborator's in-progress submissions from Section
21.127's Phase 1 deploy still key correctly. Redeployed; `/healthz` confirms
71 tasks live.

**Three known real defects fixed** (Section 21.105's original findings,
unfixed until now): `pos-trigger-ordering-0001` and `-0002` each gained
`CAST Doom Blade TARGET Grizzly Bears` in `legal_actions`, matching the
engine's finding that Doom Blade was castable and omitted. `sample-stage3-
payment-0004`'s `active_player` changed from `opp` to `you` in both
`positions.jsonl` and `position_samples_stage3_payment_batch2.jsonl` — the
attacking Goblin Guide is controlled by "you", so only "you" as active player
makes the attack legal under 508.1; changing `active_player` rather than the
`attacking` flag preserves the entire tested scenario (Shock the blocker
before damage) instead of removing it. `test_eval.py` (507 assertions),
`test_eval_positions.py` (61 assertions), and `positions.py
--check-references` all stay clean after both edits.

**Not done here, left for later**: the 65 filled boards carry an
**untrimmed** engine dump (`legal_actions_source: "xmage-engine (untrimmed)"`)
— `xmage_diff.py` says so explicitly — not the curated "list only the plays
that are genuinely available" `legal_actions` a hand-authored position has.
Curating them is a separate, manual pass; filling the field was the blocker
this section closes, not the last word on these boards' quality.

### 21.129 Blocks, closed: `Permanent.canBlock` answers the question `getAvailableBlockers` couldn't

`emitter_blind_spot` refused every blocking board on the stated grounds that
`getAvailableBlockers` is not phase-sensitive — it returns the same creature
at a main phase and at declare blockers, so it can say a creature *could ever*
block, never that a *specific* block is legal *now*. That was true of the
query the code was using. It was never true of the engine: `Permanent`
exposes `canBlock(UUID attackerId, Game game)`, which runs the engine's own
evasion and restriction system (flying, reach, menace, protection, ...) for
one attacker/blocker pair — the same kind of per-pair check `getPlayable`
already does for targets, just never asked of blocking.

**What changed.** `xmage_export.py` gained a third `@Test`,
`listAvailableBlocks()`, at `DECLARE_BLOCKERS` (one step later than
`combat_reachable`'s `DECLARE_ATTACKERS`, so a new `blocks_reachable(phase)`
mirrors it exactly). It collects the board's actually-attacking permanents
(`Permanent.isAttacking()`, no new state needed — `_combat()` already scripts
them into every test method) and, for each of `playerA`'s available blockers,
checks `canBlock` against each attacker, printing `BLOCK: <blocker> ->
<attacker>` for every legal pair. `xmage_diff.py` reads it as a fourth engine
answer (`PLAYABLE`/`ATTACKER`/`BLOCKER`/`BLOCK`), compares `BLOCK` sets on the
same footing as `ATTACKER` (a real disagreement, not a coverage gap), emits
curated `BLOCK <blocker> -> <attacker>` lines from `--emit-legal-actions`, and
no longer refuses a blocking board in `emitter_blind_spot` — only a mulligan
and `ORDER TRIGGERS` remain genuine blind spots now, for the reasons those
always had (no `PhaseStep` exists pre-game; not an `ActivatedAbility` so
`getPlayable` never reports it).

The phase-insensitive `BLOCKER:` line from `listPlayableActions` is
unchanged and still printed-only — it answers a different, weaker question
("could this creature ever block something") and the two are kept
deliberately distinct rather than one replacing the other.

**Verified against the 6 boards this closes.** Every one of the 71 recorded
boards' candidates file re-ran through the full cycle
(`run_xmage_validation.sh`, Section 21.128): the 6 boards previously REFUSED
as blocking boards are now all attempted, none refused. Results are exactly
what the boards' content predicts, not a uniform "it worked":

- Two boards (`-0033`, `-0040`) have **you** as the attacker, so there is
  nothing for you to block — correctly zero `BLOCK` lines, CAST options only.
- One board (`-0063`) has a single attacker and a single legal blocker —
  correctly one pairing: `BLOCK Spectral Sailor -> Surrak, Elusive Hunter`.
- One board (`-0055`) has two attackers and two flying blockers with no
  evasion restricting either side — correctly all four pairings, the exact
  shape a full pairwise check should produce when nothing excludes a
  combination.
- Two boards (`-0047`, `-0071`) come back empty. `-0071`'s attackers include
  an **animated land** (a Forest turned into a 21/21 by +1/+1 counters) —
  `permanent_state_problems`'s already-documented gap (`addCard` cannot place
  a land currently animated as a creature, Sections 21.117/21.123) — so the
  rebuilt board likely has fewer real attackers than the recording, and an
  empty answer here is consistent with that known limitation rather than a
  defect in the block query itself.

Net: 65 -> **71 of 71 filled**, 0 refused; scorable boards (non-empty
`legal_actions`) 54 -> **58 of 71**. `test_engine_legal_actions` in
`test_eval.py` was rewritten in the same commit — its old assertions
(`"blockers are never emitted"`, `"a blocking board is refused"`) tested
exactly the limitation this section removes, and left unfixed would have
been asserting a bug was a feature. All seven suites still pass (508
assertions, +1 from the added BLOCK-line check).

### 21.130 The next two items on the XMage backlog were not what they looked like

Blocks (21.129) closed by finding a real engine hook (`Permanent.canBlock`)
the existing query pattern had simply never been asked for. The next two
items on the backlog — `ORDER TRIGGERS` and mulligan — do not close the same
way, and both turned out to be different problems than "find the hook."

**`ORDER TRIGGERS`: a real negative result, verified by running, not
theorized.** The candidate approach was the same shape as blocks: stop the
game at the step where the position says triggers are pending
(`PhaseStep.UPKEEP`, matching `pos-trigger-ordering-0001`'s board exactly —
two upkeep triggers, "waiting to be put onto the stack" per its own
`known_information`) and read `GameState.getTriggered(controllerId)`. A
throwaway probe test (`ProbeTriggerOrderTest`, built, run, and deleted — not
committed, since it answers a question rather than testing anything) built
that exact board and printed the pending count: **0**. By the time
`setStopAt(1, PhaseStep.UPKEEP)` + `execute()` returns, the engine has
already auto-ordered and resolved both triggers — the trigger-ordering
choice happens as part of *entering* the step, before any player-facing
stop point the test framework's `setStopAt` model can catch. Unlike
`DECLARE_ATTACKERS`/`DECLARE_BLOCKERS`, which are real turn-based-action
windows the engine pauses at, "order simultaneous triggers" is not a step at
all — it is resolved inline, and the only way to observe it would be
intercepting `TestPlayer`'s own choice-handling internals (whatever backs
`chooseMulligan`-style delegation to `computerPlayer`) rather than querying
public state after the fact. That is a materially larger, more invasive
piece of engineering than anything blocks needed, and this section stops
here rather than guessing at it — a stated gap, verified rather than
assumed, per this file's own rule about a control measuring only what it
measures.

**Mulligan: not a gap to close, on inspection.** The backlog framed this as
"needs a different test shape than `setStopAt`," which is true but beside
the point once the actual rule is checked: **a mulligan decision has no
illegal option.** Any hand may always be kept, and any hand may always be
mulliganed (rule 103.5) — there is no legality condition on either choice
the way there is for a cast, an attack, or now a block. Both hand-authored
mulligan positions already reflect this exactly:
`pos-mulligan-0001`/`-0002` list `["KEEP", "MULLIGAN"]` (order differs,
content doesn't) with nothing to validate beyond "are these the only two
words that could appear here," which they trivially always are.
`emitter_blind_spot`'s mulligan refusal is therefore not a temporary
limitation waiting on engine support — it is the permanently correct answer,
because there is no engine question to ask. "Cover mulligan" was the wrong
frame for this item from the start; nothing here needs building.

**Where this leaves the backlog.** Blocks was the one item with real,
present-day payoff and a genuine missing hook, and it is closed. Of the
other three original Phase 3 items: `ORDER TRIGGERS` is a verified
engineering dead end for the current test-framework approach (not
impossible, just requiring a different and larger mechanism than anything
built so far); mulligan needed no fix because there was never a defect;
sorcery-speed step-scoping remains what it always was — a documented,
unexercised gap with no position currently triggering it, correctly left
for when one does rather than fixed speculatively ahead of the evidence.

### 21.131 The sorcery-speed gap, closed on request ahead of any position needing it — and what closing it immediately found

Closed at the user's explicit direction rather than because a position
demanded it: re-checked first, including against the 71 recorded boards
that did not exist when 21.100 first stated the gap, and the finding still
held — no position, hand-authored or recorded, has a land drop before
precombat main. The fix went in anyway because it is mechanical rather than
speculative: `getPlayable` already answers this question correctly at one
step (21.100–21.101 proved that), and asking it again one step later is not
new untested behavior, just the same proven query at a second vantage point.

**What changed.** `xmage_export.py`'s `getPlayable`+`TARGET` enumeration —
previously inlined once, in `listPlayableActions` — is now
`_emit_playable_query`, called from both there and a new fourth `@Test`,
`listPlayableAtMain`, emitted only when the stated step is strictly before
`PRECOMBAT_MAIN` (`precombat_main_reachable`, the mirror image of
`combat_reachable`/`blocks_reachable`: those ask "is the decision still
ahead", this asks "did the stated-step query run too early to see something
that opens up later"). `xmage_diff.py` needed no reading-side changes at
all: `engine_answers()` already unions every `PLAYABLE`/`TARGET` line in a
report regardless of which `@Test` method printed it, so a second vantage
point is automatically merged into the same set the first one populates.

**Verified against the full 103-class set** (32 original + 71 recorded,
freshly re-exported and compiled with `mvn clean test`, never `test` alone —
21.128's rule still applies): no compile errors, no new failure category
(still exactly the mulligan-NPE and leftover-action classes from 21.128/
21.129), 276 tests across 103 classes.

**And it found something real on the first run — a false positive, not a
new gold-set defect.** `xmage_diff.py` against the 32-position set now
reports `pos-trigger-ordering-0001` and `-0002` disagreeing: "the engine
offers 'Forest'; the position does not list it." Before treating that as an
actionable finding: both boards carry a **Phyrexian Arena** — the very
permanent whose upkeep trigger these two positions exist to test the
*ordering* of — and Phyrexian Arena's ability is "draw a card" as part of
what it does at upkeep, before precombat main. Both boards' stated hand is
`["Doom Blade"]`, no Forest. So the draw is real (it is the position's own
mechanic), but what gets drawn is not: `xmage_export.py` fills library
slots with a placeholder basic because *only the count is real, never the
content* (stated as a limitation since Section 21.100), so the card
Phyrexian Arena draws is unconditionally "Forest" regardless of what a real
recorded game's deck would have produced there.

This is the SAME limitation the docstring already names, reached through a
path that could not exist before this section: no previous query ever
looked far enough into the turn to see a card drawn mid-turn, because
nothing asked past the stated step except `DECLARE_ATTACKERS`/
`DECLARE_BLOCKERS`, neither of which triggers a draw. `listPlayableAtMain`
is the first query that can cross a draw event, and it did, on its first
real run, against exactly the two positions built around a card-drawing
trigger. **The gold set is NOT updated with "PLAY Forest"** — doing so
would encode the filler's fiction as if it were the recorded game's truth.
Anyone extending this query further (e.g., to a step after an explicit
draw step) should expect the same interaction and read a `Play <basic
land>` disagreement as a candidate instance of this, not an automatic
defect, whenever a draw could have occurred between the stated step and
the step being queried.

### 21.132 The card/ruling benchmark's first three questions, and a real retrieval-recall gap on the first measurement

PLAN_NEXT.md item 4 asks for a held-out card/ruling benchmark and says to
"measure retrieval recall before generation quality." Built the smaller,
safer half of that first: a schema, a validator, and three seed candidates —
not a full benchmark, since authoring MTG-ruling content that is subtly
wrong is worse than not having it, and a trustworthy benchmark's whole
value is being trustworthy.

**Every candidate is grounded in real, checkable data, not memory.** Each
question cites an `official_rulings_used` chunk id that must exist in the
pinned `ruling_chunks.jsonl`, and each `cr_rule_citations` entry must be a
real id in the pinned CR (`load_rule_ids`) — both asserted by
`scripts/validate_card_ruling_benchmark.py`, not just claimed. One card
(`Mana Vortex`) and its state-trigger ruling; one (`Greven, Predator
Captain`) and its life-loss/uninterrupted-resolution ruling; one pairing
(`Colossal Dreadmaw` + `Basilisk Collar`, deathtouch trampling) with **no**
ruling on either card, verified absent, to cover the "no ruling exists, CR
reasoning only" category the plan asks for. All three carry
`needs_review: true` in `data/gold/card_ruling_candidates.jsonl` — nothing
here is gold, the same rule every other candidates file in this repo
follows.

**The validator caught a real authoring mistake before it shipped.** The
first draft of the Greven question cited `702.111` — that is Menace, copied
from the card's own `keyword_rule_ids` in the ruling corpus without
checking it was relevant to the resolution-sequencing question actually
being asked. Retrieval-recall measurement flagged it as a miss, which
prompted a second look rather than accepting the miss as a pipeline
limitation — the citation itself was wrong. Corrected to `117.2e` ("no
player has priority while a spell or ability is resolving"), the actual
rule the question depends on.

**Card names must use `[[Bracket]]` syntax to resolve at all — this is not
optional.** `card_lookup.CardIndex.find_in_text` matches ONLY
`BRACKET_RE = \[\[(.*?)\]\]`; it does no plain-text scanning. The existing
gold set's Reddit-sourced questions resolve cards not because
`retrieve_hybrid.py` scans prose, but because their original authors
happened to write `[[Card Name]]` (a Reddit MTG-subreddit autolink
convention) before the question ever reached this repo. The first draft of
all three candidates here was written as plain prose and would have scored
0% card recall for a reason that has nothing to do with retrieval quality —
caught by running the validator before trusting its output, not by reading
the code.

**Measured, on the corrected three: card name recall 3/3, ruling recall
2/2, CR rule recall 1/3 (in the top-3 rules chunks; still 0/3 at top-10,
so not a ranking problem but a real miss).** `rag.retrieve` finds adjacent
chunks from the right rule GROUP (603-2, 603-6 for the Mana Vortex
question, whose target is inside rule group 603) without finding the chunk
that actually contains 603.8 or 117.2e, even widened to k=10. Both misses
share a shape: the question is phrased in card-and-scenario language
("no lands on the battlefield," "gain life back in response"), not in the
CR's own vocabulary ("state trigger," "priority... while... resolving") —
a card-grounded question does not lexically resemble the general rule it
depends on, and semantic embedding similarity does not appear to bridge
that gap reliably at n=2. This is a measurement about the CURRENT retrieval
pipeline on exactly the kind of question this benchmark exists to ask, on
the first three questions run through it — read as an n=2 signal on general-
rule recall, not yet a stable rate, and the reason "measure retrieval
recall before generation quality" is the plan's own stated order of
operations.

**Not done**: the benchmark itself. Three questions is a validated seed, not
coverage — no `ruling_relevant_insufficient` example exists yet (a case
where the ruling is necessary but not sufficient alone), and the plan's own
scope (cards with rulings, cards without, and the sufficiency split) needs
substantially more authored and reviewed content before "two judges" is
appropriate, per the plan's own ordering.

### 21.133 v4, the verified-rules-only control, was already trained and evaluated — and never narrated

PLAN_NEXT.md item 3's first bullet ("train a verified-rules-only control
first") turns out to already be done: `models/mtg-rules-adapter-v4`
(`configs/phase1_lora_v4_verified.yaml`, trained 2026-08-20 on
`data/datasets/verified`, 1,001 train examples, exactly 1,001 iterations —
2.00 epochs as planned, 10 checkpoints saved) exists, is stamped, and has
been evaluated against baseline on the full 99-question gold set under
**two independent judges**. None of that result was ever written up as its
own section — the files sat in `eval/runs`/`eval/reports` since the day
they were produced, findable only by reading the report directly.

**The comparison has one confounded arm and one clean one.** `finetuned_rag`
is not readable: `data/datasets/verified` has zero examples under
`RAG_SYSTEM_PROMPT` (`rules-verified-v1.json`'s own manifest says so), so
that arm is evaluated on a prompt shape these weights never trained on —
Section 8.7's mechanism with no prompt edit at all, exactly as CLAUDE.md's
`unseen_arms` section already warns. `finetuned` (no retrieved context, the
shape v4 actually trained on) is the valid comparison, against `base`
(same shape, no adapter).

**Under both judges, `finetuned` still loses to `base` on overall
correctness:**

| Judge | base | finetuned | winner |
| --- | --- | --- | --- |
| Qwen2.5-32B (`gold_n99_v4_32b.md`) | 1.43 | 1.15 | base |
| Mistral-Small-24B (`gold_n99_v4_mistral.md`) | 1.66 | 1.57 | base |

The direction agrees across judges — a genuine negative result, not an
artifact of which judge happened to grade it. This is the "one verified-
data adapter trained and compared against baseline" the plan's Definition
of Done asks for, closed with a still-negative answer, which is exactly
the kind of result this project's own rules say to write down plainly
rather than soften.

**But the config's own advance prediction was right, and the tension it
named is the real finding.** `phase1_lora_v4_verified.yaml`'s header
predicted, before training: refusal rate would fall, grounding would rise,
answer length would fall further — and if correctness dropped while
grounding rose, "that tension is the finding." All three happened:

- **Grounding, no-RAG arms**: `finetuned` 92/99 vs `base` 45/99 (32B judge)
  — more than double, the largest clean grounding effect measured anywhere
  in this project's history at the arm level.
- **Fabricated citations, no-RAG arms**: `finetuned` 7/99 vs `base` 35/99
  (32B judge) — a 5x reduction.
- **Citation matches reference**: `finetuned` 11/99, tied-best of all four
  arms including `base_rag` (10/99) — under a 7B model with no retrieval at
  all, citing the same rule the reference does about as often as base+RAG
  manages with rules text in the prompt.
- **Citation score, Mistral judge**: `finetuned` 2.29, second only to
  `base_rag`'s 2.39, well ahead of plain `base`'s 1.85.
- **Answer length**: `finetuned` averages 252 characters (Mistral judge)
  against `base`'s 1,163 — a 4.6x reduction, and correlation(length,
  correctness) = **+0.052**, near zero. The shorter answers are not simply
  being penalized for brevity; they are being scored on content, and losing
  on content despite citing more accurately.

**Read together: v4 fixed exactly the defect it was built to fix (the
refusal/fabrication problem measured in Section 21.6 from training on the
base model's own synthetic output) and still does not beat plain base on
the judge's correctness metric.** The most likely reading, not yet tested:
a short, well-grounded, accurately-cited answer is missing rubric key
points that a longer, more diffuse (even if more fabricated) answer
happens to brush against — which would mean the fix that was needed
(grounding, refusal rate) and the fix that would move the correctness
number (coverage of the specific claims a rubric checks for) are two
different problems, and this run only had the data to address the first
one.

**Also on file, not yet reconciled with the above**: `gold_n99_v4judge.md` /
`gold_n99_v4judge2.md` grade the SAME v4 answers under Qwen2.5-7B and
Llama-3.1-8B respectively (not the 32B/Mistral pair above), and
`gold_n99_v4_AGREEMENT.md` reports their agreement on it: Pearson r +0.46,
38% exact, on the 6 questions carrying a hand-authored (Cody Clark) rubric.
Consistent with the calibration table's standing rule that a 7B/8B pair is
weaker-but-real signal, not a substitute for the 32B/Mistral pair used for
the headline correctness numbers above.

**What item 3 still needs, per the plan's remaining bullets**: a gameplay-
examples mixture experiment and an official-rulings-only experiment, both
"separate," neither started. Given this section's finding, the more
promising next experiment is not simply "more verified rules data" but
data that suits FULL key-point coverage rather than grounding alone —
worth deciding before the next training run, not after it.

### 21.134 v5-gameplay: the mixture experiment run, and a new methodological finding it forced

PLAN_NEXT.md item 3's second bullet, closed: `models/mtg-rules-adapter-v5-gameplay`
adds the 8 board positions carrying a reviewed `reference_actions` line
(Section 21.85 — refused into the gold set unless the parser confirms the
line is legal) on top of v4's exact 1,001 verified rules pairs. 1,009 train
lines, same LoRA geometry/batch/lr/seed as v4, 2.00 epochs (recomputed for
the new count, not inherited — `configs/phase1_lora_v5_gameplay.yaml`).
Trained in ~20 minutes wall-clock on the M3 Max (this dataset's targets are
short; nowhere near the ~2h+ runs Sections 21.127-21.128 measured on larger
sets), no thermal or battery issues.

**Stamping this adapter found a real gap in `stamp_adapter.py`, not a
training defect.** It refused outright: the dataset contains
`GAMEPLAY_SYSTEM_PROMPT`, which `stamp_adapter.KNOWN` never listed, by the
same deliberate design that keeps `gameplay_fingerprint()` separate from
`prompt_fingerprint()` (a gameplay-grammar edit must not invalidate a
rules-only adapter's stamp). That separation is correct and stays — the
actual gap was narrower: `KNOWN` controls what a dataset scan RECOGNIZES,
not what the hash COVERS, and it had never been asked to recognize a mixed
dataset because the gameplay track never fed one into `stamp_adapter.py`
before. Fixed by adding `GAMEPLAY_SYSTEM_PROMPT` to `KNOWN` only —
`prompt_fingerprint()`'s hash is untouched, confirmed by the stamp landing
on the identical fingerprint v4 has (`30badae98696`). Stamp now correctly
reads `SYSTEM_PROMPTx1112, GAMEPLAY_SYSTEM_PROMPTx8`.

**Rules track: no distinguishable movement from v4**, evaluated identically
(same 99 questions, same 32B judge):

| Arm | v4 | v5-gameplay | Δ |
| --- | --- | --- | --- |
| base_rag | 1.66 | 1.68 | +0.02 |
| base | 1.43 | 1.43 | 0 |
| finetuned_rag | 1.11 | 1.16 | +0.05 |
| finetuned | 1.15 | 1.20 | +0.05 |

`base`/`base_rag` do not depend on the adapter at all and still moved
±0.02 between the two runs — see the methodological finding below before
reading anything into the ±0.05 on the finetuned arms. This project's own
sample-size guidance is ~100-150 questions to resolve a 0.4-point effect at
n=99; ±0.05 is far inside the noise floor. **Read as: correctness moved
negligibly, exactly as `phase1_lora_v5_gameplay.yaml`'s header predicted in
advance** ("8 examples out of 1,009 is under 1% of the mixture, far below
the volume that moved anything in Sections 19-21").

**Gameplay track: a new methodological finding arrived before the result
did.** The obvious comparison — this run's `ft_cards_open` (v5-gameplay)
against the stored `positions_n32_scenarios.md` baseline's same arm — is
not valid: that baseline adapter was `mtg-rules-adapter-v2-best`, not v4,
so it is not isolating the gameplay mixture at all. Worse, the arms that
use NO adapter differ between the two runs too: `base_open` correctness
1.74 (old run) vs 1.98 (this run), blunder 85% vs 94%, on the same 34
positions, same base model, same prompts. **Generation is not
run-to-run reproducible the way judging is.** `mlx_lm.generate` carries no
fixed seed the way training does (`seed: 42` is a training-only setting in
these configs), so this project's own "the judge is deterministic"
finding (re-judging identical stored text reproduces exactly) does not
extend to "re-generating text from the same prompt reproduces exactly."
Every position/rules eval comparison made ACROSS separate run files up to
now has carried this noise silently — it was never visible before because
nothing had re-run the identical non-adapter arms twice and diffed them.
Cross-run comparisons of arms that do not depend on the changed variable
should be treated as a sanity check on noise floor, not assumed to be free.

**Given that, the only valid comparison here is WITHIN this one run** —
`base_cards_open` (no adapter) against `ft_cards_open` (v5-gameplay),
both generated in the same process, same prompts, same judge, same pass:

| Arm | Blunder | Correctness | Parsed ok | Plays/answer | All legal | Legal plays | Did nothing | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_cards_open | 97% | 1.30 | 97% | 14.2 | 9% | 82% (396/483) | 3% | 21% |
| ft_cards_open | 100% | 1.20 | 88% | **3.0** | **32%** | 44% (45/103) | **32%** | 9% |

The adapter still loses on correctness and blunder rate. But the shape of
the loss is not "plays worse" — it is "plays far less." Plays per answer
fall from 14.2 to 3.0, "did nothing" (PASS as the only action) rises from
3% to 32%, and `all_legal` more than triples (9% -> 32%) for the
mechanical reason 21.58 and 21.126 both already document: an answer whose
only action is PASS always matches `legal_actions` and commits no listed
strategy error, so declining to play is nearly unbeatable on both fields
this table shows improving. Parse quality and per-attempted-play legality
both fall (97%->88% parsed, 82%->44% of attempted plays legal) — of the
plays it still tries to make, more are malformed or illegal than base
manages. **8 gameplay examples were enough to measurably shift behavior
toward passivity, not enough to teach the strategy that would make passing
less necessary or attempted plays more often legal.**

**Verdict on the mixture experiment**: negative on both tracks, for
different reasons. Rules-track: too small a fraction of the mixture (0.8%)
to move anything, as predicted. Gameplay-track: enough to change behavior,
not enough to make the changed behavior better — consistent with this
project's own repeated finding that `only_pass` inflates easy-to-read
metrics without being good play (Section 21.58's "an answer that declines
to play beats one that plays"). Neither result argues for scaling this
exact recipe up; a mixture built from many more reviewed reference lines
than 8 would be a different, better-supported experiment, and none
currently exist beyond these 8 (Section 21.85's gate is deliberately
strict about what counts).

**Still open, per PLAN_NEXT.md item 3**: the official-rulings-only
experiment. Not started — building it safely needs the same
grounded-not-memorized authoring discipline Section 21.132 used for the
card/ruling benchmark, at a volume well beyond that section's 3 seed
questions, and is a separate piece of work from what this section closes.

### 21.135 Mining XMage's own test suite, instead of authoring one probe at a time

Section 21.132's authoring approach — pick a ruling, read it, hand-build one
XMage probe to confirm it, write the question — works, but does not scale
solo: a live attempt to build a fourth probe this session (Mana Vortex's
state-trigger ruling, already in the benchmark) caught a real setup mistake
immediately (a land given to the responding player made the "no lands"
condition false from the start) but then stalled on a test-framework
target-naming ambiguity between two simultaneously-triggered abilities from
the same source. Real, useful friction — but per-question, and the user
asked directly for something that leverages more of what already exists
rather than more solo hand-construction, probe or otherwise.

**The leverage was sitting in the checkout already.** `Mage.Tests/src/test/
java/org/mage/test/cards/` holds **1,823 JUnit tests**, written and
continuously run by XMage's own contributors to verify their card
implementations — 940 single-card checks plus 883 across thematic
directories (`triggers/` 138, `continuous/` 78, `replacement/` 53, `cost/`
45, `rules/` 19, and others). Each one is already a verified `(setup,
action, assertion)` triple for a real rules interaction, checked by a real
engine and a real CI, not by one person's reading of a ruling. Mining this
is authoring-by-selection instead of authoring-by-construction: read what
already exists and passes, rather than build something new and hope it
does.

**`scripts/mine_xmage_tests.py`**, built and verified this session:

- Walks the thematic test directories (`single/` opt-in via
  `--include-single`, since it is mostly one-card sanity checks rather than
  interactions) and extracts, per `@Test` method, the card names involved
  and every `addCard`/`castSpell`/`activateAbility`/`attack`/`block`/
  `assert*` call — regex-based on purpose, not a real Java parser, since the
  goal is triage material for a person to then read the actual source file
  from, not automated question generation. A missed or noisy extraction
  costs nothing; the file path is always printed.
- Ranks candidates by (distinct card count, assertion count) as a cheap
  proxy for "genuine interaction" over "single card sanity check."
- `--verify` compiles and runs the top candidates via the same `mvn clean
  test` pattern the rest of this repo's XMage tooling already uses, so a
  candidate is not just "assertions that once passed" but "assertions that
  pass in THIS checkout right now."

**Verified working end to end, not just read.** `--category triggers
--min-cards 2 --limit 8` surfaced real, rich multi-card interactions on the
first run — `RanarTheEverWatchfulTest` (13 cards, 4 assertions, Curse of the
Swine's board-wide transform interacting with a token-generation trigger),
`ShowstopperTest` (12 cards, 8 assertions, two copies of a dies-trigger card
plus a copy effect), `WorldgorgerDragonTest` (11 cards, 6 assertions,
Animate Dead's exile-on-leaving-battlefield loop) — none hand-picked or
guessed at, all surfaced automatically. `--verify` against two `rules/`
candidates (`NamePredicateTest`, `AlternativeCostRuleTest`) confirmed both
currently pass (`errors="0" failures="0"` in the fresh surefire reports),
closing the loop: mine, rank, confirm-in-the-present-tense, all without
constructing a single new scenario by hand.

**What this changes and what it does not.** It turns "invent and verify a
scenario" into "select and confirm one that already exists," which is a
different and much more leverageable skill than either hand-authoring from
ruling prose (Section 21.132) or hand-building a probe from scratch
(this section's stalled Mana Vortex attempt) — closer to how the recorded-
game gameplay pipeline works (curate from something real, don't invent).
It does **not** remove the need for a person to read the winning
candidate's source and write the natural-language question — that
translation step is still manual, just starting from verified ground truth
instead of from a ruling's prose or a blank page, and it does not
automatically produce `key_points`/`common_errors` or CR rule citations,
which still need authoring against the confirmed scenario the same way
Section 21.132's seed questions were built.

**Not yet done**: actually converting mined candidates into
`data/gold/card_ruling_candidates.jsonl` entries or a rulings-training
mixture. This section is the mining tool and its proof of function; turning
a batch of `--verify`-confirmed candidates into real questions is the next
piece of work, and at 1,823 tests deep it can be done in batches sized to
whatever authoring capacity is actually available, rather than one probe
at a time.

**First real batch run, same session.** The initial ranking (raw card count
+ assertion count) surfaced mostly noise on a full-repo sweep — implementation
regression tests (`ConcurrentModificationExceptionTest`, a Java exception
test with nothing to do with rules) and client/server sync checks
(`cost/modaldoublefaced/`) outranked genuine interactions because inflated,
non-card string literals in assertion messages ("client must ignore side 2",
"before last cast 1") were being counted as card names. Fixed with
`_looks_like_a_card` (rejects prose: lowercase-starting, containing
"should"/"must"/"client"/"server"/etc.) and a directory+filename denylist
(`NOISY_PATH_PARTS`, `NOISY_NAME_RE`), plus a ranking change: more than 8
real cards becomes a complexity PENALTY rather than a bonus, since a
scenario that sprawling is hard to state as one clean question even when
extraction is accurate. Re-run, the same sweep surfaced clean, comprehensible,
well-known interactions with no further filtering needed.

**Two converted to real benchmark entries, both verified passing in this
checkout right now** (`ElendaTheDuskRoseTest.testKillAndReanimate`,
`ZoneChangeReplacementTest.testPermanentNewInstanceAndKumano`), added to
`data/gold/card_ruling_candidates.jsonl` with a new `xmage_test_verified`
field recording which test and when:

- **Elenda, the Dusk Rose + Angelic Renewal** — a death trigger using
  last-known information (603.10a) for its token count while the returning
  effect creates a genuinely new object (400.7) with none of that
  information; the two rules apply independently to the same event. Elenda
  has an official ruling that directly confirms the token-count half
  ("use Elenda's power as it last existed on the battlefield"), upgrading
  this from CR-only to `ruling_sufficient`.
- **Kumano's Pupils, bounced and recast** — its exile replacement tracks
  "dealt damage by this creature this turn" per OBJECT, not per card, so
  returning the creature to hand and recasting it (400.7, a new object)
  resets that tracking entirely, even for the same physical card later that
  turn. Kumano's Pupils DOES have two official rulings on file, but both
  address a different question (simultaneous death timing) — noted
  explicitly in the record's `source` field so nobody reads this as
  contradicting or restating them.

**Grounding: clean on all 5 candidates now** (`validate_card_ruling_
benchmark.py`), including the two new ones — every card, ruling, and CR
citation resolves. **CR rule recall: 1/5**, reinforcing Section 21.132's
n=2 finding at a slightly larger n=5: `rag.retrieve` still cannot surface
the specific rule chunk a card-grounded question depends on, even though
the general rule area is often nearby in the corpus. Not yet a stable rate,
but the same real gap on more evidence.

**A `--min-rulings` filter closed the last missing category the same
session.** `ruling_relevant_insufficient` — a ruling that bears on the
question without stating its answer — needed a candidate where the named
cards actually carry official rulings, which raw card/assertion counts
cannot select for. Added `cards_with_rulings()` (reads `n_rulings` per
card from the pinned corpus directly, a different field than
`validate_card_ruling_benchmark.load_ruling_chunk_ids_by_card` reads and so
not reused from it) and `--min-rulings N`, which annotates every candidate
with which of its cards have rulings on file and how many.

That surfaced `WinLoseEffectsTest.testAngelsGrace2` (Angel's Grace +
Laboratory Maniac + Ad Nauseam, verified passing, 4/4 tests): Player A
casts Angel's Grace on themselves, empties their own library with Ad
Nauseam down to -5 life, then would draw from an empty library and wins
via Laboratory Maniac instead. **Laboratory Maniac's own official ruling
addresses this exact card pairing — from the opposite direction**: "If for
some reason you can't win the game (because your OPPONENT has cast Angel's
Grace this turn, for example), you won't lose... The draw was still
replaced." That ruling is directly relevant and does not state what
happens when the ANGEL'S GRACE CASTER is the one who wins — reaching the
answer needs Angel's Grace's own wording read closely (it stops the caster
from *losing* and stops opponents from *winning*, and says nothing about
the caster winning), plus the state-based-action rule (704.5a) it
suppresses. A textbook case of "ruling relevant, not sufficient alone" —
found by filtering for it, not stumbled into.

**The benchmark is now 6 questions, all three `ruling_sufficiency`
categories represented for the first time**, grounding still fully clean.
CR rule recall: 1/6 — the same gap, same magnitude, one more data point.

**Two more mined the same session, sweeping `continuous/` and `prevention/`
with `--min-rulings`.** Both verified passing before being written up.

- **Oathsworn Knight + Polukranos, Unchained** (`PreventDamageRemoveCounters
  Test.test_OathswornKnight_CounterRemoval`, 7/7 tests) — a contrast pair
  whose abilities read almost identically ("if damage would be dealt ...
  prevent that damage and remove ... a +1/+1 counter" vs "... remove THAT
  MANY +1/+1 counters") but differ in exactly the word that matters:
  Oathsworn Knight always removes exactly one counter per damage event no
  matter the amount; Polukranos removes one counter per point of damage
  prevented. Both cards carry an official ruling that states its own half
  directly (Oathsworn Knight's ruling literally says "not one counter per 1
  damage prevented"), so this is `ruling_sufficient` on both sides at once —
  the value is in the near-identical wording being a natural place for a
  model to conflate the two rules, not in any retrieval or CR-citation gap.
- **Conspiracy + Opalescence + Enchanted Evening** (`LayerTests.
  testMultipleLayeredDependency`, 7 tests run / 2 `@Ignore`d / 0 failures) —
  a three-effect dependency chain (613.8) in layer 4: Conspiracy only
  affects creatures, so it depends on Opalescence having already made
  enchantments into creatures; Opalescence only affects enchantments, so it
  depends on Enchanted Evening having already made every permanent an
  enchantment. The forced order is Enchanted Evening -> Opalescence ->
  Conspiracy, regardless of cast order, giving a land base P/T 0/0 (mana
  value 0) and Enchanted Evening itself 5/5 (its own mana value 5) before
  Glorious Anthem's independent layer-7c +1/+1 applies on top. Opalescence's
  own official ruling walks through this exact dependency methodology in
  careful detail — but for a *different* three-card combination (Opalescence
  + Humility + Worship). It teaches how to reason about the dependency, and
  does not itself state the answer for this combination, making it the
  second `ruling_relevant_insufficient` example — found the same way as
  Angel's Grace, by filtering for cards with real rulings on file rather
  than stumbling into one.

Grounding stays clean at n=8 (`validate_card_ruling_benchmark.py`: card
recall 8/8, ruling recall 6/6). **CR rule recall: 1/8** — held at the same
1/N rate across four sample sizes now (n=2, 5, 6, 8), which is as much a
finding as any individual question: `rag.retrieve` reliably fails to surface
the specific CR rule chunk a card-grounded question depends on, independent
of sample size, source (hand-authored vs mined), or `ruling_sufficiency`
category. Worth deciding whether to fix (larger k, a retrieval strategy
that treats "general rule for a card-grounded question" as a distinct
query shape from rules-only retrieval) or document as a settled limitation
— `PLAN_NEXT.md` item 4 carries the decision forward.

**A full mining pass, same session, per the user's explicit instruction to
mine as much as reasonably possible before returning to training.** Swept
`protection`, `requirement`, `copy`, `dynamicvalue`, and `triggers` (which at
138 files had barely been touched), plus one more `targets` candidate that
had been set aside earlier for thematic overlap with Kumano's Pupils and is
now distinct enough to include on its own. Seven more converted, all
verified passing in this checkout right now:

- **Emrakul, the Aeons Torn + Murderous Cut** (`ProtectionTest.
  testProtectionFromColoredSpells`, 6/6) — protection from colored spells
  means Murderous Cut (black) can't even be CAST targeting Emrakul; the
  illegality is caught at the casting step (702.16b), not left to fizzle at
  resolution. `no_ruling_needed_cr_only`.
- **Prized Unicorn + Oppressive Rays** (`BlockRequirementTest.
  testPrizedUnicornAndOppressiveRays`, 9/9) — "all creatures able to block do
  so" versus "can't block unless its controller pays {3}": a player is never
  forced to pay a cost to satisfy a block requirement, so an unpaid Silvercoat
  Lion simply isn't "able" to block and the requirement doesn't apply to it.
  Oppressive Rays' own ruling states this exactly ("Players can't be forced
  to pay a cost to attack or block"); 509.1c states the general rule.
  `ruling_sufficient`.
- **Dualcaster Mage + Flame Slash + Walking Ballista** (`CopySpellTest.
  testOnlyCopyFizzles`, 25/25) — Dualcaster copies Flame Slash and retargets
  the COPY at Walking Ballista instead of the original's target (Atraxa).
  Walking Ballista kills itself (its own ability, activated before the copy
  resolves) to ping something else, so the copy's target is gone by the time
  it tries to resolve and it's removed from the stack doing nothing — while
  the untouched original still resolves against Atraxa normally. A copy's
  fate depends only on its OWN target (707.10, 707.10c, 608.2b), not the
  original's. `no_ruling_needed_cr_only`.
- **Armadillo Cloak** (`SavedDamageValueTest.ArmadilloCloakTest`, 1/1) —
  its "whenever enchanted creature deals damage, gain that much life" is a
  normal triggered ability, not the lifelink keyword, confirmed directly by
  its own ruling; the distinction is invisible with one life-gain source on
  the creature and only shows up if the creature also has real lifelink
  (both would trigger, gaining life twice for one damage instance).
  `ruling_sufficient`.
- **Divine Visitation + Smothering Tithe** (`DivineVisitationTest.
  testDivineVisitationDoesNotReplaceNoncreatureTokens`, 3/3) — Divine
  Visitation's replacement only catches CREATURE token creation; Smothering
  Tithe's Treasure tokens are artifacts, so the replacement event never
  triggers and three ordinary Treasures are created, not three Angels.
  `no_ruling_needed_cr_only`. Caught a real authoring slip while writing this
  one up: `cards` named "Ancestral Recall" but the question text hadn't
  wrapped it in `[[brackets]]`, which `validate_card_ruling_benchmark.py`'s
  retrieval-recall check correctly flagged as an unresolved card — every name
  listed in `cards` has to be bracketed in the question, not only the ones
  central to the interaction. Caught by running the validator, per this
  project's own rule, not by re-reading the JSON by eye.
- **Diamond Knight + Glimpse of Freedom** (`SpellCastTriggerTest.
  testDiamondKnightTrigger`, 5/5) — casting a spell via its Escape ability
  (an alternative cost, from the graveyard) is still casting that spell in
  every other respect, so it still triggers a "whenever you cast a spell of
  the chosen color" ability exactly like casting normally from hand would
  (702.138a). `no_ruling_needed_cr_only`.
- **Dream Leash + Take into Custody + Ornamental Courage**
  (`TargetRestrictionsTest.testDreamLeashUntappingAsResponseToCast`, 2/2) —
  Dream Leash's "can't choose an untapped permanent as this spell's target
  AS YOU CAST it" is a one-time restriction checked only at casting; once
  satisfied, untapping the permanent in response before Dream Leash resolves
  does not undo it, and Dream Leash still resolves and takes control. A
  genuinely different gotcha from the other two protection/targeting
  questions here: a resolution-time legality recheck (608.2b) does not
  re-examine a casting-only restriction stated in the spell's own text.
  Confirmed directly by Dream Leash's own ruling. `ruling_sufficient`.

**Grounding stays clean at n=15** (card recall 15/15, ruling recall 9/9).
**CR rule recall: 4/15** — no longer flat at 1/N. Both hits are new this
batch (Prized Unicorn/Oppressive Rays cites 509.1c; Diamond Knight cites
702.138a), and both cite a rule with dense, focused coverage of one specific
mechanic, while every miss — including the two that DID cite 800-series or
614-series general rules — cites either a broad numbered-list rule
(608.2b) or an obscure keyword sub-rule buried among a hundred similar
ones. That's a different-shaped hypothesis than "card-grounded retrieval is
broken": it may be less about the question being card-grounded and more
about how densely a rule area is represented and how distinctive its
neighborhood is in the embedding space. Not yet enough data to treat this
as settled — worth watching as the benchmark keeps growing.

**A second pass the same session, per explicit instruction to mine as much
as reasonably possible while away from a desk.** Swept `restriction`,
`asthough`, and `copy` for the first time, and checked back on `conditional`.
Four more converted, all verified passing in this checkout right now:

- **Meddling Mage + Alive // Well** (`MeddlingMageTest.
  testMeddlingMageFuseCardStopAndCastWell`, 6/6) — naming ONE half of a fuse
  card ("Well") blocks casting that half alone AND blocks casting the card
  FUSED, even though the other half ("Alive") was never named; only casting
  the unnamed half by itself remains legal. Alive // Well's own ruling states
  this directly ("the player may name either half ... but not both. A split
  card has the chosen name if one of its two names matches"). A genuinely
  different naming-effect gotcha from anything in the benchmark so far.
  `ruling_sufficient`.
- **Reflector Mage + Bronze Sable** (`ReflectorMageTest.
  testReflectorMageAllowsOwnerToCastCreatureReturnedOnSameTurn`, 2/2) —
  Reflector Mage's "can't cast" restriction names a specific player (the
  bounced creature's OWNER), not every player and not its own controller, so
  Reflector Mage's controller can cast their own same-named card the same
  turn. Answered entirely by careful reading of the printed ability, no
  ruling or CR citation needed. `no_ruling_needed_cr_only`.
- **Narset, Enlightened Master + Cathartic Reunion**
  (`PlayFromNonHandZoneTest.testNarsetEnlightenedMasterAdditionalCost`,
  13/13) — "cast without paying its mana cost" waives only the mana cost;
  Cathartic Reunion's discard-two is an additional cost and must still be
  paid. Both cards' own rulings state their half of this directly. The
  classic "additional costs survive a waived mana cost" FAQ point, now with
  a verified engine confirmation. `ruling_sufficient`.
- **Phantasmal Image + Transcendent Master** (`PhantasmalImageTest.
  testCopyCreatureWithLevelUpAbility`, 21/21) — copying a maxed-out leveler
  (12 level counters, 9/9 lifelink+indestructible) produces a 3/3 with
  neither ability, because counters are never copiable values; the copy
  starts at zero level counters of its own even though it copies the
  leveler's full printed level-up ability text. Transcendent Master's own
  ruling states this outcome directly, down to the mechanism ("the number of
  level counters on it, are not [copied]. The abilities, power, and
  toughness of the copy will be determined based on how many level counters
  are on the copy"). `ruling_sufficient`.

Two strong-looking candidates were surfaced and set aside rather than forced
into the benchmark: `TragicSlipTest.testPlayedWithFlashbackAgain` (Snapcaster
Mage granting flashback to Tragic Slip, recast later in the turn after a
creature has since died) is correct engine behavior but not a genuine
teaching point — Morbid re-evaluating at the second casting's own resolution
is simply how an intervening-if condition is supposed to work, not a trap;
and `CleverImpersonatorTest.testKindredDiscovery` (copying an "as this
enters, choose X" permanent) has both players independently choosing the
SAME creature type in its own test, which doesn't actually demonstrate that
the choice must be re-made rather than copied. Neither was confirmed with
enough confidence to state as a fact rather than a guess, so neither was
written up — a candidate surfacing is not the same as a candidate being
usable.

**Grounding stays clean at n=19** (card recall 19/19, ruling recall 12/12).
CR rule recall: 4 of 15 records that cite one — the 4 newest entries carry
no `cr_rule_citations` at all, since each is answered by the card's own
printed text rather than any single citable rule, and leaving the field
empty was judged more honest than reaching for a citation that doesn't
actually carry the answer.

**Also a process note worth keeping**: this batch was authored in four tool
calls instead of roughly twelve, by batching source reads, `mvn` verification,
and JSONL writes each into one call covering several candidates at once,
rather than looping one-candidate-at-a-time. A `for` loop over `find`/`cat`
intermittently tripped a shell cwd-reset mid-iteration on this machine
(breaking `head`/`cat` resolution inside the loop body); chaining explicit
per-file commands in one call avoided it entirely. Worth defaulting to this
shape for any future mining session, not just one done remotely.

**A third pass the same session swept `conditional` for real** (it had only
been scanned, not converted, before). Three more, all verified passing:

- **Rootwater Matriarch** (`RootwaterMatriarchTest.
  testGainControlEnchantedTargetAndRWLeavesPlay`, 4/4) — its "gain control
  for as long as that creature is enchanted" duration is self-contained on
  the TARGET, so control is kept even after Rootwater Matriarch itself
  leaves the battlefield; only the target creature losing every Aura ends
  it. Confirmed directly by its own ruling. `ruling_sufficient`.
- **The Wretched + Wall of Pine Needles** (`TheWretchedTest.
  testGainControl_One_RegenWhichRemovesBlockerFromCombat`, 4/4) —
  regeneration removes a creature from combat, so a regenerated blocker is
  no longer "blocking" by the time The Wretched's end-of-combat trigger
  resolves and checks who's still blocking it; it escapes the control
  change entirely. Confirmed directly by its own ruling. `ruling_sufficient`.
- **Mul Daya Channelers + Dryad Arbor** (`MulDayaChannelersTest.
  testBoostLossThroughPhases`, 3/3) — its two conditional continuous
  abilities (tied to the top library card being a creature, or being a
  land) are independent, not exclusive modes, so a card that's both (Dryad
  Arbor) triggers both bonuses at once. Confirmed directly by its own
  ruling. `ruling_sufficient`.

**Grounding stays clean at n=22** (card recall 22/22 — see `PLAN_NEXT.md`
item 4 for the running count). This closes out `conditional` as an actually-
mined category rather than a scanned-but-empty one.

**A fourth pass opened `single/`** (940 files, opt-in via `--include-single`,
previously entirely unmined) alongside a re-check of `replacement`. The
`--category` flag rejects `"single"` as a value — it only scans `single/`
when combined with one of the 15 `INTERACTION_DIRS` or omitted entirely — so
`--include-single` with no `--category` was the working invocation; a first
attempt passing both errored out silently under `grep`, costing one wasted
round before the fix. Three more converted, all verified passing:

- **See the Truth + Twincast** (`SeeTheTruthTest.copyOnStack`, 4/4) — a
  Twincast copy of See the Truth still only puts ONE card into hand, not all
  three, even though See the Truth's own text upgrades to "all three" when
  cast from anywhere but hand. A copy is never cast at all, so that
  condition has nothing to check and defaults to the unmodified behavior.
  Confirmed directly by its own ruling ("the copy wasn't cast at all, so you
  only get one of the cards"). The sharpest "copies aren't cast" example in
  the benchmark so far. `ruling_sufficient`.
- **Contagion Engine** (`ContagionEngineTest.testCountersDoubledByProliferate`,
  2/2) — "Proliferate twice" resolves as two full proliferate actions inside
  ONE ability resolution, with no player able to respond in between and no
  second activation needed; the two actions can target the same permanents
  (doubling their counters) or different ones. Confirmed directly by its own
  ruling. `ruling_sufficient`.
- **Hallowed Moonlight + Spiritual Visit + Reanimate + Silvercoat Lion**
  (`HallowedMoonlightTest.testGrindstoneProgenius`, 2/2) — a three-way
  contrast in one board: a token (never cast) and a reanimated creature
  (put onto the battlefield directly, never cast) are both exiled by
  Hallowed Moonlight, while a creature spell that's actually cast normally
  is completely unaffected. Confirmed directly by its own ruling
  ("won't affect any creature that was cast ... Creature tokens are never
  cast"). `ruling_sufficient`.

**Grounding stays clean at n=25** (card recall 25/25). Between this session's
four passes, the benchmark grew from 3 questions (Section 21.132's hand-
authored seed) to 25, 22 of them mined rather than hand-invented, spanning
protection, requirement, copy, dynamicvalue, triggers, restriction, asthough,
conditional, and now `single/` and `replacement`. `dynamicvalue` (beyond
Armadillo Cloak and one un-converted Sewer Nemesis candidate) and the bulk of
`single/`'s 940 files remain unmined — there is no shortage of remaining
material, only of session time.

**A fifth pass, per the user's explicit call to finish mining before
committing to a training run** — reasoning that at n=25 the benchmark is
still a pilot-sized sample (CLAUDE.md's own guidance wants ~100-150 for a
powered effect), so more verified ground truth is worth more than an early
training result right now. Went deeper into `single/`'s per-set
subdirectories and closed out `damage`. Four more, all verified passing:

- **Burrenton Forge-Tender + Flametongue Kavu + Cloudshift**
  (`BurrentonForgeTenderTest.
  testPreventDamageFromFlametongueKavuNotAfterCloudshift`, 4/4) — a
  prevention effect that names a specific chosen SOURCE stops applying the
  instant that source is blinked, because the returned permanent is a new
  object with no connection to the one that was chosen, however identical it
  looks. A third angle on the new-object rule (400.7) in this benchmark,
  after Elenda's last-known-information and Kumano's Pupils' per-object
  damage tracking — this one hits a CHOSEN source in a prevention effect
  specifically. `no_ruling_needed_cr_only`: the closest official ruling
  addresses a different case (a spell becoming a permanent), not blinking an
  existing chosen one.
- **Alania, Divergent Storm** (`AlaniaDivergentStormTest.test_TwoSorceries`,
  7/7) — "the first instant/sorcery/Otter spell you've cast this turn"
  triggers once per category per turn, not once per qualifying spell; a
  second sorcery cast after the first has already triggered Alania doesn't
  trigger her again. Answered by the card's own printed condition alone.
  `no_ruling_needed_cr_only`.
- **Unleash the Inferno + Stone Golem** (`UnleashTheInfernoTest.
  testExcessDamage`, 1/1) — excess damage (damage dealt beyond lethal) sets
  a mana-value ceiling for a linked destroy effect; a 3-mana-value artifact
  is destroyed and a 4-mana-value one on the same board is untouched from
  the same 7-damage, 4-toughness interaction (3 excess). Confirmed directly
  by its own ruling's definition of excess damage. `ruling_sufficient`.
- **Thorn Elemental + Grizzly Bears** (`AssignDamageTest.
  test_ThornElemental_Manual_DamageToPlayer`, 3/3) — "assign combat damage
  as though it weren't blocked" is all-or-nothing: choosing it sends every
  point of damage to the defending player and the blocker takes none, unlike
  trample's split assignment. Confirmed directly by its own ruling.
  `ruling_sufficient`.

**Grounding stays clean at n=29** (card recall 29/29, ruling recall 20/20).
CR rule recall: 4 of 16 records that cite one, the same rate as before —
this pass's `no_ruling_needed` entries mostly needed no citable rule at all,
same as several earlier ones. The session total: 3 -> 29 questions across
five mining passes, 26 of them mined from currently-passing XMage tests
rather than hand-invented, spanning protection, requirement, copy,
dynamicvalue, triggers, restriction, asthough, conditional, damage, and a
first slice of `single/` and `replacement`. The great majority of `single/`'s
940 files remains untouched — there is still no shortage of material, only
of session time.

**A sixth pass, continuing per the user's "keep going" after deciding to
finish mining before training.** Went further down the ranked candidate list
in `single/`, past the top ~20 already sampled. One real dead end and three
conversions:

**Mining tool blind spot found**: `AzoriusAethermageTest.testBouncedLand`
surfaced as a strong-looking candidate (8 cards, matching setup/assertion
calls extracted cleanly) but its entire method body is a Java block comment
— the test is disabled and asserts nothing. `mine_xmage_tests.py`'s regex
extraction has no notion of comments, so it read the commented-out calls as
real ones. A `--verify` run would have caught this immediately (the "test"
executes zero assertions), but the candidate looked identical to a real one
in the ranked listing. Not fixed in the tool this session — noting it here
as a known failure mode for the next person mining with it: **always check
`n_asserts` isn't secretly coming from dead code, and treat `--verify`
(or a manual read of the actual method) as mandatory, never optional, for
anything pulled from `single/`,** where one-off disabled tests are more
likely to hide than in the actively-maintained interaction directories.

Three converted, all verified passing:

- **Gluttonous Hellkite** (`GluttonousHellkiteTest.
  test_CastWithoutSac_CounterTrigger`, 5/5) — countering ONLY the "each
  player sacrifices X creatures" triggered ability (not the creature spell
  itself) lets Gluttonous Hellkite still enter the battlefield normally,
  but with ZERO +1/+1 counters, since no creatures were actually sacrificed.
  The linked trigger and the spell are separate stack objects. Gluttonous
  Hellkite's own ruling addresses the OPPOSITE direction (spell countered,
  trigger still resolves) but not this one — the benchmark's third
  `ruling_relevant_insufficient` example.
- **Baleful Mastery + Twincast** (`BalefulMasteryTest.
  test_BalefulMastery_CopyMustKeepAlternativeCost`, 5/5) — a copy of a
  spell inherits whether an alternative cost was paid, exactly like it
  inherits a chosen X value or mode; Twincast's copy of Baleful Mastery
  still makes an opponent draw a card even though nothing was paid to
  create the copy. Confirmed directly by its own ruling. `ruling_sufficient`.
- **Deification** (`DeificationTests.testDamagePrevention_Combat`, 5/5) —
  its loyalty-preserving replacement only protects planeswalkers of the ONE
  chosen type; a different planeswalker the same player controls (Tibalt,
  when "Chandra" was chosen) gets no benefit and dies normally to the same
  size of attack that the protected one survives. `ruling_sufficient`.

**Grounding stays clean at n=32** (card recall 32/32). The benchmark now
spans three `ruling_relevant_insufficient` examples (Angel's Grace,
Conspiracy/Opalescence/Enchanted Evening, Gluttonous Hellkite) — the
category that started this session with zero.
