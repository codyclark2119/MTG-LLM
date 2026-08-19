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
