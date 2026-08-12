# Development Plan — Training an LLM to Understand Magic: The Gathering

> Design rationale, experiment log, and results history. For setup, code structure, and how to run things, see [README.md](README.md).

## Phase 1: Rules Foundation

This walkthrough covers everything from base model selection through ingesting the comprehensive rules, and stops **before** card data input. The goal of this phase is a model that understands *how Magic works* — the rules engine, turn structure, priority, the stack, zones, and gameplay mechanics — as a foundation for later card-aware, deck-based text gameplay.

> **Target hardware for this build: an M3 Pro MacBook Pro with 36GB unified memory, macOS Tahoe 26.5.2.** Every recommendation below is tailored to Apple Silicon, using Apple's **MLX** framework (not the NVIDIA/CUDA `bitsandbytes` stack). With 36GB you're comfortably out of the tightest constraints: you can fine-tune **4-bit quantized 7–8B models with real headroom** (larger batches, more LoRA layers, longer sequences), and even attempt a **13–14B** fine-tune if you want. macOS Tahoe (26.x) is fully supported by current MLX. See Section 1.5 for the full hardware picture.

---

## Status (as of 2026-08-11)

**Phase 1 (Rules Foundation):**

- [x] Repository and MLX toolchain initialized (Section 2)
- [x] Comprehensive Rules corpus acquired, pinned to the 2026-08-07 release (Section 3)
- [x] Rules parsed into structured section/rule/subrule records; chunked for retrieval and training with reference-aware bundling and glossary injection (Sections 4–5)
- [x] RAG baseline stood up over the chunks — `mlx-community/all-MiniLM-L6-v2-4bit` embeddings, brute-force cosine retrieval (Section 6)
- [x] SFT dataset generated via RAG-grounded, category-balanced question/answer synthesis over `mlx-community/Qwen2.5-7B-Instruct-4bit` (Section 7) — ~700 examples across all 8 categories from 7.1, split into `data/datasets/train.jsonl` / `valid.jsonl` and a held-out `eval/rules_questions.jsonl`
- [x] Fine-tuning run (Section 8) — QLoRA over `mlx-community/Qwen2.5-7B-Instruct-4bit`, 600 iterations, rank-16 LoRA on 16 layers. **Validation loss bottomed out at iteration 200 (1.159) and climbed steadily afterward (1.631 by iteration 600)** — classic overfitting on a 569-example train set. The iteration-200 checkpoint is promoted to `models/mtg-rules-adapter-best/`. **Spot-checks after training found the adapter fabricating rule citations more confidently than the un-tuned base model, including ignoring correct RAG-retrieved context it was explicitly told to use** — see the caveat in Section 8.6. Fine-tuning as currently trained is not yet a net improvement; needs the real Section 9 eval to quantify, and likely a larger/more diverse SFT set.
- [x] Evaluation against the RAG-only and base-model baselines (Section 9) — 110 questions (70 synthetic + 40 reddit) × 4 arms, judge-scored. Run 1: `finetuned_rag` 2.37 vs `base_rag` 3.19 (Section 9.5).
- [x] Second fine-tune + re-eval after fixing a train/inference format mismatch and growing the dataset 569 → 2,646 examples (Sections 8.7, 9.6). `finetuned_rag` improved **+0.43 (2.37 → 2.80)** against a measured judge-noise floor of ±0.09, and bare-`finetuned` citation fabrication fell 58% (26 → 11). Still below `base_rag` (3.25) — but that comparison is confounded by a judge length bias (r = +0.21 between answer length and score), so it reads as "not yet demonstrated better," not "worse."
- [x] Recalibrated the judge (Section 9.7) — separate correctness/citation scores, explicit length-neutrality, anonymized candidates; both runs re-scored under it. This reversed a false result: `base_rag` (3.75) in fact beats `base` (3.04), where the old judge showed the opposite. Retrieval is worth **+0.71 to +0.85**, the v2 format fix is worth **+0.69** (not +0.43), and measured judge noise is ±0.16.
- [x] Split eval scores by question source (Section 9.8) — **65% of synthetic questions retrieve their own source chunk**, so that subset flatters RAG (`base_rag` 4.23 synthetic vs 2.92 reddit). On *real* questions `finetuned_rag` and `base_rag` are tied within noise (2.88 vs 2.92); the apparent RAG win is largely an artifact of how the synthetic set was built.
- [x] Card-augmented retrieval measured (Section 13.5) — on card-referencing questions `base_rag_cards` gains **+0.69** over `base_rag` with citation quality 3.25 → 4.00, but n=16 gives a 95% CI of [−0.17, +1.54], so it is promising rather than established. A no-card control subset (+0.00) caught a prompt-formatting bug that had produced a convincing false result.
- [x] Settled the card question at n=100 (Section 13.5) — the +0.69 **did not replicate**: −0.01, CI [−0.33, +0.31], 25 better / 28 worse / 47 tied. It was small-sample noise, correctly flagged at the time as unestablished.
- [x] Tested judge validity with an independent Llama-3.1-8B judge (Section 9.9) — **the two judges rank the arms in opposite order** on identical answers, and agree only r = +0.43 (37% exact, mean disagreement 1.18 points). The earlier ±0.16 "noise floor" measured within-judge reproducibility only; judge-choice uncertainty is far larger, so previously reported effects of +0.4 to +0.7 are provisional.
- [x] Ingested 77,931 official WotC rulings covering 19,726 cards (Section 13.6) — the most authoritative corpus available, and the fix for eval references that currently cite a rule only 21% of the time.
- [ ] **Next:** (a) build a human-labeled gold subset to establish which judge is closer to correct — no model-quality claim is safe until then; (b) use the rulings corpus for eval references and SFT targets; (c) rebuild the synthetic eval so it stops testing retrieval of its own source chunk.

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

```
mtg-llm/
├── data/
│   ├── raw/           # Original rules documents
│   ├── processed/     # Cleaned, chunked rules + RAG index
│   ├── datasets/      # Final training JSONL (train/valid)
│   └── cards/         # Phase 2: Scryfall card pulls (Section 13)
│       ├── raw/       # Format-scoped snapshots, e.g. standard_cards.jsonl
│       └── processed/
├── scripts/
│   ├── ingest.py       # Rules → structured records
│   ├── chunk.py        # Chunking + cross-refs
│   ├── rag.py           # Embed chunks, retrieve at query time
│   ├── build_sft.py    # Generate training examples via RAG-grounded synthesis
│   ├── eval.py          # Rules comprehension eval
│   └── fetch_cards.py  # Phase 2: pull a format's legal card pool from Scryfall
├── configs/
│   └── phase1_lora.yaml   # mlx_lm.lora settings
├── models/            # Adapters / checkpoints
└── eval/
    └── rules_questions.jsonl
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

**Real-world addendum:** alongside the synthetic held-out set (`eval/rules_questions.jsonl`, Section 7.4), `eval/reddit_questions.jsonl` adds 200 actual questions Magic players asked on r/MTGRules with community-vetted answers (`scripts/build_reddit_eval.py`, source: `Javier-Jimenez99/reddit-mtgrules-qa`, CC-BY-SA-4.0). Reddit's upvote score alone isn't a quality signal — it rewards jokes as readily as correct rulings (a top-scored reply to a real rules question was "Isn't Marty's cause to get back to the future?") — so candidates are filtered through an LLM judge for topical relevance before inclusion, and explicit rule citations are checked against the currently-pinned CR. Real questions surface phrasing, ambiguity, and multi-card interactions a synthetic set generated from rules text alone tends not to reproduce.

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

**A finding that needs a caveat, not a headline:** `base` outscoring `base_rag` (3.62 vs. 3.19) looks like "RAG hurts," but spot-checking the disagreements says otherwise. On "What does 'Reveal' mean?", `base_rag` quoted the actual rule 701.20a text verbatim and still scored a point below `base`'s vaguer, more general answer — docked for "missing detail" despite being the more precisely grounded response. On "Archenemy Commander," `base_rag`'s retrieval (k=3) surfaced the general Archenemy rules but missed the specific compound concept, producing a real but incomplete answer. Two distinct issues are tangled together here: the LLM judge appears to have a verbosity/completeness bias that isn't well calibrated against precise-but-concise grounded answers, and k=3 retrieval sometimes misses the ideal chunk for compound or unusual questions. Neither of these is "RAG doesn't work" — but neither should be papered over either. Section 9.3's "spot-checked by you" caveat on model-graded scoring is doing real work here; a human pass on a sample of judge disagreements (starting with `eval/EVAL_REPORT.md`'s lowest-scoring cases) is the natural next step before trusting the aggregate numbers further.

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
