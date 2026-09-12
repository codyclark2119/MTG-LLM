#!/usr/bin/env bash
# Replicate the 32B k=0 vs k=3 card+rulings result on the full 99-question gold set.
#
# This intentionally matches the 53-question deconfounding experiment's design:
#   * Qwen2.5-32B base model, no LoRA (`--base-only`)
#   * card text + official rulings on both sides
#   * no plain rules-only RAG arm (avoids a duplicate candidate at k=0)
#   * independent Mistral 24B V3 rubric judge
#   * 800 generation tokens
#   * identical arm count and seed in both runs
#
# The only intended experimental variable is --k-rules: 0 versus 3.
# Outputs are archived under descriptive names and then analyzed with the
# pre-committed paired analyzer.
#
# Restart behavior: eval.py checkpoints generated answers before judging. If a
# checkpoint exists but the scored run does not, this script resumes from it in
# a fresh process instead of regenerating the 32B answers. A completed run is
# never overwritten automatically.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -x "$ROOT/mlx_env/bin/python" ]; then
  PY="$ROOT/mlx_env/bin/python"
else
  PY="${PYTHON:-python3}"
fi

MODEL="mlx-community/Qwen2.5-32B-Instruct-4bit"
JUDGE="mlx-community/Mistral-Small-24B-Instruct-2501-4bit"
GOLD="data/gold/gold_questions.jsonl"
STEM="gold99_32b_cards_rulings"

common=(
  scripts/eval.py
  --gold-only
  --gold "$GOLD"
  --base-only
  --base-model "$MODEL"
  --with-cards
  --with-rulings
  --no-plain-rag
  --max-tokens 800
  --judge-model "$JUDGE"
  --judge-prompt v3
  --seed 42
)

run_one() {
  local k="$1"
  local out="eval/runs/${STEM}_k${k}.jsonl"
  local checkpoint="eval/runs/${STEM}_k${k}.answers.jsonl"
  local report="eval/reports/${STEM}_k${k}.md"

  if [ -f "$out" ]; then
    echo "Refusing to overwrite completed run: $out" >&2
    echo "Move/archive it explicitly if you intend to repeat the experiment." >&2
    return 2
  fi

  echo
  echo "=== k=${k} ==="
  if [ -f "$checkpoint" ]; then
    echo "Resuming judging from $checkpoint (generation already complete)."
    "$PY" "${common[@]}" \
      --k-rules "$k" \
      --answers-from "$checkpoint" \
      --out "$out" \
      --report-out "$report"
  else
    "$PY" "${common[@]}" \
      --k-rules "$k" \
      --out "$out" \
      --report-out "$report"
  fi
}

run_one 0
run_one 3

"$PY" scripts/analyze_k_rules_pair.py \
  "eval/runs/${STEM}_k0.jsonl" \
  "eval/runs/${STEM}_k3.jsonl" \
  --label-a "k=0" \
  --label-b "k=3" \
  --out "eval/reports/${STEM}_paired.md"

echo
echo "Replication complete. Review all three reports before changing the serving default."
