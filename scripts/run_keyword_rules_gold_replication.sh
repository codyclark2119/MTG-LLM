#!/usr/bin/env bash
# Replicate keyword-rule injection on the current shipped retrieval baseline:
# Qwen2.5-32B + cards+rulings + flat k=3, full 99-question gold set.
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
STEM="gold99_32b_k3_cards_rulings_keyword"

common=(
  scripts/eval.py
  --gold-only
  --gold "$GOLD"
  --base-only
  --base-model "$MODEL"
  --with-cards
  --with-rulings
  --no-plain-rag
  --k-rules 3
  --max-tokens 800
  --judge-model "$JUDGE"
  --judge-prompt v3
  --seed 42
)

write_manifest() {
  local out="$1"
  local enabled="$2"
  "$PY" - "$out" "$enabled" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]).with_suffix('.manifest.json')
enabled = sys.argv[2] == 'true'
data = {
    'experiment': 'keyword-rules-gold99-v1',
    'gold': 'data/gold/gold_questions.jsonl',
    'base_model': 'mlx-community/Qwen2.5-32B-Instruct-4bit',
    'judge_model': 'mlx-community/Mistral-Small-24B-Instruct-2501-4bit',
    'judge_prompt': 'v3',
    'k_rules': 3,
    'with_cards': True,
    'with_rulings': True,
    'base_only': True,
    'no_plain_rag': True,
    'max_tokens': 800,
    'seed': 42,
    'keyword_rules': enabled,
}
out.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
PY
}

run_one() {
  local label="$1"
  local enabled="$2"
  local out="eval/runs/${STEM}_${label}.jsonl"
  local report="eval/reports/${STEM}_${label}.md"
  local answers="${out%.jsonl}.answers.jsonl"

  if [ -f "$out" ]; then
    echo "=== ${label}: completed run exists, keeping it ==="
    write_manifest "$out" "$enabled"
    return
  fi

  write_manifest "$out" "$enabled"
  local keyword_flag="--no-keyword-rules"
  if [ "$enabled" = true ]; then
    keyword_flag="--keyword-rules"
  fi

  echo
  echo "=== keyword rules ${label} ==="
  if [ -f "$answers" ]; then
    echo "reusing generation checkpoint $answers"
    "$PY" "${common[@]}" "$keyword_flag" \
      --answers-from "$answers" --out "$out" --report-out "$report"
  else
    "$PY" "${common[@]}" "$keyword_flag" \
      --out "$out" --report-out "$report"
  fi
}

run_one off false
run_one on true

"$PY" scripts/analyze_keyword_rules_pair.py \
  "eval/runs/${STEM}_off.jsonl" \
  "eval/runs/${STEM}_on.jsonl" \
  --out "eval/reports/${STEM}_paired.md"

echo
echo "Replication complete. Review the paired report before changing keyword_rules."
