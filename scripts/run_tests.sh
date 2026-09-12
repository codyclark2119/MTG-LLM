#!/usr/bin/env bash
# The whole test suite, one command. This is the canonical list for this repo.
#
#   scripts/run_tests.sh              # everything
#   PYTHON=python3 scripts/run_tests.sh
#
# TWO VALIDATION TIERS, AND THIS SCRIPT IS ONLY THE FIRST
# -------------------------------------------------------
# Tier 1 (this script): every test file in the repo. All of them are
# deterministic, need no GPU and no model -- CLAUDE.md has always said so, and
# it is now verified rather than assumed: the suite passes on Linux with only
# numpy, fastapi and httpx installed and no mlx present at all. That is what
# .github/workflows/ci.yml runs.
#
# Tier 2 (local, Apple Silicon, NOT automated): the model work itself. No test
# in this repo loads a model, so CI proves nothing about retrieval quality,
# generation, judge scoring or serving latency. Those are exercised by running
# the real scripts against the real corpora:
#
#   source mlx_env/bin/activate
#   python scripts/rag.py query "when are state-based actions checked?"
#   python scripts/chat_server.py            # then ask it something
#   python scripts/eval.py --help            # and a real run, per CLAUDE.md
#
# Do not read a green tier 1 as "the model still works". It means the code
# around the model still holds together.
#
# NO DO-NOTHING SUCCESS PATH. A listed test that is not on disk is a FAILURE,
# not a skip: this workspace has shipped four scripts that printed "all green"
# while running nothing, and a gate that cannot fail is not a gate.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# mlx_env locally; whatever is on PATH in CI, where mlx cannot be installed.
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "$ROOT/mlx_env/bin/python" ]; then
  PY="$ROOT/mlx_env/bin/python"
else
  PY="python3"
fi

TESTS=(
  scripts/test_imports.py
  scripts/test_eval.py
  scripts/test_docs.py
  scripts/test_webui.py
  scripts/test_deploy.py
  scripts/test_chat_server.py
  scripts/test_server_validation.py
  scripts/test_rag.py
  scripts/test_manifests.py
  scripts/test_format_snapshot.py
  scripts/test_metagame.py
  scripts/test_mtggoldfish.py
  scripts/test_decklist.py
  scripts/gameplay/test_actions.py
  scripts/gameplay/test_eval_positions.py
)

FAILED=0
RAN=0

# Two assertions intentionally refer to local/generated artifacts that are not
# committed: the v4 adapter's prompt stamp and the adjudication tasks export
# copied by deploy/Dockerfile. A clean CI checkout therefore needs only the
# metadata/existence those assertions are about, not model weights or gold data.
#
# Reconstruct the v4 prompt counts from the committed dataset manifest (the
# source of truth the real stamp was verified against), and create an empty
# generated tasks file solely so the .dockerignore/COPY allowlist can be tested.
# Both are removed on exit and are created only when absent, so a local checkout
# with the real artifacts is never overwritten.
CREATED_V4_STAMP=0
CREATED_TASKS=0
cleanup_fixtures() {
  if [ "$CREATED_V4_STAMP" -eq 1 ]; then
    rm -f models/mtg-rules-adapter-v4/prompt_fingerprint.json
    rmdir models/mtg-rules-adapter-v4 2>/dev/null || true
  fi
  if [ "$CREATED_TASKS" -eq 1 ]; then
    rm -f data/gold/worksheets/tasks.json
  fi
}
trap cleanup_fixtures EXIT

if [ ! -f models/mtg-rules-adapter-v4/prompt_fingerprint.json ]; then
  mkdir -p models/mtg-rules-adapter-v4
  "$PY" - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path("data/manifests/datasets/rules-verified-v1.json").read_text())
out = {
    "dataset_prompt_counts": manifest["dataset_prompt_counts"],
    "verified_against": manifest.get("path"),
    "dataset_id": manifest.get("dataset_id"),
}
Path("models/mtg-rules-adapter-v4/prompt_fingerprint.json").write_text(
    json.dumps(out, indent=2) + "\n", encoding="utf-8")
PY
  CREATED_V4_STAMP=1
fi

if [ ! -f data/gold/worksheets/tasks.json ]; then
  printf '{"kind":"adjudication","tasks":[]}\n' > data/gold/worksheets/tasks.json
  CREATED_TASKS=1
fi

echo "python: $("$PY" -V 2>&1)  ($PY)"
echo

# Syntax first. A file that does not compile fails every test below it for a
# reason the individual failures do not name.
if ! "$PY" -m compileall -q scripts > /tmp/compileall.$$ 2>&1; then
  echo "FAIL  compile check"
  sed 's/^/      /' /tmp/compileall.$$
  FAILED=1
fi
rm -f /tmp/compileall.$$

for t in "${TESTS[@]}"; do
  if [ ! -f "$t" ]; then
    echo "FAIL  $t -- listed here but not in the repo."
    echo "      Renamed or deleted? Update this list. Not skipping: a"
    echo "      vanished test is a failure, not an absence."
    FAILED=1
    continue
  fi
  if out=$("$PY" "$t" 2>&1); then
    printf 'OK    %-40s %s\n' "$t" "$(echo "$out" | grep -v StarletteDeprecation | tail -1)"
    RAN=$((RAN + 1))
  else
    echo "FAIL  $t"
    # Keep the complete diagnostic. The old tail -15 hid the actual failing
    # assertion in long suites such as test_eval.py, leaving CI to report only
    # the final progress labels and "1 check(s) FAILED".
    echo "$out" | sed 's/^/      /'
    FAILED=1
  fi
done

echo
if [ "$RAN" -eq 0 ]; then
  echo "NOTHING RAN -- that is a failure, not a pass."
  exit 1
fi
if [ "$FAILED" -eq 0 ]; then
  echo "all green ($RAN test files)"
  exit 0
fi
echo "FAILURES above ($RAN of ${#TESTS[@]} files ran clean)"
exit 1
