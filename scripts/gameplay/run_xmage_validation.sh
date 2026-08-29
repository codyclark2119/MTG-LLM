#!/bin/bash
# Orchestrates the validated XMage legality path end to end, so filling
# legal_actions on a batch of recorded boards is one command instead of the
# five manual steps Sections 21.100-21.128 document doing by hand.
#
# Usage:
#   scripts/gameplay/run_xmage_validation.sh CANDIDATES.jsonl [--export-tasks OUT.json] [--mage-home DIR]
#
# Requires a JDK, Maven, and a magefree/mage checkout with
# MagicLlmPositionDataCollector.java already applied (Section 21.103) — this
# script does not create or patch one. It defaults to
# ~/Documents/personal_code/mage; override with --mage-home or $MAGE_HOME.
#
# Matches the manual sequence exactly (Section 21.128):
#   1. xmage_export.py               -> one JUnit test class per board
#   2. copy into the mage checkout's magicllm test package
#   3. mvn clean test, scoped to just the classes exported in step 1 —
#      `clean` is mandatory here on purpose: a shared checkout accumulates
#      test classes from every batch ever run through it, and a bare
#      `mvn test` picks up stale compiled .class files from earlier,
#      unrelated batches. That produced a run that looked like 45 of 71
#      boards had a real data defect when the true number was zero
#      (Section 21.128). `clean` removes that failure mode structurally.
#      Test failures here (mulligan boards with no PhaseStep yet, or
#      leftover-scripted-action checks on boards recorded from a real game)
#      are expected and do not stop this script — xmage_diff.py reads the
#      printed PLAYABLE:/ATTACKER: lines regardless of the test's final
#      pass/fail state.
#   4. xmage_diff.py --emit-legal-actions -> writes legal_actions back,
#      never overwriting a board that already has a curated list
#   5. (optional) re-export tasks.json for the deployed rubric form, with
#      ids unchanged so in-progress submissions still key

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAGE_HOME="${MAGE_HOME:-$HOME/Documents/personal_code/mage}"

CANDIDATES=""
EXPORT_TASKS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --export-tasks) EXPORT_TASKS="$2"; shift 2 ;;
    --mage-home) MAGE_HOME="$2"; shift 2 ;;
    -h|--help)
      echo "usage: $0 CANDIDATES.jsonl [--export-tasks OUT.json] [--mage-home DIR]"
      exit 0
      ;;
    *)
      if [[ -z "$CANDIDATES" ]]; then CANDIDATES="$1"; shift
      else echo "unrecognized argument: $1" >&2; exit 1; fi
      ;;
  esac
done

if [[ -z "$CANDIDATES" ]]; then
  echo "usage: $0 CANDIDATES.jsonl [--export-tasks OUT.json] [--mage-home DIR]" >&2
  exit 1
fi
if [[ ! -f "$CANDIDATES" ]]; then
  echo "no such candidates file: $CANDIDATES" >&2
  exit 1
fi

MAGICLLM_DIR="$MAGE_HOME/Mage.Tests/src/test/java/org/mage/test/magicllm"
if [[ ! -d "$MAGICLLM_DIR" ]]; then
  echo "no magicllm test package at $MAGICLLM_DIR" >&2
  echo "is --mage-home / \$MAGE_HOME ($MAGE_HOME) a collector-patched" >&2
  echo "magefree/mage checkout? (Section 21.103)" >&2
  exit 1
fi
if ! command -v mvn >/dev/null 2>&1; then
  echo "mvn not found on PATH — install a JDK and Maven first (Section 21.128)" >&2
  exit 1
fi
if [[ -z "${JAVA_HOME:-}" ]]; then
  export JAVA_HOME="$(/usr/libexec/java_home 2>/dev/null || true)"
fi
if [[ -z "${JAVA_HOME:-}" ]]; then
  echo "no JAVA_HOME, and /usr/libexec/java_home found nothing — link a JDK first:" >&2
  echo "  brew install openjdk" >&2
  echo "  sudo ln -sfn \$(brew --prefix openjdk)/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk" >&2
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "==> exporting JUnit tests from $CANDIDATES"
python3 "$REPO_ROOT/scripts/gameplay/xmage_export.py" --positions "$CANDIDATES" --out "$WORKDIR"

N=$(ls "$WORKDIR"/*.java 2>/dev/null | wc -l | tr -d ' ')
if [[ "$N" -eq 0 ]]; then
  echo "xmage_export.py produced no test classes — nothing to validate"
  exit 0
fi

echo "==> copying $N test class(es) into $MAGICLLM_DIR"
cp "$WORKDIR"/*.java "$MAGICLLM_DIR/"

# Scope mvn to just this batch's class names (not a glob over the whole
# shared package) so a run here can never be inflated or contaminated by
# whatever else has accumulated in the checkout from other batches.
CLASSES="$(cd "$WORKDIR" && ls *.java | sed 's/\.java$//' | paste -sd, -)"

echo "==> mvn clean test (scoped to this batch; can take a few minutes)"
set +e
(cd "$MAGE_HOME" && mvn clean test -pl Mage.Tests -am \
    -Dtest="$CLASSES" -Dsurefire.failIfNoSpecifiedTests=false)
MVN_STATUS=$?
set -e
if [[ $MVN_STATUS -ne 0 ]]; then
  echo "==> mvn reported test failures — expected for mulligan/leftover-action"
  echo "    edge cases (Section 21.128); continuing to extract what did answer."
fi

echo "==> xmage_diff.py --emit-legal-actions"
python3 "$REPO_ROOT/scripts/gameplay/xmage_diff.py" \
  --reports "$MAGE_HOME/Mage.Tests/target/surefire-reports" \
  --positions "$CANDIDATES" --emit-legal-actions

if [[ -n "$EXPORT_TASKS" ]]; then
  echo "==> re-exporting $EXPORT_TASKS"
  python3 "$REPO_ROOT/scripts/gameplay/import_game.py" \
    --export-from "$CANDIDATES" --export-rubric-tasks "$EXPORT_TASKS"
fi

echo "==> done"
