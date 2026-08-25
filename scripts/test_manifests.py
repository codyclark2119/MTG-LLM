"""Verify repository corpus manifests match the files they describe.

    python scripts/test_manifests.py

Manifests are useful only when they cannot quietly drift from their source.
This check is intentionally local and network-free: it recomputes the content
hash and JSONL record count for each checked-in knowledge artifact.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from manifest import record_count  # noqa: E402
from common import REPO_ROOT, file_sha256  # noqa: E402

ARTIFACTS = {
    "rules.json": "data/processed/rules.jsonl",
    "glossary.json": "data/processed/glossary.jsonl",
    "card_chunks.json": "data/cards/processed/card_chunks.jsonl",
    "ruling_chunks.json": "data/cards/processed/ruling_chunks.jsonl",
}


def main() -> None:
    checked = 0
    for manifest_name, artifact_name in ARTIFACTS.items():
        manifest_path = REPO_ROOT / "data/manifests" / manifest_name
        artifact_path = REPO_ROOT / artifact_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["path"] == artifact_name
        assert manifest["content_sha256"] == file_sha256(artifact_path)
        assert manifest["record_count"] == record_count(artifact_path)
        checked += 1
    print(f"all {checked} corpus manifests match their artifacts")


if __name__ == "__main__":
    main()
