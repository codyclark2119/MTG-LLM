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
from manifest import build_dataset_manifest, record_count  # noqa: E402
from common import REPO_ROOT, file_sha256  # noqa: E402

ARTIFACTS = {
    "rules.json": "data/processed/rules.jsonl",
    "glossary.json": "data/processed/glossary.jsonl",
    "card_chunks.json": "data/cards/processed/card_chunks.jsonl",
    "ruling_chunks.json": "data/cards/processed/ruling_chunks.jsonl",
}

DATASETS = {
    "rules-verified-v1.json": {
        "dataset_id": "rules-verified-v1",
        "path": "data/datasets/verified",
        "quality_tier": "human-verified",
        "source_ids": ["rulesguru-snapshot-2026-08-24", "cr-2026-08-07"],
        "split_policy": "stratified within category; eval records excluded by id, source id, and question overlap",
    },
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
    datasets_checked = 0
    for manifest_name, spec in DATASETS.items():
        manifest_path = REPO_ROOT / "data/manifests/datasets" / manifest_name
        actual = build_dataset_manifest(
            REPO_ROOT / spec["path"], spec["dataset_id"],
            spec["quality_tier"], spec["source_ids"], spec["split_policy"])
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert stored["dataset_id"] == spec["dataset_id"]
        assert stored["path"] == spec["path"]
        assert stored["quality_tier"] == actual["quality_tier"]
        assert stored["source_ids"] == actual["source_ids"]
        assert stored["split_policy"] == actual["split_policy"]
        assert stored["splits"] == actual["splits"]
        datasets_checked += 1
    print(f"all {checked} corpus manifests and {datasets_checked} dataset manifests match")


if __name__ == "__main__":
    main()
