"""Create a content manifest for an immutable JSONL source artifact.

A manifest makes source identity explicit instead of leaving it in a filename,
comment, or corpus-specific constant. It is metadata only: the source file is
never rewritten.

Usage:
    python scripts/manifest.py data/cards/processed/ruling_chunks.jsonl \
        --source-id scryfall-rulings-2026-08-24 \
        --authority official-wotc-rulings \
        --license scryfall-data \
        --out data/manifests/ruling_chunks.json

    python scripts/manifest.py --dataset-dir data/datasets/verified \
        --dataset-id rules-verified-v1 \
        --quality-tier human-verified \
        --source-ids rulesguru-snapshot-2026-08-24 cr-2026-08-07 \
        --split-policy "stratified within category" \
        --out data/manifests/datasets/rules-verified-v1.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (CARDS_RAG_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT, REPO_ROOT,
                    SYSTEM_PROMPT, file_sha256, prompt_fingerprint)  # noqa: E402


def record_count(path: Path) -> int:
    """Count nonblank JSONL records without loading the whole source."""
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def relative_path(path: Path) -> str:
    """Use a stable repository-relative path when the source is inside the repo."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def build_manifest(path: Path, source_id: str, authority: str,
                   license_name: str | None = None,
                   retrieved_at: str | None = None,
                   notes: str | None = None) -> dict:
    """Return deterministic identity and descriptive metadata for a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = {
        "schema_version": 1,
        "source_id": source_id,
        "path": relative_path(path),
        "format": "jsonl",
        "record_count": record_count(path),
        "content_sha256": file_sha256(path),
        "authority": authority,
    }
    if license_name:
        manifest["license"] = license_name
    if retrieved_at:
        manifest["retrieved_at"] = retrieved_at
    if notes:
        manifest["notes"] = notes
    return manifest


def build_dataset_manifest(dataset_dir: Path, dataset_id: str,
                           quality_tier: str, source_ids: list[str],
                           split_policy: str) -> dict:
    """Describe a train/valid JSONL dataset view without changing its files."""
    splits = {}
    prompt_counts = {name: 0 for name in
                     ("SYSTEM_PROMPT", "RAG_SYSTEM_PROMPT", "CARDS_RAG_SYSTEM_PROMPT")}
    known_prompts = {
        SYSTEM_PROMPT: "SYSTEM_PROMPT",
        RAG_SYSTEM_PROMPT: "RAG_SYSTEM_PROMPT",
        CARDS_RAG_SYSTEM_PROMPT: "CARDS_RAG_SYSTEM_PROMPT",
    }
    unknown_prompts = 0
    for split in ("train", "valid"):
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                for message in record.get("messages") or []:
                    if message.get("role") != "system":
                        continue
                    prompt_name = known_prompts.get(message.get("content"))
                    if prompt_name:
                        prompt_counts[prompt_name] += 1
                    else:
                        unknown_prompts += 1
        splits[split] = {
            "path": relative_path(path),
            "record_count": record_count(path),
            "content_sha256": file_sha256(path),
        }
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "path": relative_path(dataset_dir),
        "format": "jsonl",
        "quality_tier": quality_tier,
        "source_ids": source_ids,
        "split_policy": split_policy,
        "prompt_fingerprint": prompt_fingerprint()["prompt_fingerprint"],
        "dataset_prompt_counts": prompt_counts,
        "unknown_prompt_count": unknown_prompts,
        "splits": splits,
    }


def verify_dataset_manifest(manifest_path: Path) -> dict:
    """Verify a dataset manifest against its current train/valid files."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_dir = REPO_ROOT / manifest["path"]
    actual = build_dataset_manifest(
        dataset_dir, manifest["dataset_id"], manifest["quality_tier"],
        manifest.get("source_ids", []), manifest["split_policy"])
    if actual["splits"] != manifest.get("splits"):
        raise ValueError(
            f"dataset manifest is stale: {manifest_path} does not match "
            f"the train/valid files under {dataset_dir}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("--source-id")
    parser.add_argument("--authority")
    parser.add_argument("--license", dest="license_name")
    parser.add_argument("--retrieved-at")
    parser.add_argument("--notes")
    parser.add_argument("--dataset-dir", type=Path,
                        help="build a train/valid dataset manifest instead of a source manifest")
    parser.add_argument("--dataset-id")
    parser.add_argument("--quality-tier")
    parser.add_argument("--source-ids", nargs="*", default=[])
    parser.add_argument("--split-policy")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset_dir:
        required = (args.dataset_id, args.quality_tier, args.split_policy)
        if not all(required):
            parser.error("--dataset-dir requires --dataset-id, --quality-tier, and --split-policy")
        manifest = build_dataset_manifest(
            args.dataset_dir, args.dataset_id, args.quality_tier,
            args.source_ids, args.split_policy)
    else:
        if not args.source or not args.source_id or not args.authority:
            parser.error("source mode requires SOURCE, --source-id, and --authority")
        manifest = build_manifest(args.source, args.source_id, args.authority,
                                  args.license_name, args.retrieved_at, args.notes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Manifests are JSON, not JSONL, but still use a same-directory replacement.
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    os.replace(temp, args.out)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
