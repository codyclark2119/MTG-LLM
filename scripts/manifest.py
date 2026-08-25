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
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO_ROOT, file_sha256  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--license", dest="license_name")
    parser.add_argument("--retrieved-at")
    parser.add_argument("--notes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

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
