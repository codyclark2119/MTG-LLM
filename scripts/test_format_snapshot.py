"""Verify the versioned Standard card-pool snapshot.

    python scripts/test_format_snapshot.py

Format legality is time-dependent. This check ensures the Standard snapshot
still names the exact card pool that was used to create it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from format_snapshot import verify_snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data/manifests/formats/standard-2026-08-10.json"


def main() -> None:
    snapshot = verify_snapshot(SNAPSHOT)
    assert snapshot["format"] == "standard"
    assert snapshot["snapshot_id"] == "standard-2026-08-10"
    assert snapshot["card_count"] == 4887
    assert snapshot["source_record_count"] == 4887
    assert "Chart a Course" in snapshot["card_names"]
    print(f"verified {snapshot['snapshot_id']}: {snapshot['card_count']} cards")


if __name__ == "__main__":
    main()
