"""Test validation applied at the public rubric server boundary."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rubric_server import validate_adjudication_input, validate_rubric_input  # noqa: E402


def main() -> None:
    rubric = {"id": "q1", "author": "Reviewer", "key_points": ["one", "two"],
              "common_errors": []}
    assert validate_rubric_input(rubric, {"q1"}) == []
    assert any("author is required" in p for p in validate_rubric_input(
        {**rubric, "author": ""}, {"q1"}))
    assert any("at least 2 key points" in p for p in validate_rubric_input(
        {**rubric, "key_points": ["one"]}, {"q1"}))
    assert any("must be a list" in p for p in validate_rubric_input(
        {**rubric, "common_errors": "none"}, {"q1"}))

    task = {"key": "q1::arm", "common_errors": ["error one", "error two"]}
    clean = {"key": task["key"], "author": "Reviewer", "errors_present": [],
             "unsure": False, "not_covered": False, "note": ""}
    assert validate_adjudication_input(clean, task) == []
    assert any("author is required" in p for p in validate_adjudication_input(
        {**clean, "author": ""}, task))
    assert any("must contain integers" in p for p in validate_adjudication_input(
        {**clean, "errors_present": ["1"]}, task))
    assert any("a note is required" in p for p in validate_adjudication_input(
        {**clean, "errors_present": [1]}, task))
    assert any("a note is required" in p for p in validate_adjudication_input(
        {**clean, "not_covered": True}, task))
    assert any("must be boolean" in p for p in validate_adjudication_input(
        {**clean, "unsure": "false"}, task))
    print("server input validation passed")


if __name__ == "__main__":
    main()
