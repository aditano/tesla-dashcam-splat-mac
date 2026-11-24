from __future__ import annotations

from pathlib import Path


def test_run_tests_does_not_overwrite_tracked_fixtures():
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_tests.sh"
    text = script.read_text()
    assert "fixtures/sample_teslacam" not in text
    assert "mktemp" in text
