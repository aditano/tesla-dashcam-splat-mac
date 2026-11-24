from __future__ import annotations

from pathlib import Path


def test_ci_yml_does_not_overwrite_tracked_fixtures():
    ci = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
    text = ci.read_text()
    assert "cp fixtures/synthetic" not in text
    assert "mktemp" in text
    # Must not redirect event.json into tracked sample path
    assert 'echo \'{"city":"Palo Alto"' not in text or "fixtures/sample_teslacam" not in text.split("Generate")[1].split("Pytest")[0]
