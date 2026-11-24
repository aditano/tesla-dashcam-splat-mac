from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from dashsplat.bootstrap import (
    BOOTSTRAP_MARKER_VERSION,
    bootstrap,
    bootstrap_is_complete,
    support_layout,
)


def test_bootstrap_skips_when_marker_current(tmp_path: Path):
    paths = support_layout(tmp_path)
    paths["root"].mkdir(parents=True, exist_ok=True)
    py = paths["venv"] / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    paths["marker"].write_text(
        json.dumps({"version": BOOTSTRAP_MARKER_VERSION, "venv": str(paths["venv"])})
    )
    assert bootstrap_is_complete(tmp_path) is True
    with patch("dashsplat.bootstrap.ensure_uv") as uv:
        result = bootstrap(support_dir=tmp_path, download_model=False, with_trainers=False)
    uv.assert_not_called()
    assert result["ok"] is True
    assert result.get("skipped") is True


def test_bootstrap_runs_when_marker_stale(tmp_path: Path):
    paths = support_layout(tmp_path)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["marker"].write_text(json.dumps({"version": "0.0.1"}))
    assert bootstrap_is_complete(tmp_path) is False
