"""Optional real MapAnything smoke (weights required).

Run locally after installing the pinned package and downloading weights:

  pip install 'dashsplat[mapanything]'
  # or: pip install torch && pip install \\
  #   'git+https://github.com/facebookresearch/map-anything.git@3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9'
  huggingface-cli download facebook/map-anything-apache --local-dir ~/.cache/dashsplat-mapanything

  DASHSPLAT_MAPANYTHING_SMOKE=1 \\
    DASHSPLAT_MAPANYTHING_CACHE=$HOME/.cache/dashsplat-mapanything \\
    python -m pytest tests/test_mapanything_smoke.py -q

Skipped cleanly in CI when the env var is unset or weights are missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dashsplat.mapanything import MAPANYTHING_HF_APACHE, smoke_infer_tiny

pytestmark = pytest.mark.skipif(
    os.environ.get("DASHSPLAT_MAPANYTHING_SMOKE") != "1",
    reason="Set DASHSPLAT_MAPANYTHING_SMOKE=1 to run real MapAnything inference",
)


def test_mapanything_tiny_infer(tmp_path: Path):
    cache = os.environ.get("DASHSPLAT_MAPANYTHING_CACHE")
    cache_dir = Path(cache) if cache else None
    if cache_dir is not None and not (cache_dir / "config.json").exists():
        pytest.skip(f"cache missing config.json: {cache_dir}")

    imgs = []
    for i in range(2):
        p = tmp_path / f"v{i}.jpg"
        Image.fromarray(np.full((256, 256, 3), 80 + i * 20, dtype=np.uint8)).save(p)
        imgs.append(p)

    try:
        result = smoke_infer_tiny(imgs, cache_dir=cache_dir, model_id=MAPANYTHING_HF_APACHE)
    except ImportError as e:
        pytest.skip(f"mapanything/torch not installed: {e}")
    except Exception as e:
        # Network / missing weights — skip rather than fail CI-like local without hub
        if "offline" in str(e).lower() or "huggingface" in str(e).lower() or "not found" in str(e).lower():
            pytest.skip(f"weights unavailable: {e}")
        raise

    assert result["pts3d"].ndim == 4
    assert result["backend"].startswith("torch-mapanything")
    assert result["pkg_commit"].startswith("3d10cf7")
