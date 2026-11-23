#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/python"

export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

python3 -m pip install -q -e ".[dev]" || python3 -m pip install -q pytest numpy pillow
python3 tools/generate_synthetic_clip.py --out "$ROOT/fixtures/synthetic"
mkdir -p "$ROOT/fixtures/sample_teslacam/SavedClips/2025-11-20_14-03-12"
cp "$ROOT/fixtures/synthetic/"*.mp4 "$ROOT/fixtures/sample_teslacam/SavedClips/2025-11-20_14-03-12/" 2>/dev/null || true
echo '{"city":"Palo Alto","est_lat":37.3947,"est_lon":-122.1503}' \
  > "$ROOT/fixtures/sample_teslacam/SavedClips/2025-11-20_14-03-12/event.json"

set +e
python3 -m pytest -q
PY_CODE=$?
set -e
echo "pytest exit_code=$PY_CODE"
exit "$PY_CODE"
