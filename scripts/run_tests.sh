#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/python"

export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

python3 -m pip install -q -e ".[dev]" || python3 -m pip install -q pytest numpy pillow

# Generate synthetic clips into a temp dir — never overwrite tracked fixtures.
SYNTH_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dashsplat-synth.XXXXXX")"
cleanup() { rm -rf "$SYNTH_DIR"; }
trap cleanup EXIT
python3 tools/generate_synthetic_clip.py --out "$SYNTH_DIR"
# Optional smoke copy under an untracked work dir (not fixtures/)
WORK_EVENT="$SYNTH_DIR/sample_event"
mkdir -p "$WORK_EVENT"
cp "$SYNTH_DIR/"*.mp4 "$WORK_EVENT/" 2>/dev/null || true
echo '{"city":"Palo Alto","est_lat":37.3947,"est_lon":-122.1503}' \
  > "$WORK_EVENT/event.json"
export DASHSPLAT_SYNTH_EVENT="$WORK_EVENT"

set +e
python3 -m pytest -q
PY_CODE=$?
set -e
echo "pytest exit_code=$PY_CODE"
exit "$PY_CODE"
