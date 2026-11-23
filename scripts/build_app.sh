#!/usr/bin/env bash
# Build unsigned DashSplat.app on macOS and zip it for distribution.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "build_app.sh must run on macOS (got $(uname -s))" >&2
  exit 1
fi

command -v brew >/dev/null || { echo "Homebrew required"; exit 1; }

if ! command -v xcodegen >/dev/null; then
  brew install xcodegen
fi

xcodegen generate
xcodebuild \
  -project DashSplat.xcodeproj \
  -scheme DashSplat \
  -configuration Release \
  -derivedDataPath build/DerivedData \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGN_IDENTITY=- \
  build

APP="$(find build/DerivedData/Build/Products/Release -name 'DashSplat.app' -maxdepth 2 | head -n1)"
if [[ -z "$APP" ]]; then
  echo "DashSplat.app not found" >&2
  exit 1
fi

mkdir -p dist
rm -rf dist/DashSplat.app
cp -R "$APP" dist/DashSplat.app
# Ensure python + viewer ride along
rsync -a --delete python/ dist/DashSplat.app/Contents/Resources/python/ || true
rsync -a viewer/ dist/DashSplat.app/Contents/Resources/ || true

(
  cd dist
  zip -r -y DashSplat-unsigned.zip DashSplat.app
)
echo "Wrote dist/DashSplat-unsigned.zip"
echo "Open unsigned app: right-click DashSplat.app → Open (Gatekeeper)."
exit 0
