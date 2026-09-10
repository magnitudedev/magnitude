#!/bin/bash
# Rebuild the checked-in ICNS from the brand tile (assets/brand/icon-tile.svg, the same
# rounded blue tile the website ships). Inkscape is an asset-authoring tool only; release
# CI consumes the committed resource.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
inkscape="${INKSCAPE_BIN:-/Applications/Inkscape.app/Contents/MacOS/inkscape}"
source=assets/brand/icon-tile.svg
resources=packages/release/resources/macos
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/Magnitude.iconset" "$resources"
for size in 16 32 128 256 512; do
  "$inkscape" "$source" --export-width="$size" --export-height="$size" --export-filename="$scratch/Magnitude.iconset/icon_${size}x${size}.png" > /dev/null
  double=$((size * 2))
  "$inkscape" "$source" --export-width="$double" --export-height="$double" --export-filename="$scratch/Magnitude.iconset/icon_${size}x${size}@2x.png" > /dev/null
done
/usr/bin/iconutil -c icns "$scratch/Magnitude.iconset" -o "$resources/Magnitude.icns"
