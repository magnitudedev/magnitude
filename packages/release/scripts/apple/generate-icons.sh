#!/bin/bash
# Rebuild the checked-in ICNS from the brand tile (assets/brand/icon-tile.svg, the same
# rounded blue tile the website ships), inset on Apple's icon grid so it matches other
# Dock icons. Asset-authoring only; release CI consumes the committed resource.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
resources=packages/release/resources/macos
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$resources"
/usr/bin/swift packages/release/scripts/apple/render-iconset.swift assets/brand/icon-tile.svg "$scratch/Magnitude.iconset"
/usr/bin/iconutil -c icns "$scratch/Magnitude.iconset" -o "$resources/Magnitude.icns"
