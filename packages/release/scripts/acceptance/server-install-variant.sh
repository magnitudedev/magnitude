#!/bin/sh
# Turns a built Debian test package into one of the server-install failure cases, keeping only it:
#   preinstall  the package refuses to install in preinst (update scenario U6)
#   startup     the bundled service exits at once, so the new version never becomes ready (U7)
set -eu
artifacts=$1 variant=$2
deb=$(ls "$artifacts"/*.deb)
work=$(mktemp -d)
dpkg-deb -R "$deb" "$work/tree"
case "$variant" in
  preinstall) sed -i '/^case "$1" in install|upgrade)/a echo "This Magnitude test package refuses to install." >\&2; exit 1' "$work/tree/DEBIAN/preinst"
    grep -q 'refuses to install' "$work/tree/DEBIAN/preinst" ;;
  startup) printf '#!/bin/sh\nexit 1\n' > "$work/tree/usr/lib/magnitude-desktop/resources/magnitude-service" ;;
  *) echo "Unknown variant $variant" >&2; exit 1 ;;
esac
dpkg-deb --root-owner-group -Zzstd -z9 --build "$work/tree" "$deb"
rm -rf "$work"
# Only the Debian package is a test case; its artifact record carries the new bytes.
find "$artifacts" -maxdepth 1 \( -name '*.rpm' -o -name '*.pkg.tar.zst' -o -name '*-rpm.artifact.json' -o -name '*-pacman.artifact.json' \) -delete
record=$(ls "$artifacts"/*-deb.artifact.json)
jq --argjson bytes "$(stat -c %s "$deb")" --arg sha256 "$(sha256sum "$deb" | cut -d ' ' -f 1)" '.bytes = $bytes | .sha256 = $sha256' "$record" > "$record.new"
mv "$record.new" "$record"
