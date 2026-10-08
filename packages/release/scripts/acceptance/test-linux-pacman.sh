#!/usr/bin/env bash
# Install an accepted pacman package in a fresh Arch Linux container and exercise its admission hook and desktop ownership.
# Run from the repository root after `bun install`; Playwright and the lifecycle fixture come from this checkout.
set -euo pipefail
test "$#" -eq 3 || { echo 'Usage: test-linux-pacman.sh PACKAGE ACCEPTED_DIRECTORY SECCOMP_PROFILE' >&2; exit 1; }
package=$(realpath "$1")
accepted=$(realpath "$2")
seccomp=$(realpath "$3")
node_root="$(dirname "$(dirname "$(readlink -f "$(command -v node)")")")"
docker run --rm --init \
  --security-opt "seccomp=$seccomp" \
  -v "$PWD:/workspace:ro" \
  -v "$package:/package/$(basename "$package"):ro" \
  -v "$accepted:/accepted:ro" \
  -v "$node_root:/node:ro" \
  docker.io/library/archlinux:base bash -euc '
    set -- /package/*.pkg.tar.zst
    pacman -Syu --noconfirm --needed xorg-server-xvfb xorg-xauth xorg-xprop dbus xfwm4 desktop-file-utils diffutils
    pacman -U --noconfirm "$1"
    pacman -Qkk magnitude-desktop
    cmp /accepted/bin/magnitude-service /usr/lib/magnitude-desktop/resources/magnitude-service
    test "$(stat -c "%u:%g:%a" /usr/lib/magnitude-desktop/chrome-sandbox)" = "0:0:4755"
    # A running application holds the installation lock shared; pacman must refuse to replace it.
    # The holder itself keeps the descriptor so that killing it releases the lock.
    sh -c "exec 9</var/lib/magnitude-desktop/installation.lock && flock --shared 9 && exec sleep 60" & holder=$!
    sleep 1
    if pacman -U --noconfirm "$1"; then echo "pacman replaced a running installation" >&2; exit 1; fi
    if pacman -R --noconfirm magnitude-desktop; then echo "pacman removed a running installation" >&2; exit 1; fi
    kill "$holder"; wait "$holder" || true
    test ! -e /var/lib/magnitude-desktop/installing
    pacman -U --noconfirm "$1"
    test ! -e /var/lib/magnitude-desktop/installing
    useradd --create-home magnitude-test
    runuser -u magnitude-test -- env PATH=/node/bin:/usr/bin:/bin \
      DEBUG=pw:browser \
      MAGNITUDE_TEST_CLI_EXECUTABLE=/accepted/bin/magnitude-cli \
      xvfb-run -a dbus-run-session -- node /workspace/desktop/src/fixtures/linux-installed-lifecycle.mjs
  '
