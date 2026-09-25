#!/usr/bin/env bash
# Run only on a disposable RPM-based VM with no installed Magnitude package.
set -euo pipefail
umask 077
test "$#" -eq 3 || { echo 'Usage: test-linux-rpm.sh BASE_RPM NEXT_RPM ICN_INSTALLATION' >&2; exit 1; }
base_rpm=$(realpath "$1")
next_rpm=$(realpath "$2")
export MAGNITUDE_ICN_PATH=$(realpath "$3")
if rpm -q magnitude-desktop >/dev/null 2>&1; then
  echo 'This test requires an unused Magnitude installation.' >&2
  exit 1
fi
sudo -n true
test_root=$(mktemp -d "${TMPDIR:-/tmp}/magnitude-rpm-acceptance.XXXXXXXX")
echo "Acceptance output: $test_root"
export MAGNITUDE_DEV_DATA_DIR="$test_root/profile"
export MAGNITUDE_DESKTOP_STATE_DIR="$test_root/state"
export MAGNITUDE_DEV_PORT=11237
mkdir "$MAGNITUDE_DEV_DATA_DIR"
printf 'preserve\n' > "$MAGNITUDE_DEV_DATA_DIR/acceptance-sentinel"
owner_pid=''
trap 'if test -n "$owner_pid"; then kill -TERM "$owner_pid" 2>/dev/null || true; wait "$owner_pid" 2>/dev/null || true; fi' EXIT
start_server() {
  /usr/bin/magnitude serve > "$test_root/serve-$1.log" 2>&1 &
  owner_pid=$!
  for ((attempt=0; attempt<300; attempt++)); do
    kill -0 "$owner_pid"
    /usr/bin/magnitude status > "$test_root/status.txt"
    if grep -Eq 'Runtime[[:space:]]+Ready' "$test_root/status.txt" && grep -Eq 'Owner[[:space:]]+Headless' "$test_root/status.txt"; then return; fi
    sleep 0.2
  done
  echo 'Server did not become ready.' >&2
  return 1
}
stop_server() {
  kill -TERM "$owner_pid"
  wait "$owner_pid"
  owner_pid=''
}
sudo -n dnf -y install "$base_rpm" > "$test_root/install.log" 2>&1
base_identity=$(rpm -q magnitude-desktop)
start_server initial
if sudo -n dnf -y upgrade "$next_rpm" > "$test_root/refused-upgrade.log" 2>&1; then
  echo 'Upgrade unexpectedly succeeded while serving.' >&2; exit 1
fi
test "$(rpm -q magnitude-desktop)" = "$base_identity"
/usr/bin/magnitude models status
stop_server
sudo -n dnf -y upgrade "$next_rpm" > "$test_root/upgrade.log" 2>&1
test "$(rpm -q magnitude-desktop)" != "$base_identity"
start_server upgraded
if sudo -n dnf -y remove magnitude-desktop > "$test_root/refused-removal.log" 2>&1; then
  echo 'Removal unexpectedly succeeded while serving.' >&2; exit 1
fi
test -f /var/lib/magnitude-desktop/installing
/usr/bin/magnitude models status
stop_server
if /usr/bin/magnitude serve > "$test_root/repair-guidance.log" 2>&1; then
  echo 'Startup unexpectedly succeeded after interrupted package removal.' >&2; exit 1
fi
grep -q 'repair' "$test_root/repair-guidance.log"
sudo -n dnf -y reinstall "$next_rpm" > "$test_root/repair.log" 2>&1
test ! -e /var/lib/magnitude-desktop/installing
start_server repaired
/usr/bin/magnitude hardware
stop_server
sudo -n dnf -y remove magnitude-desktop > "$test_root/remove.log" 2>&1
test ! -e /usr/bin/magnitude
test ! -e /var/lib/magnitude-desktop/installing
test "$(cat "$MAGNITUDE_DEV_DATA_DIR/acceptance-sentinel")" = preserve
printf 'RPM install, upgrade, live-owner refusal, repair and removal passed\n' | tee "$test_root/result.txt"
