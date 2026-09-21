#!/bin/bash
set -euo pipefail
cd /opt/lab/runtime
# These daemons live only in this fresh container. Real pkexec/Polkit authorizes
# the same packaged updater operation as the disposable Azure Linux workers.
sudo -n mkdir -p /run/dbus
sudo -n dbus-daemon --system --fork
sudo -n /usr/lib/polkit-1/polkitd --no-debug > /tmp/magnitude-lab-polkit.log 2>&1 &
policy=$!
trap 'sudo -n kill "$policy" 2>/dev/null || true; wait "$policy" 2>/dev/null || true' EXIT
ready=false
for attempt in $(seq 1 30); do
  if dbus-send --system --print-reply --dest=org.freedesktop.PolicyKit1 \
    /org/freedesktop/PolicyKit1/Authority org.freedesktop.DBus.Peer.Ping >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 1
done
test "$ready" = true
xvfb-run -a -s '-screen 0 1600x1000x24' dbus-run-session -- bash -c '
  openbox >/tmp/magnitude-lab-openbox.log 2>&1 &
  manager=$!
  trap '\''kill "$manager" 2>/dev/null || true; wait "$manager" 2>/dev/null || true'\'' EXIT
  bun packages/testing-lab/src/worker-entry.ts "$1"
' -- "$1"
