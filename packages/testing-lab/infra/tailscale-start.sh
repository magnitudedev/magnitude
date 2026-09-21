#!/bin/sh
set -eu
umask 077
: "${LAB_TAILSCALE_AUTH_KEY:?Tailscale enrollment credential is required}"
mkdir -p /opt/lab/tailscale /var/run/tailscale
printf '%s' "$LAB_TAILSCALE_AUTH_KEY" > /opt/lab/tailscale/auth-key
unset LAB_TAILSCALE_AUTH_KEY
daemon_pid=''
ready=false
trap 'rm -f /opt/lab/tailscale/auth-key; if [ "$ready" != true ] && [ -n "$daemon_pid" ]; then kill "$daemon_pid" 2>/dev/null || true; fi' EXIT
# No host networking, TUN device, subnet route, SSH listener or privileged
# container is required. State is ephemeral; enrollment must permit new replicas.
tailscaled --tun=userspace-networking --state=mem: > /opt/lab/tailscale/daemon.log 2>&1 &
daemon_pid=$!
attempt=0
while [ ! -S /var/run/tailscale/tailscaled.sock ]; do
  kill -0 "$daemon_pid"
  attempt=$((attempt + 1))
  [ "$attempt" -lt 30 ] || exit 1
  sleep 1
done
tailscale up --auth-key=file:/opt/lab/tailscale/auth-key --hostname=magnitude-lab-coordinator \
  --accept-dns=false --accept-routes=false --shields-up --timeout=45s \
  > /opt/lab/tailscale/enrollment.log 2>&1
ready=true
