#!/bin/sh
set -eu
# Docker 28.1.1's SSH connection helper invokes precisely this command shape.
# Never accept an arbitrary shell command through this adapter.
if [ "$#" -ne 9 ] || [ "$1" != '-o ConnectTimeout=30' ] || [ "$2" != '-T' ] || [ "$3" != '-l' ] || [ "$5" != '--' ] || [ "$7" != 'docker' ] || [ "$8" != 'system' ] || [ "$9" != 'dial-stdio' ]; then
  echo 'Unexpected Docker SSH invocation' >&2
  exit 64
fi
case "$4" in ''|*[!a-zA-Z0-9_-]*) exit 64 ;; esac
case "$6" in ''|-*|*[!a-zA-Z0-9.-]*) exit 64 ;; esac
# tailscale ssh resolves MagicDNS in userspace mode and verifies the host key
# against Tailscale's authenticated node identity. Its system SSH must not recurse
# through this adapter.
export PATH=/usr/local/bin:/usr/bin:/bin
exec tailscale ssh "$4@$6" docker system dial-stdio
