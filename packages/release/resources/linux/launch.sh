#!/bin/sh
set -eu
unavailable() {
  echo 'Magnitude installation is in progress or needs package-manager repair. Retry after installation finishes.' >&2
  exit 75
}
exec 9</var/lib/magnitude-desktop/installation.lock || unavailable
flock -s -n 9 || unavailable
[ ! -e /var/lib/magnitude-desktop/installing ] || unavailable
# exec retains the shared descriptor; no launcher supervisor remains alive.
exec /usr/lib/magnitude-desktop/magnitude "$@"
