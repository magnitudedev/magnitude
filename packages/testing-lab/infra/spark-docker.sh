#!/bin/sh
set -eu
# Only this Docker client uses the Tailscale SSH adapter. Other providers retain
# the ordinary system PATH and SSH behavior.
export PATH=/opt/lab/spark-bin:$PATH
exec /usr/local/bin/docker "$@"
