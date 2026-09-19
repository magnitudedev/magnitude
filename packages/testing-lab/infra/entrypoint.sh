#!/bin/sh
set -eu
umask 077
: "${LAB_AZURE_CLIENT_ID:?Managed identity client ID is required}"
: "${LAB_COORDINATOR_CONFIG_BASE64:?Coordinator configuration is required}"
printf '%s' "$LAB_COORDINATOR_CONFIG_BASE64" | base64 -d > /opt/lab/config.json
unset LAB_COORDINATOR_CONFIG_BASE64
if [ -n "${LAB_WORKER_INITIALIZATION_BASE64:-}" ]; then
  printf '%s' "$LAB_WORKER_INITIALIZATION_BASE64" | base64 -d > /opt/lab/worker-cloud-init.yml
  unset LAB_WORKER_INITIALIZATION_BASE64
fi
export LAB_COORDINATOR_CONFIG=/opt/lab/config.json
az login --identity --client-id "$LAB_AZURE_CLIENT_ID" --output none
exec bun /opt/lab/coordinator.js
