#!/usr/bin/env sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
uvx --from openapi-python-client==0.29.0 openapi-python-client generate \
  --path "$repository_root/packages/icn-protocol/openapi.json" \
  --config "$repository_root/python-sdk/openapi-python-client.yaml" \
  --output-path "$repository_root/python-sdk/generated" \
  --meta uv \
  --overwrite \
  --fail-on-warning

cp \
  "$repository_root/python-sdk/streams.py" \
  "$repository_root/python-sdk/generated/magnitude_inference/streams.py"
