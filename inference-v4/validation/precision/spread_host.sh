#!/bin/bash
# llama.cpp spread against the F32 reference on a shared benchmark host, one category per hold of
# the host GPU lock (a spread saturates the GPU; other streams take timing runs under the lock).
#
#   spread_host.sh <label> <quantized gguf> <llama.cpp binary dir> [reference label]
#
# Run from inside the synced inference-v4/validation directory, with the reference directory and
# corpus under results/precision/. Writes results/precision/<label>/{spread.json,<cat>.bin,logs};
# spread.json carries kl_base's comparator (including same_top_above_margin) per category.
set -euo pipefail
LABEL=$1; MODEL=$2; BIN=$3; REFERENCE=${4:-ref-cpu-f32-b8680}
LOCK=$HOME/native-program/gpu-lock
export PATH=$HOME/.local/bin:$PATH
for category in prose code tool_json; do
  until mkdir "$LOCK" 2>/dev/null; do sleep 20; done
  trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
  echo "$category: lock acquired $(date -u +%FT%TZ)"
  uv run precision/reference.py spread --model "$MODEL" --reference "results/precision/$REFERENCE" \
    --label "$LABEL" --binary-dir "$BIN" --categories "$category" --save-base > /dev/null
  rmdir "$LOCK"
  trap - EXIT
  echo "$category: lock released $(date -u +%FT%TZ)"
done
echo "SPREAD-DONE results/precision/$LABEL/spread.json"
