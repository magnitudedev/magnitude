#!/bin/bash
# llama.cpp spread against the F32 reference, one category at a time. A spread saturates the GPU; on
# a machine shared with other timing work, set GPU_LOCK to a lock directory path that every GPU user
# agrees on, and each category then runs while holding it (mkdir to acquire, rmdir to release).
#
#   [GPU_LOCK=<lock dir>] spread_host.sh <label> <quantized gguf> <llama.cpp binary dir> [reference label [base|nobase [llama-perplexity args...]]]
#
# Requires `uv` on PATH. Run from inside the inference-v4/validation directory, with the reference
# directory and corpus under results/precision/. Writes results/precision/<label>/{spread.json,logs};
# with `base` (the default) also this backend's own <cat>.bin, and spread.json carries kl_base's
# comparator (including same_top_above_margin) per category. Extra arguments go to llama-perplexity,
# e.g. `-ctk q8_0 -ctv q4_0 -fa on` for the quantized-KV comparison on a long-context reference.
set -euo pipefail
LABEL=$1; MODEL=$2; BIN=$3; REFERENCE=${4:-ref-cpu-f32-b8680}; BASE=${5:-base}
shift $(( $# < 5 ? $# : 5 ))
case $BASE in
  base) SAVE=(--save-base) ;;
  nobase) SAVE=() ;;
  *) echo "fifth argument must be base or nobase" >&2; exit 2 ;;
esac
LOCK=${GPU_LOCK:-}
for category in prose code tool_json; do
  if [ -n "$LOCK" ]; then
    until mkdir "$LOCK" 2>/dev/null; do sleep 2; done
    trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
    echo "$category: lock acquired $(date -u +%FT%TZ)"
  fi
  uv run precision/reference.py spread --model "$MODEL" --reference "results/precision/$REFERENCE" \
    --label "$LABEL" --binary-dir "$BIN" --categories "$category" "${SAVE[@]}" --extra "$@" > /dev/null
  if [ -n "$LOCK" ]; then
    rmdir "$LOCK"
    trap - EXIT
    echo "$category: lock released $(date -u +%FT%TZ)"
  fi
done
echo "SPREAD-DONE results/precision/$LABEL/spread.json"
