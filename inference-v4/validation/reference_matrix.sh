#!/bin/bash
# Reference-matrix runner (native kernel program §3.3). Runs one engine's cells
# for one model on the current host and writes JSON under $OUT.
#
#   reference_matrix.sh llama  <gguf> <model-tag> <out-dir> <llama-bin-dir>
#   reference_matrix.sh v3     <gguf> <model-tag> <out-dir> <v3-python> <v3-source> <backend> <memory-gib> \
#                              [contexts] [prefill-lengths] [concurrent-sequences]
#   reference_matrix.sh v4     <gguf> <model-tag> <out-dir> <forward_bench-binary> <storage-gib> \
#                              [contexts] [prefill-lengths] [concurrent-sequences]
#   reference_matrix.sh probes <out-dir> <probe-dir>
#
# Long-context cells (decode after H tokens, prefill-512 after H tokens), run periodically:
#   reference_matrix.sh llama-long <gguf> <model-tag> <out-dir> <llama-bin-dir> <H>
#   reference_matrix.sh v3-long    <gguf> <model-tag> <out-dir> <v3-python> <v3-source> <backend> \
#                                  <memory-gib> <H> <dense-bf16|v3-default>
#   reference_matrix.sh v4-long    <gguf> <model-tag> <out-dir> <forward_bench-binary> <storage-gib> <H>
#
# Cells: decode at 256/4k/16k, prefill 32/128/512, concurrent 1/2/4/8 at 4k; the long cells at
# 64k are assessed periodically, not every run (owner instruction 2026-09-24).
set -euo pipefail
engine=$1; shift
host=$(hostname -s)
case $engine in
  llama)
    G=$1 TAG=$2 OUT=$3 LB=$4
    mkdir -p "$OUT"
    "$LB/llama-bench" -m "$G" -ngl 99 -fa 1 -ctk f16 -ctv f16 -p 0 -n 64 -d 256,4096,16384 -r 5 -o json \
      > "$OUT/llama-$host-$TAG-decode.json"
    "$LB/llama-bench" -m "$G" -ngl 99 -fa 1 -p 32,128,512 -n 0 -r 5 -o json \
      > "$OUT/llama-$host-$TAG-prefill.json"
    "$LB/llama-batched-bench" -m "$G" -ngl 99 -fa on -c 33792 -npp 4096 -ntg 128 -npl 1,2,4,8 \
      --output-format jsonl > "$OUT/llama-$host-$TAG-concurrent.jsonl"
    ;;
  v3)
    # Optional trailing cell sets narrow the matrix (the 35B model runs decode 256,4096 and
    # prefill 128,512 only); an empty set skips that family.
    G=$1 TAG=$2 OUT=$3 PY=$4 SRC=$5 BACKEND=$6 MEM=$7
    CONTEXTS=${8-256,4096,16384} PREFILL=${9-32,128,512} SEQUENCES=${10-1,2,4,8}
    mkdir -p "$OUT"
    here=$(cd "$(dirname "$0")" && pwd)
    # V3's generated TileLang host functions for the 35B model need more than macOS's default
    # 8 MiB main-thread stack (SIGSEGV in ___chkstk_darwin under __tvm_ffi_main).
    ulimit -s "$(ulimit -Hs)"
    for context in ${CONTEXTS//,/ }; do
      "$PY" "$here/v3_qwen_forward_bench.py" --source "$SRC" --artifact "$G" --backend "$BACKEND" \
        --memory-gib "$MEM" --cells decode --context "$context" \
        --output "$OUT/v3-$host-$TAG-decode-$context.json"
    done
    if [ -n "$PREFILL" ]; then
      "$PY" "$here/v3_qwen_forward_bench.py" --source "$SRC" --artifact "$G" --backend "$BACKEND" \
        --memory-gib "$MEM" --cells prefill --prefill "$PREFILL" --output "$OUT/v3-$host-$TAG-prefill.json"
    fi
    # One process per sequence count: V3's per-kernel inspection fails on multi-sequence batches
    # ("Kernel timestamp precedes capture") and leaves an open inspection behind, which must not
    # overlap a later count's measured steps.
    for sequences in ${SEQUENCES//,/ }; do
      "$PY" "$here/v3_qwen_forward_bench.py" --source "$SRC" --artifact "$G" --backend "$BACKEND" \
        --memory-gib "$MEM" --cells concurrent --context 4096 --sequences "$sequences" \
        --output "$OUT/v3-$host-$TAG-concurrent-$sequences.json" || status=$?
    done
    exit "${status:-0}"
    ;;
  v4)
    # Optional trailing cell sets as for v3; an empty concurrent set skips that run.
    G=$1 TAG=$2 OUT=$3 BIN=$4 STORAGE=$5
    CONTEXTS=${6-256,4096,16384} PREFILL=${7-32,128,512} SEQUENCES=${8-1,2,4,8}
    mkdir -p "$OUT"
    "$BIN" bench --model "$G" --storage-gib "$STORAGE" --cells decode,prefill \
      --context "$CONTEXTS" --prefill "$PREFILL" --output "$OUT/v4-$host-$TAG.json"
    if [ -n "$SEQUENCES" ]; then
      "$BIN" bench --model "$G" --storage-gib "$STORAGE" --cells concurrent \
        --context 4096 --sequences "$SEQUENCES" --output "$OUT/v4-$host-$TAG-concurrent.json"
    fi
    ;;
  llama-long)
    # Long-context cells: decode after H tokens and a 512-token prefill after H tokens.
    G=$1 TAG=$2 OUT=$3 LB=$4 H=$5
    mkdir -p "$OUT"
    "$LB/llama-bench" -m "$G" -ngl 99 -fa 1 -ctk f16 -ctv f16 -p 0 -n 64 -d "$H" -r 3 -o json \
      > "$OUT/llama-$host-$TAG-decode-$H.json"
    "$LB/llama-bench" -m "$G" -ngl 99 -fa 1 -ctk f16 -ctv f16 -p 512 -n 0 -d "$H" -r 3 -o json \
      > "$OUT/llama-$host-$TAG-prefill-$H.json"
    ;;
  v3-long)
    # KV is the V3 codec: dense-bf16 (as the V4 cells) or v3-default (V3's shipped K8/V4 codec).
    G=$1 TAG=$2 OUT=$3 PY=$4 SRC=$5 BACKEND=$6 MEM=$7 H=$8 KV=$9
    mkdir -p "$OUT"
    here=$(cd "$(dirname "$0")" && pwd)
    ulimit -s "$(ulimit -Hs)"
    "$PY" "$here/v3_qwen_forward_bench.py" --source "$SRC" --artifact "$G" --backend "$BACKEND" \
      --memory-gib "$MEM" --kv "$KV" --cells decode --context "$H" \
      --output "$OUT/v3-$host-$TAG-decode-$H.json"
    "$PY" "$here/v3_qwen_forward_bench.py" --source "$SRC" --artifact "$G" --backend "$BACKEND" \
      --memory-gib "$MEM" --kv "$KV" --cells prefill --prefill 512 --prefill-history "$H" \
      --output "$OUT/v3-$host-$TAG-prefill-$H.json"
    ;;
  v4-long)
    G=$1 TAG=$2 OUT=$3 BIN=$4 STORAGE=$5 H=$6
    mkdir -p "$OUT"
    "$BIN" bench --model "$G" --storage-gib "$STORAGE" --cells decode,prefill \
      --context "$H" --prefill 512 --prefill-history "$H" --output "$OUT/v4-$host-$TAG-long-$H.json"
    ;;
  probes)
    OUT=$1 DIR=$2
    mkdir -p "$OUT"
    "$DIR/stream-read" > "$OUT/bandwidth-$host.json"
    if [ -x "$DIR/simdgroup-throughput" ]; then
      "$DIR/simdgroup-throughput" > "$OUT/simdgroup-throughput-$host.json"
    fi
    if [ -x "$DIR/mma-determinism" ]; then
      "$DIR/mma-determinism" > "$OUT/mma-determinism-$host.json"
    fi
    ;;
  *) echo "unknown engine $engine" >&2; exit 2 ;;
esac
