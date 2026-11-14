#!/bin/bash
# The reference against upstream llama.cpp, at the converter's pinned release, on a slice of a checkpoint's layers
# (`magnitude-references compare`). Upstream's own converter writes the slice with every tensor F32, so llama.cpp runs
# on the values the reference computes with (its BF16 dequantization isn't exact): the reference computes MODEL, our
# conversion of the same layers, whose values must equal the F32 file's bit for bit. llama.cpp runs on the CPU, each
# prompt in one forward, without flash attention. It is judged by the storage-resolution bar: the reference over its
# stored values moved by one step of their formats, D11's draws perturbed:1..4. SEED draws the prompts.
#
#   compare-llama.sh CHECKPOINT REPO REVISION LAYERS MODEL OUT [COUNT LENGTH SEED]
#
# OUT receives upstream-f32.gguf (kept, and reused by a later run), tokens.npy, llama.npy and compare.json. llama.cpp
# v0.9.0 decodes a glm5-next file only with an attention layer in it (without one, its hybrid memory's unallocated
# K-index input aborts in `llama_kv_cache::set_input_k_idxs`), so GLM-5.3-Flash is compared on layers {0, 3}:
#   $MRIG_SRC/evaluation/references/compare-llama.sh ~/models/zai-org/GLM-5.3-Flash-layers-0-3.view \
#     zai-org/GLM-5.3-Flash 46a050bec13002ad29ee795c16e61cf4fa3c4a9c 0,3 \
#     ~/models/zai-org/GLM-5.3-Flash-layers-0-3/GLM-5.3-Flash-layers-0-3.gguf ~/models/zai-org/GLM-5.3-Flash-layers-0-3-llama
set -euo pipefail
if [ $# -lt 6 ]; then
  sed -n '2,16p' "$0"
  exit 2
fi
CHECKPOINT=$1 REPO=$2 REVISION=$3 LAYERS=$4 MODEL=$5 OUT=$6
COUNT=${7:-3} LENGTH=${8:-160} SEED=${9:-20260927}
INFERENCE=$(cd "$(dirname "$0")/../.." && pwd)
# The converter is its own project, in its own environment, never a project environment a caller set for this one.
conversion() { env -u UV_PROJECT_ENVIRONMENT uv run --frozen -q --project "$INFERENCE/conversion" python -m magnitude_conversion "$@"; }
references() { uv run --frozen -q --project "$INFERENCE/evaluation" magnitude-references "$@"; }
# llama.cpp's tools build with cmake, which a host may lack.
command -v cmake > /dev/null || PATH="$(uv tool run -q --from cmake==3.31.6 python -c 'import cmake; print(cmake.CMAKE_BIN_DIR)'):$PATH"

mkdir -p "$OUT"
F32=$OUT/upstream-f32.gguf
if [ ! -f "$F32" ]; then
  conversion convert "$CHECKPOINT" "$OUT/upstream-f32.partial.gguf" --layers "$LAYERS" --upstream-f32 \
    --source-repo "$REPO" --source-revision "$REVISION"
  mv "$OUT/upstream-f32.partial.gguf" "$F32"
fi
references prompts --model "$MODEL" --count "$COUNT" --length "$LENGTH" --seed "$SEED" --output "$OUT/tokens.npy"
conversion llama-logits "$F32" "$OUT/tokens.npy" "$OUT/llama.npy"
references compare --model "$MODEL" --tokens "$OUT/tokens.npy" --logits "$OUT/llama.npy" --same-values "$F32" \
  --yardstick-perturbed 1 2 3 4 --json "$OUT/compare.json"
