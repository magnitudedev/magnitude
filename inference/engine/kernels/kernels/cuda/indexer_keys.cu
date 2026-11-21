// indexer_keys on CUDA (contract and portable body in indexer.seismic), as
// Metal's (`metal/indexer_keys.metal`):
//
// A warp per row (8 a block), lane l holding columns [l E, (l + 1) E) of the
// row's projected key (E = W / 32): a row that completes a unit
// (latent_rows[row] >= 0) widens its key to F32, RMS-normalizes it with the
// key norm, rotates its pairs at the row's coordinates (the unit's first
// token's; `rotary::layout`), rounds it with the indexer's trained rounding
// (`rounding::round_lanes`) and stores it in the F32 key plane at
// latent_rows[row].
//
// Dynamic extents, strides and parameters read `seismic_words`.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"
#include "lib/core/head.cuh"
#include "lib/core/rotary.cuh"
#include "lib/core/rounding.cuh"
#include "lib/indexer/indexer.cuh"
#include <seismic/slab.cuh>

namespace keys {

typedef unsigned int u32;
typedef unsigned long long u64;
typedef activation::Act Act;

constexpr int W = 2 * SEISMIC_DIM_P + SEISMIC_DIM_S;
constexpr int E = W / 32;
static_assert(W % 32 == 0, "indexer key width must be a multiple of 32");

typedef ROTARY_ENTRY_LAYOUT(W) key_layout;

} // namespace keys

extern "C" __global__ void indexer_keys(SEISMIC_KERNEL_PARAMS) {
    using namespace keys;
    extern __shared__ float shared[];  // [8][W]: each warp's rotation exchange
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const u64 row = static_cast<u64>(blockIdx.x) * 8 + warp;
    if (row >= SEISMIC_DIM_M) return;
    const int destination =
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_LATENT_ROWS))[row * SEISMIC_LATENT_ROWS_STRIDE_0];
    if (destination < 0) return;
    const element::u8 *key = SEISMIC_PTR(SEISMIC_BUFFER_KEY);
    const u64 key_row_offset = row * SEISMIC_KEY_STRIDE_0;
    float x[E];
    if constexpr (E % 4 == 0) {
        const element::u8 *key_row = key + key_row_offset * Act::bytes;
        if (SEISMIC_KEY_STRIDE_1 == 1ull &&
            (reinterpret_cast<u64>(key_row) & (4ull * Act::bytes - 1ull)) == 0ull) {
#pragma unroll
            for (int i = 0; i < E; i += 4) {
                float chunk[4];
                element::span<Act, 4, true>(key_row, static_cast<u64>(lane * E + i), chunk);
                x[i + 0] = chunk[0];
                x[i + 1] = chunk[1];
                x[i + 2] = chunk[2];
                x[i + 3] = chunk[3];
            }
        } else {
#pragma unroll
            for (int i = 0; i < E; ++i)
                x[i] = element::at<Act>(key, key_row_offset + static_cast<u64>(lane * E + i) * SEISMIC_KEY_STRIDE_1);
        }
    } else {
#pragma unroll
        for (int i = 0; i < E; ++i)
            x[i] = element::at<Act>(key, key_row_offset + static_cast<u64>(lane * E + i) * SEISMIC_KEY_STRIDE_1);
    }
    // A rounded key is rounded from its norm and rotation held in A.
    const rounding::Rounding trained = rounding::of(SEISMIC_PARAM_BLOCK, SEISMIC_PARAM_ELEMENT, SEISMIC_PARAM_SCALE,
                                                   SEISMIC_PARAM_AMAX_FLOOR);
    head::rms_norm(x, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM)),
                   SEISMIC_KEY_NORM_STRIDE_1, element::word_f32(SEISMIC_PARAM_EPSILON), lane);
    rounding::hold_lanes<Act>(x, trained);
    const rotary::table table = ROTARY_ENTRY_TABLE(
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES)) + row * SEISMIC_COORDINATES_STRIDE_0,
        SEISMIC_COORDINATES_STRIDE_1);
    key_layout::rotate(x, table, shared + warp * W, lane);
    rounding::hold_lanes<Act>(x, trained);
    rounding::round_lanes<E>(x, trained);
    const element::Out<element::F32> target =
        SEISMIC_OUT_KEYS.row(static_cast<u64>(destination), SEISMIC_PARAM_SLAB_ROWS, W);
    if constexpr (E % 4 == 0) {
        if ((reinterpret_cast<u64>(target.stored(0)) & 15ull) == 0ull) {
            target.put_span<E>(static_cast<u64>(lane * E), x);
        } else {
#pragma unroll
            for (int i = 0; i < E; ++i) target.put(lane * E + i, x[i]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < E; ++i) target.put(lane * E + i, x[i]);
    }
}
