// indexer_keys on Metal (contract and portable body in indexer.seismic).
//
// A simdgroup per row (8 a threadgroup), lane l holding columns
// [l E, (l + 1) E) of the row's projected key (E = W / 32): a row that
// completes a unit (latent_rows[row] >= 0) widens its key to F32,
// RMS-normalizes it with the key norm, rotates its pairs at the row's
// coordinates (the unit's first token's; `rotary::layout`), rounds it with
// the indexer's trained rounding and stores it in the F32 key plane at
// latent_rows[row].
//
// Dynamic extents and parameters read `seismic_words`.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include "lib/core/head.h"
#include "lib/core/rotary.h"
#include "lib/core/rounding.h"
#include <seismic/slab.h>

#define KEYS_W uint(2 * SEISMIC_DIM_P + SEISMIC_DIM_S)
#define KEYS_E (KEYS_W / 32)

typedef rotary::layout<KEYS_W, SEISMIC_DIM_P, (SEISMIC_DIM_RT != 0 ? SEISMIC_DIM_S : 0), (SEISMIC_DIM_RA != 0)>
    key_layout;

kernel void indexer_keys(
    device const activation::Act::native *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const float *key_norm [[buffer(SEISMIC_BUFFER_KEY_NORM)]],
    device const int *latent_rows [[buffer(SEISMIC_BUFFER_LATENT_ROWS)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const float *rotary_amplitudes [[buffer(SEISMIC_BUFFER_ROTARY_AMPLITUDES)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    SEISMIC_OUT_KEYS_PARAM(keys),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = ulong(group.x) * 8 + simd;
    if (row >= ulong(SEISMIC_DIM_M))
        return;
    const int destination = latent_rows[row];
    if (destination < 0)
        return;
    float x[KEYS_E];
    const ulong key_offset = row * KEYS_W + ulong(lane * KEYS_E);
    if constexpr (KEYS_E % 4 == 0) {
        device const activation::Act::storage *key_base =
            reinterpret_cast<device const activation::Act::storage *>(key) + key_offset;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < KEYS_E; i += 4) {
            const float4 v = activation::Act::load4(key_base + i);
            x[i + 0] = v.x;
            x[i + 1] = v.y;
            x[i + 2] = v.z;
            x[i + 3] = v.w;
        }
    } else {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < KEYS_E; ++i)
            x[i] = float(key[key_offset + i]);
    }
    // A rounded key is rounded from its norm and rotation held in A.
    const rounding::Rounding trained = rounding::of(uint(SEISMIC_PARAM_BLOCK), uint(SEISMIC_PARAM_ELEMENT),
        uint(SEISMIC_PARAM_SCALE), uint(SEISMIC_PARAM_AMAX_FLOOR));
    head::rms_norm(x, key_norm, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), lane);
    rounding::hold_lanes<activation::Act>(x, trained);
    const key_layout::table table(coordinates + row * 4, rotary_components, rotary_frequencies, rotary_amplitudes,
        lane);
    key_layout::rotate(table, lane, x);
    rounding::hold_lanes<activation::Act>(x, trained);
    rounding::round_lanes<KEYS_E>(x, trained);
    const element::Out<element::F32> target = slab::out_row(keys, ulong(destination), ulong(SEISMIC_PARAM_SLAB_ROWS),
        KEYS_W);
    if constexpr (KEYS_E % 4 == 0) {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < KEYS_E; i += 4)
            target.put4(ulong(lane * KEYS_E + i), float4(x[i + 0], x[i + 1], x[i + 2], x[i + 3]));
    } else {
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < KEYS_E; ++i)
            target.put(lane * KEYS_E + i, x[i]);
    }
}
