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
    device const element::Act::native *key [[buffer(SEISMIC_BUFFER_KEY)]],
    device const float *key_norm [[buffer(SEISMIC_BUFFER_KEY_NORM)]],
    device const int *latent_rows [[buffer(SEISMIC_BUFFER_LATENT_ROWS)]],
    device const int *rotary_components [[buffer(SEISMIC_BUFFER_ROTARY_COMPONENTS)]],
    device const float *rotary_frequencies [[buffer(SEISMIC_BUFFER_ROTARY_FREQUENCIES)]],
    device const float *rotary_amplitudes [[buffer(SEISMIC_BUFFER_ROTARY_AMPLITUDES)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    device const ulong *keys [[buffer(SEISMIC_BUFFER_KEYS)]],
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
    for (uint i = 0; i < KEYS_E; ++i)
        x[i] = float(key[row * KEYS_W + lane * KEYS_E + i]);
    // A rounded key is rounded from its norm and rotation held in A.
    const rounding::Rounding trained = rounding::of(uint(SEISMIC_PARAM_BLOCK), uint(SEISMIC_PARAM_ELEMENT),
        uint(SEISMIC_PARAM_SCALE), uint(SEISMIC_PARAM_AMAX_FLOOR));
    head::rms_norm(x, key_norm, as_type<float>(uint(SEISMIC_PARAM_EPSILON)), lane);
    rounding::hold_lanes<element::Act>(x, trained);
    const key_layout::table table(coordinates + row * 4, rotary_components, rotary_frequencies, rotary_amplitudes,
        lane);
    key_layout::rotate(table, lane, x);
    rounding::hold_lanes<element::Act>(x, trained);
    rounding::round_lanes<KEYS_E>(x, trained);
    device float *target = slab::row<float>(keys, ulong(destination), ulong(SEISMIC_PARAM_SLAB_ROWS), KEYS_W);
    for (uint i = 0; i < KEYS_E; ++i)
        target[lane * KEYS_E + i] = x[i];
}
