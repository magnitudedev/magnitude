#include <seismic/element.h>
// candidate_blocks on Metal (contract and portable body in indexer.seismic).
//
// A threadgroup of 256 threads (8 simdgroups) per (8 blocks, row), simdgroup s
// scoring block 8 x + s: the 32 lanes read the block's scores coalesced and
// reduce with `metal::simd_max`; the newest block is +inf, -inf past the row's
// blocks. Lane 0 of block 0 writes the row's block list.
//
// Dynamic extents read `seismic_words`.

kernel void candidate_blocks(
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const int *complete [[buffer(SEISMIC_BUFFER_COMPLETE)]],
    SEISMIC_RESULT_0_PARAM(blocks_cells),
    device int *lists [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong row = group.y;
    const uint n = uint(SEISMIC_DIM_N);
    const uint nb = uint(SEISMIC_DIM_NB);
    const uint bu = uint(SEISMIC_DIM_BU);
    const uint block = group.x * 8u + simd;
    const uint units = uint(metal::clamp(complete[row], 0, int(n)));
    const uint count = (units + bu - 1) / bu;
    if (block == 0 && lane == 0) {
        lists[row * 2] = 0;
        lists[row * 2 + 1] = int(count);
    }
    if (block >= nb)
        return;
    if (block + 1 >= count) {
        if (lane == 0) {
            const float score = (block + 1 == count) ? INFINITY : -INFINITY;
            element::Out<element::F32>(blocks_cells).put(row * ulong(nb) + block, score);
        }
        return;
    }
    const uint begin = block * bu;
    const uint end = metal::min(begin + bu, units);
    const uint len = end - begin;
    device const float *base = scores + row * ulong(n) + begin;
    float score = -INFINITY;
    uint offset = 0;
    if (len >= 128u) {
        const uint vec_count = len >> 2;
        device const packed_float4 *vec = reinterpret_cast<device const packed_float4 *>(base);
        for (uint i = lane; i < vec_count; i += 32u) {
            const float4 v = float4(vec[i]);
            score = metal::max(score, metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w)));
        }
        offset = vec_count << 2;
    }
    for (uint i = offset + lane; i < len; i += 32u)
        score = metal::max(score, base[i]);
    score = metal::simd_max(score);
    if (lane == 0)
        element::Out<element::F32>(blocks_cells).put(row * ulong(nb) + block, score);
}
