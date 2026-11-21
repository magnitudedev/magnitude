// candidate_blocks on CUDA (contract and portable body in indexer.seismic), as
// Metal's (`metal/candidate_blocks.metal`):
//
// A block of 256 threads (8 warps) per (8 blocks, row), warp w scoring block
// 8 x + w: the 32 lanes read the block's scores coalesced and reduce with
// `seismic_warp_max_f32`; the newest block is +inf, -inf past the row's
// blocks. Lane 0 of block 0 writes the row's block list.
//
// Dynamic extents and strides read `seismic_words`.

#include <seismic/element.cuh>

extern "C" __global__ void __launch_bounds__(256) candidate_blocks(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned int u32;
    typedef unsigned long long u64;
    const u64 row = blockIdx.y;
    const u32 n = static_cast<u32>(SEISMIC_DIM_N);
    const u32 nb = static_cast<u32>(SEISMIC_DIM_NB);
    const u32 bu = static_cast<u32>(SEISMIC_DIM_BU);
    const u32 warp = threadIdx.x >> 5;
    const u32 lane = threadIdx.x & 31u;
    const u32 block = blockIdx.x * 8u + warp;
    const int complete =
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COMPLETE))[row * SEISMIC_COMPLETE_STRIDE_0];
    const u32 units = static_cast<u32>(min(max(complete, 0), static_cast<int>(n)));
    const u32 count = (units + bu - 1) / bu;
    if (block == 0 && lane == 0) {
        int *lists = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER)) + row * SEISMIC_RESULT_1_STRIDE_0;
        lists[0] = 0;
        lists[SEISMIC_RESULT_1_STRIDE_2] = static_cast<int>(count);
    }
    if (block >= nb) return;
    const float infinity = __int_as_float(0x7f800000);
    if (block + 1 >= count) {
        if (lane == 0) {
            const float score = (block + 1 == count) ? infinity : -infinity;
            SEISMIC_RESULT_0_OUT.put(
                row * SEISMIC_RESULT_0_STRIDE_0 + static_cast<u64>(block) * SEISMIC_RESULT_0_STRIDE_1,
                score);
        }
        return;
    }
    const float *row_scores =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)) + row * SEISMIC_SCORES_STRIDE_0;
    const u64 stride1 = SEISMIC_SCORES_STRIDE_1;
    const u32 begin = block * bu;
    const u32 end = min(begin + bu, units);
    const u32 len = end - begin;
    float score = -infinity;
    if (stride1 == 1) {
        const float *base = row_scores + begin;
        u32 offset = 0;
        if (len >= 128u && (reinterpret_cast<u64>(base) & 15ull) == 0ull) {
            const u32 vec_count = len >> 2;
            const float4 *vec = reinterpret_cast<const float4 *>(base);
            for (u32 i = lane; i < vec_count; i += 32u) {
                const float4 v = vec[i];
                score = fmaxf(score, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
            }
            offset = vec_count << 2;
        }
        for (u32 i = offset + lane; i < len; i += 32u)
            score = fmaxf(score, base[i]);
    } else {
        for (u32 unit = begin + lane; unit < end; unit += 32u)
            score = fmaxf(score, row_scores[static_cast<u64>(unit) * stride1]);
    }
    score = seismic_warp_max_f32(score);
    if (lane == 0) {
        SEISMIC_RESULT_0_OUT.put(row * SEISMIC_RESULT_0_STRIDE_0 + static_cast<u64>(block) * SEISMIC_RESULT_0_STRIDE_1,
                                 score);
    }
}
