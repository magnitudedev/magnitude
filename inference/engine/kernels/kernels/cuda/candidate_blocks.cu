// candidate_blocks on CUDA (contract and portable body in indexer.seismic), as
// Metal's (`metal/candidate_blocks.metal`):
//
// A block of 256 threads per (256 blocks, row), thread t scoring block
// 256 x + t: the largest of its listed scores, the newest block +inf, -inf
// past the row's blocks. The first thread of the row's first block writes
// its block list.
//
// Dynamic extents and strides read `seismic_words`.

extern "C" __global__ void candidate_blocks(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned int u32;
    typedef unsigned long long u64;
    const u64 row = blockIdx.y;
    const u32 n = static_cast<u32>(SEISMIC_DIM_N);
    const u32 nb = static_cast<u32>(SEISMIC_DIM_NB);
    const u32 bu = static_cast<u32>(SEISMIC_DIM_BU);
    const int complete =
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COMPLETE))[row * SEISMIC_COMPLETE_STRIDE_0];
    const u32 units = static_cast<u32>(min(max(complete, 0), static_cast<int>(n)));
    const u32 count = (units + bu - 1) / bu;
    const u32 block = blockIdx.x * 256 + threadIdx.x;
    if (block == 0) {
        int *lists = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER)) + row * SEISMIC_RESULT_1_STRIDE_0;
        lists[0] = 0;
        lists[SEISMIC_RESULT_1_STRIDE_2] = static_cast<int>(count);
    }
    if (block >= nb) return;
    const float infinity = __int_as_float(0x7f800000);
    float score = -infinity;
    if (block + 1 == count) {
        score = infinity;
    } else if (block < count) {
        const float *row_scores =
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES)) + row * SEISMIC_SCORES_STRIDE_0;
        const u32 end = min((block + 1) * bu, units);
        for (u32 unit = block * bu; unit < end; ++unit)
            score = fmaxf(score, row_scores[static_cast<u64>(unit) * SEISMIC_SCORES_STRIDE_1]);
    }
    reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER))[row * SEISMIC_RESULT_0_STRIDE_0 +
                                                                  static_cast<u64>(block) * SEISMIC_RESULT_0_STRIDE_1] =
        score;
}
