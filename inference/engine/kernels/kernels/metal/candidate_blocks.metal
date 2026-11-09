// candidate_blocks on Metal (contract and portable body in indexer.seismic).
//
// A threadgroup of 256 threads per (256 blocks, row), thread t scoring block
// 256 x + t: the largest of its listed scores, the newest block +inf, -inf
// past the row's blocks. The first thread of the row's first threadgroup
// writes its block list.
//
// Dynamic extents read `seismic_words`.

kernel void candidate_blocks(
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const int *complete [[buffer(SEISMIC_BUFFER_COMPLETE)]],
    device float *blocks [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    device int *lists [[buffer(SEISMIC_RESULT_1_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const ulong row = group.y;
    const uint n = uint(SEISMIC_DIM_N);
    const uint nb = uint(SEISMIC_DIM_NB);
    const uint bu = uint(SEISMIC_DIM_BU);
    const uint units = uint(metal::clamp(complete[row], 0, int(n)));
    const uint count = (units + bu - 1) / bu;
    const uint block = group.x * 256 + thread_index;
    if (block == 0) {
        lists[row * 2] = 0;
        lists[row * 2 + 1] = int(count);
    }
    if (block >= nb)
        return;
    float score = -INFINITY;
    if (block + 1 == count) {
        score = INFINITY;
    } else if (block < count) {
        device const float *row_scores = scores + row * ulong(n);
        const uint end = metal::min((block + 1) * bu, units);
        for (uint unit = block * bu; unit < end; ++unit)
            score = metal::max(score, row_scores[unit]);
    }
    blocks[row * ulong(nb) + block] = score;
}
