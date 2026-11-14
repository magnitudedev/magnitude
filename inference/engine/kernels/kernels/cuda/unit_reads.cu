// unit_reads on CUDA (contract and portable body in indexer.seismic), as
// Metal's (`metal/unit_reads.metal`): a block of 256 threads per row, a
// thread per bitmap word finding its word's picks by a binary search of the
// ascending valid picks; thread 0 writes the units, the picks then the tail
// unit, then -1.
//
// Dynamic extents and strides read `seismic_words`.

typedef unsigned int u32;
typedef unsigned long long u64;

extern "C" __global__ void unit_reads(SEISMIC_KERNEL_PARAMS) {
    const u64 row = blockIdx.x;
    const u32 k = static_cast<u32>(SEISMIC_DIM_K);
    const u32 b = static_cast<u32>(SEISMIC_DIM_B);
    const int *picks = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_PICKS)) + row * SEISMIC_PICKS_STRIDE_0;
    const u64 pick_stride = SEISMIC_PICKS_STRIDE_1;
    const int tail = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COMPLETE))[row * SEISMIC_COMPLETE_STRIDE_0];
    u32 *bitmap = reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + row * SEISMIC_RESULT_0_STRIDE_0;
    int *units = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER)) + row * SEISMIC_RESULT_1_STRIDE_0;
    // The valid picks: the first `count`.
    u32 lo = 0, hi = k;
    while (lo < hi) {
        const u32 mid = (lo + hi) / 2;
        if (picks[mid * pick_stride] >= 0) lo = mid + 1;
        else hi = mid;
    }
    const u32 count = lo;
    if (threadIdx.x == 0)
        for (u32 slot = 0; slot <= k; ++slot)
            units[slot * SEISMIC_RESULT_1_STRIDE_1] =
                slot < count ? picks[slot * pick_stride] : (slot == count ? tail : -1);
    for (u32 word = threadIdx.x; word < b; word += blockDim.x) {
        const int first = static_cast<int>(word * 32);
        // The first pick at or past the word's first unit.
        u32 at = 0, end = count;
        while (at < end) {
            const u32 mid = (at + end) / 2;
            if (picks[mid * pick_stride] < first) at = mid + 1;
            else end = mid;
        }
        u32 bits = 0;
        for (; at < count && picks[at * pick_stride] < first + 32; ++at)
            bits |= 1u << static_cast<u32>(picks[at * pick_stride] - first);
        // Every unit from the tail's on.
        if (tail <= first) bits = 0xffffffffu;
        else if (tail < first + 32) bits |= 0xffffffffu << static_cast<u32>(tail - first);
        bitmap[word * SEISMIC_RESULT_0_STRIDE_1] = bits;
    }
}
