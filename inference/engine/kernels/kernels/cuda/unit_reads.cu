// unit_reads on CUDA (contract and portable body in indexer.seismic), as
// Metal's (`metal/unit_reads.metal`): a block of 256 threads per row. Thread 0
// finds the count of valid picks by binary search and broadcasts it through
// shared memory; when `count <= 2048`, the block stages the valid picks into
// shared memory once so both `units` writes and per-word bitmap binary
// searches read single-cycle shared memory instead of uncoalesced DRAM.
//
// Dynamic extents and strides read `seismic_words`.

typedef unsigned int u32;
typedef unsigned long long u64;

template <class ReadPick>
__device__ __forceinline__ void emit_row(u32 k, u32 b, u32 count, int tail, u32 *bitmap, u64 bitmap_stride, int *units,
                                         u64 units_stride, const ReadPick &read_pick) {
    for (u32 slot = threadIdx.x; slot <= k; slot += blockDim.x)
        units[slot * units_stride] = slot < count ? read_pick(slot) : (slot == count ? tail : -1);
    for (u32 word = threadIdx.x; word < b; word += blockDim.x) {
        const int first = static_cast<int>(word * 32);
        // The first pick at or past the word's first unit.
        u32 at = 0, end = count;
        while (at < end) {
            const u32 mid = (at + end) / 2;
            if (read_pick(mid) < first) at = mid + 1;
            else end = mid;
        }
        u32 bits = 0;
        for (; at < count; ++at) {
            const int pick = read_pick(at);
            if (pick >= first + 32) break;
            bits |= 1u << static_cast<u32>(pick - first);
        }
        // Every unit from the tail's on.
        if (tail <= first) bits = 0xffffffffu;
        else if (tail < first + 32) bits |= 0xffffffffu << static_cast<u32>(tail - first);
        bitmap[word * bitmap_stride] = bits;
    }
}

extern "C" __global__ void unit_reads(SEISMIC_KERNEL_PARAMS) {
    constexpr u32 CACHED_PICKS_CAP = 2048;
    __shared__ u32 shared_count;
    __shared__ int cached_picks[CACHED_PICKS_CAP];
    const u64 row = blockIdx.x;
    const u32 k = static_cast<u32>(SEISMIC_DIM_K);
    const u32 b = static_cast<u32>(SEISMIC_DIM_B);
    const int *picks = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_PICKS)) + row * SEISMIC_PICKS_STRIDE_0;
    const u64 pick_stride = SEISMIC_PICKS_STRIDE_1;
    const int tail = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COMPLETE))[row * SEISMIC_COMPLETE_STRIDE_0];
    u32 *bitmap = reinterpret_cast<u32 *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + row * SEISMIC_RESULT_0_STRIDE_0;
    const u64 bitmap_stride = SEISMIC_RESULT_0_STRIDE_1;
    int *units = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER)) + row * SEISMIC_RESULT_1_STRIDE_0;
    const u64 units_stride = SEISMIC_RESULT_1_STRIDE_1;
    // The valid picks: the first `count`, found once by thread 0 and broadcast.
    if (threadIdx.x == 0) {
        u32 lo = 0, hi = k;
        while (lo < hi) {
            const u32 mid = (lo + hi) / 2;
            if (picks[mid * pick_stride] >= 0) lo = mid + 1;
            else hi = mid;
        }
        shared_count = lo;
    }
    __syncthreads();
    const u32 count = shared_count;
    if (count <= CACHED_PICKS_CAP) {
        for (u32 slot = threadIdx.x; slot < count; slot += blockDim.x)
            cached_picks[slot] = picks[slot * pick_stride];
        __syncthreads();
        emit_row(k, b, count, tail, bitmap, bitmap_stride, units, units_stride,
                 [&](u32 idx) { return cached_picks[idx]; });
    } else {
        emit_row(k, b, count, tail, bitmap, bitmap_stride, units, units_stride,
                 [&](u32 idx) { return picks[idx * pick_stride]; });
    }
}
