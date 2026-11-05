// The row a reader of a gathered table reads (`gathered_row`,
// target.seismic). The counterpart of `metal/lib/core/gathered.h`,
// `vulkan/lib/core/gathered.glsl` and `cpu/lib/core/gathered.rs`.
//
// `selected` is a row of a (row, status) selection. Through an empty
// directory (`entries` 0) the reader reads it itself. Else the directory,
// `entries` rows of `fields` I32 (`stride` elements apart, fields adjacent),
// is dense (`fields` 1: every table row's slot or -1) or hashed (`fields` 2:
// `entries` linearly probed (row, slot) buckets, a power of two; row -1 is
// empty and ends a probe, -2 a tombstone a probe passes). A selection that
// names no row (-1) and a row without a slot give -1: the reader reads zeros.
//
// This file is independent of any entry ABI.

namespace gathered {

// murmur3's 32-bit finalizer of `row`, masked to `buckets` (a power of two).
__device__ __forceinline__ unsigned bucket(int row, unsigned buckets) {
    unsigned h = (unsigned)row;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h & (buckets - 1u);
}

__device__ __forceinline__ long long row(const int *directory, unsigned long long stride,
                                         unsigned long long entries, unsigned long long fields,
                                         int selected) {
    if (selected < 0)
        return -1ll;
    if (entries == 0)
        return (long long)selected;
    if (fields == 1)
        return (long long)directory[(unsigned long long)selected * stride];
    const unsigned mask = (unsigned)entries - 1u;
    unsigned probe = bucket(selected, (unsigned)entries);
    for (unsigned long long visited = 0; visited < entries; ++visited, probe = (probe + 1u) & mask) {
        const int key = directory[(unsigned long long)probe * stride];
        if (key == selected)
            return (long long)directory[(unsigned long long)probe * stride + 1];
        if (key == -1)
            break;
    }
    return -1ll;
}

} // namespace gathered
