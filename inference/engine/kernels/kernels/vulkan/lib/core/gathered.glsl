// The row a reader of a gathered table reads (`gathered_row`,
// target.seismic). The counterpart of `metal/lib/core/gathered.h`,
// `cuda/lib/core/gathered.cuh` and `cpu/lib/core/gathered.rs`.
//
// `selected` is a row of a (row, status) selection. Through an empty
// directory (`entries` 0) the reader reads it itself. Else the directory,
// `entries` rows of `fields` I32 (`stride` elements apart, fields adjacent),
// is dense (`fields` 1: every table row's slot or -1) or hashed (`fields` 2:
// `entries` linearly probed (row, slot) buckets, a power of two; row -1 is
// empty and ends a probe, -2 a tombstone a probe passes). A failed selection (-1) and a row without a slot read row 0.
//
// Reads through <seismic/element.glsl>, which the including entry provides.

// murmur3's 32-bit finalizer of `row`, masked to `buckets` (a power of two).
uint gathered_bucket(int row, uint buckets) {
    uint h = uint(row);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h & (buckets - 1u);
}

uint64_t gathered_row(uint64_t directory, uint64_t stride, uint64_t entries, uint64_t fields,
    int selected) {
    if (selected < 0)
        return 0ul;
    if (entries == 0ul)
        return uint64_t(selected);
    if (fields == 1ul)
        return uint64_t(max(element_i32_index(directory, uint64_t(selected) * stride), 0));
    const uint mask = uint(entries) - 1u;
    uint probe = gathered_bucket(selected, uint(entries));
    for (uint64_t visited = 0ul; visited < entries; ++visited, probe = (probe + 1u) & mask) {
        const int key = element_i32_index(directory, uint64_t(probe) * stride);
        if (key == selected)
            return uint64_t(max(element_i32_index(directory, uint64_t(probe) * stride + 1ul), 0));
        if (key == -1)
            break;
    }
    return 0ul;
}
