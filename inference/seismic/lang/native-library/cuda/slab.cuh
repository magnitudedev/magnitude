// Address a component region in a slab-stored tensor. The table is a U32
// [capacity, 2] tensor; each pair stores one little-endian 64-bit device
// pointer, already offset to this component's region. The caller passes an
// allocated slot; a span uses `region` once, while a write uses `row`.
#pragma once

namespace slab {
using u32 = unsigned int;
using u64 = unsigned long long;
using u8 = unsigned char;

__device__ __forceinline__ u8 *region(const u32 *table, u64 slab_index) {
    const u64 address = (u64)table[2 * slab_index] | ((u64)table[2 * slab_index + 1] << 32);
    return reinterpret_cast<u8 *>(address);
}

__device__ __forceinline__ u8 *row(const u32 *table, u64 row_index, u64 rows_per_slab, u64 row_bytes) {
    return region(table, row_index / rows_per_slab)
        + (row_index % rows_per_slab) * row_bytes;
}
} // namespace slab
