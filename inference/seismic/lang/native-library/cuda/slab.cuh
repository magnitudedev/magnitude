// Address a component region in a slab-stored tensor, or a record of a
// record port. The table is a U32 [capacity, 2] tensor; each pair stores one
// little-endian 64-bit device pointer, already offset to this component's
// region, or to that record. The caller passes an allocated slot; a span uses
// `region` once, while a write uses `row`.
#pragma once

namespace slab {
using u32 = unsigned int;
using u64 = unsigned long long;
using u8 = unsigned char;

__device__ __forceinline__ u8 *region(const u32 *table, u64 slab_index) {
    const u64 address = (u64)table[2 * slab_index] | ((u64)table[2 * slab_index + 1] << 32);
    return reinterpret_cast<u8 *>(address);
}

// `region` of a table bound as a buffer's bytes (a record port's).
__device__ __forceinline__ u8 *region(const u8 *table, u64 slab_index) {
    return region(reinterpret_cast<const u32 *>(table), slab_index);
}

// Where an expert-stacked weight's matrix is read for one use (a routed
// choice's, a grouped block's): `code` >= 0 is matrix `code` of the primary
// stack, <= -2 matrix -2 - code of the secondary, and -1 none (the use reads
// no matrix and contributes nothing). The routed entries' `decode_source`.
struct Source {
    int code;
    __device__ __forceinline__ bool none() const { return code == -1; }
};

// An expert's matrix: matrix `record` of `stack` (a weight slot's
// `KERNEL_Wn_MATRIX(stack, record)`), or no stack for a source naming none.
struct Matrix {
    const u8 *stack;
    u64 record;
};

// An expert-stacked weight bound as two stacks of matrices: `primary` (a
// bank's records, a held layer's slots, a resident layer's experts) and
// `secondary` (the host tier's records, read in place over the link, or the
// experts' homes by index).
struct Experts {
    const u8 *primary;
    const u8 *secondary;
    __device__ __forceinline__ Matrix matrix(Source source) const {
        if (source.none())
            return Matrix{nullptr, 0};
        return source.code >= 0 ? Matrix{primary, (u64)source.code} : Matrix{secondary, (u64)(-2 - source.code)};
    }
};

__device__ __forceinline__ u8 *row(const u32 *table, u64 row_index, u64 rows_per_slab, u64 row_bytes) {
    return region(table, row_index / rows_per_slab)
        + (row_index % rows_per_slab) * row_bytes;
}
} // namespace slab
