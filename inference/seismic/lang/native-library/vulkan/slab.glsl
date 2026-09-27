// Address a component region in a slab-stored tensor. The table is a U32
// [capacity, 2] tensor; each pair stores one little-endian 64-bit device
// address, already offset to this component's region. The caller passes an
// allocated slot; a span uses `slab_region` once, while a write uses `slab_row`.
#include <seismic/element.glsl>

uint64_t slab_region(uint64_t table, uint64_t slab_index) {
    const uint64_t pair = table + slab_index * 8ul;
    const uint64_t address = uint64_t(element_u32_at(pair))
        | (uint64_t(element_u32_at(pair + 4ul)) << 32);
    return address;
}

uint64_t slab_row(uint64_t table, uint64_t row_index, uint64_t rows_per_slab,
    uint64_t row_bytes) {
    return slab_region(table, row_index / rows_per_slab)
        + (row_index % rows_per_slab) * row_bytes;
}

// Row maps use i32 indices and slab counts use u32. Keep their quotient and
// remainder in that domain before widening the byte address.
uint64_t slab_row(uint64_t table, uint row_index, uint rows_per_slab,
    uint64_t row_bytes) {
    return slab_region(table, uint64_t(row_index / rows_per_slab))
        + uint64_t(row_index % rows_per_slab) * row_bytes;
}
