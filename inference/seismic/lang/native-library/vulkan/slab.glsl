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

// A floating slab-stored tensor a kernel mutates (`&slab mut`), as the kernel writes it: the slab
// address table and the tensor's element kind, from the ABI macro `SEISMIC_OUT_<NAME>`. Its
// regions and rows are reached only as `Out`.
struct SlabOut {
    uint64_t seismic_table;
    int seismic_kind;
};

Out slab_out_region(SlabOut slab, uint64_t slab_index) {
    return Out(slab_region(slab.seismic_table, slab_index), slab.seismic_kind);
}
Out slab_out_row(SlabOut slab, uint64_t row_index, uint64_t rows_per_slab, uint64_t row_elements) {
    return Out(slab_row(slab.seismic_table, row_index, rows_per_slab,
        row_elements * uint64_t(ELEMENT_BYTES(slab.seismic_kind))), slab.seismic_kind);
}
// The 32-bit form (row maps use i32 indices and slab counts u32), as `slab_row`'s.
Out slab_out_row(SlabOut slab, uint row_index, uint rows_per_slab, uint64_t row_elements) {
    return Out(slab_row(slab.seismic_table, row_index, rows_per_slab,
        row_elements * uint64_t(ELEMENT_BYTES(slab.seismic_kind))), slab.seismic_kind);
}
