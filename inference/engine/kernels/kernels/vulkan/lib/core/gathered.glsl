// The row a reader of a gathered table reads (`gathered_row`,
// target.seismic). The counterpart of `metal/lib/core/gathered.h`,
// `cuda/lib/core/gathered.cuh` and `cpu/lib/core/gathered.rs`.
//
// `selected` is a row of a (row, status) selection. Through an empty
// directory (`entries` 0) the reader reads it itself; else its slot in
// `directory` (`stride` elements apart), every row's slot or -1. A failed
// selection (-1) and a row without a slot read row 0.
//
// Reads through <seismic/element.glsl>, which the including entry provides.

uint64_t gathered_row(uint64_t directory, uint64_t stride, uint64_t entries, int selected) {
    int row = selected;
    if (entries != 0ul && selected >= 0)
        row = element_i32_index(directory, uint64_t(selected) * stride);
    return uint64_t(max(row, 0));
}
