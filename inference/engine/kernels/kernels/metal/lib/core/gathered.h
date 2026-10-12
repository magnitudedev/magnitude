// The row a reader of a gathered table reads (`gathered_row`,
// target.seismic). The counterpart of `cuda/lib/core/gathered.cuh`,
// `vulkan/lib/core/gathered.glsl` and `cpu/lib/core/gathered.rs`.
//
// `selected` is a row of a (row, status) selection. Through an empty
// directory (`entries` 0) the reader reads it itself; else its slot in
// `directory` (`stride` elements apart), every row's slot or -1. A failed
// selection (-1) and a row without a slot read row 0.
//
// This file is independent of any entry ABI.

namespace gathered {

inline ulong row(device const int *directory, ulong stride, ulong entries, int selected) {
    int row = selected;
    if (entries != 0 && selected >= 0)
        row = directory[ulong(selected) * stride];
    return ulong(metal::max(row, 0));
}

} // namespace gathered
