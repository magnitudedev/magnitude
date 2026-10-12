// The row a reader of a gathered table reads (`gathered_row`,
// target.seismic). The counterpart of `metal/lib/core/gathered.h`,
// `vulkan/lib/core/gathered.glsl` and `cpu/lib/core/gathered.rs`.
//
// `selected` is a row of a (row, status) selection. Through an empty
// directory (`entries` 0) the reader reads it itself; else its slot in
// `directory` (`stride` elements apart), every row's slot or -1. A failed
// selection (-1) and a row without a slot read row 0.
//
// This file is independent of any entry ABI.

namespace gathered {

__device__ __forceinline__ unsigned long long row(const int *directory, unsigned long long stride,
                                                  unsigned long long entries, int selected) {
    int row = selected;
    if (entries != 0 && selected >= 0)
        row = directory[(unsigned long long)selected * stride];
    return (unsigned long long)max(row, 0);
}

} // namespace gathered
