// The row a reader of a gathered table reads (`gathered_row`,
// target.seismic). The counterpart of `metal/lib/core/gathered.h`,
// `cuda/lib/core/gathered.cuh` and `vulkan/lib/core/gathered.glsl`.
//
// `selected` is a row of a (row, status) selection. Through an empty
// directory (`entries` 0) the reader reads it itself; else its slot,
// `slot(selected)`, of every row's slot or -1. A failed selection (-1) and a
// row without a slot read row 0.

#[inline(always)]
pub fn row(selected: i32, entries: u64, slot: impl FnOnce(usize) -> i32) -> usize {
    let row = match usize::try_from(selected) {
        Ok(selected) if entries != 0 => slot(selected),
        _ => selected,
    };
    row.max(0) as usize
}
