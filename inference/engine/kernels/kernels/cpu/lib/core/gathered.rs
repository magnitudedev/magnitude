// The row a reader of a gathered table reads (`gathered_row`,
// target.seismic). The counterpart of `metal/lib/core/gathered.h`,
// `cuda/lib/core/gathered.cuh` and `vulkan/lib/core/gathered.glsl`.
//
// `selected` is a row of a (row, status) selection. Through an empty
// directory (`entries` 0) the reader reads it itself. Else the directory,
// `entries` rows of `fields` I32 read by `entry(row, field)`, is dense
// (`fields` 1: every table row's slot or -1) or hashed (`fields` 2: `entries`
// linearly probed (row, slot) buckets, a power of two; row -1 is empty and
// ends a probe, -2 a tombstone a probe passes). A failed selection (-1) and a
// row without a slot read row 0.

/// murmur3's 32-bit finalizer of `row`, masked to `buckets` (a power of two).
#[inline(always)]
pub fn bucket(row: i32, buckets: u32) -> u32 {
    let mut h = row as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h & buckets.wrapping_sub(1)
}

#[inline(always)]
pub fn row(selected: i32, entries: u64, fields: u64, entry: impl Fn(usize, usize) -> i32) -> usize {
    let Ok(index) = usize::try_from(selected) else {
        return 0;
    };
    if entries == 0 {
        return index;
    }
    if fields == 1 {
        return entry(index, 0).max(0) as usize;
    }
    let mask = (entries as u32).wrapping_sub(1);
    let mut probe = bucket(selected, entries as u32);
    for _ in 0..entries {
        let key = entry(probe as usize, 0);
        if key == selected {
            return entry(probe as usize, 1).max(0) as usize;
        }
        if key == -1 {
            break;
        }
        probe = probe.wrapping_add(1) & mask;
    }
    0
}
