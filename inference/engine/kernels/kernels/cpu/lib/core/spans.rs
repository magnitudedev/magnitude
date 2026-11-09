// A row's span list (`[R, 2]` I32 bounds), as the portable bodies walk it
// (indexer.seismic, pooled.seismic): span (first, last) lists the rows
// first..last when `first >= 0 and last >= first`, and counts toward the
// row's total when `last > first`. The two agree exactly when every span
// holding rows starts at a row (first >= 0), which every producer ensures, so
// a span holding rows below row 0 is an invariant violation.

use std::ops::Range;

/// The rows the span (first, last) lists.
#[inline(always)]
pub fn span(first: i32, last: i32) -> Range<i32> {
    assert!(first >= 0 || last <= first, "the span {first}..{last} holds rows below row 0");
    first..last.max(first)
}

/// The rows of a row's `count` spans in order, span `s`'s bounds
/// `bounds(s, 0)` and `bounds(s, 1)`.
#[inline(always)]
pub fn rows(bounds: &impl Fn(usize, usize) -> i32, count: usize) -> impl Iterator<Item = i32> + '_ {
    (0..count).flat_map(move |s| span(bounds(s, 0), bounds(s, 1)))
}
