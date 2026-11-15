// A launch's split of `rows` rows of `width` values, taken as one flattened
// sequence, into `spans` spans of equal length (ceil(rows * width / spans)
// values, the last shorter and any after it empty): every work item of a grid
// of the pool's participants takes one span, so each does the same work
// whatever the row count. A span's pieces are its parts within each row it
// touches; piece (span s, row r) has slot s + r, distinct for every piece (a
// span starting in a row after another's ends there), so per-piece partials
// need at most `spans + rows - 1` slots.

use std::ops::Range;

/// The flattened split of `rows` rows of `width` values into spans.
#[derive(Clone, Copy)]
pub struct Flat {
    width: usize,
    total: usize,
    span: usize,
}

/// One span's part of a row.
pub struct Piece {
    pub row: usize,
    /// The row's values the piece holds.
    pub columns: Range<usize>,
    /// Its partial's slot (`span + row`).
    pub slot: usize,
}

impl Flat {
    /// The split into `spans` spans; `None` when there are no values.
    #[inline(always)]
    pub fn new(rows: usize, width: usize, spans: usize) -> Option<Self> {
        let total = rows * width;
        assert!(spans > 0, "a grid of the pool's participants has one");
        (total > 0).then(|| Self { width, total, span: total.div_ceil(spans) })
    }

    /// The pieces of span `span`, in row order.
    #[inline(always)]
    pub fn pieces(self, span: usize) -> impl Iterator<Item = Piece> {
        let start = (span * self.span).min(self.total);
        let end = (start + self.span).min(self.total);
        let rows = if start < end { start / self.width..(end - 1) / self.width + 1 } else { 0..0 };
        rows.map(move |row| Piece {
            row,
            columns: start.max(row * self.width) - row * self.width..end.min((row + 1) * self.width) - row * self.width,
            slot: span + row,
        })
    }

    /// The spans that hold values of row `row`, in order; a row's piece of
    /// span `s` has slot `s + row`.
    #[inline(always)]
    pub fn spans_of(self, row: usize) -> Range<usize> {
        row * self.width / self.span..((row + 1) * self.width - 1) / self.span + 1
    }
}
