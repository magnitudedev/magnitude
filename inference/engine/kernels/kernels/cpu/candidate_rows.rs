// candidate_rows on CPU (contract and portable body in indexer.seismic): one
// work item per row. Its taken units form ascending runs: before each picked
// block (ascending, clipped to the row's units) the fill takes the gap's
// lowest units while it lasts, then the block's own; after the last, the
// rest of the fill. Each run is mapped through the row's spans, split where
// a span ends, and adjacent runs within one span join.

use lib::core::spans;
use seismic::cpu::Scalars;

#[inline(always)]
fn candidate_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (r, b, bu, k) = (cx.dim_r() as usize, cx.dim_b() as usize, cx.dim_bu() as usize, cx.dim_k() as usize);
    let bounds = cx.arg_spans();
    let mut units = 0usize;
    for span in 0..r {
        let rows = spans::span(bounds.get([row, span, 0]), bounds.get([row, span, 1]));
        units += (rows.end - rows.start) as usize;
    }
    let picks = cx.arg_blocks();
    let mut pooled_units = 0usize;
    for pick in 0..b {
        let block = picks.get([row, pick]);
        if block >= 0 {
            let block = block as usize;
            let start = (block * bu).min(units);
            let end = ((block + 1) * bu).min(units);
            pooled_units += end - start;
        }
    }
    let mut fill = k.min(units).saturating_sub(pooled_units);
    let result = cx.result_0();
    let mut out = Pool { bounds, result, row, limit: b + r, entry: 0, span: 0, offset: 0, run: None };
    let mut cursor = 0usize;
    for pick in 0..b {
        let block = picks.get([row, pick]);
        if block < 0 {
            continue;
        }
        let block = block as usize;
        let start = (block * bu).min(units);
        let end = ((block + 1) * bu).min(units);
        assert!(start >= cursor, "the picked blocks ascend");
        let take = fill.min(start - cursor);
        out.take(cursor, cursor + take);
        fill -= take;
        out.take(start, end);
        cursor = end;
    }
    assert!(units >= cursor, "the picked blocks ascend");
    let take = fill.min(units - cursor);
    out.take(cursor, cursor + take);
    out.flush();
    for entry in out.entry..b + r {
        // SAFETY: each work item writes its own row.
        unsafe {
            result.set([row, entry, 0], -1);
            result.set([row, entry, 1], -1);
        }
    }
}

/// A row's taken units as plane-row spans: the pending run (joined with an
/// adjacent next), mapped through the list's spans and written directly to
/// `result` on flush.
struct Pool<'a> {
    bounds: Scalars<'a, i32, 3>,
    result: Scalars<'a, i32, 3>,
    row: usize,
    /// At most B + 1 runs (the fill takes every gap unit below its end, so
    /// the units below it are one run, then one run per later block), split
    /// only where one of the R list spans ends.
    limit: usize,
    entry: usize,
    /// The list span the next run starts in, and the units before it.
    span: usize,
    offset: usize,
    run: Option<(usize, usize)>,
}

impl Pool<'_> {
    /// Takes the units [start, end), ascending after every earlier run.
    #[inline(always)]
    fn take(&mut self, start: usize, end: usize) {
        if start == end {
            return;
        }
        match &mut self.run {
            Some((_, last)) if *last == start => *last = end,
            _ => {
                self.flush();
                self.run = Some((start, end));
            }
        }
    }

    #[inline(always)]
    fn flush(&mut self) {
        let Some((mut start, end)) = self.run.take() else {
            return;
        };
        while start < end {
            let rows = spans::span(
                self.bounds.get([self.row, self.span, 0]),
                self.bounds.get([self.row, self.span, 1]),
            );
            let length = (rows.end - rows.start) as usize;
            if self.offset + length <= start {
                self.offset += length;
                self.span += 1;
                continue;
            }
            let stop = end.min(self.offset + length);
            assert!(self.entry < self.limit, "the pool's runs exceed B + R spans");
            let lo = rows.start + (start - self.offset) as i32;
            let hi = rows.start + (stop - self.offset) as i32;
            // SAFETY: each work item writes its own row.
            unsafe {
                self.result.set([self.row, self.entry, 0], lo);
                self.result.set([self.row, self.entry, 1], hi);
            }
            self.entry += 1;
            start = stop;
            if stop == self.offset + length {
                self.offset += length;
                self.span += 1;
            }
        }
    }
}
