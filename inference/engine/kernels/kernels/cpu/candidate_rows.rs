// candidate_rows on CPU (contract and portable body in indexer.seismic): one
// work item per row. Its taken units form ascending runs: before each picked
// block (ascending, clipped to the row's units) the fill takes the gap's
// lowest units while it lasts, then the block's own; after the last, the
// rest of the fill. Each run is mapped through the row's spans, split where
// a span ends, and adjacent runs within one span join.

use lib::core::spans;

#[inline(always)]
fn candidate_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (r, b, bu, k) = (cx.dim_r() as usize, cx.dim_b() as usize, cx.dim_bu() as usize, cx.dim_k() as usize);
    let bounds = cx.arg_spans();
    let list = (0..r)
        .map(|span| spans::span(bounds.get([row, span, 0]), bounds.get([row, span, 1])))
        .filter(|rows| !rows.is_empty())
        .map(|rows| (rows.start, rows.end))
        .collect::<Vec<_>>();
    let units = list.iter().map(|(lo, hi)| (hi - lo) as usize).sum::<usize>();
    let picks = cx.arg_blocks();
    let pooled = (0..b)
        .map(|pick| picks.get([row, pick]))
        .filter(|block| *block >= 0)
        .map(|block| ((block as usize * bu).min(units), ((block as usize + 1) * bu).min(units)))
        .collect::<Vec<_>>();
    let mut fill = k.min(units).saturating_sub(pooled.iter().map(|(start, end)| end - start).sum());
    let mut out = Pool { list: &list, spans: Vec::new(), span: 0, offset: 0, run: None };
    let mut cursor = 0;
    for (start, end) in pooled.into_iter().chain([(units, units)]) {
        assert!(start >= cursor, "the picked blocks ascend");
        let take = fill.min(start - cursor);
        out.take(cursor, cursor + take);
        fill -= take;
        out.take(start, end);
        cursor = end;
    }
    out.flush();
    // At most B + 1 runs (the fill takes every gap unit below its end, so
    // the units below it are one run, then one run per later block), split
    // only where one of the R list spans ends.
    assert!(out.spans.len() <= b + r, "the pool's runs exceed B + R spans");
    let result = cx.result_0();
    for entry in 0..b + r {
        let [lo, hi] = out.spans.get(entry).copied().unwrap_or([-1, -1]);
        // SAFETY: each work item writes its own row.
        unsafe {
            result.set([row, entry, 0], lo);
            result.set([row, entry, 1], hi);
        }
    }
}

/// A row's taken units as plane-row spans: the pending run (joined with an
/// adjacent next), mapped through the list's spans on flush.
struct Pool<'a> {
    list: &'a [(i32, i32)],
    spans: Vec<[i32; 2]>,
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
            let (lo, hi) = self.list[self.span];
            let length = (hi - lo) as usize;
            if self.offset + length <= start {
                self.offset += length;
                self.span += 1;
                continue;
            }
            let stop = end.min(self.offset + length);
            self.spans.push([lo + (start - self.offset) as i32, lo + (stop - self.offset) as i32]);
            start = stop;
        }
    }
}
