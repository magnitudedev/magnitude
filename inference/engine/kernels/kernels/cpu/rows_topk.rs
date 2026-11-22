// rows_topk on CPU (contract and portable body in indexer.seismic): one work
// item per row. A dense row (complete units or scored rows at most K) streams
// the first K rows of its spans directly with no heap allocation; a selecting
// row packs (!ordered_key(score), ordinal) into a single Vec<u64>, partially
// sorts the K smallest entries (score descending, ordinal ascending), sorts
// those K entries by ordinal, and resolves them in one linear walk over its
// spans; then -1 to K. Scores rank by their order keys
// (`functions::ordered_key`), as the Metal and CUDA forms rank them.

use lib::core::{functions, spans};

#[inline(always)]
fn rows_topk<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (n, r, k) = (cx.dim_n() as usize, cx.dim_r() as usize, cx.dim_k() as usize);
    if k == 0 {
        return;
    }
    let bounds = cx.arg_spans();
    let mut total_len = 0usize;
    for span in 0..r {
        let range = spans::span(bounds.get([row, span, 0]), bounds.get([row, span, 1]));
        total_len = total_len.saturating_add((range.end - range.start) as usize);
    }
    let scored = total_len.min(n);
    let selected = cx.result_0();
    if i64::from(cx.arg_complete().get([row])) <= k as i64 || scored <= k {
        let mut pick = 0;
        for key_row in spans::rows(&|span, bound| bounds.get([row, span, bound]), r).take(scored.min(k)) {
            // SAFETY: each work item writes its own row.
            unsafe { selected.set([row, pick], key_row) };
            pick += 1;
        }
        for slot in pick..k {
            // SAFETY: each work item writes its own row.
            unsafe { selected.set([row, slot], -1) };
        }
        return;
    }
    assert!(scored <= u32::MAX as usize, "scored ordinals exceed u32::MAX");
    let mut packed = cx.arg_scores().row([row, 0])[..scored]
        .iter()
        .enumerate()
        .map(|(ordinal, &score)| (u64::from(!functions::ordered_key(score)) << 32) | ordinal as u64)
        .collect::<Vec<u64>>();
    packed.select_nth_unstable(k - 1);
    let chosen = &mut packed[..k];
    chosen.sort_unstable_by_key(|&entry| entry as u32);
    let mut pick = 0;
    let mut span_start_ordinal = 0usize;
    for span in 0..r {
        let range = spans::span(bounds.get([row, span, 0]), bounds.get([row, span, 1]));
        let span_end_ordinal = span_start_ordinal + (range.end - range.start) as usize;
        while pick < k {
            let ordinal = (chosen[pick] as u32) as usize;
            if ordinal >= span_end_ordinal {
                break;
            }
            let key_row = range.start + (ordinal - span_start_ordinal) as i32;
            // SAFETY: each work item writes its own row.
            unsafe { selected.set([row, pick], key_row) };
            pick += 1;
        }
        if pick == k {
            break;
        }
        span_start_ordinal = span_end_ordinal;
    }
    debug_assert_eq!(pick, k);
}
