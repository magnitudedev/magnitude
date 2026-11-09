// rows_topk on CPU (contract and portable body in indexer.seismic): one work
// item per row. A dense row (complete units at most K) reads the first K
// rows of its list; a selecting row partially sorts its scored ordinals
// (score descending, ordinal ascending), keeps the first K and emits their
// rows in ordinal order; then -1 to K.

fn rows_topk<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (n, r, k) = (cx.dim_n() as usize, cx.dim_r() as usize, cx.dim_k() as usize);
    let spans = cx.arg_spans();
    let list = (0..r)
        .flat_map(|span| {
            let (lo, hi) = (spans.get([row, span, 0]), spans.get([row, span, 1]));
            if lo >= 0 && hi >= lo { lo..hi } else { 0..0 }
        })
        .take(n)
        .collect::<Vec<_>>();
    let picks = if i64::from(cx.arg_complete().get([row])) <= k as i64 || list.len() <= k || k == 0 {
        list.into_iter().take(k).collect::<Vec<_>>()
    } else {
        let scores = cx.arg_scores().row([row, 0]);
        let mut order = (0..list.len()).collect::<Vec<_>>();
        // `-0.0` and `0.0` compare equal, as their order keys are.
        let better = |left: &usize, right: &usize| {
            scores[*right].partial_cmp(&scores[*left]).expect("scores are not NaN").then(left.cmp(right))
        };
        order.select_nth_unstable_by(k - 1, better);
        let mut chosen = order[..k].to_vec();
        chosen.sort_unstable();
        chosen.into_iter().map(|ordinal| list[ordinal]).collect()
    };
    let selected = cx.result_0();
    for pick in 0..k {
        // SAFETY: each work item writes its own row.
        unsafe { selected.set([row, pick], picks.get(pick).copied().unwrap_or(-1)) };
    }
}
