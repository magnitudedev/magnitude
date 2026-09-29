// draft_top_k on CPU (contract and portable body in dflash.seismic): one work
// item per row. Round r scans the tokens ordered strictly after round r - 1's
// choice in (value descending, token ascending) order.

fn draft_top_k<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let count = cx.dim_k() as usize;
    let line = cx.arg_logits().row([row, 0]);
    let (mut previous, mut previous_token) = (f32::INFINITY, -1i64);
    for rank in 0..count {
        let (mut best, mut best_token) = (f32::NEG_INFINITY, i64::MAX);
        for (token, &value) in line.iter().enumerate() {
            let token = token as i64;
            let after = value < previous || (value == previous && token > previous_token);
            if after && (value > best || (value == best && token < best_token)) {
                best = value;
                best_token = token;
            }
        }
        (previous, previous_token) = (best, best_token);
        let at = row * count + rank;
        // SAFETY: each work item writes its own rows.
        unsafe {
            cx.arg_candidates().set([at, 0], best_token as i32);
            cx.arg_candidates().set([at, 1], 0);
            cx.arg_unary().set([row, rank], best);
        }
    }
}
