// draft_path_step on CPU (contract and portable body in dflash.seismic): one
// work item per row scores its candidates in order and selects the first
// highest-scoring one.

fn draft_path_step<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (count, rank) = (cx.dim_k() as usize, cx.dim_r() as usize);
    let values = seismic::cpu::tensor::floats(shared, 3 * rank);
    let (joint, rest) = values.split_at_mut(rank);
    let (hidden, successor) = rest.split_at_mut(rank);
    cx.arg_predecessor().decode_row(row, joint);
    cx.arg_hidden().decode_row(row, hidden);
    for (value, hidden) in joint.iter_mut().zip(hidden.iter()) {
        *value *= hidden;
    }
    let (mut best, mut chosen) = (f32::NEG_INFINITY, row * count);
    for candidate in 0..count {
        let at = row * count + candidate;
        cx.arg_successor().decode_row(at, successor);
        let score = joint
            .iter()
            .zip(successor.iter())
            .fold(0.0f32, |sum, (joint, successor)| joint.mul_add(*successor, sum))
            + cx.arg_unary().get([row, candidate]);
        if score > best {
            best = score;
            chosen = at;
        }
    }
    // SAFETY: each work item writes its own row.
    unsafe {
        cx.arg_selection().set([row, 0], cx.arg_candidates().get([chosen, 0]));
        cx.arg_selection().set([row, 1], 0);
    }
}
