// corrector_path on CPU (contract and portable body in corrector.seismic): one
// work item per slot walks its lattice greedily, each mask row selecting the
// first best-scoring candidate after the previous row's selection.

fn corrector_path<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let slot = group[0] as usize;
    let (count, steps) = (cx.dim_k() as usize, cx.dim_p() as usize);
    let (incoming, outgoing, anchor, candidates) = (cx.arg_incoming(), cx.arg_outgoing(), cx.arg_anchor(), cx.arg_candidates());
    let mut previous = 0usize;
    for step in 0..steps {
        let other = if step == 0 {
            anchor.row([slot, 0])
        } else {
            outgoing.row([slot, step - 1, previous, 0])
        };
        let (mut best, mut chosen) = (f32::NEG_INFINITY, 0usize);
        for candidate in 0..count {
            let score = incoming
                .row([slot, step, candidate, 0])
                .iter()
                .zip(other)
                .map(|(a, b)| a * b)
                .sum::<f32>();
            if score > best {
                best = score;
                chosen = candidate;
            }
        }
        previous = chosen;
        // SAFETY: each work item writes its own slot's selections.
        unsafe {
            cx.arg_selection().set([step, slot, 0], candidates.get([slot, step, chosen, 0]));
            cx.arg_selection().set([step, slot, 1], 0);
        }
    }
}
