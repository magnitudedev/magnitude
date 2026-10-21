// corrector_nodes on CPU (contract and portable body in corrector.seismic): one
// work item per node (slot s, mask row j, candidate k), the sum of its token,
// pass, feature, slot and rank rows with their biases, in the portable body's
// order.

fn corrector_nodes<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let node = group[0] as usize;
    let (rows, count) = (cx.dim_j() as usize, cx.dim_k() as usize);
    let (slot, mask, candidate) = (node / (rows * count), node / count % rows, node % count);
    let token = cx.arg_token().row([slot, mask, candidate, 0]);
    let pass = cx.arg_pass().row([slot, mask, 0]);
    let pass_bias = cx.arg_pass_bias().row([0]);
    let feature = cx.arg_feature().row([slot, mask, candidate, 0]);
    let feature_bias = cx.arg_feature_bias().row([0]);
    let slot_row = cx.arg_slot().row([mask, 0]);
    let rank = cx.arg_rank().row([candidate, 0]);
    let token_bias = (cx.dim_tb() == 1).then(|| cx.arg_token_bias().row([0, 0]));
    // SAFETY: each work item writes its own node row.
    let out = unsafe { cx.result_0().row_mut([slot, mask, candidate, 0]) };
    for (column, target) in out.iter_mut().enumerate() {
        let mut value = token[column]
            + pass[column]
            + pass_bias[column]
            + feature[column]
            + feature_bias[column]
            + slot_row[column]
            + rank[column];
        if let Some(bias) = token_bias {
            value += bias[column];
        }
        *target = value;
    }
}
