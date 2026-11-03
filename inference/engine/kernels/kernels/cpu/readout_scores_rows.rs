// readout_scores_rows on CPU (contract and portable body in readout.seismic):
// one work item per (key, query) pair, which widens both A rows to F32 and
// sums their products in column order.

fn readout_scores_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (key, row) = (group[0] as usize, group[1] as usize);
    let d = cx.dim_d() as usize;
    let (query, keys) = shared.split_at_mut(4 * d);
    let query = seismic::cpu::tensor::floats(query, d);
    let keys = seismic::cpu::tensor::floats(keys, d);
    cx.arg_query().decode_row(row, query);
    cx.arg_keys().decode_row(key, keys);
    let sum = query
        .iter()
        .zip(keys.iter())
        .fold(0.0_f32, |sum, (q, k)| q.mul_add(*k, sum));
    // SAFETY: each work item writes its own element.
    unsafe { cx.result_0().set([row, key], sum * cx.arg_scale()) };
}
