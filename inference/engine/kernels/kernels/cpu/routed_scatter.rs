// routed_scatter on CPU (contract and portable body in routed.seismic): one
// work item per row unpermutes the row's grouped expert outputs (each decoded
// as F32 into its shared bytes; row b * T + t of the [B, T, H] outputs is
// position b * T + t), weights them in slot order and publishes
// base + selected.

fn routed_scatter<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (h, k) = (cx.dim_h() as usize, cx.dim_k() as usize);
    let (inverse, weights, expert_output) = (cx.arg_inverse(), cx.arg_weights(), cx.arg_expert_output());
    let (selected, published) = shared.split_at_mut(4 * h);
    let selected = seismic::cpu::tensor::floats(selected, h);
    let published = seismic::cpu::tensor::floats(published, h);
    selected.fill(0.0);
    for slot in 0..k {
        let position = usize::try_from(inverse.get([row, slot])).expect("a grouped position is non-negative");
        expert_output.decode_row(position, published);
        let weight = weights.get([row, slot]);
        for (selected, published) in selected.iter_mut().zip(published.iter()) {
            *selected = weight.mul_add(*published, *selected);
        }
    }
    let base = cx.arg_base().row([row, 0]);
    // SAFETY: each work item writes its own row.
    let out = unsafe { cx.result_0().row_mut([row, 0]) };
    for ((target, base), selected) in out.iter_mut().zip(base).zip(selected.iter()) {
        *target = E::R::narrow(base + selected);
    }
}
