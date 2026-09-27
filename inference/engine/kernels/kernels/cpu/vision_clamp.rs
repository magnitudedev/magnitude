// vision_clamp on CPU (contract and portable body in vision.seismic): one work
// item per row, each value clamped to [minimum, maximum] and published to A.

fn vision_clamp<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let n = cx.dim_n() as usize;
    let (minimum, maximum) = (cx.arg_minimum().get([0]), cx.arg_maximum().get([0]));
    let values = seismic::cpu::tensor::floats(shared, n);
    seismic::cpu::tensor::widen_row::<E::A>(cx.arg_x().row([row, 0]), values);
    let result = cx.result_0();
    // SAFETY: each work item writes its own row.
    let out = unsafe { result.span_mut([row, 0], n) };
    for (target, value) in out.iter_mut().zip(values.iter()) {
        *target = E::A::narrow(value.max(minimum).min(maximum));
    }
}
