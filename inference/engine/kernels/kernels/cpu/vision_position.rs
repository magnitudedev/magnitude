// vision_position on CPU (contract and portable body in vision.seismic): one
// work item per row, the row plus the blend of its four position-table rows
// (in corner order), all F32.

fn vision_position<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    let (table, indices, coefficients) = (cx.arg_table(), cx.arg_indices(), cx.arg_coefficients());
    let shared = seismic::cpu::tensor::floats(shared, 2 * h);
    let (decoded, blended) = shared.split_at_mut(h);
    blended.fill(0.0);
    for corner in 0..4 {
        table.decode_row(indices.get([row, corner]) as usize, decoded);
        let coefficient = coefficients.get([row, corner]);
        for (target, value) in blended.iter_mut().zip(decoded.iter()) {
            *target += value * coefficient;
        }
    }
    let source = cx.arg_source().row([row, 0]);
    let result = cx.result_0();
    // SAFETY: each work item writes its own row.
    let out = unsafe { result.span_mut([row, 0], h) };
    for ((target, value), blended) in out.iter_mut().zip(source).zip(blended.iter()) {
        *target = value + blended;
    }
}
