// post_norm_residual on CPU (contract and portable body in
// residual.seismic): one work item per output row, which normalizes its
// projected row and adds it to its `out_rows` residual row, all F32.

use lib::core::reduce;

fn post_norm_residual<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    let norm = seismic::cpu::tensor::floats(shared, d);
    cx.arg_norm().decode_row(0, norm);
    let projected = cx.arg_projected().row([row, 0]);
    let source = cx.arg_out_rows().get([row]) as usize;
    let residual = cx.arg_residual().row([source, 0]);
    let inverse = reduce::rms_inverse(projected, cx.arg_epsilon());
    let scale = cx.arg_scale();
    // SAFETY: each work item writes its own row.
    let out = unsafe { cx.result_0().row_mut([row, 0]) };
    for (((target, r), p), w) in out.iter_mut().zip(residual).zip(projected).zip(norm.iter()) {
        *target = (r + p * inverse * w) * scale;
    }
}
