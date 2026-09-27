// moe_tail on CPU (contract and portable body in residual.seismic): one work
// item per row, which normalizes both branch rows into the combined row, then
// normalizes that and adds it to the residual row, all F32.

use lib::core::reduce;

fn moe_tail<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    let (combined, norms) = shared.split_at_mut(4 * d);
    let combined = seismic::cpu::tensor::floats(combined, d);
    let (dense_norm, norms) = norms.split_at_mut(4 * d);
    let (routed_norm, norm) = norms.split_at_mut(4 * d);
    let (dense_norm, routed_norm, norm) = (
        seismic::cpu::tensor::floats(dense_norm, d),
        seismic::cpu::tensor::floats(routed_norm, d),
        seismic::cpu::tensor::floats(norm, d),
    );
    cx.arg_dense_norm().decode_row(0, dense_norm);
    cx.arg_routed_norm().decode_row(0, routed_norm);
    cx.arg_norm().decode_row(0, norm);
    let epsilon = cx.arg_epsilon();
    let dense = cx.arg_dense().row([row, 0]);
    let routed = cx.arg_routed().row([row, 0]);
    let dense_inverse = reduce::rms_inverse(dense, epsilon);
    let routed_inverse = reduce::rms_inverse(routed, epsilon);
    for i in 0..d {
        combined[i] = dense[i] * dense_inverse * dense_norm[i] + routed[i] * routed_inverse * routed_norm[i];
    }
    let inverse = reduce::rms_inverse(combined, epsilon);
    let scale = cx.arg_scale();
    let residual = cx.arg_residual().row([row, 0]);
    // SAFETY: each work item writes its own row.
    let out = unsafe { cx.result_0().row_mut([row, 0]) };
    for i in 0..d {
        out[i] = (residual[i] + combined[i] * inverse * norm[i]) * scale;
    }
}
