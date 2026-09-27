// vision_linear on CPU (contract and portable body in vision.seismic).
// `vision_linear_stage` stages each input row (and its gate row) as F32, one
// work item per row. `vision_linear_rows` gives each work item eight weight
// rows, projected against every staged row: the F32 product plus the bias,
// clamped, then published as the portable body's epilogue computes it.

use lib::core::activation;
use lib::projection::projection;
use lib::vision::vision;

fn vision_linear_stage<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (k, n) = (cx.dim_k() as usize, cx.dim_n() as usize);
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * row * k, k) };
    seismic::cpu::tensor::widen_row::<E::A>(cx.arg_x().row([row, 0]), staged);
    if cx.dim_ng() == 1 {
        // SAFETY: each work item writes its own row of the scratch.
        let gate = unsafe { cx.scratch_gates().slice_mut::<f32>(4 * row * n, n) };
        seismic::cpu::tensor::widen_row::<E::A>(cx.arg_gate().row([0, row, 0]), gate);
    }
}

fn vision_linear_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (m, n, k) = (cx.dim_m() as usize, cx.dim_n() as usize, cx.dim_k() as usize);
    let rows = projection::item_rows(group[0], 8, n);
    let shared = seismic::cpu::tensor::floats(shared, n);
    let bias = (cx.dim_nb() == 1).then(|| vision::decode_vector(&cx.arg_bias(), shared));
    let bounds = (cx.dim_nc() == 1).then(|| (cx.arg_minimum().get([0]), cx.arg_maximum().get([0])));
    let code = cx.arg_activation();
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * k) };
    // SAFETY: the stage launch wrote every gate row before this launch.
    let gates = (cx.dim_ng() == 1).then(|| unsafe { cx.scratch_gates().slice::<f32>(0, m * n) });
    let (weight, residual, result) = (cx.arg_weight(), cx.arg_residual(), cx.result_0());
    let mut projected = [0.0f32; projection::MAX_ROWS];
    let projected = &mut projected[..rows.len()];
    for (row, x) in staged.chunks_exact(k).enumerate() {
        projection::project(&weight, rows.start, x, projected);
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { result.span_mut([row, rows.start], rows.len()) };
        for ((target, &product), output) in out.iter_mut().zip(projected.iter()).zip(rows.clone()) {
            let mut value = product;
            if let Some(bias) = bias {
                value += bias[output];
            }
            if let Some((minimum, maximum)) = bounds {
                value = value.max(minimum).min(maximum);
            }
            if code != 0 {
                value = vision::activate(code, activation::publish::<E::A>(value));
            } else if let Some(gates) = gates {
                value = activation::publish::<E::A>(value) * gates[row * n + output];
            } else if cx.dim_nr() == 1 {
                value = residual.get([0, row, output]) + value;
            }
            *target = E::Y::narrow(value);
        }
    }
}
