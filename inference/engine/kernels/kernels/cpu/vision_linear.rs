// vision_linear on CPU (contract and portable body in vision.seismic).
// The operator's form launch forms each input row as F32 (`rows`) and its
// Q8 blocks under the INT8 form.
// `vision_linear_rows` gives each work item ROWS weight rows, projected
// against every operand row: the F32 product plus the bias, clamped, then
// published as the portable body's epilogue computes it.

use lib::projection::projection;
use lib::vision::vision;

#[inline(always)]
fn vision_linear_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (m, n, k) = (cx.dim_m() as usize, cx.dim_n() as usize, cx.dim_k() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), n);
    let shared = seismic::cpu::tensor::floats(shared, n);
    let bias = (cx.dim_nb() == 1).then(|| vision::decode_vector(&cx.arg_bias(), shared));
    let bounds = (cx.dim_nc() == 1).then(|| (cx.arg_minimum().get([0]), cx.arg_maximum().get([0])));
    let code = cx.arg_activation();
    // SAFETY: the form launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_rows().slice::<f32>(0, m * k) };
    let (weight, residual, gate_rows, result) = (cx.arg_weight(), cx.arg_residual(), cx.arg_gate(), cx.result_0());
    let quantized = (cx.param_int8() == 1)
        .then(|| unsafe { projection::quantized(cx.scratch_quantized(), 0, m, k, projection::tiled(m, &[&weight])) });
    projection::project_staged_arithmetic(&weight, rows.clone(), staged, quantized, |row, projected| {
        // SAFETY: each work item writes its own columns of every row.
        let mut out = unsafe { result.span_mut([row, rows.start], rows.len()) };
        // The row's gates of these outputs, in F32.
        let mut gates = [0.0f32; projection::MAX_ROWS];
        let gates = &mut gates[..rows.len()];
        if cx.dim_ng() == 1 {
            seismic::cpu::tensor::widen_row::<E::A>(gate_rows.span([0, row, rows.start], rows.len()), gates);
        }
        for (((target, &product), output), gate) in out.iter_mut().zip(projected.iter()).zip(rows.clone()).zip(gates.iter()) {
            let mut value = product;
            if let Some(bias) = bias {
                value += bias[output];
            }
            if let Some((minimum, maximum)) = bounds {
                value = value.max(minimum).min(maximum);
            }
            if code != 0 {
                value = vision::activate(code, value);
            } else if cx.dim_ng() == 1 {
                value *= gate;
            } else if cx.dim_nr() == 1 {
                value = residual.get([0, row, output]) + value;
            }
            target.set(value);
        }
    });
}
