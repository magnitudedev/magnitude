// per_layer_gate on CPU (contract and portable body in per_layer_gate.seismic).
// `per_layer_gate_stage` stages each hidden row rounded to A (as F32, and its
// q8 blocks for the INT8 variant), one work item per row.
// `per_layer_gate_rows` gives each work item ROWS gate rows, projected
// against every staged row and published A(A(act(A(gate))) * input) with the
// row's per-layer input of `layer`.

use lib::core::{activation, functions};
use lib::projection::projection;

#[inline(always)]
fn per_layer_gate_stage<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * row * d, d) };
    for (target, hidden) in staged.iter_mut().zip(cx.arg_hidden().row([row, 0])) {
        *target = activation::publish::<E::A>(*hidden);
    }
    if cx.param_int8() == 1 {
        // SAFETY: this work item alone writes row `row`'s quantized bytes.
        unsafe { projection::quantize_row(staged, row, projection::tiled(cx.dim_m() as usize, &[&cx.arg_gate_weight()]), cx.scratch_quantized(), 0) };
    }
    projection::operand_row(&[&cx.arg_gate_weight()], staged);
}

#[inline(always)]
fn per_layer_gate_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (m, d, p) = (cx.dim_m() as usize, cx.dim_d() as usize, cx.dim_p() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), p);
    let (inputs, result) = (cx.arg_inputs(), cx.result_0());
    let (layer, function) = (usize::try_from(cx.arg_layer()).expect("a layer index is non-negative"), cx.arg_activation());
    let scale = if cx.dim_gs() == 0 { 1.0 } else { cx.arg_gate_scale().get([0]) };
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * d) };
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        projection::quantized(cx.scratch_quantized(), 0, m, d, projection::tiled(m, &[&cx.arg_gate_weight()]))
    });
    projection::project_staged_arithmetic(&cx.arg_gate_weight(), rows.clone(), staged, quantized, |row, gate| {
        let multiplier = inputs.span([row, layer, rows.start], rows.len());
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { result.span_mut([row, rows.start], rows.len()) };
        for ((target, gate), multiplier) in out.iter_mut().zip(gate).zip(multiplier) {
            *target = E::A::narrow(E::A::widen(functions::activated::<E::A>(function, scale * *gate)) * multiplier);
        }
    });
}
