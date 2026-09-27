// per_layer_gate on CPU (contract and portable body in dense_rows.seismic).
// `per_layer_gate_stage` stages each hidden row rounded to A (as F32, and its
// q8 blocks for the INT8 variant), one work item per row.
// `per_layer_gate_rows` gives each work item ROWS gate rows, projected
// against every staged row and published A(A(act(A(gate))) * input) with the
// row's per-layer input of `layer`.

use lib::core::{activation, functions};
use lib::projection::projection;

fn per_layer_gate_stage<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * row * d, d) };
    for (target, hidden) in staged.iter_mut().zip(cx.arg_hidden().row([row, 0])) {
        *target = activation::publish::<E::A>(*hidden);
    }
    if cx.param_int8() == 1 {
        let blocks = seismic::cpu::quant::blocks(d);
        // SAFETY: this work item owns the corresponding quantized row.
        let q8 = unsafe {
            cx.scratch_quantized()
                .slice_mut::<seismic::cpu::quant::Q8Block>(row * blocks * projection::Q8_BYTES, blocks)
        };
        projection::quantize(staged, q8);
    }
}

fn per_layer_gate_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (m, d, p) = (cx.dim_m() as usize, cx.dim_d() as usize, cx.dim_p() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), p);
    let (inputs, result) = (cx.arg_inputs(), cx.result_0());
    let (layer, function) = (usize::try_from(cx.arg_layer()).expect("a layer index is non-negative"), cx.arg_activation());
    let scale = if cx.dim_gs() == 0 { 1.0 } else { cx.arg_gate_scale().get([0]) };
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * d) };
    let blocks = seismic::cpu::quant::blocks(d);
    let quantized = (cx.param_int8() == 1)
        .then(|| unsafe { cx.scratch_quantized().slice::<seismic::cpu::quant::Q8Block>(0, m * blocks) });
    projection::project_staged_arithmetic(&cx.arg_gate_weight(), rows.clone(), staged, quantized, |row, gate| {
        let multiplier = inputs.span([row, layer, rows.start], rows.len());
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { result.span_mut([row, rows.start], rows.len()) };
        for ((target, gate), multiplier) in out.iter_mut().zip(gate).zip(multiplier) {
            let activated = activation::publish::<E::A>(functions::activate(function, activation::publish::<E::A>(scale * *gate)));
            *target = E::A::narrow(activated * multiplier);
        }
    });
}
