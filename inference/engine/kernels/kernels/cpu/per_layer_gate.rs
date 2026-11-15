// per_layer_gate on CPU (contract and portable body in per_layer_gate.seismic).
// The operator's form launch forms each hidden row rounded to A (as F32,
// `rows`, and its Q8 blocks for the INT8 variant).
// `per_layer_gate_rows` gives each work item ROWS gate rows, projected
// against every staged row and published A(A(act(A(gate))) * input) with the
// row's per-layer input of `layer`.

use lib::core::functions;
use lib::projection::projection;

#[inline(always)]
fn per_layer_gate_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (m, d, p) = (cx.dim_m() as usize, cx.dim_d() as usize, cx.dim_p() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), p);
    let (inputs, result) = (cx.arg_inputs(), cx.result_0());
    let (layer, function) = (usize::try_from(cx.arg_layer()).expect("a layer index is non-negative"), cx.arg_activation());
    let scale = if cx.dim_gs() == 0 { 1.0 } else { cx.arg_gate_scale().get([0]) };
    // SAFETY: the form launch (or the operand's producer) wrote every row before this launch.
    let staged = unsafe { cx.scratch_rows().slice::<f32>(0, m * d) };
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        projection::quantized(cx.scratch_quantized(), 0, m, d, projection::tiled(m, &[&cx.arg_gate_weight()]))
    });
    projection::project_staged_arithmetic(&cx.arg_gate_weight(), rows.clone(), staged, quantized, |row, gate| {
        let multiplier = inputs.span([row, layer, rows.start], rows.len());
        // SAFETY: each work item writes its own columns of every row.
        let mut out = unsafe { result.span_mut([row, rows.start], rows.len()) };
        for ((target, gate), multiplier) in out.iter_mut().zip(gate).zip(multiplier) {
            target.set(E::A::widen(functions::activated::<E::A>(function, scale * *gate)) * multiplier);
        }
    });
}
