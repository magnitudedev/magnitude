// dense_expand on CPU (contract and portable body in feed_forward.seismic).
// `dense_expand_normalize` publishes the RMS-normalized residual row of each
// output row (rounded to A) as F32 into scratch, one work item per row.
// `dense_expand_rows` gives each work item ROWS gate and up weight rows, which
// it projects against every normalized row and combines as the portable body
// publishes: the `glu` of gate and up, `act` by the `activation` code.

use lib::core::functions;
use lib::projection::projection;

#[inline(always)]
fn dense_expand_normalize<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    let source = cx.arg_out_rows().get([row]) as usize;
    let weight = seismic::cpu::tensor::floats(shared, h);
    cx.arg_norm().decode_row(0, weight);
    // SAFETY: each work item writes its own row of the scratch.
    let normalized = unsafe { cx.scratch_normalized().slice_mut::<f32>(4 * row * h, h) };
    projection::rms_row::<E::A>(
        cx.arg_residual().row([source, 0]),
        weight,
        cx.arg_eps(),
        normalized,
    );
    if cx.param_int8() == 1 {
        // SAFETY: this work item alone writes row `row`'s quantized bytes.
        unsafe { projection::quantize_row(normalized, row, projection::tiled(cx.dim_o() as usize, &[&cx.arg_gate_weight(), &cx.arg_up_weight()]), cx.scratch_quantized(), 0) };
    }
    projection::operand_row(&[&cx.arg_gate_weight(), &cx.arg_up_weight()], normalized);
}

#[inline(always)]
fn dense_expand_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (o, h, f) = (
        cx.dim_o() as usize,
        cx.dim_h() as usize,
        cx.dim_f() as usize,
    );
    let rows = projection::item_rows(group[0], cx.param_rows(), f);
    let (gate_weight, up_weight, result) =
        (cx.arg_gate_weight(), cx.arg_up_weight(), cx.result_0());
    let (function, limit) = (cx.arg_activation(), functions::glu_limit(cx.dim_lb()));
    let gate_scale = if cx.dim_gs() == 0 { 1.0 } else { cx.arg_gate_scale().get([0]) };
    let up_scale = if cx.dim_us() == 0 { 1.0 } else { cx.arg_up_scale().get([0]) };
    // SAFETY: the normalize launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, o * h) };
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        projection::quantized(cx.scratch_quantized(), 0, o, h, projection::tiled(o, &[&cx.arg_gate_weight(), &cx.arg_up_weight()]))
    });
    projection::project_pair_staged_arithmetic(
        (&gate_weight, rows.start),
        (&up_weight, rows.start),
        rows.len(),
        normalized,
        quantized,
        |row, gate, up| {
            // SAFETY: each work item writes its own columns of every row.
            let out = unsafe { result.span_mut([row, rows.start], rows.len()) };
            let pairs = out.iter_mut().zip(gate).zip(up).map(|((target, gate), up)| (target, *gate, *up));
            functions::glu_row::<E::A, _>(function, limit, (gate_scale, up_scale), pairs, |target, value| {
                *target = E::A::narrow(value)
            });
        },
    );
}
