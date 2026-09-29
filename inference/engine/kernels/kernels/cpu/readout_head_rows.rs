// readout_head_rows on CPU (contract and portable body in readout.seismic).
// `readout_head_rows_normalize` publishes the RMS-normalized hidden row of
// each output row (rounded to A) as F32 into scratch, one work item per row.
// `readout_head_rows_rows` gives each work item ROWS vocabulary weight rows,
// which it projects against every normalized row into the F32 logits (scaled
// by a present `weight_scale`, then softcapped when `softcap` > 0).

use lib::core::functions;
use lib::projection::projection;

fn readout_head_rows_normalize<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    let source = cx.arg_out_rows().get([row]) as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let normalized = unsafe { cx.scratch_normalized().slice_mut::<f32>(4 * row * d, d) };
    projection::normalize::<E::A>(
        cx.arg_hidden().row([source, 0]),
        &cx.arg_norm(),
        cx.arg_epsilon(),
        shared,
        normalized,
    );
    if cx.param_int8() == 1 {
        let blocks = seismic::cpu::quant::blocks(d);
        // SAFETY: this work item owns the corresponding quantized row.
        let q8 = unsafe {
            cx.scratch_quantized()
                .slice_mut::<seismic::cpu::quant::Q8Block>(
                    row * blocks * projection::Q8_BYTES,
                    blocks,
                )
        };
        projection::quantize(normalized, q8);
    }
}

fn readout_head_rows_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (o, v, d) = (
        cx.dim_o() as usize,
        cx.dim_v() as usize,
        cx.dim_d() as usize,
    );
    let rows = projection::item_rows(group[0], cx.param_rows(), v);
    let result = cx.result_0();
    // SAFETY: the normalize launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, o * d) };
    let blocks = seismic::cpu::quant::blocks(d);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, o * blocks)
    });
    let cap = cx.arg_softcap();
    let scale = if cx.dim_ws() == 0 { 1.0 } else { cx.arg_weight_scale().get([0]) };
    projection::project_staged_arithmetic(
        &cx.arg_weight(),
        rows.clone(),
        normalized,
        quantized,
        |row, projected| {
            // SAFETY: each work item writes its own columns of every row.
            let logits = unsafe { result.span_mut([row, rows.start], rows.len()) };
            for (logit, value) in logits.iter_mut().zip(projected) {
                let value = *value * scale;
                *logit = if cap > 0.0 { functions::softcap(cap, value) } else { value };
            }
        },
    );
}
