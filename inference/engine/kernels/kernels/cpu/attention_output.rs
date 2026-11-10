// attention_output on CPU (contract and portable body in attention.seismic).
// `attention_output_stage` stages each row's Q gated heads as one F32 row of
// Q * W values in scratch, one work item per row. `attention_output_rows`
// gives each work item ROWS output weight rows, which it projects against
// every staged row: result = the hidden row plus the projection (times the
// weight's second-level scale, where it has one) published to A, as the
// portable body adds it.

use lib::core::{activation, functions};
use lib::projection::projection;

#[inline(always)]
fn attention_output_stage<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let (q, w) = (cx.dim_q() as usize, cx.dim_w() as usize);
    let gated = cx.arg_gated();
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * row * q * w, q * w) };
    for (head, target) in staged.chunks_exact_mut(w).enumerate() {
        projection::stage::<E::A>(gated.row([row, head, 0]), target);
    }
    if cx.param_int8() == 1 {
        // SAFETY: this work item alone writes row `row`'s quantized bytes.
        unsafe { projection::quantize_row(staged, row, projection::tiled(cx.dim_m() as usize, &[&cx.arg_output_weight()]), cx.scratch_quantized(), 0) };
    }
    projection::operand_row(&[&cx.arg_output_weight()], staged);
}

#[inline(always)]
fn attention_output_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, d, k) = (
        cx.dim_m() as usize,
        cx.dim_d() as usize,
        (cx.dim_q() * cx.dim_w()) as usize,
    );
    let rows = projection::item_rows(group[0], cx.param_rows(), d);
    let (hidden, result, inject) = (cx.arg_hidden(), cx.result_0(), cx.arg_inject());
    let (streams, inject_width) = (cx.dim_sn() as usize, cx.dim_in() as usize);
    let output_scale = if cx.dim_os() == 0 { 1.0 } else { cx.arg_output_scale().get([0]) };
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * k) };
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        projection::quantized(cx.scratch_quantized(), 0, m, k, projection::tiled(m, &[&cx.arg_output_weight()]))
    });
    projection::project_staged_arithmetic(
        &cx.arg_output_weight(),
        rows.clone(),
        staged,
        quantized,
        |row, projected| {
            // `scatter` of the projection into each stream.
            for stream in 0..streams {
                // SAFETY: each work item writes its own columns of every row.
                let out = unsafe { result.span_mut([row, stream * d + rows.start], rows.len()) };
                functions::scatter(
                    streams,
                    inject_width,
                    stream,
                    |column| activation::publish::<E::A>(output_scale * projected[column]),
                    |at| inject.get([row, at]),
                    |from| hidden.span([row, from * d + rows.start], rows.len()),
                    out,
                );
            }
        },
    );
}
