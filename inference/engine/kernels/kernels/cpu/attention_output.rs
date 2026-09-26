// attention_output on CPU (contract and portable body in attention.seismic).
// `attention_output_stage` stages each row's Q gated heads as one F32 row of
// Q * W values in scratch, one work item per row. `attention_output_rows`
// gives each work item ROWS output weight rows, which it projects against
// every staged row: result = the hidden row plus the projection published to
// A, as the portable body adds it.

use lib::core::activation;
use lib::projection::projection;

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
        let blocks = seismic::cpu::quant::blocks(q * w);
        // SAFETY: this work item owns the corresponding quantized row.
        let q8 = unsafe {
            cx.scratch_quantized()
                .slice_mut::<seismic::cpu::quant::Q8Block>(
                    row * blocks * projection::Q8_BYTES,
                    blocks,
                )
        };
        projection::quantize(staged, q8);
    }
}

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
    let (hidden, result) = (cx.arg_hidden(), cx.result_0());
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * k) };
    let blocks = seismic::cpu::quant::blocks(k);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * blocks)
    });
    projection::project_staged_arithmetic(
        &cx.arg_output_weight(),
        rows.clone(),
        staged,
        quantized,
        |row, projected| {
            let source = hidden.span([row, rows.start], rows.len());
            // SAFETY: each work item writes its own columns of every row.
            let out = unsafe { result.span_mut([row, rows.start], rows.len()) };
            for ((target, hidden), projected) in out.iter_mut().zip(source).zip(projected) {
                *target = hidden + activation::publish::<E::A>(*projected);
            }
        },
    );
}
