// head_logits_rows on CPU (contract and portable body in draft.seismic).
// The entry only reads its A features, so they reach the kernel as a dense
// operand: `head_logits_rows_stage` decodes each feature row as F32 into
// scratch, one work item per row. `head_logits_rows_rows` gives each work item
// ROWS vocabulary weight rows, which it projects against every staged row
// into the F32 logits.

use lib::projection::projection;

fn head_logits_rows_stage<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * row * d, d) };
    cx.arg_features().decode_row(row, staged);
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
        projection::quantize(staged, q8);
    }
}

fn head_logits_rows_rows<L: Isa, E: Elements>(
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
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, o * d) };
    let blocks = seismic::cpu::quant::blocks(d);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, o * blocks)
    });
    projection::project_staged_arithmetic(
        &cx.arg_weight(),
        rows.clone(),
        staged,
        quantized,
        |row, projected| {
            // SAFETY: each work item writes its own columns of every row.
            unsafe { result.span_mut([row, rows.start], rows.len()) }.copy_from_slice(projected);
        },
    );
}
