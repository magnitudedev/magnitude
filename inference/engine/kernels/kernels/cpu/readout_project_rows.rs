// readout_project_rows on CPU (contract and portable body in draft.seismic).
// The entry only reads its A features, so they reach the kernel as a dense
// operand: `readout_project_rows_stage` decodes each feature row as F32 into
// scratch, one work item per row. `readout_project_rows_rows` gives each work item
// ROWS vocabulary weight rows, which it projects against every staged row
// into the F32 logits.

use lib::core::functions;
use lib::projection::projection;

fn readout_project_rows_stage<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * row * d, d) };
    projection::stage::<E::A>(cx.arg_features().row([row, 0]), staged);
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
    projection::operand_row(&[&cx.arg_weight()], staged);
}

fn readout_project_rows_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (o, v, d) = (
        cx.dim_o() as usize,
        cx.dim_s() as usize,
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
    let cap = cx.arg_softcap();
    let scale = if cx.dim_ws() == 0 { 1.0 } else { cx.arg_weight_scale().get([0]) };
    for column in rows {
        for (row, x) in staged.chunks_exact(d).enumerate() {
            let vocabulary = if cx.dim_i() == 0 { column } else { cx.arg_token_rows().get([row, 0, column]) as usize };
            if vocabulary >= cx.dim_v() as usize {
                unsafe { result.set([row, column], f32::NEG_INFINITY) };
                continue;
            }
            let q8 = quantized.map(|q8| &q8[row * blocks..(row + 1) * blocks]);
            let mut projected = [0.0];
            projection::project_arithmetic(&cx.arg_weight(), vocabulary, x, q8, &mut projected);
            let value = projected[0] * scale;
            let value = if cap > 0.0 { functions::softcap(cap, value) } else { value };
            // SAFETY: each work item owns its columns of every row.
            unsafe { result.set([row, column], value) };
        }
    }
}
