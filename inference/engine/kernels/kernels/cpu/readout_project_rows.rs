// readout_project_rows on CPU (contract and portable body in draft.seismic).
// The entry only reads its A features, so they reach the kernel as a dense
// operand: `readout_project_rows_stage` decodes each feature row as F32 into
// scratch, one work item per row. `readout_project_rows_rows` gives each work item
// ROWS vocabulary weight rows, which it projects against every staged row
// into the F32 logits: all the rows at once where every row reads the same
// vocabulary rows (no token rows), else each row its own.

use lib::core::functions;
use lib::projection::projection;

#[inline(always)]
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
        // SAFETY: this work item alone writes row `row`'s quantized bytes.
        unsafe { projection::quantize_row(staged, row, tiles(cx), cx.scratch_quantized(), 0) };
    }
    projection::operand_row(&[&cx.arg_weight()], staged);
}

/// Whether the rows are quantized as tiles: every row reads the same
/// vocabulary rows (`projection::tiled`).
#[inline(always)]
fn tiles<E: Elements>(cx: &Context<'_, E>) -> bool {
    cx.dim_i() == 0 && projection::tiled(cx.dim_o() as usize, &[&cx.arg_weight()])
}

#[inline(always)]
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
    let quantized = (cx.param_int8() == 1).then(|| unsafe { projection::quantized(cx.scratch_quantized(), 0, o, d, tiles(cx)) });
    let cap = cx.arg_softcap();
    let scale = if cx.dim_ws() == 0 { 1.0 } else { cx.arg_weight_scale().get([0]) };
    let finish = |value: f32| {
        let value = value * scale;
        if cap > 0.0 { functions::softcap(cap, value) } else { value }
    };
    if cx.dim_i() == 0 {
        // Every row reads the work item's vocabulary rows, those past the
        // vocabulary none.
        let real = rows.start.min(cx.dim_v() as usize)..rows.end.min(cx.dim_v() as usize);
        for row in 0..o {
            for column in real.end..rows.end {
                unsafe { result.set([row, column], f32::NEG_INFINITY) };
            }
        }
        if !real.is_empty() {
            projection::project_staged_arithmetic(&cx.arg_weight(), real.clone(), staged, quantized, |row, projected| {
                for (column, value) in real.clone().zip(projected) {
                    // SAFETY: each work item owns its columns of every row.
                    unsafe { result.set([row, column], finish(*value)) };
                }
            });
        }
        return;
    }
    let quantized = quantized.map(|quantized| match quantized {
        projection::Quantized::Rows(q8) => q8,
        projection::Quantized::Tiles(_) => unreachable!("token rows are quantized row by row"),
    });
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
            // SAFETY: each work item owns its columns of every row.
            unsafe { result.set([row, column], finish(projected[0])) };
        }
    }
}
