// readout_head_rows on CPU (contract and portable body in readout.seismic).
// The operator's form launch forms the RMS-normalized hidden row of each
// output row (rounded to A) as F32 (`rows`) and its Q8 blocks.
// `readout_head_rows_rows` gives each work item ROWS vocabulary weight rows,
// which it projects against every normalized row into the F32 logits (scaled
// by a present `weight_scale`, then softcapped when `softcap` > 0).

use lib::core::functions;
use lib::projection::projection;

#[inline(always)]
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
    // SAFETY: the form launch (or the operand's producer) wrote every row before this launch.
    let normalized = unsafe { cx.scratch_rows().slice::<f32>(0, o * d) };
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        projection::quantized(cx.scratch_quantized(), 0, o, d, projection::tiled(o, &[&cx.arg_weight()]))
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
            let mut logits = unsafe { result.span_mut([row, rows.start], rows.len()) };
            for (logit, value) in logits.iter_mut().zip(projected) {
                let value = *value * scale;
                logit.set(if cap > 0.0 { functions::softcap(cap, value) } else { value });
            }
        },
    );
}
