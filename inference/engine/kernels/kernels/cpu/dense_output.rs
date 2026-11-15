// dense_output on CPU (contract and portable body in feed_forward.seismic).
// The operator's form launch forms each product row as F32 (`rows`) and its
// Q8 blocks. `dense_output_rows` gives each work item ROWS down weight rows,
// which it projects against every operand row: result = the residual row
// plus the projection published to A, as the portable body adds it.

use lib::core::activation;
use lib::projection::projection;

#[inline(always)]
fn dense_output_rows<L: Isa, E: Elements>(
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
    let rows = projection::item_rows(group[0], cx.param_rows(), h);
    let (down_weight, residual, out_rows, result) = (
        cx.arg_down_weight(),
        cx.arg_residual(),
        cx.arg_out_rows(),
        cx.result_0(),
    );
    let down_scale = if cx.dim_ds() == 0 { 1.0 } else { cx.arg_down_scale().get([0]) };
    // SAFETY: the form launch (or the operand's producer) wrote every row before this launch.
    let staged = unsafe { cx.scratch_rows().slice::<f32>(0, o * f) };
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        projection::quantized(cx.scratch_quantized(), 0, o, f, projection::tiled(o, &[&cx.arg_down_weight()]))
    });
    projection::project_staged_arithmetic(
        &down_weight,
        rows.clone(),
        staged,
        quantized,
        |row, projected| {
            let source = residual.span([out_rows.get([row]) as usize, rows.start], rows.len());
            // SAFETY: each work item writes its own columns of every row.
            let mut out = unsafe { result.span_mut([row, rows.start], rows.len()) };
            for ((target, residual), projected) in out.iter_mut().zip(source).zip(projected.iter()) {
                target.set(residual + activation::publish::<E::A>(down_scale * *projected));
            }
        },
    );
}
