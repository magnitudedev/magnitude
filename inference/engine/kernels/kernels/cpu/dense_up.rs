// dense_up on CPU (contract and portable body in feed_forward.seismic).
// The operator's form launch forms the RMS-normalized residual row of each
// output row (rounded to A) as F32 (`rows`) and its Q8 blocks.
// `dense_up_rows` gives each work item ROWS up weight rows, projected against
// every normalized row and published A(act(A(up_scale * up))).

use lib::core::functions;
use lib::projection::projection;

#[inline(always)]
fn dense_up_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (o, h, f) = (cx.dim_o() as usize, cx.dim_h() as usize, cx.dim_f() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), f);
    let result = cx.result_0();
    let function = cx.arg_activation();
    let scale = if cx.dim_us() == 0 { 1.0 } else { cx.arg_up_scale().get([0]) };
    // SAFETY: the form launch (or the operand's producer) wrote every row before this launch.
    let normalized = unsafe { cx.scratch_rows().slice::<f32>(0, o * h) };
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        projection::quantized(cx.scratch_quantized(), 0, o, h, projection::tiled(o, &[&cx.arg_up_weight()]))
    });
    projection::project_staged_arithmetic(&cx.arg_up_weight(), rows.clone(), normalized, quantized, |row, up| {
        // SAFETY: each work item writes its own columns of every row.
        let mut out = unsafe { result.span_mut([row, rows.start], rows.len()) };
        for (target, up) in out.iter_mut().zip(up) {
            target.set(functions::activated_value::<E::A>(function, scale * *up));
        }
    });
}
