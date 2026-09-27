// routed_route on CPU (contract and portable body in routed.seismic).
// `routed_route_normalize` publishes each row's RMS normalization (rounded to
// A) as the `normalized` result and as F32 into scratch, one work item per
// row. `routed_route_logits` gives each work item ROWS router rows, which it
// projects against every normalized row into the `logits` scratch [M, E].
// `routed_route_select` routes one row per work item: the softmax, the top-K
// selection, the optional renormalization and the shared-expert coefficient
// sigmoid(normalized . shared_router).

use lib::core::{activation, reduce};
use lib::projection::projection;
use lib::routed::routed;

fn routed_route_normalize<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    let weight = seismic::cpu::tensor::floats(shared, h);
    cx.arg_norm().decode_row(0, weight);
    // SAFETY: each work item writes its own row of the scratch.
    let normalized = unsafe { cx.scratch_normalized().slice_mut::<f32>(4 * row * h, h) };
    projection::rms_row::<E::A>(cx.arg_residual().row([row, 0]), weight, cx.arg_eps(), normalized);
    // SAFETY: each work item writes its own row of the result.
    activation::store::<E::A>(normalized, unsafe { cx.result_0().row_mut([row, 0]) });
}

fn routed_route_logits<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (m, h, e) = (cx.dim_m() as usize, cx.dim_h() as usize, cx.dim_e() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), e);
    let router = cx.arg_router();
    // SAFETY: the normalize launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, m * h) };
    for row in 0..m {
        // SAFETY: each work item writes its own experts of every row.
        let logits = unsafe { cx.scratch_logits().slice_mut::<f32>(4 * (row * e + rows.start), rows.len()) };
        projection::project(&router, rows.start, &normalized[row * h..(row + 1) * h], logits);
    }
}

fn routed_route_select<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (h, e) = (cx.dim_h() as usize, cx.dim_e() as usize);
    // SAFETY: the normalize and logits launches wrote every row before this
    // launch.
    let (normalized, logits) = unsafe {
        (cx.scratch_normalized().slice::<f32>(4 * row * h, h), cx.scratch_logits().slice::<f32>(4 * row * e, e))
    };
    // SAFETY: each work item writes its own row of the routes and scores.
    let (routes, scores) = unsafe { (cx.arg_routes().row_mut([row, 0]), cx.arg_scores().row_mut([row, 0])) };
    routed::select(logits, seismic::cpu::tensor::floats(shared, e), cx.arg_normalize() != 0, routes, scores);
    let gate = reduce::dot(normalized, cx.arg_shared_router().row([0]));
    // SAFETY: each work item writes its own row's coefficient.
    unsafe { cx.result_1().set([row], activation::sigmoid(gate)) };
}
