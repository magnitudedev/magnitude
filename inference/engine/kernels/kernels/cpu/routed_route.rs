// routed_route on CPU (contract and portable body in routed.seismic).
// `routed_route_normalize` publishes each row's RMS normalization (rounded to
// A) as the `normalized` result and as F32 into scratch, one work item per
// row, and emits the row's operand (`projection::emit`: the stored row
// staged, its Q8 blocks, its NVFP4 operand rows) for the projection that
// reads `normalized` (`routed_expand`), in the forms its graph pairs
// (`emits_result_0_rows()`, `emits_result_0_quantized()`,
// `nvfp4_result_0_rows()`).
// `routed_route_logits` splits the router rows evenly over its grid (the
// pool's participants, at most E), each work item projecting its rows against
// every normalized row into the `logits` scratch [M, E].
// `routed_route_select` routes one row per work item: the softmax, the top-K
// selection, the optional renormalization, each winner's `expert_scale` and
// the shared-expert coefficient sigmoid(normalized . shared_router).

use lib::core::activation;
use lib::projection::projection;
use lib::routed::{ranked, routed};

#[inline(always)]
fn routed_route_normalize<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let normalized = unsafe { cx.scratch_normalized().slice_mut::<f32>(4 * row * h, h) };
    let norm = cx.arg_norm();
    projection::form_input::<E::A>(
        cx.arg_residual().row([row, 0]),
        (cx.dim_nm() != 0).then_some(&norm),
        cx.arg_eps(),
        shared,
        normalized,
    );
    let result = cx.result_0();
    // SAFETY: each work item writes its own row of the result.
    activation::store::<E::A>(normalized, unsafe { result.row_mut([row, 0]) });
    if cx.emits_result_0_rows() {
        // SAFETY: the emitted operand's buffers; this item stored the whole
        // row, which arrives once.
        unsafe {
            projection::emit(
                cx.scratch_result_0_rows(),
                cx.emits_result_0_quantized().then(|| cx.scratch_result_0_quantized()),
                cx.nvfp4_result_0_rows(),
                cx.scratch_result_0_arrivals(),
                h,
                row,
                h,
                |out| projection::stage::<E::A>(result.row([row, 0]), out),
            )
        };
    }
}

#[inline(always)]
fn routed_route_logits<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (m, h, e) = (cx.dim_m() as usize, cx.dim_h() as usize, cx.dim_e() as usize);
    let rows = projection::part_rows(group[0], cx.groups()[0], e);
    let router = cx.arg_router();
    // SAFETY: the normalize launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, m * h) };
    for row in 0..m {
        // SAFETY: each work item writes its own experts of every row.
        let logits = unsafe { cx.scratch_logits().slice_mut::<f32>(4 * (row * e + rows.start), rows.len()) };
        projection::project(&router, rows.start, &normalized[row * h..(row + 1) * h], logits);
    }
}

#[inline(always)]
fn routed_route_select<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (h, e) = (cx.dim_h() as usize, cx.dim_e() as usize);
    // SAFETY: the normalize and logits launches wrote every row before this
    // launch.
    let (normalized, logits) = unsafe {
        (cx.scratch_normalized().slice::<f32>(4 * row * h, h), cx.scratch_logits().slice::<f32>(4 * row * e, e))
    };
    // SAFETY: each work item writes its own row of the routes and scores.
    let (routes, mut scores) = unsafe { (cx.arg_routes().row_mut([row, 0]), cx.arg_scores().row_mut([row, 0])) };
    let expert_scale = cx.arg_expert_scale();
    let (probabilities, order) = shared.split_at_mut(4 * e);
    scores.compute(|scores| {
        routed::select(
            logits,
            seismic::cpu::tensor::floats(probabilities, e),
            ranked::indices(order, e),
            cx.arg_normalize() != 0,
            (cx.dim_es(), |expert| expert_scale.get([expert])),
            routes,
            scores,
        )
    });
    // The shared gate's row is a weight as the router's rows are: the same
    // dot, in the same order.
    let mut gate = 0.0f32;
    projection::project(&cx.arg_shared_router(), 0, normalized, std::slice::from_mut(&mut gate));
    // SAFETY: each work item writes its own row's coefficient.
    unsafe { cx.result_1().set([row], activation::sigmoid(gate)) };
}
