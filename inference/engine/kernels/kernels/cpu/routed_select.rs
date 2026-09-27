// routed_select on CPU (contract and portable body in routed.seismic).
// `routed_select_normalize` publishes each row's RMS normalization with
// `norm` (rounded to A) as the `normalized` result, and with `router_norm` as
// F32 into scratch, one work item per row. `routed_select_logits` gives each
// work item ROWS router rows, which it projects against every router input
// row into the `logits` scratch [M, E]. `routed_select_select` selects one
// row per work item: scores, biased top-K, normalization, scales.

use lib::core::{activation, reduce};
use lib::projection::projection;
use lib::routed::select;

fn routed_select_normalize<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    let (norm, rest) = shared.split_at_mut(4 * h);
    let (router_norm, published) = rest.split_at_mut(4 * h);
    let (norm, router_norm, published) = (
        seismic::cpu::tensor::floats(norm, h),
        seismic::cpu::tensor::floats(router_norm, h),
        seismic::cpu::tensor::floats(published, h),
    );
    cx.arg_norm().decode_row(0, norm);
    cx.arg_router_norm().decode_row(0, router_norm);
    let residual = cx.arg_residual().row([row, 0]);
    let inverse = reduce::rms_inverse(residual, cx.arg_epsilon());
    // SAFETY: each work item writes its own row of the scratch.
    let router_rows = unsafe { cx.scratch_router_rows().slice_mut::<f32>(4 * row * h, h) };
    for (((target, router), value), (weight, router_weight)) in
        published.iter_mut().zip(router_rows.iter_mut()).zip(residual).zip(norm.iter().zip(router_norm.iter()))
    {
        *target = activation::publish::<E::A>(value * inverse * weight);
        *router = activation::publish::<E::A>(value * inverse * router_weight);
    }
    // SAFETY: each work item writes its own row of the result.
    activation::store::<E::A>(published, unsafe { cx.result_0().row_mut([row, 0]) });
}

fn routed_select_logits<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (m, h, e) = (cx.dim_m() as usize, cx.dim_h() as usize, cx.dim_e() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), e);
    let router = cx.arg_router();
    // SAFETY: the normalize launch wrote every row before this launch.
    let router_rows = unsafe { cx.scratch_router_rows().slice::<f32>(0, m * h) };
    for row in 0..m {
        // SAFETY: each work item writes its own experts of every row.
        let logits = unsafe { cx.scratch_logits().slice_mut::<f32>(4 * (row * e + rows.start), rows.len()) };
        projection::project(&router, rows.start, &router_rows[row * h..(row + 1) * h], logits);
    }
}

fn routed_select_select<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let e = cx.dim_e() as usize;
    let (scores, order) = shared.split_at_mut(4 * e);
    let scores = seismic::cpu::tensor::floats(scores, e);
    let order = select::indices(order, e);
    // SAFETY: the logits launch wrote every row before this launch.
    scores.copy_from_slice(unsafe { cx.scratch_logits().slice::<f32>(4 * row * e, e) });
    select::scores(cx.arg_score(), scores);
    // SAFETY: each work item writes its own row of the routes and weights.
    let (routes, weights) = unsafe { (cx.arg_routes().row_mut([row, 0]), cx.arg_weights().row_mut([row, 0])) };
    let bias = cx.arg_bias();
    select::top_k(scores, |expert| bias.get([expert]), order, routes, weights);
    let denominator = weights.iter().sum::<f32>();
    let (normalization, epsilon, scale) =
        (cx.arg_normalization(), cx.arg_normalization_epsilon(), cx.arg_scale());
    let expert_scale = cx.arg_expert_scale();
    for (weight, expert) in weights.iter_mut().zip(routes.iter()) {
        *weight = select::normalized(normalization, *weight, denominator, epsilon) * scale
            * expert_scale.get([*expert as usize]);
    }
}
