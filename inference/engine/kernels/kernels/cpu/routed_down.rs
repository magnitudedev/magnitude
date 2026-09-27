// routed_down on CPU (contract and portable body in routed.seismic).
// `routed_down_stage` stages each row's choice products as F32 in scratch,
// one work item per row. `routed_down_rows`: work item (x, m) owns ROWS
// output channels of row m. It projects each choice's expert down rows,
// weighting each published (A-rounded) projection in slot order, and
// publishes base + selected.

use lib::core::activation;
use lib::projection::projection;
use lib::routed::routed;

fn routed_down_stage<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (k, f) = (cx.dim_k() as usize, cx.dim_f() as usize);
    let product = cx.arg_product();
    for slot in 0..k {
        let choice = row * k + slot;
        // SAFETY: each work item writes its own row's choices of the scratch.
        let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * choice * f, f) };
        projection::stage::<E::A>(product.row([row, slot, 0]), staged);
        if cx.param_int8() == 1 {
            let blocks = seismic::cpu::quant::blocks(f);
            // SAFETY: this work item owns the corresponding choice row.
            let q8 = unsafe {
                cx.scratch_quantized()
                    .slice_mut::<seismic::cpu::quant::Q8Block>(choice * blocks * projection::Q8_BYTES, blocks)
            };
            projection::quantize(staged, q8);
        }
    }
}

fn routed_down_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (m, h, k, f) = (cx.dim_m() as usize, cx.dim_h() as usize, cx.dim_k() as usize, cx.dim_f() as usize);
    let row = group[1] as usize;
    let columns = projection::item_rows(group[0], cx.param_rows(), h);
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * k * f) };
    let blocks = seismic::cpu::quant::blocks(f);
    let quantized = (cx.param_int8() == 1)
        .then(|| unsafe { cx.scratch_quantized().slice::<seismic::cpu::quant::Q8Block>(0, m * k * blocks) });
    let (routes, weights, down) = (cx.arg_routes(), cx.arg_weights(), cx.arg_expert_down());
    let mut selected = [0.0f32; projection::MAX_ROWS];
    let mut projected = [0.0f32; projection::MAX_ROWS];
    let (selected, projected) = (&mut selected[..columns.len()], &mut projected[..columns.len()]);
    for slot in 0..k {
        let choice = row * k + slot;
        let first = routed::expert_row(routes.get([row, slot]), h) + columns.start;
        let q8 = quantized.map(|q8| &q8[choice * blocks..(choice + 1) * blocks]);
        projection::project_arithmetic(&down, first, &staged[choice * f..(choice + 1) * f], q8, projected);
        let weight = weights.get([row, slot]);
        for (selected, projected) in selected.iter_mut().zip(projected.iter()) {
            *selected = weight.mul_add(activation::publish::<E::A>(*projected), *selected);
        }
    }
    let base = cx.arg_base().span([row, columns.start], columns.len());
    // SAFETY: each work item writes its own channels of its own row.
    let out = unsafe { cx.result_0().span_mut([row, columns.start], columns.len()) };
    for ((target, base), selected) in out.iter_mut().zip(base).zip(selected.iter()) {
        *target = E::R::narrow(base + selected);
    }
}
