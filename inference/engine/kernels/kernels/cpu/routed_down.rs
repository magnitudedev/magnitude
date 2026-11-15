// routed_down on CPU (contract and portable body in routed.seismic).
// The operator's form launch forms each row's choice products as F32
// (`rows`) and their Q8 blocks. `routed_down_rows`: work item (x, m) owns
// ROWS output channels of row m. It projects each choice's expert down rows
// (none for a choice its sources name none for), weighting each published
// (A-rounded) projection in slot order, and publishes base + selected.

use lib::core::activation;
use lib::projection::projection;
use lib::routed::routed;

#[inline(always)]
fn routed_down_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (m, h, k, f) = (cx.dim_m() as usize, cx.dim_h() as usize, cx.dim_k() as usize, cx.dim_f() as usize);
    let row = group[1] as usize;
    let columns = projection::item_rows(group[0], cx.param_rows(), h);
    // SAFETY: the form launch (or the operand's producer) wrote every row before this launch.
    let staged = unsafe { cx.scratch_rows().slice::<f32>(0, m * k * f) };
    let blocks = seismic::cpu::quant::blocks(f);
    let quantized = (cx.param_int8() == 1)
        .then(|| unsafe { cx.scratch_quantized().slice::<seismic::cpu::quant::Q8Block>(0, m * k * blocks) });
    let (routes, weights, down) = (cx.arg_routes(), cx.arg_weights(), cx.arg_expert_down());
    let tier = cx.arg_tier_down();
    let sources = cx.arg_sources();
    let record = |slot: usize| routed::decode_source(cx.dim_xs(), || sources.get([0, row, slot]), routes.get([row, slot]));
    let mut selected = [0.0f32; projection::MAX_ROWS];
    let mut projected = [0.0f32; projection::MAX_ROWS];
    let (selected, projected) = (&mut selected[..columns.len()], &mut projected[..columns.len()]);
    for slot in 0..k {
        let choice = row * k + slot;
        // A choice its slot map names no record for reads none: a zero
        // projection.
        match record(slot) {
            Some(place) => {
                let down = place.of(&down, &tier);
                let q8 = quantized.map(|q8| &q8[choice * blocks..(choice + 1) * blocks]);
                projection::project_arithmetic(&down, columns.start, &staged[choice * f..(choice + 1) * f], q8, projected);
            }
            None => projected.fill(0.0),
        }
        let weight = weights.get([row, slot]);
        for (selected, projected) in selected.iter_mut().zip(projected.iter()) {
            *selected = weight.mul_add(activation::publish::<E::A>(*projected), *selected);
        }
    }
    let base = cx.arg_base().span([row, columns.start], columns.len());
    // SAFETY: each work item writes its own channels of its own row.
    let mut out = unsafe { cx.result_0().span_mut([row, columns.start], columns.len()) };
    for ((target, base), selected) in out.iter_mut().zip(base).zip(selected.iter()) {
        target.set(base + selected);
    }
}
