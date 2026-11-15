// routed_experts_up on CPU (contract and portable body in routed.seismic):
// the `routed_experts` launches, grids and weights' places with one expanding
// weight. The operator's form launch forms the grouped normalized rows (the
// window from row first[0]) as F32 and their Q8 blocks.
// `routed_experts_up_expand`: work item (x, b) owns ROWS features of block b
// and projects its expert's up rows against every live row of the block
// (from a lane of its own) into the product rows `edw_rows` [B * T, F], each
// A(act(A(us * up))) as F32 (`us` the block's expert's scale, 1 where the
// port is absent); a row's last stored feature forms the down projection's
// operand of it in place (`routed::product_arrival`).
// `routed_experts_up_down`: work item (x, b) owns ROWS output channels of
// block b and projects its expert's down rows against every live row's
// product, published in A. Blocks past `used` and padding rows carry no
// contract and are left unwritten.

use lib::core::{activation, functions};
use lib::projection::projection;
use lib::routed::routed;

#[inline(always)]
fn routed_experts_up_grids<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], _shared: &mut [u8]) {
    let items = |extent: u64| extent.div_ceil(cx.param_rows()) as usize;
    routed::expert_grids(
        [cx.scratch_expand_grid(), cx.scratch_down_grid()],
        cx.scratch_product_arrivals(),
        cx.arg_used().get([0]),
        cx.dim_t() as usize,
        items(cx.dim_f()),
        items(cx.dim_h()),
    );
}

#[inline(always)]
fn routed_experts_up_expand<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (m, h, f, t) = (cx.dim_m() as usize, cx.dim_h() as usize, cx.dim_f() as usize, cx.dim_t() as usize);
    let block = group[1] as usize;
    let expert = cx.arg_blocks().get([block]);
    let rows = projection::item_rows(group[0], cx.param_rows(), f);
    let ((map, step), experts, records) = ((cx.arg_map(), cx.arg_step()), cx.arg_expert_up(), cx.arg_records_up());
    let slot = routed::expert_slot(cx.dim_x(), |e| routed::held(|w| map.get([e, w]), |w| step.get([w])), expert);
    let Some(up) = routed::expert_weights(&experts, &records, slot) else {
        return;
    };
    let function = cx.arg_activation();
    let up_scale = cx.arg_up_scale();
    let scale = routed::scale_factor(cx.dim_us(), |e| up_scale.get([e]), expert);
    let order = cx.arg_order().row([block, 0]);
    // SAFETY: the form launch (or the rows' producer) wrote every row of the
    // window before this launch.
    let staged = unsafe { cx.scratch_rows().slice::<f32>(0, m * h) };
    let blocks = seismic::cpu::quant::blocks(h);
    let quantized = (cx.param_int8() == 1)
        .then(|| unsafe { cx.scratch_quantized().slice::<seismic::cpu::quant::Q8Block>(0, m * blocks) });
    let (products, down_q8, nvfp4) = (
        cx.scratch_edw_rows(),
        (cx.param_int8() == 1).then(|| cx.scratch_edw_quantized()),
        cx.arg_expert_down().nvfp4_operands(),
    );
    let live = routed::block_rows(order);
    for step in 0..live {
        let lane = (group[0] as usize + step) % live;
        let source = order[lane] as usize;
        let row = block * t + lane;
        // SAFETY: each work item writes its own features of its own block's rows.
        let out = unsafe { products.slice_mut::<f32>(4 * (row * f + rows.start), rows.len()) };
        let q8 = quantized.map(|q8| &q8[source * blocks..(source + 1) * blocks]);
        projection::project_arithmetic(&up, rows.start, &staged[source * h..(source + 1) * h], q8, out);
        for value in out.iter_mut() {
            *value = E::A::widen(functions::activated::<E::A>(function, scale * *value));
        }
        // SAFETY: the grids launch cleared the row's counter; these
        // features are this item's, stored above.
        unsafe { routed::product_arrival(products, down_q8, nvfp4, cx.scratch_product_arrivals(), f, row, rows.len()) };
    }
}

#[inline(always)]
fn routed_experts_up_down<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (h, f, t, b) = (cx.dim_h() as usize, cx.dim_f() as usize, cx.dim_t() as usize, cx.dim_b() as usize);
    let block = group[1] as usize;
    let expert = cx.arg_blocks().get([block]);
    let columns = projection::item_rows(group[0], cx.param_rows(), h);
    let ((map, step), experts, records) = ((cx.arg_map(), cx.arg_step()), cx.arg_expert_down(), cx.arg_records_down());
    let slot = routed::expert_slot(cx.dim_x(), |e| routed::held(|w| map.get([e, w]), |w| step.get([w])), expert);
    let Some(down) = routed::expert_weights(&experts, &records, slot) else {
        return;
    };
    let output = cx.result_0();
    let order = cx.arg_order().row([block, 0]);
    // SAFETY: the expand launch wrote and formed every live row before this
    // launch.
    let product = unsafe { cx.scratch_edw_rows().slice::<f32>(0, b * t * f) };
    let blocks = seismic::cpu::quant::blocks(f);
    let quantized = (cx.param_int8() == 1)
        .then(|| unsafe { cx.scratch_edw_quantized().slice::<seismic::cpu::quant::Q8Block>(0, b * t * blocks) });
    let mut projected = [0.0f32; projection::MAX_ROWS];
    let projected = &mut projected[..columns.len()];
    for lane in 0..routed::block_rows(order) {
        let row = block * t + lane;
        let q8 = quantized.map(|q8| &q8[row * blocks..(row + 1) * blocks]);
        projection::project_arithmetic(&down, columns.start, &product[row * f..(row + 1) * f], q8, projected);
        // SAFETY: each work item writes its own channels of its own block's rows.
        let out = unsafe { output.span_mut([block, lane, columns.start], columns.len()) };
        activation::store::<E::A>(projected, out);
    }
}
