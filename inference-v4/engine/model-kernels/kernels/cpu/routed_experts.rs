// routed_experts on CPU (contract and portable body in routed.seismic).
// `routed_experts_stage` stages each normalized row as F32 in scratch, one
// work item per row. `routed_experts_expand`: work item (x, b) owns ROWS
// features of block b and expands its expert's gate/up rows against every
// live row of the block (the source rows `order` names) into the `product`
// scratch [B * T, F], each A(silu(gate) * up) as F32.
// `routed_experts_down`: work item (x, b) owns ROWS output channels of block
// b and projects its expert's down rows against every live row's product,
// published in A. Blocks of expert -1 and padding rows carry no contract and
// are left unwritten.

use lib::core::activation;
use lib::projection::projection;
use lib::routed::routed;

fn routed_experts_stage<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * row * h, h) };
    projection::stage::<E::A>(cx.arg_normalized().row([row, 0]), staged);
    if cx.param_int8() == 1 {
        let blocks = seismic::cpu::quant::blocks(h);
        // SAFETY: this work item owns the corresponding quantized row.
        let q8 = unsafe {
            cx.scratch_quantized()
                .slice_mut::<seismic::cpu::quant::Q8Block>(
                    row * blocks * projection::Q8_BYTES,
                    blocks,
                )
        };
        projection::quantize(staged, q8);
    }
}

fn routed_experts_expand<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, h, f, t) = (
        cx.dim_m() as usize,
        cx.dim_h() as usize,
        cx.dim_f() as usize,
        cx.dim_t() as usize,
    );
    let block = group[1] as usize;
    let expert = cx.arg_blocks().get([block]);
    if expert < 0 {
        return;
    }
    let rows = projection::item_rows(group[0], cx.param_rows(), f);
    let first = routed::expert_row(expert, f) + rows.start;
    let (gate, up) = (cx.arg_expert_gate(), cx.arg_expert_up());
    let order = cx.arg_order().row([block, 0]);
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * h) };
    let blocks = seismic::cpu::quant::blocks(h);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * blocks)
    });
    for (lane, source) in order[..routed::block_rows(order)].iter().enumerate() {
        let source = *source as usize;
        // SAFETY: each work item writes its own features of its own block's rows.
        let out = unsafe {
            cx.scratch_product()
                .slice_mut::<f32>(4 * ((block * t + lane) * f + rows.start), rows.len())
        };
        let q8 = quantized.map(|q8| &q8[source * blocks..(source + 1) * blocks]);
        routed::expand::<E::A>(
            &gate,
            &up,
            first,
            &staged[source * h..(source + 1) * h],
            q8,
            out,
        );
    }
}

fn routed_experts_quantize<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (f, t) = (cx.dim_f() as usize, cx.dim_t() as usize);
    let (block, lane) = (group[0] as usize / t, group[0] as usize % t);
    if cx.arg_blocks().get([block]) < 0
        || lane >= routed::block_rows(cx.arg_order().row([block, 0]))
    {
        return;
    }
    let row = block * t + lane;
    let blocks = seismic::cpu::quant::blocks(f);
    // SAFETY: the expand launch wrote this product row; this work item owns
    // its quantized row.
    let product = unsafe { cx.scratch_product().slice::<f32>(4 * row * f, f) };
    let q8 = unsafe {
        cx.scratch_product_q8()
            .slice_mut::<seismic::cpu::quant::Q8Block>(row * blocks * projection::Q8_BYTES, blocks)
    };
    projection::quantize(product, q8);
}

fn routed_experts_down<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (h, f, t, b) = (
        cx.dim_h() as usize,
        cx.dim_f() as usize,
        cx.dim_t() as usize,
        cx.dim_b() as usize,
    );
    let block = group[1] as usize;
    let expert = cx.arg_blocks().get([block]);
    if expert < 0 {
        return;
    }
    let columns = projection::item_rows(group[0], cx.param_rows(), h);
    let first = routed::expert_row(expert, h) + columns.start;
    let (down, output) = (cx.arg_expert_down(), cx.result_0());
    let order = cx.arg_order().row([block, 0]);
    // SAFETY: the expand launch wrote every live row before this launch.
    let product = unsafe { cx.scratch_product().slice::<f32>(0, b * t * f) };
    let blocks = seismic::cpu::quant::blocks(f);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_product_q8()
            .slice::<seismic::cpu::quant::Q8Block>(0, b * t * blocks)
    });
    let mut projected = [0.0f32; projection::MAX_ROWS];
    let projected = &mut projected[..columns.len()];
    for lane in 0..routed::block_rows(order) {
        let at = (block * t + lane) * f;
        let row = block * t + lane;
        let q8 = quantized.map(|q8| &q8[row * blocks..(row + 1) * blocks]);
        projection::project_arithmetic(&down, first, &product[at..at + f], q8, projected);
        // SAFETY: each work item writes its own channels of its own block's rows.
        let out = unsafe { output.span_mut([block, lane, columns.start], columns.len()) };
        activation::store::<E::A>(projected, out);
    }
}
