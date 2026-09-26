// routed_combine on CPU (contract and portable body in routed.seismic).
// `routed_combine_stage` stages each normalized row as F32 in scratch, one
// work item per row. `routed_combine_shared` gives each work item ROWS shared
// features, which it expands against every row into the `shared_product`
// scratch [M, S] (A(silu(gate) * up) as F32). `routed_combine` gives each
// work item ROWS output channels: per row it projects the shared expert's
// down rows, gathers the row's grouped expert outputs (`inverse`) weighted by
// their scores in slot order, and publishes
//     residual + selected + A(shared) * coefficient.

use lib::core::activation;
use lib::projection::projection;
use lib::routed::routed;

fn routed_combine_stage<L: Isa, E: Elements>(
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

fn routed_combine_shared<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, h, s) = (
        cx.dim_m() as usize,
        cx.dim_h() as usize,
        cx.dim_s() as usize,
    );
    let rows = projection::item_rows(group[0], cx.param_rows(), s);
    let (gate, up) = (cx.arg_shared_gate(), cx.arg_shared_up());
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * h) };
    let blocks = seismic::cpu::quant::blocks(h);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * blocks)
    });
    for row in 0..m {
        // SAFETY: each work item writes its own features of every row.
        let out = unsafe {
            cx.scratch_shared_product()
                .slice_mut::<f32>(4 * (row * s + rows.start), rows.len())
        };
        let q8 = quantized.map(|q8| &q8[row * blocks..(row + 1) * blocks]);
        routed::expand::<E::A>(
            &gate,
            &up,
            rows.start,
            &staged[row * h..(row + 1) * h],
            q8,
            out,
        );
    }
}

fn routed_combine_quantize<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let s = cx.dim_s() as usize;
    let blocks = seismic::cpu::quant::blocks(s);
    // SAFETY: the shared launch wrote this product row; this work item owns
    // its quantized row.
    let product = unsafe { cx.scratch_shared_product().slice::<f32>(4 * row * s, s) };
    let q8 = unsafe {
        cx.scratch_shared_q8()
            .slice_mut::<seismic::cpu::quant::Q8Block>(row * blocks * projection::Q8_BYTES, blocks)
    };
    projection::quantize(product, q8);
}

fn routed_combine<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, h, k) = (
        cx.dim_m() as usize,
        cx.dim_h() as usize,
        cx.dim_k() as usize,
    );
    let (s, t) = (cx.dim_s() as usize, cx.dim_t() as usize);
    let columns = projection::item_rows(group[0], cx.param_rows(), h);
    let (down, residual, expert_output) = (
        cx.arg_shared_down(),
        cx.arg_residual(),
        cx.arg_expert_output(),
    );
    let (inverse, scores, coefficient, result) = (
        cx.arg_inverse(),
        cx.arg_scores(),
        cx.arg_coefficient(),
        cx.result_0(),
    );
    // SAFETY: the shared launch wrote every row before this launch.
    let shared = unsafe { cx.scratch_shared_product().slice::<f32>(0, m * s) };
    let blocks = seismic::cpu::quant::blocks(s);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_shared_q8()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * blocks)
    });
    let mut projected = [0.0f32; projection::MAX_ROWS];
    let projected = &mut projected[..columns.len()];
    for row in 0..m {
        let q8 = quantized.map(|q8| &q8[row * blocks..(row + 1) * blocks]);
        projection::project_arithmetic(
            &down,
            columns.start,
            &shared[row * s..(row + 1) * s],
            q8,
            projected,
        );
        let mut selected = [0.0f32; projection::MAX_ROWS];
        let selected = &mut selected[..columns.len()];
        for slot in 0..k {
            let position = usize::try_from(inverse.get([row, slot]))
                .expect("a grouped position is non-negative");
            let score = scores.get([row, slot]);
            let published =
                expert_output.span([position / t, position % t, columns.start], columns.len());
            for (selected, published) in selected.iter_mut().zip(published) {
                *selected = score.mul_add(<E::A as Dense>::widen(*published), *selected);
            }
        }
        let coefficient = coefficient.get([row]);
        let source = residual.span([row, columns.start], columns.len());
        // SAFETY: each work item writes its own channels of every row.
        let out = unsafe { result.span_mut([row, columns.start], columns.len()) };
        for (((target, residual), selected), projected) in out
            .iter_mut()
            .zip(source)
            .zip(selected.iter())
            .zip(projected.iter())
        {
            *target = residual + selected + activation::publish::<E::A>(*projected) * coefficient;
        }
    }
}
