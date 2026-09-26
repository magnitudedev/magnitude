// routed_output on CPU (contract and portable body in routed.seismic).
// `routed_output_stage` stages each row's expert and shared products as F32
// in scratch, one work item per row. `routed_output_rows`: work item (x, m)
// owns ROWS output channels of row m. It projects each choice's expert down
// rows, weighting each published (A-rounded) projection by its score in slot
// order, then the shared expert's down rows, and publishes
//     residual + selected + A(shared) * coefficient.

use lib::core::activation;
use lib::projection::projection;
use lib::routed::routed;

fn routed_output_stage<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let (k, f, s) = (
        cx.dim_k() as usize,
        cx.dim_f() as usize,
        cx.dim_s() as usize,
    );
    let expert_product = cx.arg_expert_product();
    for slot in 0..k {
        let choice = row * k + slot;
        // SAFETY: each work item writes its own row's choices of the scratch.
        let staged = unsafe { cx.scratch_expert().slice_mut::<f32>(4 * choice * f, f) };
        projection::stage::<E::A>(expert_product.row([row, slot, 0]), staged);
        if cx.param_int8() == 1 {
            let blocks = seismic::cpu::quant::blocks(f);
            // SAFETY: this work item owns the corresponding choice row.
            let q8 = unsafe {
                cx.scratch_expert_q8()
                    .slice_mut::<seismic::cpu::quant::Q8Block>(
                        choice * blocks * projection::Q8_BYTES,
                        blocks,
                    )
            };
            projection::quantize(staged, q8);
        }
    }
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_shared().slice_mut::<f32>(4 * row * s, s) };
    projection::stage::<E::A>(cx.arg_shared_product().row([row, 0]), staged);
    if cx.param_int8() == 1 {
        let blocks = seismic::cpu::quant::blocks(s);
        // SAFETY: this work item owns the corresponding shared row.
        let q8 = unsafe {
            cx.scratch_shared_q8()
                .slice_mut::<seismic::cpu::quant::Q8Block>(
                    row * blocks * projection::Q8_BYTES,
                    blocks,
                )
        };
        projection::quantize(staged, q8);
    }
}

fn routed_output_rows<L: Isa, E: Elements>(
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
    let (f, s) = (cx.dim_f() as usize, cx.dim_s() as usize);
    let row = group[1] as usize;
    let columns = projection::item_rows(group[0], cx.param_rows(), h);
    // SAFETY: the stage launch wrote every row before this launch.
    let (expert, shared) = unsafe {
        (
            cx.scratch_expert().slice::<f32>(0, m * k * f),
            cx.scratch_shared().slice::<f32>(0, m * s),
        )
    };
    let (expert_blocks, shared_blocks) = (
        seismic::cpu::quant::blocks(f),
        seismic::cpu::quant::blocks(s),
    );
    let expert_q8 = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_expert_q8()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * k * expert_blocks)
    });
    let shared_q8 = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_shared_q8()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * shared_blocks)
    });
    let (routes, scores, down) = (cx.arg_routes(), cx.arg_scores(), cx.arg_expert_down());
    let mut selected = [0.0f32; projection::MAX_ROWS];
    let mut projected = [0.0f32; projection::MAX_ROWS];
    let (selected, projected) = (
        &mut selected[..columns.len()],
        &mut projected[..columns.len()],
    );
    for slot in 0..k {
        let choice = row * k + slot;
        let first = routed::expert_row(routes.get([row, slot]), h) + columns.start;
        let q8 = expert_q8.map(|q8| &q8[choice * expert_blocks..(choice + 1) * expert_blocks]);
        projection::project_arithmetic(
            &down,
            first,
            &expert[choice * f..(choice + 1) * f],
            q8,
            projected,
        );
        let score = scores.get([row, slot]);
        for (selected, projected) in selected.iter_mut().zip(projected.iter()) {
            *selected = score.mul_add(activation::publish::<E::A>(*projected), *selected);
        }
    }
    let q8 = shared_q8.map(|q8| &q8[row * shared_blocks..(row + 1) * shared_blocks]);
    projection::project_arithmetic(
        &cx.arg_shared_down(),
        columns.start,
        &shared[row * s..(row + 1) * s],
        q8,
        projected,
    );
    let coefficient = cx.arg_coefficient().get([row]);
    let residual = cx.arg_residual().span([row, columns.start], columns.len());
    // SAFETY: each work item writes its own channels of its own row.
    let out = unsafe { cx.result_0().span_mut([row, columns.start], columns.len()) };
    for (((target, residual), selected), projected) in out
        .iter_mut()
        .zip(residual)
        .zip(selected.iter())
        .zip(projected.iter())
    {
        *target = residual + selected + activation::publish::<E::A>(*projected) * coefficient;
    }
}
