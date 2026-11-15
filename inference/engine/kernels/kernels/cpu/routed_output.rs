// routed_output on CPU (contract and portable body in routed.seismic).
// The operator's form launch forms each row's expert and shared products as
// F32 (`edw_rows`, `sdw_rows`), their Q8 blocks where INT8 = 1, and NVFP4
// operands where the down weights take them; `routed_expand` emits the same
// operands. `routed_output_rows`: work item
// (x, m) owns ROWS output channels of row m. It projects each choice's
// expert down rows, weighting each published (A-rounded) projection by its
// score in slot order, then the shared expert's down rows, and publishes
//     residual + selected + A(shared) * coefficient.
// The row's last item then forms the next projection's operand from the
// whole row under `operand_norm` (`projection::emit`), in the forms its
// graph pairs (`emits_result_0_rows()`, `emits_result_0_quantized()`,
// `nvfp4_result_0_rows()`): the normalized rows, their Q8 blocks and their
// NVFP4 operand rows, which that projection's form launch would otherwise
// form.

use lib::core::activation;
use lib::projection::projection;
use lib::routed::routed;

#[inline(always)]
fn routed_output_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared_bytes: &mut [u8],
) {
    let (m, h, k) = (
        cx.dim_m() as usize,
        cx.dim_h() as usize,
        cx.dim_k() as usize,
    );
    let (f, s) = (cx.dim_f() as usize, cx.dim_s() as usize);
    let row = group[1] as usize;
    let columns = projection::item_rows(group[0], cx.param_rows(), h);
    // SAFETY: the form launch (or the products' producer) wrote every row before this launch.
    let (expert, shared) = unsafe {
        (
            cx.scratch_edw_rows().slice::<f32>(0, m * k * f),
            cx.scratch_sdw_rows().slice::<f32>(0, m * s),
        )
    };
    let (expert_blocks, shared_blocks) = (
        seismic::cpu::quant::blocks(f),
        seismic::cpu::quant::blocks(s),
    );
    let expert_q8 = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_edw_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * k * expert_blocks)
    });
    let shared_q8 = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_sdw_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * shared_blocks)
    });
    let (routes, scores, down) = (cx.arg_routes(), cx.arg_scores(), cx.arg_expert_down());
    let tier = cx.arg_tier_down();
    let sources = cx.arg_sources();
    let record = |slot: usize| routed::decode_source(cx.dim_xs(), || sources.get([0, row, slot]), routes.get([row, slot]));
    let mut selected = [0.0f32; projection::MAX_ROWS];
    let mut projected = [0.0f32; projection::MAX_ROWS];
    let (selected, projected) = (
        &mut selected[..columns.len()],
        &mut projected[..columns.len()],
    );
    for slot in 0..k {
        let choice = row * k + slot;
        // A choice its slot map names no record for reads none: a zero
        // projection.
        match record(slot) {
            Some(place) => {
                let down = place.of(&down, &tier);
                let q8 = expert_q8.map(|q8| &q8[choice * expert_blocks..(choice + 1) * expert_blocks]);
                projection::project_arithmetic(&down, columns.start, &expert[choice * f..(choice + 1) * f], q8, projected);
            }
            None => projected.fill(0.0),
        }
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
    let (h, streams, injected) = (cx.dim_h() as usize, cx.dim_sn() as usize, cx.dim_in() != 0);
    for stream in 0..streams {
        let first = stream * h + columns.start;
        let factor = injected.then(|| cx.arg_inject().get([row, stream]));
        let residual = cx.arg_residual().span([row, first], columns.len());
        // SAFETY: each work item writes its own channels of its own row.
        let mut out = unsafe { cx.result_0().span_mut([row, first], columns.len()) };
        for (((target, residual), selected), projected) in out
            .iter_mut()
            .zip(residual)
            .zip(selected.iter())
            .zip(projected.iter())
        {
            let shared = activation::publish::<E::A>(*projected) * coefficient;
            target.set(match factor {
                None => residual + selected + shared,
                Some(factor) => residual + (selected + shared) * factor,
            });
        }
    }
    if cx.emits_result_0_rows() {
        let norm = cx.arg_operand_norm();
        // SAFETY: the emitted operand's buffers over rows of H values (one
        // stream: the contract's `NO == H and SN == 1`); this item stored
        // its channels of the row, and the row's last arrival reads it whole
        // once every item has.
        unsafe {
            projection::emit(
                cx.scratch_result_0_rows(),
                cx.emits_result_0_quantized().then(|| cx.scratch_result_0_quantized()),
                cx.nvfp4_result_0_rows(),
                cx.scratch_result_0_arrivals(),
                h,
                row,
                columns.len(),
                |out| {
                    projection::form_input::<E::A>(
                        cx.result_0().span([row, 0], h),
                        (cx.dim_no() != 0).then_some(&norm),
                        cx.arg_operand_epsilon(),
                        shared_bytes,
                        out,
                    )
                },
            );
        }
    }
}
