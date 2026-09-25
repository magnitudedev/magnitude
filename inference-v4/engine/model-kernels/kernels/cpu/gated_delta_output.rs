// gated_delta_output on CPU (contract and portable body in
// recurrent.seismic). `gated_delta_output_stage` forms each row's gated
// prologue as F32 in scratch, one work item per row: per value head, the
// mixed values RMS-normalized over W and scaled by the recurrent norm (A),
// times SiLU(z) (A), published to A. `gated_delta_output_rows` gives each
// work item ROWS output weight rows, which it projects against every staged
// row: result = the hidden row plus the projection published to A.

use lib::core::{activation, reduce};
use lib::projection::projection;

fn gated_delta_output_stage<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let (nk, nv, w) = (
        cx.dim_nk() as usize,
        cx.dim_nv() as usize,
        cx.dim_w() as usize,
    );
    let z = (2 * nk + nv) * w;
    let (mixed, projected, epsilon) = (cx.arg_mixed(), cx.arg_projection(), cx.arg_epsilon());
    let (norm, values) = seismic::cpu::tensor::floats(shared, 2 * w).split_at_mut(w);
    cx.arg_recurrent_norm().decode_row(0, norm);
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe {
        cx.scratch_staged()
            .slice_mut::<f32>(4 * row * nv * w, nv * w)
    };
    for (head, target) in staged.chunks_exact_mut(w).enumerate() {
        activation::widen::<E::A>(mixed.row([row, head, 0]), values);
        let inverse = reduce::rms_inverse(values, epsilon);
        let gates = projected.span([row, z + head * w], w);
        for (((target, value), weight), gate) in target
            .iter_mut()
            .zip(values.iter())
            .zip(norm.iter())
            .zip(gates)
        {
            let normalized = activation::publish::<E::A>(value * inverse * weight);
            let activated = activation::publish::<E::A>(activation::silu(E::A::widen(*gate)));
            *target = activation::publish::<E::A>(normalized * activated);
        }
    }
    if cx.param_int8() == 1 {
        let blocks = seismic::cpu::quant::blocks(nv * w);
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

fn gated_delta_output_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, h, k) = (
        cx.dim_m() as usize,
        cx.dim_h() as usize,
        (cx.dim_nv() * cx.dim_w()) as usize,
    );
    let rows = projection::item_rows(group[0], cx.param_rows(), h);
    let (hidden, result) = (cx.arg_hidden(), cx.result_0());
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * k) };
    let blocks = seismic::cpu::quant::blocks(k);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * blocks)
    });
    projection::project_staged_arithmetic(
        &cx.arg_output_weight(),
        rows.clone(),
        staged,
        quantized,
        |row, projected| {
            let source = hidden.span([row, rows.start], rows.len());
            // SAFETY: each work item writes its own columns of every row.
            let out = unsafe { result.span_mut([row, rows.start], rows.len()) };
            for ((target, hidden), projected) in out.iter_mut().zip(source).zip(projected) {
                *target = hidden + activation::publish::<E::A>(*projected);
            }
        },
    );
}
