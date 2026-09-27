// gated_delta_project on CPU (contract and portable body in
// recurrent.seismic). `gated_delta_project_normalize` publishes each row's
// RMS-normalized hidden row (rounded to A) as F32 into scratch, one work item
// per row. `gated_delta_project_rows` enumerates the segments
// qkv | z | alpha | beta in order, ROWS weight rows per work item, each
// projected against every normalized row and published in A at its segment's
// columns of the projection row.

use lib::core::activation;
use lib::projection::projection;

fn gated_delta_project_normalize<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let normalized = unsafe { cx.scratch_normalized().slice_mut::<f32>(4 * row * h, h) };
    projection::normalize::<E::A>(
        cx.arg_hidden().row([row, 0]),
        &cx.arg_input_norm(),
        cx.arg_epsilon(),
        shared,
        normalized,
    );
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
        projection::quantize(normalized, q8);
    }
}

fn gated_delta_project_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, h) = (cx.dim_m() as usize, cx.dim_h() as usize);
    let (nk, nv, w) = (
        cx.dim_nk() as usize,
        cx.dim_nv() as usize,
        cx.dim_w() as usize,
    );
    let totals = [(2 * nk + nv) * w, nv * w, nv, nv];
    let (segment, rows) = projection::segment_rows(group[0], cx.param_rows(), &totals);
    let weights = [
        cx.arg_qkv_weight(),
        cx.arg_gate_weight(),
        cx.arg_alpha_weight(),
        cx.arg_beta_weight(),
    ][segment];
    let column = totals[..segment].iter().sum::<usize>() + rows.start;
    let result = cx.result_0();
    // SAFETY: the normalize launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, m * h) };
    let blocks = seismic::cpu::quant::blocks(h);
    // The alpha and beta segments project exactly under either arithmetic:
    // they set the recurrence's decay and write strength, where activation
    // rounding compounds over the whole sequence, and they are two of the
    // projection's (2 NK + 2 NV) W + 2 NV rows.
    let quantized = (cx.param_int8() == 1 && segment < 2).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * blocks)
    });
    projection::project_staged_arithmetic(
        &weights,
        rows.clone(),
        normalized,
        quantized,
        |row, projected| {
            // SAFETY: each work item writes its own columns of every row.
            activation::store::<E::A>(projected, unsafe {
                result.span_mut([row, column], rows.len())
            });
        },
    );
}
