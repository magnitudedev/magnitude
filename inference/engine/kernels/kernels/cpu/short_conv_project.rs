// short_conv_project on CPU (contract and portable body in
// short_conv.seismic). `short_conv_project_normalize` publishes each
// RMS-normalized residual row (rounded to A) as F32 into scratch.
// `short_conv_project_rows`: items [0, S) each own ROWS channels of the u
// segment (B and X rows paired, u = B * X in F32), the rest ROWS channels of
// the C segment (stored F32 after the CH u columns).

use lib::projection::projection;

#[inline(always)]
fn short_conv_project_normalize<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    let weight = seismic::cpu::tensor::floats(shared, h);
    cx.arg_norm().decode_row(0, weight);
    // SAFETY: each work item writes its own row of the scratch.
    let normalized = unsafe { cx.scratch_normalized().slice_mut::<f32>(4 * row * h, h) };
    projection::rms_row::<E::A>(cx.arg_residual().row([row, 0]), weight, cx.arg_eps(), normalized);
    if cx.param_int8() == 1 {
        // SAFETY: this work item alone writes row `row`'s quantized bytes.
        unsafe { projection::quantize_row(normalized, row, projection::tiled(cx.dim_m() as usize, &[&cx.arg_b_weight(), &cx.arg_x_weight(), &cx.arg_c_weight()]), cx.scratch_quantized(), 0) };
    }
    projection::operand_row(
        &[&cx.arg_b_weight(), &cx.arg_x_weight(), &cx.arg_c_weight()],
        normalized,
    );
}

#[inline(always)]
fn short_conv_project_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, h, ch) = (cx.dim_m() as usize, cx.dim_h() as usize, cx.dim_ch() as usize);
    let segment = ch.div_ceil(cx.param_rows() as usize) as u64;
    let result = cx.result_0();
    let b_scale = if cx.dim_bs() == 0 { 1.0 } else { cx.arg_b_scale().get([0]) };
    let c_scale = if cx.dim_cs() == 0 { 1.0 } else { cx.arg_c_scale().get([0]) };
    let x_scale = if cx.dim_xs() == 0 { 1.0 } else { cx.arg_x_scale().get([0]) };
    // SAFETY: the normalize launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, m * h) };
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        projection::quantized(cx.scratch_quantized(), 0, m, h, projection::tiled(m, &[&cx.arg_b_weight(), &cx.arg_x_weight(), &cx.arg_c_weight()]))
    });
    if group[0] < segment {
        let rows = projection::item_rows(group[0], cx.param_rows(), ch);
        projection::project_pair_staged_arithmetic(
            (&cx.arg_b_weight(), rows.start),
            (&cx.arg_x_weight(), rows.start),
            rows.len(),
            normalized,
            quantized,
            |row, b, x| {
                // SAFETY: each work item writes its own columns of every row.
                let out = unsafe { result.span_mut([row, rows.start], rows.len()) };
                for ((target, b), x) in out.iter_mut().zip(b).zip(x) {
                    *target = (b_scale * b) * (x_scale * x);
                }
            },
        );
    } else {
        let rows = projection::item_rows(group[0] - segment, cx.param_rows(), ch);
        projection::project_staged_arithmetic(&cx.arg_c_weight(), rows.clone(), normalized, quantized, |row, c| {
            // SAFETY: each work item writes its own columns of every row.
            let out = unsafe { result.span_mut([row, ch + rows.start], rows.len()) };
            for (target, value) in out.iter_mut().zip(c) {
                *target = c_scale * value;
            }
        });
    }
}
