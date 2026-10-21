// attention_project on CPU (contract and portable body in attention.seismic).
// `attention_project_normalize` publishes each row's RMS-normalized hidden row
// (rounded to A) as F32 into scratch, one work item per row.
// `attention_project_rows` enumerates the segments query | gate | key | value in
// order, ROWS weight rows per work item, each projected against every
// normalized row, times the segment weight's second-level scale where it has
// one, and published to its result in A.

use lib::core::activation;
use lib::projection::projection;

fn attention_project_normalize<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let normalized = unsafe { cx.scratch_normalized().slice_mut::<f32>(4 * row * d, d) };
    let norm = cx.arg_input_norm();
    projection::form_input::<E::A>(
        cx.arg_hidden().row([row, 0]),
        (cx.dim_nm() != 0).then_some(&norm),
        cx.arg_epsilon(),
        shared,
        normalized,
    );
    if cx.param_int8() == 1 {
        let blocks = seismic::cpu::quant::blocks(d);
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

fn attention_project_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, d) = (cx.dim_m() as usize, cx.dim_d() as usize);
    let segments = [cx.dim_q() as usize, cx.dim_gr() as usize, cx.dim_k() as usize, cx.dim_v() as usize];
    let (segment, rows) = projection::segment_rows(group[0], cx.param_rows(), &segments);
    let weights = [
        cx.arg_query_weight(),
        cx.arg_gate_weight(),
        cx.arg_key_weight(),
        cx.arg_value_weight(),
    ][segment];
    let result = [cx.result_0(), cx.result_1(), cx.result_2(), cx.result_3()][segment];
    // The segment weight's second-level scale, 1 where its port is absent.
    let scale = match segment {
        0 if cx.dim_qs() != 0 => cx.arg_query_scale().get([0]),
        1 if cx.dim_gs() != 0 => cx.arg_gate_scale().get([0]),
        2 if cx.dim_ks() != 0 => cx.arg_key_scale().get([0]),
        3 if cx.dim_vs() != 0 => cx.arg_value_scale().get([0]),
        _ => 1.0,
    };
    if segment < 2 && cx.arg_project_mode() != 0 {
        let zero = vec![0.0f32; rows.len()];
        for row in 0..m {
            // SAFETY: this work item owns these columns of this segment's result.
            activation::store::<E::A>(&zero, unsafe {
                result.span_mut([row, rows.start], rows.len())
            });
        }
        return;
    }
    // SAFETY: the normalize launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, m * d) };
    let blocks = seismic::cpu::quant::blocks(d);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * blocks)
    });
    projection::project_staged_arithmetic(
        &weights,
        rows.clone(),
        normalized,
        quantized,
        |row, projected| {
            let mut scaled = [0.0f32; projection::MAX_ROWS];
            let scaled = &mut scaled[..projected.len()];
            for (target, value) in scaled.iter_mut().zip(projected) {
                *target = scale * value;
            }
            // SAFETY: each work item writes its own columns of every row of its segment's result.
            activation::store::<E::A>(scaled, unsafe {
                result.span_mut([row, rows.start], rows.len())
            });
        },
    );
}
