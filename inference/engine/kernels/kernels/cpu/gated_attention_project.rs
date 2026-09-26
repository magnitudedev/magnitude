// gated_attention_project on CPU (contract and portable body in
// attention.seismic). `gated_attention_project_normalize` publishes each
// row's RMS-normalized hidden row (rounded to A) as F32 into scratch, one work
// item per row. `gated_attention_project_rows` enumerates the segments
// query+gate | key | value in order, ROWS weight rows per work item, each
// projected against every normalized row and published to its result in A.

use lib::core::activation;
use lib::projection::projection;

fn gated_attention_project_normalize<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let normalized = unsafe { cx.scratch_normalized().slice_mut::<f32>(4 * row * d, d) };
    projection::normalize::<E::A>(
        cx.arg_hidden().row([row, 0]),
        &cx.arg_input_norm(),
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

fn gated_attention_project_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, d) = (cx.dim_m() as usize, cx.dim_d() as usize);
    let (kv, g, w) = (
        cx.dim_kv() as usize,
        cx.dim_g() as usize,
        cx.dim_w() as usize,
    );
    let (segment, rows) =
        projection::segment_rows(group[0], cx.param_rows(), &[kv * g * 2 * w, kv * w, kv * w]);
    let weights = [
        cx.arg_query_gate_weight(),
        cx.arg_key_weight(),
        cx.arg_value_weight(),
    ][segment];
    let result = [cx.result_0(), cx.result_1(), cx.result_2()][segment];
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
            // SAFETY: each work item writes its own columns of every row of its segment's result.
            activation::store::<E::A>(projected, unsafe {
                result.span_mut([row, rows.start], rows.len())
            });
        },
    );
}
