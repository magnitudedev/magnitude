// readout_selected_rows on CPU (contract and portable body in
// readout.seismic). `readout_selected_rows_normalize` publishes the
// RMS-normalized hidden row of each output row (rounded to A) as F32 into
// scratch, one work item per row. `readout_selected_rows_rows` gives each work
// item ROWS selected vocabulary rows; the selected weight rows are not
// adjacent, so each is one single-row component call per normalized row.

use lib::projection::projection;

fn readout_selected_rows_normalize<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    let source = cx.arg_out_rows().get([row]) as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let normalized = unsafe { cx.scratch_normalized().slice_mut::<f32>(4 * row * d, d) };
    projection::normalize::<E::A>(
        cx.arg_hidden().row([source, 0]),
        &cx.arg_norm(),
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

fn readout_selected_rows_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (o, d, sv) = (
        cx.dim_o() as usize,
        cx.dim_d() as usize,
        cx.dim_sv() as usize,
    );
    let columns = projection::item_rows(group[0], cx.param_rows(), sv);
    let (weight, selected, result) = (cx.arg_weight(), cx.arg_selected(), cx.result_0());
    // SAFETY: the normalize launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, o * d) };
    let blocks = seismic::cpu::quant::blocks(d);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, o * blocks)
    });
    for column in columns {
        let vocabulary = selected.get([column]) as usize;
        for (row, x) in normalized.chunks_exact(d).enumerate() {
            let mut logit = [0.0f32];
            let q8 = quantized.map(|q8| &q8[row * blocks..(row + 1) * blocks]);
            projection::project_arithmetic(&weight, vocabulary, x, q8, &mut logit);
            // SAFETY: each work item writes its own columns of every row.
            unsafe { result.set([row, column], logit[0]) };
        }
    }
}
