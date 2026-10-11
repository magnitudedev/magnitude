// project_rows on CPU (contract and portable body in dense_rows.seismic).
// `project_rows_stage` stages each source row as F32 in scratch (and its q8
// blocks for the INT8 variant), one work item per row. `project_rows_rows`
// gives each work item ROWS weight rows, projected against every staged row
// and published in Y.

use lib::projection::projection;

fn project_rows_stage<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let k = cx.dim_k() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * row * k, k) };
    cx.arg_source().decode_row(row, staged);
    if cx.param_int8() == 1 {
        let blocks = seismic::cpu::quant::blocks(k);
        // SAFETY: this work item owns the corresponding quantized row.
        let q8 = unsafe {
            cx.scratch_quantized()
                .slice_mut::<seismic::cpu::quant::Q8Block>(row * blocks * projection::Q8_BYTES, blocks)
        };
        projection::quantize(staged, q8);
    }
}

fn project_rows_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (m, k, n) = (cx.dim_m() as usize, cx.dim_k() as usize, cx.dim_n() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), n);
    let result = cx.result_0();
    let scale = if cx.dim_ws() == 0 { 1.0 } else { cx.arg_weight_scale().get([0]) };
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * k) };
    let blocks = seismic::cpu::quant::blocks(k);
    let quantized = (cx.param_int8() == 1)
        .then(|| unsafe { cx.scratch_quantized().slice::<seismic::cpu::quant::Q8Block>(0, m * blocks) });
    projection::project_staged_arithmetic(&cx.arg_weight(), rows.clone(), staged, quantized, |row, projected| {
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { result.span_mut([row, rows.start], rows.len()) };
        for (target, projected) in out.iter_mut().zip(projected) {
            *target = E::Y::narrow(scale * *projected);
        }
    });
}
