// project_rows on CPU (contract and portable body in dense_rows.seismic).
// `project_rows_stage` stages each source row's run of group g as F32 in
// scratch (and its q8 blocks for the INT8 variant), one work item per (row,
// group); the staged rows are group-major, G blocks of M rows. Work item
// (item, g) of `project_rows_rows` projects ROWS of group g's weight rows
// against its staged rows and publishes them in Y at the group's columns.

use lib::projection::projection;

fn project_rows_stage<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (row, g) = (group[0] as usize, group[1] as usize);
    let (m, k) = (cx.dim_m() as usize, cx.dim_k() as usize);
    let staged_row = g * m + row;
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * staged_row * k, k) };
    projection::stage::<E::A>(&cx.arg_source().row([row, 0])[g * k..][..k], staged);
    if cx.param_int8() == 1 {
        let blocks = seismic::cpu::quant::blocks(k);
        // SAFETY: this work item owns the corresponding quantized row.
        let q8 = unsafe {
            cx.scratch_quantized()
                .slice_mut::<seismic::cpu::quant::Q8Block>(
                    staged_row * blocks * seismic::cpu::quant::Q8_BYTES,
                    blocks,
                )
        };
        projection::quantize(staged, q8);
    }
    projection::operand_row(&[&cx.arg_weight()], staged);
}

fn project_rows_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (m, k, n) = (cx.dim_m() as usize, cx.dim_k() as usize, cx.dim_n() as usize);
    let g = group[1] as usize;
    let rows = projection::item_rows(group[0], cx.param_rows(), n);
    let (first, columns) = (g * n + rows.start, rows.len());
    let result = cx.result_0();
    let scale = if cx.dim_ws() == 0 { 1.0 } else { cx.arg_weight_scale().get([0]) };
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(4 * g * m * k, m * k) };
    let blocks = seismic::cpu::quant::blocks(k);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized().slice::<seismic::cpu::quant::Q8Block>(
            g * m * blocks * seismic::cpu::quant::Q8_BYTES,
            m * blocks,
        )
    });
    projection::project_staged_arithmetic(&cx.arg_weight(), first..first + columns, staged, quantized, |row, projected| {
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { result.span_mut([row, first], columns) };
        for (target, projected) in out.iter_mut().zip(projected) {
            *target = E::Y::narrow(scale * *projected);
        }
    });
}
