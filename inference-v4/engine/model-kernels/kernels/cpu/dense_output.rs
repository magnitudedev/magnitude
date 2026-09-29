// dense_output on CPU (contract and portable body in dense_rows.seismic).
// `dense_output_stage` stages each product row as F32 in scratch, one work
// item per row. `dense_output_rows` gives each work item ROWS down weight
// rows, which it projects against every staged row: result = the residual row
// plus the projection published to A, as the portable body adds it.

use lib::core::activation;
use lib::projection::projection;

fn dense_output_stage<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let f = cx.dim_f() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * row * f, f) };
    projection::stage::<E::A>(cx.arg_product().row([row, 0]), staged);
    if cx.param_int8() == 1 {
        let blocks = seismic::cpu::quant::blocks(f);
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

fn dense_output_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (o, h, f) = (
        cx.dim_o() as usize,
        cx.dim_h() as usize,
        cx.dim_f() as usize,
    );
    let rows = projection::item_rows(group[0], cx.param_rows(), h);
    let (down_weight, residual, out_rows, result) = (
        cx.arg_down_weight(),
        cx.arg_residual(),
        cx.arg_out_rows(),
        cx.result_0(),
    );
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, o * f) };
    let blocks = seismic::cpu::quant::blocks(f);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, o * blocks)
    });
    projection::project_staged_arithmetic(
        &down_weight,
        rows.clone(),
        staged,
        quantized,
        |row, projected| {
            let source = residual.span([out_rows.get([row]) as usize, rows.start], rows.len());
            // SAFETY: each work item writes its own columns of every row.
            let out = unsafe { result.span_mut([row, rows.start], rows.len()) };
            for ((target, residual), projected) in out.iter_mut().zip(source).zip(projected.iter()) {
                *target = residual + activation::publish::<E::A>(*projected);
            }
        },
    );
}
