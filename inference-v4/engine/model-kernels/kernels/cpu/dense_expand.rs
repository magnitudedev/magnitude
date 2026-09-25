// dense_expand on CPU (contract and portable body in dense_rows.seismic).
// `dense_expand_normalize` publishes the RMS-normalized residual row of each
// output row (rounded to A) as F32 into scratch, one work item per row.
// `dense_expand_rows` gives each work item ROWS gate and up weight rows, which
// it projects against every normalized row and combines as the portable body
// publishes: A(A(silu(A(gate))) * A(up)).

use lib::core::activation;
use lib::projection::projection;

fn dense_expand_normalize<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    let source = cx.arg_out_rows().get([row]) as usize;
    let weight = seismic::cpu::tensor::floats(shared, h);
    cx.arg_norm().decode_row(0, weight);
    // SAFETY: each work item writes its own row of the scratch.
    let normalized = unsafe { cx.scratch_normalized().slice_mut::<f32>(4 * row * h, h) };
    projection::rms_row::<E::A>(
        cx.arg_residual().row([source, 0]),
        weight,
        cx.arg_eps(),
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

fn dense_expand_rows<L: Isa, E: Elements>(
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
    let rows = projection::item_rows(group[0], cx.param_rows(), f);
    let (gate_weight, up_weight, result) =
        (cx.arg_gate_weight(), cx.arg_up_weight(), cx.result_0());
    // SAFETY: the normalize launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, o * h) };
    let blocks = seismic::cpu::quant::blocks(h);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, o * blocks)
    });
    let mut gate = [0.0f32; projection::MAX_ROWS];
    let mut up = [0.0f32; projection::MAX_ROWS];
    let (gate, up) = (&mut gate[..rows.len()], &mut up[..rows.len()]);
    for row in 0..o {
        let x = &normalized[row * h..(row + 1) * h];
        let q8 = quantized.map(|q8| &q8[row * blocks..(row + 1) * blocks]);
        projection::project_arithmetic(&gate_weight, rows.start, x, q8, gate);
        projection::project_arithmetic(&up_weight, rows.start, x, q8, up);
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { result.span_mut([row, rows.start], rows.len()) };
        for ((target, gate), up) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
            let gate = activation::publish::<E::A>(*gate);
            let activated = activation::publish::<E::A>(activation::silu(gate));
            *target = E::A::narrow(activated * activation::publish::<E::A>(*up));
        }
    }
}
