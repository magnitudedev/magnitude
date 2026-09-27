// routed_expand on CPU (contract and portable body in routed.seismic).
// `routed_expand_stage` stages each normalized row as F32 in scratch, one
// work item per row. `routed_expand_rows`: work item (x, y) owns feature
// block x of ROWS features; y < M * K is choice (m, k), whose expert gate/up
// rows it expands against row m, and y = M * K is the shared expert, whose
// rows it expands against every row. Each product is A(silu(gate) * up).

use lib::projection::projection;
use lib::routed::routed;

fn routed_expand_stage<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * row * h, h) };
    projection::stage::<E::A>(cx.arg_normalized().row([row, 0]), staged);
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
        projection::quantize(staged, q8);
    }
}

fn routed_expand_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, h, k) = (
        cx.dim_m() as usize,
        cx.dim_h() as usize,
        cx.dim_k() as usize,
    );
    let (f, s) = (cx.dim_f() as usize, cx.dim_s() as usize);
    let choice = group[1] as usize;
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * h) };
    let blocks = seismic::cpu::quant::blocks(h);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * blocks)
    });
    if choice < m * k {
        let rows = projection::item_rows(group[0], cx.param_rows(), f);
        if rows.is_empty() {
            return;
        }
        let (row, slot) = (choice / k, choice % k);
        let first = routed::expert_row(cx.arg_routes().get([row, slot]), f) + rows.start;
        // SAFETY: each work item writes its own features of its own choice.
        let out = unsafe { cx.result_0().span_mut([row, slot, rows.start], rows.len()) };
        let x = &staged[row * h..(row + 1) * h];
        let q8 = quantized.map(|q8| &q8[row * blocks..(row + 1) * blocks]);
        routed::expand_into::<E::A>(
            &cx.arg_expert_gate(),
            &cx.arg_expert_up(),
            first,
            x,
            q8,
            out,
        );
        return;
    }
    let rows = projection::item_rows(group[0], cx.param_rows(), s);
    if rows.is_empty() {
        return;
    }
    let (gate, up) = (cx.arg_shared_gate(), cx.arg_shared_up());
    for row in 0..m {
        // SAFETY: each work item writes its own shared features of every row.
        let out = unsafe { cx.result_1().span_mut([row, rows.start], rows.len()) };
        let q8 = quantized.map(|q8| &q8[row * blocks..(row + 1) * blocks]);
        routed::expand_into::<E::A>(
            &gate,
            &up,
            rows.start,
            &staged[row * h..(row + 1) * h],
            q8,
            out,
        );
    }
}
