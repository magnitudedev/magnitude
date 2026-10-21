// routed_gate_up on CPU (contract and portable body in routed.seismic).
// `routed_gate_up_stage` stages each normalized row as F32 in scratch, one
// work item per row. `routed_gate_up_rows`: work item (x, y) owns feature
// block x of ROWS features of choice y = (m, k), whose expert gate/up rows it
// expands against row m: A(A(act(A(gs * gate))) * A(us * up)), the scales the
// choice's expert's, 1 where a port is absent.

use lib::projection::projection;
use lib::routed::routed;

fn routed_gate_up_stage<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
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
                .slice_mut::<seismic::cpu::quant::Q8Block>(row * blocks * projection::Q8_BYTES, blocks)
        };
        projection::quantize(staged, q8);
    }
}

fn routed_gate_up_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (m, h, k, f) = (cx.dim_m() as usize, cx.dim_h() as usize, cx.dim_k() as usize, cx.dim_f() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), f);
    if rows.is_empty() {
        return;
    }
    let (row, slot) = (group[1] as usize / k, group[1] as usize % k);
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * h) };
    let blocks = seismic::cpu::quant::blocks(h);
    let q8 = (cx.param_int8() == 1).then(|| unsafe {
        &cx.scratch_quantized().slice::<seismic::cpu::quant::Q8Block>(0, m * blocks)[row * blocks..(row + 1) * blocks]
    });
    let expert = cx.arg_routes().get([row, slot]);
    let first = routed::expert_row(expert, f) + rows.start;
    let (gate_scale, up_scale) = (cx.arg_gate_scale(), cx.arg_up_scale());
    // SAFETY: each work item writes its own features of its own choice.
    let out = unsafe { cx.result_0().span_mut([row, slot, rows.start], rows.len()) };
    routed::glu_into::<E::A>(
        &cx.arg_expert_gate(),
        &cx.arg_expert_up(),
        first,
        &staged[row * h..(row + 1) * h],
        q8,
        cx.arg_activation(),
        (
            routed::scale_factor(cx.dim_gs(), |e| gate_scale.get([e]), expert),
            routed::scale_factor(cx.dim_us(), |e| up_scale.get([e]), expert),
        ),
        out,
    );
}
