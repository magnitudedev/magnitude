// routed_expand on CPU (contract and portable body in routed.seismic).
// The operator's form launch forms each normalized row as F32 (`rows`), its
// Q8 blocks where INT8 = 1, and its NVFP4 operand rows for the experts that
// take them: in place where both the routed and the shared experts do, else
// as a copy after the rows; `routed_route` emits the same operand.
// `routed_expand_rows`: work item (x, y) owns feature block x of ROWS
// features; y < M * K is choice (m, k), whose expert gate/up rows it expands
// against row m, and y = M * K is the shared expert, whose rows it expands
// against every row. Each product is the SiLU `glu` of gs * gate and us *
// up: a choice's scales its expert's, 1 where a port is absent and for the
// shared expert. Each item emits the operands of the products it stored
// (`projection::emit`) toward `routed_output`, in the forms its graph pairs
// (`emits_result_N_rows()`, `emits_result_N_quantized()`,
// `nvfp4_result_N_rows()`): the choice rows' and the shared rows'.

use lib::core::functions;
use lib::projection::projection;
use lib::routed::routed;

/// The operand rows a gate/up pair takes, the recipe of its `gate` (the
/// entry declares that the pair shares it; `projection::operand_rows`).
#[inline(always)]
fn rows_for<'a, E: Elements>(cx: &'a Context<'_, E>, gate: &impl projection::Recipe, m: usize, h: usize) -> &'a [f32] {
    let (eg, eu, sg, su) = (cx.arg_expert_gate(), cx.arg_expert_up(), cx.arg_shared_gate(), cx.arg_shared_up());
    let all: [&dyn projection::Recipe; 4] = [&eg, &eu, &sg, &su];
    // SAFETY: `routed_expand_form` ran before this launch.
    unsafe { projection::operand_rows(cx.scratch_rows(), m, h, gate, &all) }
}

#[inline(always)]
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
        let expert = cx.arg_routes().get([row, slot]);
        let (gate_scale, up_scale) = (cx.arg_gate_scale(), cx.arg_up_scale());
        // SAFETY: each work item writes its own features of its own choice.
        let out = unsafe { cx.result_0().span_mut([row, slot, rows.start], rows.len()) };
        let (gate, up) = (cx.arg_expert_gate(), cx.arg_expert_up());
        let x = &rows_for(cx, &gate, m, h)[row * h..(row + 1) * h];
        let q8 = quantized.map(|q8| &q8[row * blocks..(row + 1) * blocks]);
        routed::expand_into::<E::A>(
            &routed::record(&gate, expert),
            &routed::record(&up, expert),
            rows.start,
            x,
            q8,
            functions::SILU,
            None,
            (
                routed::scale_factor(cx.dim_gs(), |e| gate_scale.get([e]), expert),
                routed::scale_factor(cx.dim_us(), |e| up_scale.get([e]), expert),
            ),
            out,
        );
        if cx.emits_result_0_rows() {
            let products = cx.result_0();
            // SAFETY: the emitted operand's buffers; this item stored these
            // features of choice row `choice`, which arrive once.
            unsafe {
                projection::emit(
                    cx.scratch_result_0_rows(),
                    cx.emits_result_0_quantized().then(|| cx.scratch_result_0_quantized()),
                    cx.nvfp4_result_0_rows(),
                    cx.scratch_result_0_arrivals(),
                    f,
                    choice,
                    rows.len(),
                    |out| projection::stage::<E::A>(products.row([row, slot, 0]), out),
                )
            };
        }
        return;
    }
    let rows = projection::item_rows(group[0], cx.param_rows(), s);
    if rows.is_empty() {
        return;
    }
    let (gate, up) = (cx.arg_shared_gate(), cx.arg_shared_up());
    let staged = rows_for(cx, &gate, m, h);
    let products = cx.result_1();
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
            functions::SILU,
            None,
            // The shared expert binds no per-expert scale.
            (1.0, 1.0),
            out,
        );
        if cx.emits_result_1_rows() {
            // SAFETY: the emitted operand's buffers; this item stored these
            // shared features of row `row`, which arrive once.
            unsafe {
                projection::emit(
                    cx.scratch_result_1_rows(),
                    cx.emits_result_1_quantized().then(|| cx.scratch_result_1_quantized()),
                    cx.nvfp4_result_1_rows(),
                    cx.scratch_result_1_arrivals(),
                    s,
                    row,
                    rows.len(),
                    |out| projection::stage::<E::A>(products.row([row, 0]), out),
                )
            };
        }
    }
}
