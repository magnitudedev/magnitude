// gated_delta_step on CPU (contract and portable body `gated_delta_rows` in
// recurrent.seismic): the library's step over Gated DeltaNet's rule
// (`recurrent::Recurrence`; `Rule::step`, `Rule::gate`), emitting each gated
// row's operand (`projection::emit`) for the output projection in the forms
// its graph pairs. ROWS never changes result bits, and `gated_delta_chunk`
// computes the same bits.

use lib::projection::projection;
use lib::recurrent::recurrent::{self, Rule};

#[inline(always)]
fn gated_delta_step<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    recurrent::recurrence!(cx).step(group, cx.param_rows(), shared);
}

#[inline(always)]
fn gated_delta_step_gate<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let norm = cx.arg_recurrent_norm();
    let (row, gated) = (group[0] as usize, cx.result_0());
    recurrent::recurrence!(cx).gate(
        row,
        |out| norm.decode_row(0, out),
        cx.arg_epsilon(),
        cx.arg_gate_function(),
        &gated,
        shared,
    );
    if cx.emits_result_0_rows() {
        let width = (cx.dim_nv() * cx.dim_w()) as usize;
        // SAFETY: the emitted operand's buffers; this item stored the whole
        // row, which arrives once.
        unsafe {
            projection::emit(
                cx.scratch_result_0_rows(),
                cx.emits_result_0_quantized().then(|| cx.scratch_result_0_quantized()),
                cx.nvfp4_result_0_rows(),
                cx.scratch_result_0_arrivals(),
                width,
                row,
                width,
                |out| projection::stage_heads::<E::A>(&gated, row, out),
            )
        };
    }
}
