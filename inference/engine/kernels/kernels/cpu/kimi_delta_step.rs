// kimi_delta_step on CPU (contract and portable body `kimi_delta_rows` in
// recurrent.seismic): the library's step over Kimi's rule
// (`kimi::Kimi`; `Rule::step`, `Rule::gate`). ROWS never changes
// result bits, and `kimi_delta_chunk` computes the same bits.

use lib::recurrent::kimi::{self, kimi_rule};
use lib::recurrent::recurrent::{self, Rule};

#[inline(always)]
fn kimi_delta_step<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    kimi_rule!(cx).step(group, cx.param_rows(), shared);
}

#[inline(always)]
fn kimi_delta_step_gate<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let norm = cx.arg_recurrent_norm();
    kimi_rule!(cx).gate(
        group[0] as usize,
        |out| norm.decode_row(0, out),
        cx.arg_epsilon(),
        cx.arg_gate_function(),
        &cx.result_0(),
        shared,
    );
}
