// gated_delta_chunk on CPU (contract and portable body `gated_delta_rows` in
// recurrent.seismic): the library's chunk over Gated DeltaNet's rule
// (`recurrent::Recurrence`). `gated_delta_chunk_inputs` stages each row of
// every slot once (`Rule::chunk_inputs`), `gated_delta_chunk_scan` advances
// the state row-sequentially from the staged rows (`Rule::chunk_scan`) and
// `gated_delta_chunk_gate` gates each row (`Rule::gate`). Every slot gets
// `gated_delta_step`'s bits; ROWS never changes result bits.

use lib::recurrent::recurrent::{self, Rule};

#[inline(always)]
fn gated_delta_chunk_inputs<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    recurrent::recurrence!(cx).chunk_inputs(group[0] as usize, cx.scratch_inputs());
}

#[inline(always)]
fn gated_delta_chunk_scan<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    recurrent::recurrence!(cx).chunk_scan(group, cx.param_rows(), cx.scratch_inputs(), shared);
}

#[inline(always)]
fn gated_delta_chunk_gate<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let norm = cx.arg_recurrent_norm();
    recurrent::recurrence!(cx).gate(
        group[0] as usize,
        |out| norm.decode_row(0, out),
        cx.arg_epsilon(),
        cx.arg_gate_function(),
        &cx.result_0(),
        shared,
    );
}
