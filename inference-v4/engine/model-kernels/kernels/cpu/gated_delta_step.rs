// gated_delta_step on CPU (contract and portable body `gated_delta_rows` in
// recurrent.seismic): the row-sequential gated delta rule with its prologue
// fused (`recurrent::Fused`). A work item owns (value head, ROWS state rows,
// slot) and advances its private state rows over every row of the slot; the
// head's first work item publishes the head's share of the successor window.
// Grid z = B zeroes the mixed rows no slot covers. ROWS never changes result
// bits, and `gated_delta_chunk` computes the same bits.

use lib::recurrent::recurrent::{self, Recurrence};

fn recurrence<'a, E: Elements>(cx: &Context<'a, E>) -> Recurrence<'a, E::A> {
    Recurrence {
        projection: cx.arg_projection(),
        convolution: cx.arg_convolution(),
        rate: cx.arg_rate(),
        time_bias: cx.arg_time_bias(),
        segments: cx.arg_segments(),
        stop: cx.arg_stop(),
        previous_bank: cx.arg_previous_bank(),
        previous_tape: cx.arg_previous_tape(),
        following_bank: cx.arg_following_bank(),
        window: cx.arg_window(),
        delta: cx.arg_delta(),
        tape: cx.arg_tape(),
        mixed: cx.result_0(),
        rows: cx.dim_m() as usize,
        slots: cx.dim_b() as usize,
        key_heads: cx.dim_nk() as usize,
        value_heads: cx.dim_nv() as usize,
        width: cx.dim_w() as usize,
        taps: cx.dim_c() as usize,
        tape_rows: cx.dim_t() as usize,
        epsilon: cx.arg_norm_epsilon(),
        grouped: cx.arg_grouped(),
    }
}

fn gated_delta_step<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let r = recurrence(cx);
    let head = group[0] as usize;
    let rows = recurrent::state_rows(group[1], cx.param_rows(), r.width);
    if group[2] as usize == r.slots {
        r.zero_uncovered(head, rows);
        return;
    }
    let slot = r.slot(group[2] as usize);
    if rows.start == 0 {
        r.publish_window(&slot, head);
    }
    let floats = seismic::cpu::tensor::floats(shared, rows.len() * r.width + 2 * r.width + rows.len());
    let (state, buffers) = floats.split_at_mut(rows.len() * r.width);
    let mut prologue = recurrent::Fused::new(&r, slot, head, rows.clone(), buffers);
    r.advance(&slot, head, rows, state, &mut prologue);
}
