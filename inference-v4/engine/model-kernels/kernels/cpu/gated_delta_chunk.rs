// gated_delta_chunk on CPU (contract and portable body `gated_delta_rows` in
// recurrent.seismic). `gated_delta_chunk_inputs` convolves and normalizes
// each row of every slot once into the `inputs` scratch (q, k and v
// channels, then beta and the decay of each value head), one work item per
// row. `gated_delta_chunk_scan` advances the state row-sequentially from the
// staged rows (`recurrent::Staged`): a work item owns (value head, ROWS state
// rows, slot), the head's first publishes the head's share of the successor
// window, and grid z = B zeroes the mixed rows no slot covers. The staged
// values are the step's, so every slot gets `gated_delta_step`'s bits (the
// CPU needs no WY form: the row-sequential rule is its cheapest schedule);
// ROWS never changes result bits.

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

fn gated_delta_chunk_inputs<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let r = recurrence(cx);
    let row = group[0] as usize;
    // Rows after the slots are no slot's input.
    let Some(slot) = r.slot_of_row(row) else { return };
    let stride = r.staged_width();
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_inputs().slice_mut::<f32>(4 * row * stride, stride) };
    r.stage_row(&slot, row - slot.lo, staged);
}

fn gated_delta_chunk_scan<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
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
    // SAFETY: the inputs launch wrote the rows of every slot before this
    // launch; this one only reads them.
    let staged = unsafe { cx.scratch_inputs().slice::<f32>(0, r.rows * r.staged_width()) };
    let state = seismic::cpu::tensor::floats(shared, rows.len() * r.width);
    let mut prologue = recurrent::Staged::new(&r, staged, head, rows.clone());
    r.advance(&slot, head, rows, state, &mut prologue);
}
