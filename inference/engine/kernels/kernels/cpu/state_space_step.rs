// state_space_step on CPU (contract and portable body `state_space_rows` in
// state_space.seismic): the row-sequential state-space rule with its prologue
// fused (`state_space::Fused`). A work item owns (head, ROWS state rows,
// slot) and advances its private state rows over every row of the slot; the
// head's first work item publishes the head's share of the successor window.
// Grid z = B zeroes the mixed rows no slot covers. ROWS never changes result
// bits, and `state_space_chunk` computes the same bits.

use lib::recurrent::state_space::{self, StateSpace};
use lib::recurrent::versions::Slots;
use seismic::cpu::slab::SlabTensor;

fn space<'a, E: Elements>(cx: &Context<'a, E>) -> StateSpace<'a, E::A> {
    let slab_banks = cx.arg_slab_banks() as usize;
    StateSpace {
        projection: cx.arg_projection(),
        convolution: cx.arg_convolution(),
        convolution_bias: cx.arg_convolution_bias(),
        rate: cx.arg_rate(),
        time_bias: cx.arg_time_bias(),
        skip: cx.arg_skip(),
        slots: Slots {
            segments: cx.arg_segments(),
            stop: cx.arg_stop(),
            previous_bank: cx.arg_previous_bank(),
            previous_tape: cx.arg_previous_tape(),
            following_bank: cx.arg_following_bank(),
            count: cx.dim_b() as usize,
        },
        window: SlabTensor::from_bound(cx.arg_window(), slab_banks),
        state: SlabTensor::from_bound(cx.arg_state(), slab_banks),
        tape: SlabTensor::from_bound(cx.arg_tape(), slab_banks),
        mixed: cx.result_0(),
        rows: cx.dim_m() as usize,
        heads: cx.dim_nh() as usize,
        head_width: cx.dim_p() as usize,
        groups: cx.dim_g() as usize,
        state_width: cx.dim_n() as usize,
        taps: cx.dim_c() as usize,
        tape_rows: cx.dim_t() as usize,
    }
}

fn state_space_step<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let s = space(cx);
    let head = group[0] as usize;
    let rows = state_space::state_rows(group[1], cx.param_rows(), s.head_width);
    if group[2] as usize == s.slots.count {
        s.zero_uncovered(head, rows);
        return;
    }
    let slot = s.slots.slot(group[2] as usize);
    if rows.start == 0 {
        s.publish_window(&slot, head);
    }
    let n = s.state_width;
    let floats = seismic::cpu::tensor::floats(shared, rows.len() * n + 2 * n + rows.len());
    let (state, buffers) = floats.split_at_mut(rows.len() * n);
    let mut prologue = state_space::Fused::new(&s, slot, head, rows.clone(), buffers);
    s.advance(&slot, head, rows, state, &mut prologue);
}
