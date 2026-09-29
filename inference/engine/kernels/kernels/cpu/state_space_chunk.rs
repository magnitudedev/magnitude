// state_space_chunk on CPU (contract and portable body `state_space_rows` in
// state_space.seismic). `state_space_chunk_inputs` convolves each row of
// every slot once into the `inputs` scratch (x | B | C after the bias and
// SiLU, then each head's step size and log decay), one work item per row.
// `state_space_chunk_scan` advances the state row-sequentially from the
// staged rows (`state_space::Staged`): a work item owns (head, ROWS state
// rows, slot), the head's first publishes the head's share of the successor
// window, and grid z = B zeroes the mixed rows no slot covers. The staged
// values are the step's, so every slot gets `state_space_step`'s bits: on the
// CPU the row-sequential rule is the cheapest schedule, so the chunk has no
// dual form. ROWS never changes result bits.

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

fn state_space_chunk_inputs<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let s = space(cx);
    let row = group[0] as usize;
    // Rows after the slots are no slot's input.
    let Some(slot) = s.slots.slot_of_row(row) else { return };
    let stride = s.staged_width();
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_inputs().slice_mut::<f32>(4 * row * stride, stride) };
    s.stage_row(&slot, row - slot.lo, staged);
}

fn state_space_chunk_scan<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
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
    // SAFETY: the inputs launch wrote the rows of every slot before this
    // launch; this one only reads them.
    let staged = unsafe { cx.scratch_inputs().slice::<f32>(0, s.rows * s.staged_width()) };
    let state = seismic::cpu::tensor::floats(shared, rows.len() * s.state_width);
    let mut prologue = state_space::Staged::new(&s, staged, head, rows.clone());
    s.advance(&slot, head, rows, state, &mut prologue);
}
