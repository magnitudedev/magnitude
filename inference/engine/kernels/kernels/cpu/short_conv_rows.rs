// short_conv_rows on CPU (contract and portable body in short_conv.seismic):
// the gated causal depthwise convolution of one short-convolution layer. Work
// item r < M computes every channel of row r (zero past the slots); work item
// M + b publishes slot b's successor window. Each output is the body's
// tap-ascending F32 FMA chain times the gate, the bits of every other
// backend. The successor window is no slot's source, so the publishing work
// items run beside the row work items.

use lib::recurrent::versions::{Slot, Slots};
use seismic::cpu::slab::SlabTensor;

fn short_conv_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let slots = Slots {
        segments: cx.arg_segments(),
        stop: cx.arg_stop(),
        previous_bank: cx.arg_previous_bank(),
        previous_tape: cx.arg_previous_tape(),
        following_bank: cx.arg_following_bank(),
        count: cx.dim_b() as usize,
    };
    let projection = cx.arg_projection();
    let convolution = cx.arg_convolution();
    let window = SlabTensor::from_bound(cx.arg_window(), cx.arg_slab_banks() as usize);
    let gated = cx.result_0();
    let channels = cx.dim_ch() as usize;
    let taps = cx.dim_c() as usize - 1;
    let rows = cx.dim_m() as usize;
    let item = group[0] as usize;
    // The input row at slot-local `position`: the source version's window rows
    // before the slot, the projection's u after.
    let input = |slot: &Slot, position: isize| -> &[f32] {
        if position < 0 {
            window.row([slot.source, slot.taped + taps - position.unsigned_abs(), 0])
        } else {
            &projection.row([slot.lo + position as usize, 0])[..channels]
        }
    };
    if item >= rows {
        let slot = slots.slot(item - rows);
        for tap in 0..taps + slot.recorded(cx.dim_t() as usize) {
            let source = input(&slot, (slot.stop + tap) as isize - taps as isize);
            // SAFETY: each slot's work item writes its own successor bank,
            // which no slot reads.
            let target = unsafe { window.row_mut([slot.target, tap, 0]) };
            target.copy_from_slice(source);
        }
        return;
    }
    // SAFETY: each work item writes its own row.
    let out = unsafe { gated.row_mut([item, 0]) };
    let Some(slot) = slots.slot_of_row(item) else {
        out.fill(E::A::narrow(0.0));
        return;
    };
    let local = (item - slot.lo) as isize;
    let gate = &projection.row([item, 0])[channels..];
    let sum = seismic::cpu::tensor::floats(shared, channels);
    sum.fill(0.0);
    for tap in 0..=taps {
        let values = input(&slot, local + tap as isize - taps as isize);
        for ((sum, value), channel) in sum.iter_mut().zip(values).zip(0..channels) {
            *sum = convolution.get([channel, tap]).mul_add(*value, *sum);
        }
    }
    for ((out, sum), gate) in out.iter_mut().zip(sum.iter()).zip(gate) {
        *out = E::A::narrow(gate * sum);
    }
}
