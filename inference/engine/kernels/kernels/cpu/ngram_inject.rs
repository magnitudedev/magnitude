// ngram_inject on CPU (contract and portable body in ngram.seismic): the
// n-gram input's dilated causal convolution over its banked window, added
// with the gated value into the streams. Work item r < M computes every
// channel of row r (a copy of `hidden` past the slots); work item M + b
// publishes slot b's successor window. A launch row's u value is formed from
// the scales, the value row and the norm, as the body forms it.

use lib::core::activation;
use lib::recurrent::versions::{Slot, Slots};
use seismic::cpu::slab::SlabTensor;

fn ngram_inject<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let slots = Slots {
        segments: cx.arg_segments(),
        stop: cx.arg_stop(),
        previous_bank: cx.arg_previous_bank(),
        previous_tape: cx.arg_previous_tape(),
        following_bank: cx.arg_following_bank(),
        count: cx.dim_b() as usize,
    };
    let (h, s) = (cx.dim_h() as usize, cx.dim_s() as usize);
    let channels = s * h;
    let dilation = cx.arg_dilation() as usize;
    let taps = cx.dim_k() as usize;
    let reach = dilation * (taps - 1);
    let norm = seismic::cpu::tensor::floats(shared, channels);
    for stream in 0..s {
        cx.arg_norm()
            .decode_row(stream, &mut norm[stream * h..(stream + 1) * h]);
    }
    let scales = cx.arg_scales();
    let value = cx.arg_value();
    let convolution = cx.arg_convolution();
    let window = SlabTensor::from_bound(cx.arg_window(), cx.arg_slab_banks() as usize);
    let rows = cx.dim_m() as usize;
    let item = group[0] as usize;
    // u of channel `channel` at launch row `row`.
    let u = |row: usize, channel: usize| -> f32 {
        let (stream, column) = (channel / h, channel % h);
        let scales = scales.row([row, 0]);
        scales[stream] * value.get([row, column]) * scales[s + stream] * norm[channel]
    };
    // The input at slot-local `position` of channel `channel`: the source
    // version's window rows before the slot, the launch's u after.
    let input = |slot: &Slot, position: isize, channel: usize| -> f32 {
        if position < 0 {
            window.row([slot.source, slot.taped + reach - position.unsigned_abs(), 0])[channel]
        } else {
            u(slot.lo + position as usize, channel)
        }
    };
    if item >= rows {
        let slot = slots.slot(item - rows);
        for tap in 0..reach + slot.recorded(cx.dim_r() as usize - reach) {
            let position = (slot.stop + tap) as isize - reach as isize;
            // SAFETY: each slot's work item writes its own successor bank,
            // which no slot reads.
            let target = unsafe { window.row_mut([slot.target, tap, 0]) };
            for (channel, target) in target[..channels].iter_mut().enumerate() {
                *target = input(&slot, position, channel);
            }
        }
        return;
    }
    let hidden = &cx.arg_hidden().row([item, 0])[..channels];
    // SAFETY: each work item writes its own row.
    let out = unsafe { cx.result_0().row_mut([item, 0]) };
    let Some(slot) = slots.slot_of_row(item) else {
        out[..channels].copy_from_slice(hidden);
        return;
    };
    let local = (item - slot.lo) as isize;
    let scales = scales.row([item, 0]);
    for channel in 0..channels {
        let mut sum = 0.0f32;
        for tap in 0..taps {
            let position = local + (dilation * tap) as isize - reach as isize;
            sum = convolution
                .get([channel, tap])
                .mul_add(input(&slot, position, channel), sum);
        }
        let (stream, column) = (channel / h, channel % h);
        let gated = scales[stream] * value.get([item, column]);
        out[channel] = hidden[channel] + (gated + activation::silu(sum));
    }
}
