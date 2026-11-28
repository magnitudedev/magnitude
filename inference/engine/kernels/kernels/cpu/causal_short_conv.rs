// causal_short_conv on CPU (contract and portable body in short_conv.seismic):
// generalized depthwise causal short convolution with residual add, optional
// per-head dynamic taps, and variance normalization:
//   y = residual_scale * (x + static_conv(x) + dyn_conv(x))
// Work item r < M computes row r; work item M + b publishes slot b's successor
// window.

use lib::recurrent::versions::Slots;
use seismic::cpu::slab::SlabTensor;

#[inline(always)]
fn causal_short_conv<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let slots = Slots {
        segments: cx.arg_segments(),
        stop: cx.arg_stop(),
        previous_bank: cx.arg_previous_bank(),
        previous_tape: cx.arg_previous_tape(),
        following_bank: cx.arg_following_bank(),
        count: cx.dim_b() as usize,
    };
    let input_tensor = cx.arg_input();
    let convolution = cx.arg_convolution();
    let dynamic_taps = cx.arg_dynamic_taps();
    let window = SlabTensor::from_bound(cx.arg_window(), cx.arg_slab_banks() as usize);
    let output = cx.result_0();
    let channels = cx.dim_ch() as usize;
    let taps = cx.dim_c() as usize - 1;
    let rows = cx.dim_m() as usize;
    let hd = cx.dim_hd() as usize;
    let residual_scale = cx.arg_residual_scale();
    let item = group[0] as usize;

    let buf = seismic::cpu::tensor::floats(shared, channels * 3);
    let (static_sum, rest) = buf.split_at_mut(channels);
    let (dyn_sum, row_buf) = rest.split_at_mut(channels);

    if item >= rows {
        let slot = slots.slot(item - rows);
        if slot.lo == slot.hi {
            return;
        }
        for tap in 0..taps + slot.recorded(cx.dim_t() as usize) {
            let position = (slot.stop + tap) as isize - taps as isize;
            // SAFETY: each slot's work item writes its own successor bank,
            // which no slot reads.
            let mut target = unsafe { window.row_mut([slot.target, tap, 0]) };
            if position < 0 {
                let source = window.stored([slot.source, slot.taped + taps - position.unsigned_abs(), 0]);
                target.move_stored(source);
            } else {
                let src_row = slot.lo + position as usize;
                input_tensor.decode_row(src_row, row_buf);
                for (cell, &v) in target.iter_mut().zip(row_buf.iter()) {
                    cell.set(v);
                }
            }
        }
        return;
    }

    // SAFETY: each work item writes its own row.
    let mut out = unsafe { output.row_mut([item, 0]) };
    let Some(slot) = slots.slot_of_row(item) else {
        out.fill(0.0);
        return;
    };
    let local = (item - slot.lo) as isize;
    static_sum.fill(0.0);
    dyn_sum.fill(0.0);
    let head_width = if hd > 0 { channels / hd } else { 1 };

    for tap in 0..=taps {
        let position = local + tap as isize - taps as isize;
        let lag = taps - tap;
        if position < 0 {
            let win_row = window.row([slot.source, slot.taped + taps - position.unsigned_abs(), 0]);
            for ch in 0..channels {
                let v = win_row[ch];
                static_sum[ch] = convolution.get([ch, tap]).mul_add(v, static_sum[ch]);
                if hd > 0 {
                    let head = ch / head_width;
                    dyn_sum[ch] = dynamic_taps.get([item, head, lag]).mul_add(v, dyn_sum[ch]);
                }
            }
        } else {
            input_tensor.decode_row(slot.lo + position as usize, row_buf);
            for ch in 0..channels {
                let v = row_buf[ch];
                static_sum[ch] = convolution.get([ch, tap]).mul_add(v, static_sum[ch]);
                if hd > 0 {
                    let head = ch / head_width;
                    dyn_sum[ch] = dynamic_taps.get([item, head, lag]).mul_add(v, dyn_sum[ch]);
                }
            }
        }
    }
    input_tensor.decode_row(item, row_buf);
    for (ch, target) in out.iter_mut().enumerate() {
        target.set((row_buf[ch] + static_sum[ch] + dyn_sum[ch]) * residual_scale);
    }
}
