// ngram_inject on CPU (contract and portable body in ngram.seismic): the
// n-gram input's dilated causal convolution over its banked window, added
// with the gated value into the streams. Work item r < M computes every
// channel of row r (a copy of `hidden` past the slots); work item M + b
// publishes slot b's successor window. A launch row's u value is formed from
// the scales, the value row and the norm, as the body forms it.

use lib::core::activation;
use lib::recurrent::versions::Slots;
use seismic::cpu::slab::SlabTensor;

#[inline(always)]
fn decode_norm<'a, E: Elements>(
    cx: &Context<'_, E>,
    shared: &'a mut [u8],
    s: usize,
    h: usize,
) -> &'a [f32] {
    let norm = seismic::cpu::tensor::floats(shared, s * h);
    for stream in 0..s {
        cx.arg_norm()
            .decode_row(stream, &mut norm[stream * h..(stream + 1) * h]);
    }
    norm
}

#[inline(always)]
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
    let scales = cx.arg_scales();
    let value = cx.arg_value();
    let convolution = cx.arg_convolution();
    let window = SlabTensor::from_bound(cx.arg_window(), cx.arg_slab_banks() as usize);
    let rows = cx.dim_m() as usize;
    let item = group[0] as usize;
    if item >= rows {
        let slot = slots.slot(item - rows);
        if slot.lo == slot.hi {
            return;
        }
        let norm = decode_norm(cx, shared, s, h);
        for tap in 0..reach + slot.recorded(cx.dim_r() as usize - reach) {
            let position = (slot.stop + tap) as isize - reach as isize;
            // SAFETY: each slot's work item writes its own successor bank,
            // which no slot reads.
            let mut target = unsafe { window.row_mut([slot.target, tap, 0]) };
            if position < 0 {
                let source = window.stored([
                    slot.source,
                    slot.taped + reach - position.unsigned_abs(),
                    0,
                ]);
                target.move_stored(source);
            } else {
                let row = slot.lo + position as usize;
                let row_scales = scales.row([row, 0]);
                let row_value = &value.row([row, 0])[..h];
                target.compute(|dst| {
                    for stream in 0..s {
                        let stream_base = stream * h;
                        let gate = row_scales[stream];
                        let inv = row_scales[s + stream];
                        let stream_norm = &norm[stream_base..stream_base + h];
                        let stream_dst = &mut dst[stream_base..stream_base + h];
                        for column in 0..h {
                            stream_dst[column] =
                                gate * row_value[column] * inv * stream_norm[column];
                        }
                    }
                });
            }
        }
        return;
    }
    // SAFETY: each work item writes its own row.
    let mut out = unsafe { cx.result_0().row_mut([item, 0]) };
    let Some(slot) = slots.slot_of_row(item) else {
        out.move_stored(cx.arg_hidden().stored([item, 0]));
        return;
    };
    let norm = decode_norm(cx, shared, s, h);
    let hidden = &cx.arg_hidden().row([item, 0])[..channels];
    let local = (item - slot.lo) as isize;
    let item_scales = scales.row([item, 0]);
    let item_value = &value.row([item, 0])[..h];
    out.compute(|dst| {
        for stream in 0..s {
            let stream_base = stream * h;
            let stream_dst = &mut dst[stream_base..stream_base + h];
            let stream_norm = &norm[stream_base..stream_base + h];
            let stream_hidden = &hidden[stream_base..stream_base + h];
            stream_dst.fill(0.0);
            for tap in 0..taps {
                let position = local + (dilation * tap) as isize - reach as isize;
                if position < 0 {
                    let src = &window.row([
                        slot.source,
                        slot.taped + reach - position.unsigned_abs(),
                        0,
                    ])[stream_base..stream_base + h];
                    for column in 0..h {
                        stream_dst[column] = convolution
                            .get([stream_base + column, tap])
                            .mul_add(src[column], stream_dst[column]);
                    }
                } else {
                    let row = slot.lo + position as usize;
                    let tap_scales = scales.row([row, 0]);
                    let gate = tap_scales[stream];
                    let inv = tap_scales[s + stream];
                    let tap_value = &value.row([row, 0])[..h];
                    for column in 0..h {
                        let u = gate * tap_value[column] * inv * stream_norm[column];
                        stream_dst[column] = convolution
                            .get([stream_base + column, tap])
                            .mul_add(u, stream_dst[column]);
                    }
                }
            }
            let gate = item_scales[stream];
            for column in 0..h {
                let gated = gate * item_value[column];
                stream_dst[column] =
                    stream_hidden[column] + (gated + activation::silu(stream_dst[column]));
            }
        }
    });
}
