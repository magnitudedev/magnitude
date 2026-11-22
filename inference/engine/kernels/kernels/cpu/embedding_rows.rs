// embedding_rows on CPU (contract and portable body in target.seismic): one
// work item per token row, which decodes its table row, multiplies it by
// `scale`, with `normalize` applies its weightless RMS, rounds it to A and
// stores it both as A and, widened again, as F32.

use lib::core::{activation, gathered};
use seismic::cpu::reduce::{LANES, combine};

#[inline(always)]
fn embedding_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let token = gathered::row(cx.arg_tokens().get([row, 0]), cx.dim_r(), cx.dim_f(), |entry, field| {
        cx.arg_directory().get([entry, field])
    });
    // SAFETY: each work item writes its own row of both results.
    let (mut embedded, mut widened) = unsafe { (cx.result_0().row_mut([row, 0]), cx.result_1().row_mut([row, 0])) };
    widened.compute(|widened| {
        // A row that names no table row reads zeros.
        match token {
            Some(token) => cx.arg_table().decode_row(token, widened),
            None => widened.fill(0.0),
        }
        let scale = cx.arg_scale();
        let factor = if cx.arg_normalize() != 0 {
            let mut lanes = [0.0f32; LANES];
            let mut chunks = widened.chunks_exact_mut(LANES);
            for chunk in &mut chunks {
                for lane in 0..LANES {
                    let scaled = chunk[lane] * scale;
                    chunk[lane] = scaled;
                    lanes[lane] = scaled.mul_add(scaled, lanes[lane]);
                }
            }
            for (lane, value) in chunks.into_remainder().iter_mut().enumerate() {
                let scaled = *value * scale;
                *value = scaled;
                lanes[lane] = scaled.mul_add(scaled, lanes[lane]);
            }
            1.0 / (combine(lanes) / widened.len() as f32 + cx.arg_epsilon()).sqrt()
        } else {
            scale
        };
        for (slot, value) in embedded.iter_mut().zip(widened.iter_mut()) {
            let scaled = *value * factor;
            slot.set(scaled);
            *value = activation::publish::<E::A>(scaled);
        }
    });
}
