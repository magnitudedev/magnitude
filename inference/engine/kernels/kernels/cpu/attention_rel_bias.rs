// attention_rel_bias on CPU (contract and portable body in attention.seismic):
// attention with in-kernel relative-position bias, LSE-fused static/dynamic
// per-token sinks, and optional output gate. One work item per (row, kv_head).

use seismic::cpu::slab::SlabTensor;

#[inline(always)]
fn gated_column(attended: f32, gate: f32, gate_function: i32) -> f32 {
    if gate_function != 0 {
        attended * (gate.max(0.0) + (-gate.abs()).exp().ln_1p())
    } else {
        attended / (1.0 + (-gate).exp())
    }
}

#[inline(always)]
fn attention_rel_bias<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let kv = cx.dim_kv() as usize;
    let g = cx.dim_g() as usize;
    let w = cx.dim_w() as usize;
    let ext = cx.dim_ext() as usize;
    let u = cx.dim_u() as usize;
    let sk = cx.dim_sk() as usize;
    let r = cx.dim_r() as usize;
    let m = cx.dim_m() as usize;
    let item = group[0] as usize;
    let row = item / kv;
    let kv_head = item % kv;

    let query = cx.arg_query();
    let gate = cx.arg_gate();
    let key = cx.arg_key();
    let value = cx.arg_value();
    let rel_bias = cx.arg_rel_bias();
    let sinks = cx.arg_sinks();
    let visible = cx.arg_visible();
    let fresh = cx.arg_fresh();
    let destinations = cx.arg_destinations();
    let slab_rows = cx.arg_slab_rows() as usize;
    let history_key = SlabTensor::from_bound(cx.arg_history_key(), slab_rows);
    let history_value = SlabTensor::from_bound(cx.arg_history_value(), slab_rows);
    let scale = cx.arg_scale();
    let gate_function = cx.arg_gate_function();

    let mut total_keys: isize = 0;
    for span in 0..r {
        let lo = visible.get([row, span, 0]) as isize;
        let hi = visible.get([row, span, 1]) as isize;
        if hi > lo && lo >= 0 {
            total_keys += hi - lo;
        }
    }
    let fresh_lo = fresh.get([row, 0]) as isize;
    let fresh_hi = fresh.get([row, 1]) as isize;
    if fresh_hi > fresh_lo && fresh_lo >= 0 {
        total_keys += fresh_hi - fresh_lo;
    }

    let buf = seismic::cpu::tensor::floats(shared, g * (2 + w));
    let (maximum, rest) = buf.split_at_mut(g);
    let (denominator, acc) = rest.split_at_mut(g);
    maximum.fill(f32::NEG_INFINITY);
    denominator.fill(0.0);
    acc.fill(0.0);

    let mut ordinal: isize = 0;
    for span in 0..r {
        let lo = visible.get([row, span, 0]) as isize;
        let hi = visible.get([row, span, 1]) as isize;
        if hi > lo && lo >= 0 {
            for token in lo..hi {
                let tok = token as usize;
                let d = (total_keys - 1) - ordinal;
                for head in 0..g {
                    let q_head = kv_head * g + head;
                    let mut dot = 0.0f32;
                    for c in 0..w {
                        dot = query.get([row, q_head, c]).mul_add(history_key.get([tok, kv_head, c]), dot);
                    }
                    let mut score = dot * scale;
                    if ext > 0 && d >= 0 && (d as usize) < ext {
                        score += rel_bias.get([row, q_head, d as usize]);
                    }
                    let next = maximum[head].max(score);
                    let p = (score - next).exp();
                    let carry = (maximum[head] - next).exp();
                    denominator[head] = denominator[head].mul_add(carry, p);
                    let acc_head = &mut acc[head * w..(head + 1) * w];
                    for c in 0..w {
                        acc_head[c] = p.mul_add(history_value.get([tok, kv_head, c]), acc_head[c] * carry);
                    }
                    maximum[head] = next;
                }
                ordinal += 1;
            }
        }
    }

    if fresh_hi > fresh_lo && fresh_lo >= 0 {
        for token in fresh_lo..fresh_hi {
            let tok = token as usize;
            if tok < m {
                let d = (total_keys - 1) - ordinal;
                for head in 0..g {
                    let q_head = kv_head * g + head;
                    let mut dot = 0.0f32;
                    for c in 0..w {
                        dot = query.get([row, q_head, c]).mul_add(key.get([tok, kv_head, c]), dot);
                    }
                    let mut score = dot * scale;
                    if ext > 0 && d >= 0 && (d as usize) < ext {
                        score += rel_bias.get([row, q_head, d as usize]);
                    }
                    let next = maximum[head].max(score);
                    let p = (score - next).exp();
                    let carry = (maximum[head] - next).exp();
                    denominator[head] = denominator[head].mul_add(carry, p);
                    let acc_head = &mut acc[head * w..(head + 1) * w];
                    for c in 0..w {
                        acc_head[c] = p.mul_add(value.get([tok, kv_head, c]), acc_head[c] * carry);
                    }
                    maximum[head] = next;
                }
                ordinal += 1;
            }
        }
    }

    let result = cx.result_0();
    for head in 0..g {
        let q_head = kv_head * g + head;
        let mut total = denominator[head];
        if sk > 0 {
            let sink = sinks.get([row % sk, q_head]);
            total += (sink - maximum[head]).exp();
        }
        let denom = total.max(1e-30);
        let acc_head = &acc[head * w..(head + 1) * w];
        // SAFETY: each work item owns its unique (row, kv_head) query heads.
        let mut out = unsafe { result.row_mut([row, q_head, 0]) };
        for (c, target) in out.iter_mut().enumerate() {
            let mut val = acc_head[c] / denom;
            if u > 0 {
                val = gated_column(val, gate.get([row, q_head, c % u]), gate_function);
            }
            target.set(val);
        }
    }

    let dest = destinations.get([row]);
    if dest >= 0 {
        let dst_row = dest as usize;
        // SAFETY: each (row, kv_head) owns its unique (dst_row, kv_head) history slot.
        let mut hk = unsafe { history_key.row_mut([dst_row, kv_head, 0]) };
        let mut hv = unsafe { history_value.row_mut([dst_row, kv_head, 0]) };
        for (c, target) in hk.iter_mut().enumerate() {
            target.set(key.get([row, kv_head, c]));
        }
        for (c, target) in hv.iter_mut().enumerate() {
            target.set(value.get([row, kv_head, c]));
        }
    }
}
