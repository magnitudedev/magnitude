// ngram_gate on CPU (contract and portable body in ngram.seismic): one work
// item per row, which publishes each stream's gate sigmoid(g) and the
// inverse RMS of its gated value row. Under the stored product (PR = 1) the
// query weight is one, so `query_norm` is skipped and the dot product omits
// that multiply.

use lib::core::{activation, reduce};
use seismic::cpu::reduce::{combine, LANES};

/// Single-pass fused `(dot, sum_sq_k, sum_sq_x)` in `seismic::cpu::reduce`'s
/// 8-lane order so `k` and `x` are each read once and every tier produces the
/// same bits.
#[inline(always)]
fn stream_sums(
    k: &[f32],
    x: &[f32],
    key_norm: &[f32],
    query_norm: Option<&[f32]>,
) -> (f32, f32, f32) {
    let full = k.len() / LANES * LANES;
    let mut dot = [0.0f32; LANES];
    let mut keys = [0.0f32; LANES];
    let mut queries = [0.0f32; LANES];
    match query_norm {
        Some(qn) => {
            for (((k, x), kn), qn) in k[..full]
                .chunks_exact(LANES)
                .zip(x[..full].chunks_exact(LANES))
                .zip(key_norm[..full].chunks_exact(LANES))
                .zip(qn[..full].chunks_exact(LANES))
            {
                for lane in 0..LANES {
                    let kv = k[lane];
                    let xv = x[lane];
                    keys[lane] = kv.mul_add(kv, keys[lane]);
                    queries[lane] = xv.mul_add(xv, queries[lane]);
                    dot[lane] = (kv * kn[lane]).mul_add(xv * qn[lane], dot[lane]);
                }
            }
            for (lane, (((&kv, &xv), &kn), &qn)) in k[full..]
                .iter()
                .zip(&x[full..])
                .zip(&key_norm[full..])
                .zip(&qn[full..])
                .enumerate()
            {
                keys[lane] = kv.mul_add(kv, keys[lane]);
                queries[lane] = xv.mul_add(xv, queries[lane]);
                dot[lane] = (kv * kn).mul_add(xv * qn, dot[lane]);
            }
        }
        None => {
            for ((k, x), kn) in k[..full]
                .chunks_exact(LANES)
                .zip(x[..full].chunks_exact(LANES))
                .zip(key_norm[..full].chunks_exact(LANES))
            {
                for lane in 0..LANES {
                    let kv = k[lane];
                    let xv = x[lane];
                    keys[lane] = kv.mul_add(kv, keys[lane]);
                    queries[lane] = xv.mul_add(xv, queries[lane]);
                    dot[lane] = (kv * kn[lane]).mul_add(xv, dot[lane]);
                }
            }
            for (lane, ((&kv, &xv), &kn)) in k[full..]
                .iter()
                .zip(&x[full..])
                .zip(&key_norm[full..])
                .enumerate()
            {
                keys[lane] = kv.mul_add(kv, keys[lane]);
                queries[lane] = xv.mul_add(xv, queries[lane]);
                dot[lane] = (kv * kn).mul_add(xv, dot[lane]);
            }
        }
    }
    (combine(dot), combine(keys), combine(queries))
}

#[inline(always)]
fn ngram_gate<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (h, s) = (cx.dim_h() as usize, cx.dim_s() as usize);
    let product = cx.dim_pr() != 0;
    let norms = seismic::cpu::tensor::floats(shared, if product { h } else { 2 * h });
    let (key_norm, query_norm) = norms.split_at_mut(h);
    let epsilon = cx.arg_epsilon();
    let h_f32 = h as f32;
    let sqrt_h = h_f32.sqrt();
    let values = reduce::sum_squares(&cx.arg_value().row([row, 0])[..h]) / h_f32;
    // SAFETY: each work item writes its own row.
    let mut out = unsafe { cx.result_0().row_mut([row, 0]) };
    for stream in 0..s {
        cx.arg_key_norm().decode_row(stream, key_norm);
        let qn = if product {
            None
        } else {
            cx.arg_query_norm().decode_row(stream, query_norm);
            Some(&*query_norm)
        };
        let k = &cx.arg_key().row([row, stream, 0])[..h];
        let x = &cx.arg_hidden().row([row, stream, 0])[..h];
        let (dot, keys, queries) = stream_sums(k, x, key_norm, qn);
        let key_inv = 1.0 / (keys / h_f32 + epsilon).sqrt();
        let query_inv = 1.0 / (queries / h_f32 + epsilon).sqrt();
        let g = dot * key_inv * query_inv / sqrt_h;
        let root = g.abs().max(1e-6).sqrt();
        let signed = if g > 0.0 {
            root
        } else if g < 0.0 {
            -root
        } else {
            0.0
        };
        let gate = activation::sigmoid(signed);
        out.set(stream, gate);
        out.set(s + stream, 1.0 / (gate * gate * values + epsilon).sqrt());
    }
}
