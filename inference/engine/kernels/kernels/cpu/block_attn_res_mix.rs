// block_attn_res_mix on CPU (contract and portable body in attn_res.seismic):
// one work item per (row, query), computing per-source RMS inverse and
// query dot-product in a single pass, softmax weights, and weighted F32 sum.

#[inline(always)]
fn block_attn_res_mix<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let m = cx.dim_m() as usize;
    let s_count = cx.dim_s() as usize;
    let h = cx.dim_h() as usize;
    let item = group[0] as usize;
    let q = item / m;
    let row = item % m;
    let eps = cx.arg_epsilon();
    let scale = cx.arg_scale();
    let norm_v = cx.arg_normalize_values() != 0;

    let sources = cx.arg_sources();
    let query = cx.arg_queries().row([q, 0]);
    let buf = seismic::cpu::tensor::floats(shared, s_count * 2);
    let (inv, logits) = buf.split_at_mut(s_count);

    let mut maximum = f32::NEG_INFINITY;
    for s in 0..s_count {
        let src = sources.row([s, row, 0]);
        let mut sq = 0.0f32;
        let mut dot = 0.0f32;
        for i in 0..h {
            let v = src[i];
            sq = v.mul_add(v, sq);
            dot = query[i].mul_add(v, dot);
        }
        let r = 1.0f32 / (sq / (h as f32) + eps).sqrt();
        inv[s] = r;
        let l = dot * r * scale;
        logits[s] = l;
        if l > maximum {
            maximum = l;
        }
    }

    let mut total = 0.0f32;
    for s in 0..s_count {
        let e = (logits[s] - maximum).exp();
        logits[s] = e;
        total += e;
    }
    for s in 0..s_count {
        let w = logits[s] / total;
        logits[s] = if norm_v { w * inv[s] } else { w };
    }

    // SAFETY: each work item owns its unique (q, row) output row.
    let mut out = unsafe { cx.result_0().row_mut([q, row, 0]) };
    out.compute(|dst| {
        dst.fill(0.0);
        for s in 0..s_count {
            let coeff = logits[s];
            let src = sources.row([s, row, 0]);
            for (acc, &v) in dst.iter_mut().zip(src.iter()) {
                *acc = coeff.mul_add(v, *acc);
            }
        }
    });
}
