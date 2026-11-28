// rel_bias_project on CPU (contract and portable body in attn_res.seismic):
// `[M, H, D] A @ [E, D]^T W -> [M, H, E] f32`, with optional fused
// `head_l2_clip` when `threshold > 0.0` (rounding clipped R to `E::A` before
// the projection to preserve the exact intermediate dtype boundary).

#[inline(always)]
fn rel_bias_project<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let h_count = cx.dim_h() as usize;
    let d_len = cx.dim_d() as usize;
    let sc = cx.dim_sc() as usize;
    let item = group[0] as usize;
    let row = item / h_count;
    let head = item % h_count;

    let buf = seismic::cpu::tensor::floats(shared, d_len * 2);
    let (r_buf, w_buf) = buf.split_at_mut(d_len);
    seismic::cpu::tensor::widen_row::<E::A>(cx.arg_r().row([row, head, 0]), r_buf);

    let threshold = cx.arg_threshold();
    if threshold > 0.0 {
        let eps = cx.arg_epsilon();
        let log_floor = cx.arg_log_floor();
        let log_alpha = cx.arg_log_alpha();
        let tau = if log_alpha != 0.0 {
            let pos = cx.arg_coordinates().get([row, 0]) as f32;
            let ratio = (pos + 1.0) / log_floor;
            1.0 + log_alpha * ratio.max(1.0).ln()
        } else {
            1.0
        };
        let scale = cx.arg_scale();
        let scale_row = if sc > 1 { head } else { 0 };
        let mut sq = 0.0f32;
        for d in 0..d_len {
            let mut v = r_buf[d];
            if sc > 0 {
                v *= scale.get([scale_row, d]);
                r_buf[d] = v;
            }
            sq = v.mul_add(v, sq);
        }
        let clip = (threshold / (sq.sqrt() + eps)).min(1.0);
        let factor = clip * tau;
        for v in r_buf.iter_mut() {
            *v = E::A::round(*v * factor);
        }
    }

    let weight = cx.arg_weight();

    // SAFETY: each work item owns its unique (row, head) output slice.
    let mut out = unsafe { cx.result_0().row_mut([row, head, 0]) };
    for (ext, target) in out.iter_mut().enumerate() {
        weight.decode_row(ext, w_buf);
        let mut acc = 0.0f32;
        for d in 0..d_len {
            acc = r_buf[d].mul_add(w_buf[d], acc);
        }
        target.set(acc);
    }
}
