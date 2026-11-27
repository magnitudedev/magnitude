// head_l2_clip on CPU (contract and portable body in attention.seismic):
// per-head optional channel scale, L2 norm clip, and optional position-
// dependent log-scaling `tau`.

#[inline(always)]
fn head_l2_clip<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let h_count = cx.dim_h() as usize;
    let w = cx.dim_w() as usize;
    let sc = cx.dim_sc() as usize;
    let item = group[0] as usize;
    let row = item / h_count;
    let head = item % h_count;

    let threshold = cx.arg_threshold();
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

    let buf = seismic::cpu::tensor::floats(shared, w);
    cx.arg_x().decode_row(item, buf);
    let scale = cx.arg_scale();
    let scale_row = if sc > 1 { head } else { 0 };
    let mut sq = 0.0f32;
    for c in 0..w {
        let mut v = buf[c];
        if sc > 0 {
            v *= scale.get([scale_row, c]);
            buf[c] = v;
        }
        sq = v.mul_add(v, sq);
    }
    let clip = (threshold / (sq.sqrt() + eps)).min(1.0);
    let factor = clip * tau;

    // SAFETY: each work item owns its unique (row, head) output slice.
    let mut out = unsafe { cx.result_0().row_mut([row, head, 0]) };
    for (target, &v) in out.iter_mut().zip(buf.iter()) {
        target.set(v * factor);
    }
}
