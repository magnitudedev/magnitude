// message_gate_mix_norm on CPU (contract and portable body in attn_res.seismic):
// fused cross-pass message gate + RMSNorm to A, one work item per row.

#[inline(always)]
fn message_gate_mix_norm<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    let eps = cx.arg_epsilon();
    let dest = cx.arg_dest().row([row, 0]);
    let msg = cx.arg_msg().row([row, 0]);
    let gain = cx.arg_gain().row([row, 0]);
    let forget = cx.arg_forget().row([row, 0]);

    let buf = seismic::cpu::tensor::floats(shared, h * 2);
    let (y_buf, norm_buf) = buf.split_at_mut(h);
    let mut sq = 0.0f32;
    for i in 0..h {
        let tf = 2.0 / (1.0 + (-2.0 * forget[i]).exp()) - 1.0;
        let tg = 2.0 / (1.0 + (-2.0 * gain[i]).exp()) - 1.0;
        let y = (1.0 - tf) * dest[i] + tg * msg[i];
        y_buf[i] = y;
        sq = y.mul_add(y, sq);
    }
    let inv = (sq / (h as f32) + eps).sqrt().recip();
    cx.arg_norm().decode_row(0, norm_buf);

    // SAFETY: each work item writes its own row.
    let mut out = unsafe { cx.result_0().row_mut([row, 0]) };
    for ((target, &y), &nw) in out.iter_mut().zip(y_buf.iter()).zip(norm_buf.iter()) {
        target.set((y * inv) * nw);
    }
}
