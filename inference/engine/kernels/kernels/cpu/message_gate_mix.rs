// message_gate_mix on CPU (contract and portable body in attn_res.seismic):
// `(1 - tanh(forget)) * dest + tanh(gain) * msg` in F32.

#[inline(always)]
fn message_gate_mix<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let dest = cx.arg_dest().row([row, 0]);
    let msg = cx.arg_msg().row([row, 0]);
    let gain = cx.arg_gain().row([row, 0]);
    let forget = cx.arg_forget().row([row, 0]);
    // SAFETY: each work item writes its own row.
    let mut out = unsafe { cx.result_0().row_mut([row, 0]) };
    for ((((target, &d), &m), &g), &f) in out.iter_mut().zip(dest).zip(msg).zip(gain).zip(forget) {
        let tf = 2.0 / (1.0 + (-2.0 * f).exp()) - 1.0;
        let tg = 2.0 / (1.0 + (-2.0 * g).exp()) - 1.0;
        target.set((1.0 - tf) * d + tg * m);
    }
}
