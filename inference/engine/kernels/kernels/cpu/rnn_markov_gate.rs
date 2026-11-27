// rnn_markov_gate on CPU (contract and portable body in dflash.seismic):
// fused DSpark MarkovHead::RnnHidden recurrent gate step.

#[inline(always)]
fn rnn_markov_gate<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let s = group[0] as usize;
    let r_dim = cx.dim_r() as usize;
    let h_prev = cx.arg_h_prev().row([s, 0]);
    let base_frnn = cx.arg_base_frnn().row([s, 0]);
    let base_x = cx.arg_base_x().row([s, 0]);
    let he_h = cx.arg_he_h().row([s, 0]);
    let he_e = cx.arg_he_e().row([s, 0]);
    // SAFETY: each work item writes its own row of all three results.
    let (mut h_next, mut h_next_a, mut latent) = unsafe {
        (
            cx.result_0().row_mut([s, 0]),
            cx.result_1().row_mut([s, 0]),
            cx.result_2().row_mut([s, 0]),
        )
    };
    for (r, ((hn, hna), lat)) in h_next
        .iter_mut()
        .zip(h_next_a.iter_mut())
        .zip(latent.iter_mut())
        .enumerate()
    {
        let r1 = r + r_dim;
        let r2 = r + 2 * r_dim;
        let g0 = base_frnn[r] + base_x[r] + he_h[r] + he_e[r];
        let g1 = base_frnn[r1] + base_x[r1] + he_h[r1] + he_e[r1];
        let g2 = base_frnn[r2] + base_x[r2] + he_h[r2] + he_e[r2];
        let gate = 1.0 / (1.0 + (-g0).exp());
        let cand = 2.0 / (1.0 + (-2.0 * g1).exp()) - 1.0;
        let next = gate * h_prev[r] + (1.0 - gate) * cand;
        let lat_val = 2.0 / (1.0 + (-2.0 * g2).exp()) - 1.0;
        hn.set(next);
        hna.set(next);
        lat.set(lat_val);
    }
}
