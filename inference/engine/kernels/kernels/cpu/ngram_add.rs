// ngram_add on CPU (contract and portable body in ngram.seismic): one work
// item per row adds each stream's gated value row into its streams.

fn ngram_add<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (h, s) = (cx.dim_h() as usize, cx.dim_s() as usize);
    let hidden = &cx.arg_hidden().row([row, 0])[..s * h];
    let gates = &cx.arg_scales().row([row, 0])[..s];
    let value = &cx.arg_value().row([row, 0])[..h];
    // SAFETY: each work item writes its own row.
    let out = unsafe { cx.result_0().row_mut([row, 0]) };
    for stream in 0..s {
        for column in 0..h {
            let channel = stream * h + column;
            out[channel] = hidden[channel] + gates[stream] * value[column];
        }
    }
}
