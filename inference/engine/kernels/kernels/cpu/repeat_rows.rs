// repeat_rows on CPU (contract and portable body in streams.seismic): one
// work item per row, which writes its S scaled copies.

fn repeat_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    let source = cx.arg_rows().row([row, 0]);
    // SAFETY: each work item writes its own row.
    let out = unsafe { cx.result_0().row_mut([row, 0]) };
    for (stream, copy) in out.chunks_exact_mut(h).enumerate() {
        let scale = cx.arg_scales().get([stream]);
        for (target, value) in copy.iter_mut().zip(source) {
            *target = value * scale;
        }
    }
}
