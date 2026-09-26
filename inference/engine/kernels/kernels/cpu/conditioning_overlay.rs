// conditioning_overlay on CPU (contract and portable body in target.seismic).
// One work item per row copies the row's D conditioned F32 values into `out`.

fn conditioning_overlay<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    // SAFETY: each work item writes its own row.
    let target = unsafe { cx.arg_out().row_mut([row, 0]) };
    target.copy_from_slice(cx.arg_input().row([row, 0]));
}
