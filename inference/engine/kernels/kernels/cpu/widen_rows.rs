// widen_rows on CPU (contract and portable body in dflash.seismic): one work
// item per row, which widens its A row to F32.

fn widen_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    // SAFETY: each work item writes its own row.
    cx.arg_rows().decode_row(row, unsafe { cx.result_0().row_mut([row, 0]) });
}
