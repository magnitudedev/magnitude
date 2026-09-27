// copy_rows on CPU (contract and portable body in state.seismic). One work
// item per copied item moves its KV rows of W stored elements bit for bit
// (storage words, never decoded). Items whose source or destination row lies
// outside its tensor are skipped, as on Metal and CUDA.

fn copy_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let item = group[0] as usize;
    let (src, dst) = (cx.arg_src(), cx.arg_dst());
    let (from, to) = (cx.arg_from().get([item]), cx.arg_to().get([item]));
    if from < 0 || to < 0 || from as u64 >= cx.dim_ts() || to as u64 >= cx.dim_td() {
        return;
    }
    let (from, to) = (from as usize, to as usize);
    for head in 0..cx.dim_kv() as usize {
        // SAFETY: compaction names distinct destination rows, so each item
        // writes its own rows.
        let target = unsafe { dst.row_mut([to, head, 0]) };
        target.copy_from_slice(src.row([from, head, 0]));
    }
}
