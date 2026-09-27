// repack_weight on CPU (contract and portable body in import.seismic). Each
// work item converts eight of the B * N rows of the [B, N, K] view with the
// registered conversion of its external source into a resident row layout,
// moving codes and coefficients bit for bit.

fn repack_weight<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (source, destination) = (cx.arg_source(), cx.result_0());
    let n = cx.dim_n() as usize;
    let first = group[1] as usize * n + group[0] as usize * 8;
    let end = (group[1] as usize + 1) * n;
    let rows = first.min(end)..(first + 8).min(end);
    // SAFETY: each work item writes its own rows.
    unsafe { seismic::cpu::repack::rows(&source, &destination, rows) };
}
