// import_dense on CPU (contract and portable body in import.seismic). Each
// work item converts 16 of the B * N rows of the [B, N, K] view: it decodes
// a source row exactly to F32 and stores each value rounded once to U.

fn import_dense<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (n, k) = (cx.dim_n() as usize, cx.dim_k() as usize);
    let total = cx.dim_b() as usize * n;
    let first = group[0] as usize * 16;
    let rows = first.min(total)..(first + 16).min(total);
    let (source, destination) = (cx.arg_source(), cx.result_0());
    let values = seismic::cpu::tensor::floats(shared, k);
    for row in rows {
        source.decode_row(row, values);
        // SAFETY: each work item writes its own rows.
        let target = unsafe { destination.row_mut([row / n, row % n, 0]) };
        seismic::cpu::tensor::narrow_row::<E::U>(values, target);
    }
}
