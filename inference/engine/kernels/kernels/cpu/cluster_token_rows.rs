fn cluster_token_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let count = cx.dim_k() as usize;
    let width = cx.dim_b() as usize;
    for rank in 0..count {
        let cluster = cx.arg_selected().get([row, rank, 0]);
        for offset in 0..width {
            let token = if cluster >= 0 && (cluster as u64) < cx.dim_c() {
                cx.arg_token_order().get([cluster as usize, offset])
            } else {
                -1
            };
            // SAFETY: each work item owns one complete output row.
            unsafe {
                cx.result_0().set([row, rank * width + offset], token);
            }
        }
    }
}
