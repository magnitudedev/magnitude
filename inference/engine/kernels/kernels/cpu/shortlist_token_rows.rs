#[inline(always)]
fn shortlist_token_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let count = cx.dim_k() as usize;
    for rank in 0..count {
        let token = cx.arg_selected().get([row, rank, 0]);
        // SAFETY: each work item owns one complete output row.
        unsafe {
            cx.result_0().set([row, rank], token);
        }
    }
}
