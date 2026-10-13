// selection_notice on CPU (contract and portable body in target.seismic). One
// work item copies every row's (token, status) pair.

fn selection_notice<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], _shared: &mut [u8]) {
    let (selection, notice) = (cx.arg_selection(), cx.arg_notice());
    for row in 0..cx.dim_m() as usize {
        for field in 0..2 {
            // SAFETY: the single work item writes every element.
            unsafe { notice.set([row, field], selection.get([row, field])) };
        }
    }
}
