// candidate_blocks on CPU (contract and portable body in indexer.seismic):
// one work item per row scores its blocks: the largest of each block's
// listed scores, the newest block +inf, -inf past the row's blocks; and
// writes its block list.

fn candidate_blocks<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (n, nb, bu) = (cx.dim_n() as usize, cx.dim_nb() as usize, cx.dim_bu() as usize);
    let units = (cx.arg_complete().get([row]).max(0) as usize).min(n);
    let count = units.div_ceil(bu);
    let scores = cx.arg_scores().row([row, 0]);
    let (blocks, lists) = (cx.result_0(), cx.result_1());
    for block in 0..nb {
        let score = match block + 1 == count {
            true => f32::INFINITY,
            false if block < count => scores[block * bu..((block + 1) * bu).min(units)]
                .iter()
                .fold(f32::NEG_INFINITY, |best, score| best.max(*score)),
            false => f32::NEG_INFINITY,
        };
        // SAFETY: each work item writes its own row.
        unsafe { blocks.set([row, block], score) };
    }
    // SAFETY: as above.
    unsafe {
        lists.set([row, 0, 0], 0);
        lists.set([row, 0, 1], count as i32);
    }
}
