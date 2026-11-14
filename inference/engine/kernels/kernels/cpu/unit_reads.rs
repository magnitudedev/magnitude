// unit_reads on CPU (contract and portable body in indexer.seismic): one work
// item per row. Its picks are ascending and -1 padded (`rows_topk`): the units
// list them, then the row's tail unit; the bitmap sets each picked unit's bit
// and every bit from the tail unit on.

#[inline(always)]
fn unit_reads<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (k, b) = (cx.dim_k() as usize, cx.dim_b() as usize);
    let tail = cx.arg_complete().get([row]);
    let source = cx.arg_picks();
    let picks = (0..k).map(|pick| source.get([row, pick])).filter(|unit| *unit >= 0).collect::<Vec<_>>();
    let units = cx.result_1();
    for (slot, unit) in picks.iter().copied().chain([tail]).chain(std::iter::repeat(-1)).take(k + 1).enumerate() {
        // SAFETY: each work item writes its own row.
        unsafe { units.set([row, slot], unit) };
    }
    let bitmap = cx.result_0();
    for word in 0..b {
        let mut bits = 0u32;
        for bit in 0..32 {
            let unit = (word * 32 + bit) as i32;
            if unit >= tail || picks.binary_search(&unit).is_ok() {
                bits |= 1 << bit;
            }
        }
        // SAFETY: as above.
        unsafe { bitmap.set([row, word], bits) };
    }
}
