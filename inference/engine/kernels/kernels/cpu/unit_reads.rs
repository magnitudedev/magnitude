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
    let bitmap: &mut [u32] = if b > 0 {
        // SAFETY: each work item writes its own row.
        let bitmap = unsafe { cx.result_0().row_mut([row, 0]) };
        let tail_unit = tail.max(0) as usize;
        let split = tail_unit / 32;
        let below = split.min(b);
        bitmap[..below].fill(0);
        if split < b {
            bitmap[split] = !0u32 << (tail_unit % 32);
            bitmap[split + 1..].fill(!0u32);
        }
        bitmap
    } else {
        &mut []
    };
    // SAFETY: each work item writes its own row.
    let units = unsafe { cx.result_1().row_mut([row, 0]) };
    let mut count = 0usize;
    for pick in 0..k {
        let unit = source.get([row, pick]);
        if unit >= 0 {
            units[count] = unit;
            count += 1;
            if let Some(bits) = bitmap.get_mut((unit as usize) / 32) {
                *bits |= 1u32 << (unit as u32 & 31);
            }
        }
    }
    units[count] = tail;
    units[count + 1..].fill(-1);
}
