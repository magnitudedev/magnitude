// routed_group on CPU (contract and portable body in routed.seismic). One
// work item builds every table: it counts each expert's grouped choices (the
// rows[0] rows from row first[0]), places each expert's tiles after the previous
// experts' (ceil(count / T) blocks each), then walks the grouped choices in
// (row, choice) order so every expert's rows keep that order. Tile slots no
// choice claims are -1; blocks past `used` are left as they were.

const INLINE_EXPERTS: usize = 512;

#[inline(always)]
fn routed_group<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], _shared: &mut [u8]) {
    let (e, t) = (cx.dim_e() as usize, cx.dim_t() as usize);
    let (routes, counts, order, inverse, blocks, used) =
        (cx.arg_routes(), cx.arg_counts(), cx.arg_order(), cx.arg_inverse(), cx.arg_blocks(), cx.arg_used());
    let first = usize::try_from(cx.arg_first().get([0])).expect("a first row is not negative");
    let rows = usize::try_from(cx.arg_rows().get([0])).expect("a row count is not negative");
    let row_range = first..first + rows;
    // SAFETY: the launch has one work item, which writes every table.
    unsafe {
        let counts = counts.row_mut([0]);
        counts.fill(0);
        for row in row_range.clone() {
            for &route in routes.row([row, 0]) {
                let expert = usize::try_from(route).expect("a route is an expert index");
                counts[expert] += 1;
            }
        }
        let blocks = blocks.row_mut([0]);
        // The next (tile, slot) of each expert.
        let mut inline_next = [(0usize, 0usize); INLINE_EXPERTS];
        let mut heap_next = Vec::new();
        let next = if e <= INLINE_EXPERTS {
            &mut inline_next[..e]
        } else {
            heap_next.resize(e, (0, 0));
            heap_next.as_mut_slice()
        };
        let mut block = 0;
        for (expert, &count) in counts.iter().enumerate() {
            let count = count as usize;
            if count == 0 {
                continue;
            }
            let tiles = if count <= t { 1 } else { count.div_ceil(t) };
            next[expert] = (block, 0);
            blocks[block..block + tiles].fill(expert as i32);
            let last_filled = count - (tiles - 1) * t;
            if last_filled < t {
                order.row_mut([block + tiles - 1, 0])[last_filled..].fill(-1);
            }
            block += tiles;
        }
        used.set([0], block as i32);
        for row in row_range {
            let row_i32 = row as i32;
            for (&route, inv) in routes.row([row, 0]).iter().zip(inverse.row_mut([row, 0])) {
                let expert = usize::try_from(route).expect("a route is an expert index");
                let (tile, slot) = &mut next[expert];
                let (cur_tile, cur_slot) = (*tile, *slot);
                *slot += 1;
                if *slot == t {
                    *slot = 0;
                    *tile += 1;
                }
                order.set([cur_tile, cur_slot], row_i32);
                *inv = (cur_tile * t + cur_slot) as i32;
            }
        }
    }
}
