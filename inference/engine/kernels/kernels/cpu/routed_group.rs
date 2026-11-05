// routed_group on CPU (contract and portable body in routed.seismic). One
// work item builds every table: it counts each expert's grouped choices (the
// rows[0] rows' from row first[0]), places each expert's tiles after the previous
// experts' (ceil(count / T) blocks each), then walks the grouped choices in
// flat (row, choice) order so every expert's rows keep that order. Tile
// slots no choice claims are -1; blocks past `used` are left as they were.

fn routed_group<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], _shared: &mut [u8]) {
    let (e, k, t) = (cx.dim_e() as usize, cx.dim_k() as usize, cx.dim_t() as usize);
    let (routes, counts, order, inverse, blocks, used) =
        (cx.arg_routes(), cx.arg_counts(), cx.arg_order(), cx.arg_inverse(), cx.arg_blocks(), cx.arg_used());
    let first = usize::try_from(cx.arg_first().get([0])).expect("a first row is not negative");
    let rows = usize::try_from(cx.arg_rows().get([0])).expect("a row count is not negative");
    let grouped = first * k..(first + rows) * k;
    let expert_of = |flat: usize| usize::try_from(routes.get([flat / k, flat % k])).expect("a route is an expert index");
    // SAFETY: the launch has one work item, which writes every table.
    unsafe {
        let counts = counts.row_mut([0]);
        counts.fill(0);
        for flat in grouped.clone() {
            counts[expert_of(flat)] += 1;
        }
        let blocks = blocks.row_mut([0]);
        // The next tile position of each expert.
        let mut next = Vec::with_capacity(e);
        let mut block = 0;
        for (expert, count) in counts.iter().enumerate() {
            let tiles = (*count as usize).div_ceil(t);
            next.push(block * t);
            blocks[block..block + tiles].fill(expert as i32);
            for tile in block..block + tiles {
                order.row_mut([tile, 0]).fill(-1);
            }
            block += tiles;
        }
        used.set([0], block as i32);
        for flat in grouped {
            let expert = expert_of(flat);
            let position = next[expert];
            next[expert] += 1;
            order.set([position / t, position % t], (flat / k) as i32);
            inverse.set([flat / k, flat % k], position as i32);
        }
    }
}
