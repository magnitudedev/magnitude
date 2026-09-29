fn slab_scale<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    if row == 0 && cx.arg_delay_ms() != 0 {
        std::thread::sleep(std::time::Duration::from_millis(cx.arg_delay_ms() as u64));
    }
    let slab_rows = cx.arg_slab_rows() as usize;
    let slab = row / slab_rows;
    let local = row % slab_rows;
    let table = cx.arg_x().pointer() as *const u64;
    // SAFETY: Seismic bound every backed slab for the call and the checked
    // row domain names only initialized rows. The address table is fixed for
    // the full submission and each work item writes a distinct output row.
    let base = unsafe { *table.add(slab) as *const f32 };
    let output = cx.result_0();
    let factor = cx.arg_factor();
    for column in 0..cx.dim_n() as usize {
        let value = unsafe { *base.add(local * cx.dim_n() as usize + column) };
        unsafe { output.set([row, column], value * factor) };
    }
}
