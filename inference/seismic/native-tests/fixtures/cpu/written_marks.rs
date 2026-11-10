// `written_marks` (conditional.seismic): the grid written, every mark
// cleared, then each work item of the written grid marks its own.
#[inline(always)]
fn written_marks_grid<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], _shared: &mut [u8]) {
    // SAFETY: the one work item owns the grid.
    let grid = unsafe { cx.scratch_grid().slice_mut::<i32>(0, 3) };
    grid.copy_from_slice(&[cx.arg_count().get([0]), 2, 1]);
}

#[inline(always)]
fn written_marks_clear<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let n = cx.dim_n();
    // SAFETY: each work item clears its own mark.
    unsafe { cx.arg_marks().set([(group[0] / n) as usize, (group[0] % n) as usize], 0) };
}

#[inline(always)]
fn written_marks<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    // SAFETY: each work item writes its own mark.
    unsafe { cx.arg_marks().set([group[1] as usize, group[0] as usize], 1) };
}
