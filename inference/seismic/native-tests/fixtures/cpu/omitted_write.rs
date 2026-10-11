// One work item copies every element, or (`WRITE` 0) writes nothing at all.
fn omitted_write<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], _shared: &mut [u8]) {
    if cx.param_write() == 0 {
        return;
    }
    let x = cx.arg_x();
    let output = cx.result_0();
    for index in 0..cx.dim_n() as usize {
        // SAFETY: the one work item owns the result.
        unsafe { output.set([index], x.get([index])) };
    }
}
