// One work item copies every element; `STRAY` 1 writes a wrong last one.
fn stray_write<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], _shared: &mut [u8]) {
    let state = cx.arg_state();
    let x = cx.arg_x();
    let n = cx.dim_n() as usize;
    for index in 0..n {
        let stray = cx.param_stray() != 0 && index == n - 1;
        // SAFETY: the one work item owns the state.
        unsafe { state.set([index], if stray { 42.0 } else { x.get([index]) }) };
    }
}
