// One work item over all `N` elements in chunks of `WIDTH`, matching the GPU
// kernels bit for bit. The whole call is a few hundred additions: spreading
// it over cores would cost more in synchronization than the work itself.
fn accumulate<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], _shared: &mut [u8]) {
    let state = cx.arg_state();
    let x = cx.arg_x();
    let (bias, width) = (cx.param_bias(), cx.param_width() as usize);
    let n = cx.dim_n() as usize;
    for chunk in 0..n.div_ceil(width) {
        for lane in 0..width {
            let index = chunk * width + lane;
            if index >= n {
                return;
            }
            // SAFETY: the one work item owns the state.
            unsafe { state.set([index], state.get([index]) + (x.get([index]) + bias as f32 * 0.5)) };
        }
    }
}
