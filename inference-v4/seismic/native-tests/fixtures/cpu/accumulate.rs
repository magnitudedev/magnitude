// One work item per `WIDTH` elements, matching the GPU kernels bit for bit.
fn accumulate<const BIAS: u64, const WIDTH: u64>(
    context: &Context<'_>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let state = context.arg_state();
    let x = context.arg_x();
    for lane in 0..WIDTH {
        let index = group[0] * WIDTH + lane;
        if index >= context.dim_n() {
            return;
        }
        unsafe {
            let at = state.pointer.cast::<f32>().add((index * state.strides[0]) as usize);
            let value = x.pointer.cast::<f32>().add((index * x.strides[0]) as usize).read();
            at.write(at.read() + (value + BIAS as f32 * 0.5));
        }
    }
}
