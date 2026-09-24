// Each part is summed in element order, staged `WIDTH` elements at a time
// through the work item's shared buffer, matching the GPU kernels bit for bit.
fn split_partial<const PARTS: u64, const WIDTH: u64>(
    context: &Context<'_>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let x = context.arg_x();
    let n = context.dim_n();
    let per = n.div_ceil(PARTS);
    let begin = (group[0] * per).min(n);
    let end = (begin + per).min(n);
    let mut total = 0.0f32;
    let mut base = begin;
    while base < end {
        let count = WIDTH.min(end - base);
        for lane in 0..count {
            let value = unsafe {
                x.pointer
                    .cast::<f32>()
                    .add(((base + lane) * x.strides[0]) as usize)
                    .read()
            };
            shared[lane as usize * 4..lane as usize * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        for lane in 0..count as usize {
            total += f32::from_le_bytes(shared[lane * 4..lane * 4 + 4].try_into().unwrap());
        }
        base += WIDTH;
    }
    unsafe {
        context
            .scratch_partials()
            .cast::<f32>()
            .add(group[0] as usize)
            .write(total)
    };
}

fn split_merge<const PARTS: u64, const WIDTH: u64>(
    context: &Context<'_>,
    _group: [u64; 3],
    _shared: &mut [u8],
) {
    let partials = context.scratch_partials().cast::<f32>();
    let mut total = 0.0f32;
    for part in 0..PARTS as usize {
        total += unsafe { partials.add(part).read() };
    }
    unsafe { context.result_0().pointer.cast::<f32>().write(total) };
}
