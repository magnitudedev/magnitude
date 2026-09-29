// Each part is summed in element order, staged `WIDTH` elements at a time
// through the work item's shared buffer, matching the GPU kernels bit for bit.
fn split_partial<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let x = cx.arg_x();
    let (parts, width) = (cx.param_parts(), cx.param_width());
    let n = cx.dim_n();
    let per = n.div_ceil(parts);
    let begin = (group[0] * per).min(n);
    let end = (begin + per).min(n);
    let mut total = 0.0f32;
    let mut base = begin;
    while base < end {
        let count = width.min(end - base) as usize;
        for lane in 0..count {
            let value = x.get([base as usize + lane]);
            shared[lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        for lane in 0..count {
            total += f32::from_le_bytes(shared[lane * 4..lane * 4 + 4].try_into().unwrap());
        }
        base += width;
    }
    // SAFETY: each work item writes its own partial.
    unsafe { cx.scratch_partials().slice_mut::<f32>(4 * group[0] as usize, 1)[0] = total };
}

fn split_merge<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], _shared: &mut [u8]) {
    // SAFETY: the partials launch wrote PARTS floats before this launch.
    let partials = unsafe { cx.scratch_partials().slice::<f32>(0, cx.param_parts() as usize) };
    let mut total = 0.0f32;
    for partial in partials {
        total += partial;
    }
    // SAFETY: the one work item writes the one result element.
    unsafe { cx.result_0().set([0], total) };
}
