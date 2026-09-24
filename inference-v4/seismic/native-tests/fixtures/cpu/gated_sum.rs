// Both launches sum in element order, like the GPU kernels: the staged one
// through the work item's shared buffer, the mirrored one through scratch.
fn read_x(context: &Context<'_>, index: u64) -> f32 {
    let x = context.arg_x();
    unsafe { x.pointer.cast::<f32>().add((index * x.strides[0]) as usize).read() }
}

fn gated_staged<const SMALL: u64>(context: &Context<'_>, _group: [u64; 3], shared: &mut [u8]) {
    let n = context.dim_n() as usize;
    for index in 0..n {
        shared[index * 4..index * 4 + 4].copy_from_slice(&read_x(context, index as u64).to_le_bytes());
    }
    let mut total = 0.0f32;
    for index in 0..n {
        total += f32::from_le_bytes(shared[index * 4..index * 4 + 4].try_into().unwrap());
    }
    unsafe { context.result_0().pointer.cast::<f32>().write(total) };
}

fn gated_mirrored<const SMALL: u64>(context: &Context<'_>, _group: [u64; 3], _shared: &mut [u8]) {
    let n = context.dim_n() as usize;
    let mirror = context.scratch_mirror().cast::<f32>();
    for index in 0..n {
        unsafe { mirror.add(index).write(read_x(context, index as u64)) };
    }
    let mut total = 0.0f32;
    for index in 0..n {
        total += unsafe { mirror.add(index).read() };
    }
    unsafe { context.result_0().pointer.cast::<f32>().write(total) };
}
