// Both launches sum in element order, like the GPU kernels: the staged one
// through the work item's shared buffer, the mirrored one through scratch.
fn gated_staged<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], shared: &mut [u8]) {
    let n = cx.dim_n() as usize;
    let x = cx.arg_x();
    for index in 0..n {
        shared[index * 4..index * 4 + 4].copy_from_slice(&x.get([index]).to_le_bytes());
    }
    let mut total = 0.0f32;
    for index in 0..n {
        total += f32::from_le_bytes(shared[index * 4..index * 4 + 4].try_into().unwrap());
    }
    // SAFETY: the one work item writes the one result element.
    unsafe { cx.result_0().set([0], total) };
}

fn gated_mirrored<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], _shared: &mut [u8]) {
    let n = cx.dim_n() as usize;
    let x = cx.arg_x();
    // SAFETY: the scratch holds N floats and the one work item owns it.
    let mirror = unsafe { cx.scratch_mirror().slice_mut::<f32>(0, n) };
    for (index, value) in mirror.iter_mut().enumerate() {
        *value = x.get([index]);
    }
    let mut total = 0.0f32;
    for value in mirror.iter() {
        total += value;
    }
    // SAFETY: the one work item writes the one result element.
    unsafe { cx.result_0().set([0], total) };
}
