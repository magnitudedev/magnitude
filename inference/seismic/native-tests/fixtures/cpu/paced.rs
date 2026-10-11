// One work item over all `N` elements, matching the GPU kernels bit for
// bit: `PACE` rounds of 2^13 generator steps for each element (a CPU takes
// the elements in turn, so a round is shorter than a GPU thread's), then
// the sum with the pace's bias and the generator's low bit against the
// index's, which an even number of steps leaves equal.
fn paced<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, _group: [u64; 3], _shared: &mut [u8]) {
    let state = cx.arg_state();
    let x = cx.arg_x();
    let pace = cx.param_pace();
    let bias = match pace {
        0 => 0.0f32,
        1..=5 => 0.5,
        _ => 1.0,
    };
    for index in 0..cx.dim_n() as usize {
        let mut generated = index as u32;
        for _ in 0..pace << 13 {
            generated = generated.wrapping_mul(1664525).wrapping_add(1013904223);
        }
        let unchanged = ((generated ^ index as u32) & 1) as f32;
        // SAFETY: the one work item owns the state.
        unsafe { state.set([index], state.get([index]) + (x.get([index]) + bias + unchanged)) };
    }
}
