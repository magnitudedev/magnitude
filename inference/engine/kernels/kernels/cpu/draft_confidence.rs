// draft_confidence on CPU (contract and portable body in dflash.seismic): one
// work item per row, which widens its features and Markov memory to F32,
// reduces their dot product with the confidence weight, and marks the row's
// selection declined (status 3) when the sigmoid falls below the threshold.

use lib::core::activation;

fn draft_confidence<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (d, r) = (cx.dim_d() as usize, cx.dim_r() as usize);
    let (features, memory) = seismic::cpu::tensor::floats(shared, d + r).split_at_mut(d);
    cx.arg_features().decode_row(row, features);
    cx.arg_memory().decode_row(row, memory);
    let weight = cx.arg_weight().row([0]);
    let dot = |values: &[f32], weight: &[f32]| values.iter().zip(weight).map(|(x, w)| x * w).sum::<f32>();
    let score = dot(features, &weight[..d]) + dot(memory, &weight[d..]) + cx.arg_bias().get([0]);
    if activation::sigmoid(score) < cx.arg_threshold() {
        // SAFETY: each work item writes its own row.
        unsafe { cx.arg_selection().set([row, 1], 3) };
    }
}
