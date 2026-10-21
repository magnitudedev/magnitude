// corrector_features on CPU (contract and portable body in corrector.seismic):
// one work item per proposing row, which reduces its softmax normalizer, then
// forms each candidate's five features, their layer norm and their projection
// silu(up · normed + up_bias), rounded once to A.

use lib::core::activation;

fn corrector_features<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (count, width) = (cx.dim_k() as usize, cx.dim_h() as usize);
    let line = cx.arg_logits().row([row, 0]);
    let maximum = line.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total = line.iter().map(|logit| (logit - maximum).exp()).sum::<f32>();
    let normalizer = maximum + total.ln();
    let unary = cx.arg_unary().row([row, 0]);
    let (norm_weight, norm_bias, up_weight) = (cx.arg_norm_weight(), cx.arg_norm_bias(), cx.arg_up_weight());
    let up_bias = cx.arg_up_bias().row([0]);
    let epsilon = cx.arg_epsilon();
    let out = seismic::cpu::tensor::floats(shared, width);
    for rank in 0..count {
        let lp = unary[rank] - normalizer;
        let features = [
            lp,
            lp.exp(),
            unary[rank] - unary[0],
            if count > 1 { rank as f32 / (count - 1) as f32 } else { 0.0 },
            if rank == 0 { 1.0 } else { 0.0 },
        ];
        let mean = features.iter().sum::<f32>() / 5.0;
        let variance = features.iter().map(|feature| (feature - mean) * (feature - mean)).sum::<f32>() / 5.0;
        let inverse = 1.0 / (variance + epsilon).sqrt();
        let normed: [f32; 5] =
            core::array::from_fn(|index| (features[index] - mean) * inverse * norm_weight.get([index]) + norm_bias.get([index]));
        for (column, value) in out.iter_mut().enumerate() {
            let weights = up_weight.row([column, 0]);
            let mut sum = up_bias[column];
            for (weight, normed) in weights.iter().zip(normed.iter()) {
                sum = weight.mul_add(*normed, sum);
            }
            *value = activation::silu(sum);
        }
        // SAFETY: each work item writes its own candidates' rows.
        activation::store::<E::A>(out, unsafe { cx.result_0().row_mut([row * count + rank, 0]) });
    }
}
