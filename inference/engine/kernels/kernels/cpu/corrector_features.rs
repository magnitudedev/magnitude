// corrector_features on CPU (contract and portable body in corrector.seismic):
// one work item per proposing row, which reduces its softmax normalizer, then
// forms each candidate's five features, their layer norm and their projection
// silu(up · normed + up_bias), rounded once to A.

use lib::core::activation;
use seismic::cpu::reduce::{LANES, combine};

#[inline(always)]
fn max_f32(values: &[f32]) -> f32 {
    let mut lanes = [f32::NEG_INFINITY; LANES];
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    for chunk in chunks {
        for lane in 0..LANES {
            lanes[lane] = lanes[lane].max(chunk[lane]);
        }
    }
    for (lane, &value) in tail.iter().enumerate() {
        lanes[lane] = lanes[lane].max(value);
    }
    let a = lanes[0].max(lanes[4]).max(lanes[2].max(lanes[6]));
    let b = lanes[1].max(lanes[5]).max(lanes[3].max(lanes[7]));
    a.max(b)
}

#[inline(always)]
fn sum_exp_shifted(values: &[f32], maximum: f32) -> f32 {
    let mut lanes = [0.0f32; LANES];
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    for chunk in chunks {
        for lane in 0..LANES {
            lanes[lane] += (chunk[lane] - maximum).exp();
        }
    }
    for (lane, &value) in tail.iter().enumerate() {
        lanes[lane] += (value - maximum).exp();
    }
    combine(lanes)
}

#[inline(always)]
fn corrector_features<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (count, width) = (cx.dim_k() as usize, cx.dim_h() as usize);
    let line = cx.arg_logits().row([row, 0]);
    let maximum = max_f32(line);
    let total = sum_exp_shifted(line, maximum);
    let normalizer = maximum + total.ln();
    let unary = cx.arg_unary().row([row, 0]);
    let norm_weight: [f32; 5] = core::array::from_fn(|index| cx.arg_norm_weight().get([index]));
    let norm_bias: [f32; 5] = core::array::from_fn(|index| cx.arg_norm_bias().get([index]));
    let up_weight = cx.arg_up_weight();
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
            core::array::from_fn(|index| (features[index] - mean) * inverse * norm_weight[index] + norm_bias[index]);
        for (column, (value, &bias)) in out.iter_mut().zip(up_bias.iter()).enumerate() {
            let weights = &up_weight.row([column, 0])[..5];
            let mut sum = bias;
            sum = weights[0].mul_add(normed[0], sum);
            sum = weights[1].mul_add(normed[1], sum);
            sum = weights[2].mul_add(normed[2], sum);
            sum = weights[3].mul_add(normed[3], sum);
            sum = weights[4].mul_add(normed[4], sum);
            *value = activation::silu(sum);
        }
        // SAFETY: each work item writes its own candidates' rows.
        activation::store::<E::A>(out, unsafe { cx.result_0().row_mut([row * count + rank, 0]) });
    }
}
