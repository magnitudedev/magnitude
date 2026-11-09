// stream_mix_sinkhorn on CPU (contract and portable body in streams.seismic):
// one work item per row, which projects its flattened streams by every
// function row under the row's one RMS statistic, forms pre, post and the
// Sinkhorn comb in place, and mixes the streams by the previous site's pre.

use lib::core::{activation, reduce};

/// The most streams a row's coefficients hold in place.
const MAX_STREAMS: usize = 8;

fn stream_mix_sinkhorn<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let (h, s) = (cx.dim_h() as usize, cx.dim_s() as usize);
    let (epsilon, norm_epsilon) = (cx.arg_epsilon(), cx.arg_norm_epsilon());
    let (hidden, function, base, scale) =
        (cx.arg_hidden(), cx.arg_function(), cx.arg_base(), cx.arg_scale());
    let streams: [&[f32]; MAX_STREAMS] =
        std::array::from_fn(|stream| if stream < s { hidden.row([row, stream, 0]) } else { &[] });
    let streams = &streams[..s];
    let squares: f32 = streams.iter().map(|x| reduce::sum_squares(x)).sum();
    let inverse = 1.0 / (squares / (s * h) as f32 + norm_epsilon).sqrt();
    // Mix j of the row: function row j against the flattened streams.
    let mix = |j: usize| -> f32 {
        let weights = function.row([j, 0]);
        let projected: f32 = streams
            .iter()
            .enumerate()
            .map(|(stream, x)| reduce::dot(&weights[stream * h..(stream + 1) * h], x))
            .sum();
        projected * inverse
    };
    // SAFETY: each work item writes its own row of every result.
    let (input, pre, join) = unsafe {
        (cx.result_0().row_mut([row, 0]), cx.result_1().row_mut([row, 0]), cx.result_2().row_mut([row, 0]))
    };
    for stream in 0..s {
        pre[stream] = activation::sigmoid(mix(stream) * scale.get([0]) + base.get([stream])) + epsilon;
        join[stream] = 2.0 * activation::sigmoid(mix(s + stream) * scale.get([1]) + base.get([s + stream]));
    }
    let mut comb = [[0.0f32; MAX_STREAMS]; MAX_STREAMS];
    for (source, values) in comb.iter_mut().enumerate().take(s) {
        let values = &mut values[..s];
        for (target, value) in values.iter_mut().enumerate() {
            let at = 2 * s + source * s + target;
            *value = mix(at) * scale.get([2]) + base.get([at]);
        }
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0f32;
        for value in values.iter_mut() {
            *value = (*value - maximum).exp();
            total += *value;
        }
        for value in values.iter_mut() {
            *value = *value / total + epsilon;
        }
    }
    let columns = |comb: &mut [[f32; MAX_STREAMS]; MAX_STREAMS]| {
        for target in 0..s {
            let sum: f32 = (0..s).fold(0.0, |sum, source| sum + comb[source][target]);
            for values in comb.iter_mut().take(s) {
                values[target] /= sum + epsilon;
            }
        }
    };
    for sweep in 0..cx.arg_iterations() {
        if sweep > 0 {
            for values in comb.iter_mut().take(s) {
                let sum: f32 = values[..s].iter().fold(0.0, |sum, value| sum + value);
                for value in &mut values[..s] {
                    *value /= sum + epsilon;
                }
            }
        }
        columns(&mut comb);
    }
    for source in 0..s {
        join[s + source * s..s + (source + 1) * s].copy_from_slice(&comb[source][..s]);
    }
    let previous = cx.arg_previous().row([row, 0]);
    for (column, target) in input.iter_mut().enumerate() {
        *target = (0..s).fold(0.0, |sum, stream| sum + previous[stream] * streams[stream][column]);
    }
}
