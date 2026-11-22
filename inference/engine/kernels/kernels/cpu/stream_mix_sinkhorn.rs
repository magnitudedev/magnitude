// stream_mix_sinkhorn on CPU (contract and portable body in streams.seismic):
// one work item per row, which projects its flattened streams by every
// function row under the row's one RMS statistic, forms pre, post and the
// Sinkhorn comb in place, and mixes the streams by the previous site's pre
// (P = 1, publishing its own for the next) or its own (P = 0).

use lib::core::{activation, functions, reduce};

/// The most streams a row's coefficients hold in place.
const MAX_STREAMS: usize = 8;

/// The most mix outputs a row holds in place (`2 * S + S * S` at `MAX_STREAMS`).
const MAX_MIXES: usize = 2 * MAX_STREAMS + MAX_STREAMS * MAX_STREAMS;

/// Columns of one stream processed together so the active input slice (1 KiB)
/// stays hot in L1D across `sum_squares` and all `2 * S + S * S` mix dots.
const CHUNK: usize = 256;

#[inline(always)]
fn stream_mix_sinkhorn<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let (h, s) = (cx.dim_h() as usize, cx.dim_s() as usize);
    let mix_count = 2 * s + s * s;
    let (epsilon, norm_epsilon) = (cx.arg_epsilon(), cx.arg_norm_epsilon());
    let (hidden, function, base, scale) =
        (cx.arg_hidden(), cx.arg_function(), cx.arg_base(), cx.arg_scale());
    let streams: [&[f32]; MAX_STREAMS] =
        std::array::from_fn(|stream| if stream < s { hidden.row([row, stream, 0]) } else { &[] });
    let streams = &streams[..s];
    let function_rows: [&[f32]; MAX_MIXES] =
        std::array::from_fn(|j| if j < mix_count { function.row([j, 0]) } else { &[] });
    let function_rows = &function_rows[..mix_count];
    // Plain loops, not closures: a closure is compiled outside the entry's
    // target features, so the vector reductions inside it would not inline.
    // Cache-block each stream in `CHUNK`-float slices (1 KiB) so `x_chunk` is
    // loaded once into L1D and reused across `squares` and every mix row.
    let mut squares = 0.0f32;
    let mut mix = [0.0f32; MAX_MIXES];
    let mix = &mut mix[..mix_count];
    for (stream, x) in streams.iter().enumerate() {
        let base_col = stream * h;
        let mut c0 = 0;
        while c0 < h {
            let c1 = (c0 + CHUNK).min(h);
            let x_chunk = &x[c0..c1];
            squares += reduce::sum_squares(x_chunk);
            let col0 = base_col + c0;
            let col1 = base_col + c1;
            for (m, weights) in mix.iter_mut().zip(function_rows) {
                *m += reduce::dot(&weights[col0..col1], x_chunk);
            }
            c0 = c1;
        }
    }
    let inverse = 1.0 / (squares / (s * h) as f32 + norm_epsilon).sqrt();
    for m in mix.iter_mut() {
        *m *= inverse;
    }
    // SAFETY: each work item writes its own row of every result.
    let (input, mut join) = unsafe { (cx.result_0().row_mut([row, 0]), cx.result_2().row_mut([row, 0])) };
    let mut pre = [0.0f32; MAX_STREAMS];
    for stream in 0..s {
        pre[stream] = activation::sigmoid(mix[stream] * scale.get([0]) + base.get([stream])) + epsilon;
        join.set(stream, 2.0 * activation::sigmoid(mix[s + stream] * scale.get([1]) + base.get([s + stream])));
    }
    let pre = &pre[..s];
    let mut comb = [[0.0f32; MAX_STREAMS]; MAX_STREAMS];
    for (source, values) in comb.iter_mut().enumerate().take(s) {
        let values = &mut values[..s];
        for (target, value) in values.iter_mut().enumerate() {
            let at = 2 * s + source * s + target;
            *value = mix[at] * scale.get([2]) + base.get([at]);
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
    for sweep in 0..cx.arg_iterations() {
        if sweep > 0 {
            for values in comb.iter_mut().take(s) {
                let sum: f32 = values[..s].iter().fold(0.0, |sum, value| sum + value);
                for value in &mut values[..s] {
                    *value /= sum + epsilon;
                }
            }
        }
        for target in 0..s {
            let sum: f32 = (0..s).fold(0.0, |sum, source| sum + comb[source][target]);
            for values in comb.iter_mut().take(s) {
                values[target] /= sum + epsilon;
            }
        }
    }
    for source in 0..s {
        join.part_mut(s + source * s, s).store(&comb[source][..s]);
    }
    let carried = if cx.dim_p() > 0 {
        // SAFETY: each work item writes its own row of every result.
        unsafe { cx.result_1().row_mut([row, 0]) }.store(pre);
        cx.arg_previous().row([row, 0])
    } else {
        pre
    };
    functions::collapse(carried.len(), |stream| carried[stream], |stream| streams[stream], input);
}
