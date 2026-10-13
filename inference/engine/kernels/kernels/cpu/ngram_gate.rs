// ngram_gate on CPU (contract and portable body in ngram.seismic): one work
// item per row, which publishes each stream's gate sigmoid(g) and the
// inverse RMS of its gated value row.

use lib::core::{activation, reduce};

fn ngram_gate<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (h, s) = (cx.dim_h() as usize, cx.dim_s() as usize);
    let norms = seismic::cpu::tensor::floats(shared, 2 * s * h);
    let (key_norm, query_norm) = norms.split_at_mut(s * h);
    for stream in 0..s {
        cx.arg_key_norm()
            .decode_row(stream, &mut key_norm[stream * h..(stream + 1) * h]);
        cx.arg_query_norm()
            .decode_row(stream, &mut query_norm[stream * h..(stream + 1) * h]);
    }
    let epsilon = cx.arg_epsilon();
    let values = reduce::sum_squares(&cx.arg_value().row([row, 0])[..h]) / h as f32;
    // SAFETY: each work item writes its own row.
    let out = unsafe { cx.result_0().row_mut([row, 0]) };
    for stream in 0..s {
        let k = &cx.arg_key().row([row, stream, 0])[..h];
        let x = &cx.arg_hidden().row([row, stream, 0])[..h];
        let mut dot = 0.0f32;
        for column in 0..h {
            dot += k[column] * key_norm[stream * h + column] * x[column] * query_norm[stream * h + column];
        }
        let g = dot * reduce::rms_inverse(k, epsilon) * reduce::rms_inverse(x, epsilon) / (h as f32).sqrt();
        let root = g.abs().max(1e-6).sqrt();
        let signed = if g > 0.0 {
            root
        } else if g < 0.0 {
            -root
        } else {
            0.0
        };
        let gate = activation::sigmoid(signed);
        out[stream] = gate;
        out[s + stream] = 1.0 / (gate * gate * values + epsilon).sqrt();
    }
}
