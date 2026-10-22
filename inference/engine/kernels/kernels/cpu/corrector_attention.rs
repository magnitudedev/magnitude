// corrector_attention on CPU (contract and portable body in corrector.seismic):
// one work item per (node, head): its biased query against every key of its
// block with the head's additive bias, the softmax, and the weighted values,
// rounded once to A.

use lib::core::activation;

fn corrector_attention<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (row, head) = (group[0] as usize, group[1] as usize);
    let (nodes, width) = (cx.dim_n() as usize, cx.dim_w() as usize);
    let (block, query_node) = (row / nodes, row % nodes);
    let work = seismic::cpu::tensor::floats(shared, nodes + 2 * width);
    let (scores, work) = work.split_at_mut(nodes);
    let (query, out) = work.split_at_mut(width);
    let (qkv, input_bias, bias) = (cx.arg_qkv(), cx.arg_input_bias(), cx.arg_bias());
    let (query_bias, key_bias, value_bias) = (
        input_bias.row([0, head, 0]),
        input_bias.row([1, head, 0]),
        input_bias.row([2, head, 0]),
    );
    for ((value, q), b) in query.iter_mut().zip(qkv.row([block, query_node, 0, head, 0])).zip(query_bias) {
        *value = q + b;
    }
    let scale = 1.0 / (width as f32).sqrt();
    for (key, score) in scores.iter_mut().enumerate() {
        let dot = query
            .iter()
            .zip(qkv.row([block, key, 1, head, 0]))
            .zip(key_bias)
            .map(|((q, k), b)| q * (k + b))
            .sum::<f32>();
        *score = dot * scale + bias.get([head, query_node, key]);
    }
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    for score in scores.iter_mut() {
        *score = (*score - maximum).exp();
        total += *score;
    }
    out.fill(0.0);
    for (key, weight) in scores.iter().enumerate() {
        for ((value, v), b) in out.iter_mut().zip(qkv.row([block, key, 2, head, 0])).zip(value_bias) {
            *value = weight.mul_add(v + b, *value);
        }
    }
    for value in out.iter_mut() {
        *value /= total;
    }
    // SAFETY: each work item writes its own head of its node.
    activation::store::<E::A>(out, unsafe { cx.result_0().row_mut([block, query_node, head, 0]) });
}
