// vision_gated_convolution on CPU (contract and portable body in
// vision.seismic): one work item per row. The row's present neighbour rows
// are widened once; each channel's weight row is decoded once and its taps
// summed in tap order in F32, published to A, then times the row's gate and
// published again.

use lib::core::activation;

fn vision_gated_convolution<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (h, t) = (cx.dim_h() as usize, cx.dim_t() as usize);
    let (value, weight, neighbors) = (cx.arg_value(), cx.arg_weight(), cx.arg_neighbors());
    let shared = seismic::cpu::tensor::floats(shared, (t + 1) * h + t);
    let (sources, rest) = shared.split_at_mut(t * h);
    let (gate, taps) = rest.split_at_mut(h);
    for tap in 0..t {
        let neighbor = neighbors.get([row, tap]);
        if neighbor >= 0 {
            seismic::cpu::tensor::widen_row::<E::A>(value.row([neighbor as usize, 0]), &mut sources[tap * h..][..h]);
        }
    }
    seismic::cpu::tensor::widen_row::<E::A>(cx.arg_gate().row([row, 0]), gate);
    let result = cx.result_0();
    // SAFETY: each work item writes its own row.
    let out = unsafe { result.span_mut([row, 0], h) };
    for (channel, target) in out.iter_mut().enumerate() {
        weight.decode_row(channel, taps);
        let mut sum = 0.0f32;
        for tap in 0..t {
            if neighbors.get([row, tap]) >= 0 {
                sum = sources[tap * h + channel].mul_add(taps[tap], sum);
            }
        }
        *target = E::A::narrow(activation::publish::<E::A>(sum) * gate[channel]);
    }
}
