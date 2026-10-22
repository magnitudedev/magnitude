// corrector_rows on CPU (contract and portable body in corrector.seismic): one
// work item per row: values plus bias, then by form the identity, silu or unit
// length, plus the residual row when there is one, stored to Y.

use lib::core::activation;

fn corrector_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let width = cx.dim_n() as usize;
    let values = seismic::cpu::tensor::floats(shared, width);
    let (input, bias) = (cx.arg_values().row([row, 0]), cx.arg_bias().row([0]));
    for ((value, input), bias) in values.iter_mut().zip(input).zip(bias) {
        *value = input + bias;
    }
    match cx.arg_form() {
        1 => {
            for value in values.iter_mut() {
                *value = activation::silu(*value);
            }
        }
        2 => {
            let length = values.iter().map(|value| value * value).sum::<f32>().sqrt().max(cx.arg_epsilon());
            for value in values.iter_mut() {
                *value /= length;
            }
        }
        _ => {}
    }
    if cx.dim_r() == 1 {
        for (value, residual) in values.iter_mut().zip(cx.arg_residual().row([0, row, 0])) {
            *value += residual;
        }
    }
    // SAFETY: each work item writes its own row.
    activation::store::<E::Y>(values, unsafe { cx.result_0().row_mut([row, 0]) });
}
