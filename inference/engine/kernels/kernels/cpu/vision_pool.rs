// vision_pool on CPU (contract and portable body in vision.seismic): one work
// item per cell, the sum of its G rows each times `weight`, times `scale`,
// then standardized when NS = 1.

use lib::vision::vision;

#[inline(always)]
fn vision_pool<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (g, h) = (cx.dim_g() as usize, cx.dim_h() as usize);
    let (weight, scale) = (cx.arg_weight(), cx.arg_scale());
    let source = cx.arg_source();
    let standardize = (cx.dim_ns() == 1).then(|| {
        let shared = seismic::cpu::tensor::floats(shared, 2 * h);
        let (bias, standard_scale) = shared.split_at_mut(h);
        (
            vision::decode_vector(&cx.arg_standard_bias(), bias),
            vision::decode_vector(&cx.arg_standard_scale(), standard_scale),
        )
    });
    // SAFETY: each work item writes its own row.
    unsafe { cx.result_0().row_mut([row, 0]) }.compute(|out| {
        if g == 0 {
            out.fill(0.0);
        } else {
            for (acc, &value) in out.iter_mut().zip(source.row([row, 0, 0])) {
                *acc = value * weight;
            }
            for member in 1..g {
                for (acc, &value) in out.iter_mut().zip(source.row([row, member, 0])) {
                    *acc = value.mul_add(weight, *acc);
                }
            }
        }
        if let Some((bias, standard_scale)) = standardize {
            for ((value, &b), &s) in out.iter_mut().zip(bias).zip(standard_scale) {
                *value = (*value * scale - b) * s;
            }
        } else {
            for value in out.iter_mut() {
                *value *= scale;
            }
        }
    });
}
