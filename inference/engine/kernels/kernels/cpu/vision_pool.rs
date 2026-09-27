// vision_pool on CPU (contract and portable body in vision.seismic): one work
// item per cell, the sum of its G rows each times `weight`, times `scale`,
// then standardized when NS = 1.

use lib::vision::vision;

fn vision_pool<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (g, h) = (cx.dim_g() as usize, cx.dim_h() as usize);
    let (weight, scale) = (cx.arg_weight(), cx.arg_scale());
    let source = cx.arg_source();
    let shared = seismic::cpu::tensor::floats(shared, 2 * h);
    let (bias, standard_scale) = shared.split_at_mut(h);
    let standardize = (cx.dim_ns() == 1).then(|| {
        (
            vision::decode_vector(&cx.arg_standard_bias(), bias),
            vision::decode_vector(&cx.arg_standard_scale(), standard_scale),
        )
    });
    let result = cx.result_0();
    // SAFETY: each work item writes its own row.
    let out = unsafe { result.span_mut([row, 0], h) };
    for (column, target) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for member in 0..g {
            sum += source.get([row, member, column]) * weight;
        }
        let mut value = sum * scale;
        if let Some((bias, standard_scale)) = standardize {
            value = (value - bias[column]) * standard_scale[column];
        }
        *target = value;
    }
}
