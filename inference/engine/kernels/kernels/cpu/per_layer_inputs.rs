// per_layer_inputs on CPU (contract and portable body in residual.seismic):
// one work item per row, which decodes its gathered row and the shared chunk
// norm, then forms every chunk
//   (p * rms_inverse(p) * norm + gathered * gathered_scale) * scale
// with p the chunk's projection values times `projected_scale`.

use lib::core::reduce;

fn per_layer_inputs<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let chunk = cx.dim_p() as usize;
    let width = cx.dim_l() as usize * chunk;
    let (gathered, rest) = shared.split_at_mut(4 * width);
    let (norm, scaled) = rest.split_at_mut(4 * chunk);
    let gathered = seismic::cpu::tensor::floats(gathered, width);
    let norm = seismic::cpu::tensor::floats(norm, chunk);
    let p = seismic::cpu::tensor::floats(scaled, chunk);
    cx.arg_gathered().decode_row(row, gathered);
    cx.arg_norm().decode_row(0, norm);
    let (epsilon, gathered_scale, projected_scale, scale) =
        (cx.arg_epsilon(), cx.arg_gathered_scale(), cx.arg_projected_scale(), cx.arg_scale());
    let projected = cx.arg_projected().row([row, 0]);
    // SAFETY: each work item writes its own row.
    let out = unsafe { cx.result_0().row_mut([row, 0]) };
    for first in (0..width).step_by(chunk) {
        for (target, value) in p.iter_mut().zip(&projected[first..first + chunk]) {
            *target = value * projected_scale;
        }
        let inverse = reduce::rms_inverse(p, epsilon);
        for i in 0..chunk {
            out[first + i] = (p[i] * inverse * norm[i] + gathered[first + i] * gathered_scale) * scale;
        }
    }
}
