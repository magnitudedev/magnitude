// per_layer_inputs on CPU (contract and portable body in residual.seismic):
// one work item per row, which decodes its gathered row and the shared chunk
// norm, then forms every chunk
//   (p * rms_inverse(p) * norm + gathered * gathered_scale) * scale
// with p the chunk's projection values times `projected_scale`.

use seismic::cpu::reduce::{LANES, combine};

#[inline(always)]
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
    let projected = &cx.arg_projected().row([row, 0])[..width];
    let full = chunk / LANES * LANES;
    // SAFETY: each work item writes its own row.
    unsafe { cx.result_0().row_mut([row, 0]) }.compute(|out| {
        let out = &mut out[..width];
        for first in (0..width).step_by(chunk) {
            let proj = &projected[first..first + chunk];
            // Fuse chunk scaling and its defined-order 8-lane square sum in one pass.
            let mut lanes = [0.0f32; LANES];
            for (dst, src) in p[..full].chunks_exact_mut(LANES).zip(proj[..full].chunks_exact(LANES)) {
                for lane in 0..LANES {
                    let value = src[lane] * projected_scale;
                    dst[lane] = value;
                    lanes[lane] = value.mul_add(value, lanes[lane]);
                }
            }
            for (lane, (dst, src)) in p[full..].iter_mut().zip(&proj[full..]).enumerate() {
                let value = *src * projected_scale;
                *dst = value;
                lanes[lane] = value.mul_add(value, lanes[lane]);
            }
            let inverse = 1.0 / (combine(lanes) / chunk as f32 + epsilon).sqrt();
            let g = &gathered[first..first + chunk];
            let dst = &mut out[first..first + chunk];
            for i in 0..chunk {
                dst[i] = (p[i] * inverse * norm[i] + g[i] * gathered_scale) * scale;
            }
        }
    });
}
