// vision_patch_stem on CPU (contract and portable body in vision.seismic).
// `vision_patch_stem_stage` gathers each patch row's 1 + S frames as
// contiguous F32 rows [1 + S][C * P * P] and blends its four position-table
// rows (in corner order), one work item per patch row.
// `vision_patch_stem_rows` gives each work item eight outputs, whose frame
// weights (decoded as F32) it projects against every staged patch row:
// result = (frame 0 . w0 + frame 1 . w1 + bias) + blended, all F32 as the
// portable body computes it.

use lib::core::reduce;
use lib::projection::projection;
use lib::vision::vision;

fn vision_patch_stem_stage<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (c, s, p, h) = (cx.dim_c() as usize, cx.dim_s() as usize, cx.dim_p() as usize, cx.dim_h() as usize);
    let patch = c * p * p;
    let pixels = cx.arg_pixels();
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_pixels().slice_mut::<f32>(4 * row * (1 + s) * patch, (1 + s) * patch) };
    for (frame, staged) in staged.chunks_exact_mut(patch).enumerate() {
        for (channel, staged) in staged.chunks_exact_mut(p * p).enumerate() {
            for (y, staged) in staged.chunks_exact_mut(p).enumerate() {
                staged.copy_from_slice(pixels.row([row, channel, frame, y, 0]));
            }
        }
    }
    let (table, indices, coefficients) = (cx.arg_table(), cx.arg_indices(), cx.arg_coefficients());
    let decoded = seismic::cpu::tensor::floats(shared, h);
    // SAFETY: each work item writes its own row of the scratch.
    let blended = unsafe { cx.scratch_blended().slice_mut::<f32>(4 * row * h, h) };
    blended.fill(0.0);
    for corner in 0..4 {
        table.decode_row(indices.get([row, corner]) as usize, decoded);
        let coefficient = coefficients.get([row, corner]);
        for (target, value) in blended.iter_mut().zip(decoded.iter()) {
            *target += value * coefficient;
        }
    }
}

fn vision_patch_stem_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (m, c, s, p, h) =
        (cx.dim_m() as usize, cx.dim_c() as usize, cx.dim_s() as usize, cx.dim_p() as usize, cx.dim_h() as usize);
    let (patch, frames) = (c * p * p, 1 + s);
    let outputs = projection::item_rows(group[0], 8, h);
    let (frame_weight, next_frame_weight, result) = (cx.arg_frame_weight(), cx.arg_next_frame_weight(), cx.result_0());
    let shared = seismic::cpu::tensor::floats(shared, h + 8 * frames * patch);
    let (bias, weights) = shared.split_at_mut(h);
    let bias = (cx.dim_nb() == 1).then(|| vision::decode_vector(&cx.arg_bias(), bias));
    // Weight rows of output o: its C * P rows of P values, frame by frame.
    for (output, weights) in outputs.clone().zip(weights.chunks_exact_mut(frames * patch)) {
        for (frame, weights) in weights.chunks_exact_mut(patch).enumerate() {
            for (index, row) in weights.chunks_exact_mut(p).enumerate() {
                if frame == 0 {
                    frame_weight.decode_row(output * c * p + index, row);
                } else {
                    next_frame_weight.decode_row(((frame - 1) * h + output) * c * p + index, row);
                }
            }
        }
    }
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_pixels().slice::<f32>(0, m * frames * patch) };
    // SAFETY: the stage launch wrote every row before this launch.
    let blended = unsafe { cx.scratch_blended().slice::<f32>(0, m * h) };
    for (row, pixels) in staged.chunks_exact(frames * patch).enumerate() {
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { result.span_mut([row, outputs.start], outputs.len()) };
        for ((target, output), weights) in out.iter_mut().zip(outputs.clone()).zip(weights.chunks_exact(frames * patch)) {
            let mut value = 0.0;
            for (pixels, weights) in pixels.chunks_exact(patch).zip(weights.chunks_exact(patch)) {
                value += reduce::dot(pixels, weights);
            }
            if let Some(bias) = bias {
                value += bias[output];
            }
            *target = value + blended[row * h + output];
        }
    }
}
