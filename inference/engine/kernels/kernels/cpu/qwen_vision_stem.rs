// qwen_vision_stem on CPU (contract and portable body in vision.seismic).
// `qwen_vision_stem_stage` gathers each patch row's two temporal frames as
// contiguous F32 rows [2][C * P * P] and blends its four position-table rows
// (in corner order), one work item per patch row. `qwen_vision_stem_rows`
// gives each work item ROWS outputs, whose two frame weights (decoded as F32)
// it projects against every staged patch row: result = (frame 0 . w0 +
// frame 1 . w1 + bias) + blended, all F32 as the portable body computes it.

use lib::core::reduce;
use lib::projection::projection;
use lib::vision::vision;

fn qwen_vision_stem_stage<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (c, p, h) = (cx.dim_c() as usize, cx.dim_p() as usize, cx.dim_h() as usize);
    let patch = c * p * p;
    let pixels = cx.arg_pixels();
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_pixels().slice_mut::<f32>(4 * row * 2 * patch, 2 * patch) };
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

fn qwen_vision_stem_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (m, c, p, h) = (cx.dim_m() as usize, cx.dim_c() as usize, cx.dim_p() as usize, cx.dim_h() as usize);
    let patch = c * p * p;
    let outputs = projection::item_rows(group[0], cx.param_rows(), h);
    let (weight_0, weight_1, result) = (cx.arg_temporal_weight_0(), cx.arg_temporal_weight_1(), cx.result_0());
    let shared = seismic::cpu::tensor::floats(shared, h + 2 * outputs.len() * patch);
    let (bias, weights) = shared.split_at_mut(h);
    let bias = vision::decode_vector(&cx.arg_bias(), bias);
    // Weight rows of output o: its C * P rows of P values, frame 0 then frame 1.
    for (output, weights) in outputs.clone().zip(weights.chunks_exact_mut(2 * patch)) {
        let (frame_0, frame_1) = weights.split_at_mut(patch);
        for (index, (row_0, row_1)) in frame_0.chunks_exact_mut(p).zip(frame_1.chunks_exact_mut(p)).enumerate() {
            weight_0.decode_row(output * c * p + index, row_0);
            weight_1.decode_row(output * c * p + index, row_1);
        }
    }
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_pixels().slice::<f32>(0, m * 2 * patch) };
    // SAFETY: the stage launch wrote every row before this launch.
    let blended = unsafe { cx.scratch_blended().slice::<f32>(0, m * h) };
    for row in 0..m {
        let (frame_0, frame_1) = staged[row * 2 * patch..(row + 1) * 2 * patch].split_at(patch);
        // SAFETY: each work item writes its own columns of every row.
        let out = unsafe { result.span_mut([row, outputs.start], outputs.len()) };
        for ((target, output), weights) in out.iter_mut().zip(outputs.clone()).zip(weights.chunks_exact(2 * patch)) {
            let (weight_0, weight_1) = weights.split_at(patch);
            let value = reduce::dot(frame_0, weight_0) + reduce::dot(frame_1, weight_1);
            *target = (value + bias[output]) + blended[row * h + output];
        }
    }
}
