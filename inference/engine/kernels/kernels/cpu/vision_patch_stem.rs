// vision_patch_stem on CPU (contract and portable body in vision.seismic).
// Each frame's rows are its weight's operand: the operator's form launch forms
// them from the entry's gathers of the patch rows' frames as F32 rows of
// C * P * P values (`vision_patch_stem_first` for `frame_weight`,
// `vision_patch_stem_next` frame-major for `next_frame_weight`), and their Q8
// blocks under the INT8 form. `vision_patch_stem_rows` gives each work item
// ROWS outputs, each frame's weights (`[H, C, P, P]` as rows of C * P * P
// values) projected against that frame's rows, then blends its outputs' four
// position-table values (in corner order): result = (frame 0 . w0 + frame 1
// . w1 + bias) + blended, as the portable body adds them.

use lib::projection::projection;
use lib::vision::vision;

/// Row `row` of a frame's operand rows (`values (..)`, C * P * P values in
/// channel, y, x order): patch row `patch`'s frame `frame` of `pixels`.
#[inline(always)]
fn frame_row<E: Elements>(cx: &Context<'_, E>, patch: usize, frame: usize, out: &mut [f32]) {
    let p = cx.dim_p() as usize;
    let pixels = cx.arg_pixels();
    for (channel, out) in out.chunks_exact_mut(p * p).enumerate() {
        for (y, out) in out.chunks_exact_mut(p).enumerate() {
            out.copy_from_slice(pixels.row([patch, channel, frame, y, 0]));
        }
    }
}

/// Frame 0's operand row `row`, which `frame_weight` multiplies.
#[inline(always)]
fn vision_patch_stem_first<E: Elements>(cx: &Context<'_, E>, row: usize, out: &mut [f32], _shared: &mut [u8]) {
    frame_row(cx, row, 0, out)
}

/// The next frames' operand row `row` (frame-major: frame `1 + row / M`'s
/// patch row `row % M`), which `next_frame_weight` multiplies.
#[inline(always)]
fn vision_patch_stem_next<E: Elements>(cx: &Context<'_, E>, row: usize, out: &mut [f32], _shared: &mut [u8]) {
    let m = cx.dim_m() as usize;
    frame_row(cx, row % m, 1 + row / m, out)
}

#[inline(always)]
fn vision_patch_stem_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (m, c, s, p, h) =
        (cx.dim_m() as usize, cx.dim_c() as usize, cx.dim_s() as usize, cx.dim_p() as usize, cx.dim_h() as usize);
    let patch = c * p * p;
    let outputs = projection::item_rows(group[0], cx.param_rows(), h);
    let result = cx.result_0();
    let (bias_space, decoded) = seismic::cpu::tensor::floats(shared, 2 * h).split_at_mut(h);
    let bias = (cx.dim_nb() == 1).then(|| vision::decode_vector(&cx.arg_bias(), bias_space));
    // Each frame's weight rows of C * P rows of P values, as one row each
    // (the next frames' only where there are any: S = 0 binds none).
    let frame_weight = cx.arg_frame_weight().joined(c * p);
    let next_frame_weight = (s > 0).then(|| cx.arg_next_frame_weight().joined(c * p));
    let quantized = cx.param_int8() == 1;
    for frame in 0..1 + s {
        // SAFETY: the form launch wrote every row before this launch.
        let staged = unsafe {
            match frame {
                0 => cx.scratch_w0_rows().slice::<f32>(0, m * patch),
                _ => cx.scratch_w1_rows().slice::<f32>(4 * (frame - 1) * m * patch, m * patch),
            }
        };
        // Frame 0's rows against `frame_weight`, frame f's against rows
        // (f - 1) H .. f H of `next_frame_weight`, each sum added in frame order.
        let (weights, first, rows) = match frame {
            0 => (&frame_weight, outputs.start, quantized.then(|| unsafe { projection::quantized(cx.scratch_w0_quantized(), 0, m, patch, false) })),
            _ => {
                let at = (frame - 1) * m * seismic::cpu::quant::blocks(patch) * seismic::cpu::quant::Q8_BYTES;
                let rows = quantized.then(|| unsafe { projection::quantized(cx.scratch_w1_quantized(), at, m, patch, false) });
                let weights = next_frame_weight.as_ref().expect("a frame past the first has its weights");
                (weights, (frame - 1) * h + outputs.start, rows)
            }
        };
        projection::project_staged_arithmetic(weights, first..first + outputs.len(), staged, rows, |row, projected| {
            // SAFETY: each work item writes its own columns of every row.
            let mut out = unsafe { result.span_mut([row, outputs.start], outputs.len()) };
            for (column, projected) in projected.iter().enumerate().take(out.len()) {
                let sum = if frame == 0 { *projected } else { out.get(column) + projected };
                out.set(column, sum);
            }
        });
    }
    // The position table's columns of these outputs, decoded per corner.
    let (table, indices, coefficients) = (cx.arg_table(), cx.arg_indices(), cx.arg_coefficients());
    let (first, view) = table.columns(outputs.clone());
    let at = outputs.start - first;
    for row in 0..m {
        let mut blended = [0.0f32; projection::MAX_ROWS];
        let blended = &mut blended[..outputs.len()];
        if cx.dim_nt() == 1 {
            for corner in 0..4 {
                let decoded = &mut decoded[..view.k()];
                view.decode_row(indices.get([0, row, corner]) as usize, decoded);
                let coefficient = coefficients.get([0, row, corner]);
                for (target, value) in blended.iter_mut().zip(&decoded[at..]) {
                    *target += value * coefficient;
                }
            }
        }
        // SAFETY: each work item writes its own columns of every row.
        let mut out = unsafe { result.span_mut([row, outputs.start], outputs.len()) };
        for ((column, output), blended) in outputs.clone().enumerate().zip(blended.iter()) {
            let projected = out.get(column);
            let value = bias.map_or(projected, |bias| projected + bias[output]);
            out.set(column, value + blended);
        }
    }
}
