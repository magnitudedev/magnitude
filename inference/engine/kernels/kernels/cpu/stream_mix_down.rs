// stream_mix_down on CPU (contract and portable body in streams.seismic).
// `stream_mix_down_normalize` normalizes each stream of a row (rounded to A)
// into F32 scratch and publishes the streams' inverses, one work item per
// row. `stream_mix_down_rows` enumerates the segments down | inject, ROWS
// weight rows per work item, each projected against every normalized row:
// the down rows publish `silu(v / S)` in A, the inject rows `2 sigmoid(v / S)`.

use lib::core::{activation, reduce};
use lib::projection::projection;

fn stream_mix_down_normalize<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let (h, s) = (cx.dim_h() as usize, cx.dim_s() as usize);
    let width = s * h;
    let norm = seismic::cpu::tensor::floats(shared, h);
    // SAFETY: each work item writes its own row of the scratch and of the
    // inverses.
    let normalized = unsafe {
        cx.scratch_normalized()
            .slice_mut::<f32>(4 * row * width, width)
    };
    let inverses = unsafe { cx.result_2().span_mut([row, 0], s) };
    for stream in 0..s {
        cx.arg_norm().decode_row(stream, norm);
        let x = cx.arg_hidden().row([row, stream, 0]);
        let inverse = reduce::rms_inverse(x, cx.arg_epsilon());
        inverses[stream] = inverse;
        for ((target, value), w) in normalized[stream * h..(stream + 1) * h]
            .iter_mut()
            .zip(x)
            .zip(&*norm)
        {
            *target = activation::publish::<E::A>(value * inverse * w);
        }
    }
}

fn stream_mix_down_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let m = cx.dim_m() as usize;
    let (width, streams) = ((cx.dim_s() * cx.dim_h()) as usize, cx.dim_s() as f32);
    let segments = [cx.dim_lr() as usize, (cx.dim_c() * cx.dim_s()) as usize];
    let (segment, rows) = projection::segment_rows(group[0], cx.param_rows(), &segments);
    // SAFETY: the normalize launch wrote every row before this launch.
    let normalized = unsafe { cx.scratch_normalized().slice::<f32>(0, m * width) };
    if segment == 0 {
        let result = cx.result_0();
        projection::project_staged(
            &cx.arg_down_weight(),
            rows.clone(),
            normalized,
            |row, projected| {
                let mut low = [0.0f32; projection::MAX_ROWS];
                for (target, v) in low.iter_mut().zip(projected) {
                    *target = activation::silu(v / streams);
                }
                // SAFETY: each work item writes its own columns of every row.
                activation::store::<E::A>(&low[..rows.len()], unsafe {
                    result.span_mut([row, rows.start], rows.len())
                });
            },
        );
    } else {
        let result = cx.result_1();
        projection::project_staged(
            &cx.arg_inject_weight(),
            rows.clone(),
            normalized,
            |row, projected| {
                // SAFETY: each work item writes its own columns of every row.
                let out = unsafe { result.span_mut([row, rows.start], rows.len()) };
                for (target, v) in out.iter_mut().zip(projected) {
                    *target = 2.0 * activation::sigmoid(v / streams);
                }
            },
        );
    }
}
