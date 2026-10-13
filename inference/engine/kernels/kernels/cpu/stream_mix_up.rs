// stream_mix_up on CPU (contract and portable body in streams.seismic).
// `stream_mix_up_stage` stages each low-rank row as F32 in scratch, one work
// item per row. `stream_mix_up_rows` gives each work item ROWS weight rows of
// `up`, projected against every staged row: column j of stream s = j / H
// publishes `sigmoid(v) * x[j] * inverse[s] * norm[j] / S` in F32.

use lib::core::activation;
use lib::projection::projection;

fn stream_mix_up_stage<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let r = cx.dim_lr() as usize;
    // SAFETY: each work item writes its own row of the scratch.
    let staged = unsafe { cx.scratch_staged().slice_mut::<f32>(4 * row * r, r) };
    projection::stage::<E::A>(cx.arg_low().row([row, 0]), staged);
}

fn stream_mix_up_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let (m, h, r) = (
        cx.dim_m() as usize,
        cx.dim_h() as usize,
        cx.dim_lr() as usize,
    );
    let width = cx.dim_s() as usize * h;
    let streams = cx.dim_s() as f32;
    let rows = projection::item_rows(group[0], cx.param_rows(), width);
    let norm = seismic::cpu::tensor::floats(shared, width);
    for stream in 0..cx.dim_s() as usize {
        cx.arg_norm()
            .decode_row(stream, &mut norm[stream * h..(stream + 1) * h]);
    }
    let result = cx.result_0();
    // SAFETY: the stage launch wrote every row before this launch.
    let staged = unsafe { cx.scratch_staged().slice::<f32>(0, m * r) };
    projection::project_staged(
        &cx.arg_up_weight(),
        rows.clone(),
        staged,
        |row, projected| {
            let inverses = cx.arg_inverse().row([row, 0]);
            // SAFETY: each work item writes its own columns of every row.
            let out = unsafe { result.span_mut([row, rows.start], rows.len()) };
            for ((target, column), v) in out.iter_mut().zip(rows.clone()).zip(projected) {
                let (stream, at) = (column / h, column % h);
                let normalized =
                    cx.arg_hidden().get([row, stream, at]) * inverses[stream] * norm[column];
                *target = activation::sigmoid(*v) * normalized / streams;
            }
        },
    );
}
