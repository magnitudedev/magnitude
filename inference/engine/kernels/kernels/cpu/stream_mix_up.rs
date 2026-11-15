// stream_mix_up on CPU (contract and portable body in streams.seismic).
// The low-rank rows are the projection's operand, formed by the operator's
// form launch (F32, and their Q8 blocks under the INT8 form).
// `stream_mix_up_rows` gives each work item ROWS weight rows of `up`,
// projected against every operand row: column j of stream s = j / H
// publishes `sigmoid(v) * x[j] * inverse[s] * norm[j] / S` in F32.

use lib::core::activation;
use lib::projection::projection;

#[inline(always)]
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
    // The norm of these columns, a stream's columns at a time (each decoded
    // through its whole storage groups, at most a stream's H values).
    let mut norm = [0.0f32; projection::MAX_ROWS];
    let norm = &mut norm[..rows.len()];
    let decoded = seismic::cpu::tensor::floats(shared, h);
    let mut column = rows.start;
    while column < rows.end {
        let stream = column / h;
        let end = rows.end.min((stream + 1) * h);
        let (first, view) = cx.arg_norm().columns(column - stream * h..end - stream * h);
        let decoded = &mut decoded[..view.k()];
        view.decode_row(stream, decoded);
        norm[column - rows.start..end - rows.start]
            .copy_from_slice(&decoded[column - stream * h - first..end - stream * h - first]);
        column = end;
    }
    let (up, result) = (cx.arg_up_weight(), cx.result_0());
    // SAFETY: the form launch (or the low rows' producer) wrote every row
    // before this launch.
    let staged = unsafe { cx.scratch_rows().slice::<f32>(0, m * r) };
    let quantized = (cx.param_int8() == 1)
        .then(|| unsafe { projection::quantized(cx.scratch_quantized(), 0, m, r, projection::tiled(m, &[&up])) });
    projection::project_staged_arithmetic(&up, rows.clone(), staged, quantized, |row, projected| {
        let inverses = cx.arg_inverse().row([row, 0]);
        // SAFETY: each work item writes its own columns of every row.
        let mut out = unsafe { result.span_mut([row, rows.start], rows.len()) };
        for (((target, column), v), norm) in out.iter_mut().zip(rows.clone()).zip(projected).zip(norm.iter()) {
            let (stream, at) = (column / h, column % h);
            let normalized = cx.arg_hidden().get([row, stream, at]) * inverses[stream] * norm;
            target.set(activation::sigmoid(*v) * normalized / streams);
        }
    });
}
