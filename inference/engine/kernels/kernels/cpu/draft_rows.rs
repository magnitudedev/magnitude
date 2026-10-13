// draft_rows on CPU (contract and portable body in draft.seismic).
// `draft_rows_join` forms each row's joined [2D] input as F32 in scratch, one
// work item per row: the successor token's embedding (A) RMS-normalized with
// the embedding norm, then the conditioning row RMS-normalized with the hidden
// norm, each published to A. `draft_rows_rows` gives each work item ROWS
// combine weight rows, which it projects against every joined row into the
// F32 result.

use lib::core::{activation, gathered};
use lib::projection::projection;

fn draft_rows_join<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    let streams = cx.dim_s() as usize;
    let (source_row, stream) = (row / streams, row % streams);
    let (staged, wide) = seismic::cpu::tensor::floats(shared, 2 * d + 2 * streams * d).split_at_mut(2 * d);
    let (values, norm) = staged.split_at_mut(d);
    let (condition, hidden_weight) = wide.split_at_mut(streams * d);
    cx.arg_conditioning().decode_row(source_row, condition);
    cx.arg_hidden_norm().decode_row(0, hidden_weight);
    // SAFETY: each work item writes its own row of the scratch.
    let joined = unsafe { cx.scratch_joined().slice_mut::<f32>(4 * row * 2 * d, 2 * d) };
    let (embedding, hidden) = joined.split_at_mut(d);
    let token = gathered::row(cx.arg_tokens().get([source_row, 0]), cx.dim_r(), cx.dim_f(), |entry, field| {
        cx.arg_directory().get([entry, field])
    });
    cx.arg_table().decode_row(token, values);
    for value in values.iter_mut() {
        *value = activation::publish::<E::A>(*value);
    }
    cx.arg_embedding_norm().decode_row(0, norm);
    projection::rms_row::<E::A>(values, norm, cx.arg_epsilon(), embedding);
    values.copy_from_slice(&condition[stream * d..(stream + 1) * d]);
    norm.copy_from_slice(&hidden_weight[stream * d..(stream + 1) * d]);
    projection::rms_row::<E::A>(values, norm, cx.arg_epsilon(), hidden);
    if cx.param_int8() == 1 {
        let blocks = seismic::cpu::quant::blocks(2 * d);
        // SAFETY: this work item owns the corresponding quantized row.
        let q8 = unsafe {
            cx.scratch_quantized()
                .slice_mut::<seismic::cpu::quant::Q8Block>(
                    row * blocks * projection::Q8_BYTES,
                    blocks,
                )
        };
        projection::quantize(joined, q8);
    }
}

fn draft_rows_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, d) = ((cx.dim_m() * cx.dim_s()) as usize, cx.dim_d() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), d);
    let result = cx.result_0();
    // SAFETY: the join launch wrote every row before this launch.
    let joined = unsafe { cx.scratch_joined().slice::<f32>(0, m * 2 * d) };
    let blocks = seismic::cpu::quant::blocks(2 * d);
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        cx.scratch_quantized()
            .slice::<seismic::cpu::quant::Q8Block>(0, m * blocks)
    });
    projection::project_staged_arithmetic(
        &cx.arg_combine(),
        rows.clone(),
        joined,
        quantized,
        |row, projected| {
            // SAFETY: each work item writes its own columns of every row.
            unsafe { result.span_mut([row / cx.dim_s() as usize, (row % cx.dim_s() as usize) * d + rows.start], rows.len()) }.copy_from_slice(projected);
        },
    );
}
