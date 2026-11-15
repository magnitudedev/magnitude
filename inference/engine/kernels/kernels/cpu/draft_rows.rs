// draft_rows on CPU (contract and portable body in draft.seismic).
// The joined [2D] input of each row is the projection's operand, its values
// `draft_rows_joined`'s and its forms the operator's form launch's.
// `draft_rows_rows` gives each work item ROWS combine weight rows, which it
// projects against every joined row into the F32 result.

use lib::core::{activation, gathered};
use lib::projection::projection;

/// Row `row` of the joined operand [2D] (`values (draft_rows_joined)`),
/// which the operator's form launch forms: the successor token's embedding
/// (A) RMS-normalized with the embedding norm, then the conditioning row's
/// stream RMS-normalized with its columns of the hidden norm, each
/// published to A. `shared` holds 2D values (the launch's `FORM_WIDTH`).
#[inline(always)]
fn draft_rows_joined<E: Elements>(cx: &Context<'_, E>, row: usize, out: &mut [f32], shared: &mut [u8]) {
    let d = cx.dim_d() as usize;
    let streams = cx.dim_s() as usize;
    let (source_row, stream) = (row / streams, row % streams);
    let (values, wide) = seismic::cpu::tensor::floats(shared, 2 * d).split_at_mut(d);
    let (embedding, hidden) = out.split_at_mut(d);
    let token = gathered::row(cx.arg_tokens().get([source_row, 0]), cx.dim_r(), cx.dim_f(), |entry, field| {
        cx.arg_directory().get([entry, field])
    });
    // A row that names no table row embeds zeros.
    match token {
        Some(token) => cx.arg_table().decode_row(token, values),
        None => values.fill(0.0),
    }
    for value in values.iter_mut() {
        *value = activation::publish::<E::A>(*value);
    }
    // The embedding norm, decoded where the hidden half goes next.
    cx.arg_embedding_norm().decode_row(0, hidden);
    projection::rms_row::<E::A>(values, hidden, cx.arg_epsilon(), embedding);
    let columns = stream * d..(stream + 1) * d;
    values.copy_from_slice(decode_columns(&cx.arg_conditioning(), source_row, columns.clone(), wide));
    let weight = decode_columns(&cx.arg_hidden_norm(), 0, columns, wide);
    projection::rms_row::<E::A>(values, weight, cx.arg_epsilon(), hidden);
}

/// Columns `columns` of row `row` of `source`, decoded into `buffer` (their
/// whole storage groups, which fit it: a stream's columns of dense rows are
/// exactly its own).
#[inline(always)]
fn decode_columns<'b>(source: &seismic::cpu::Weights<'_>, row: usize, columns: std::ops::Range<usize>, buffer: &'b mut [f32]) -> &'b [f32] {
    let (first, view) = source.columns(columns.clone());
    let decoded = &mut buffer[..view.k()];
    view.decode_row(row, decoded);
    &decoded[columns.start - first..columns.end - first]
}

#[inline(always)]
fn draft_rows_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, d) = ((cx.dim_m() * cx.dim_s()) as usize, cx.dim_d() as usize);
    let rows = projection::item_rows(group[0], cx.param_rows(), d);
    let result = cx.result_0();
    // SAFETY: the form launch wrote every row before this launch.
    let joined = unsafe { cx.scratch_rows().slice::<f32>(0, m * 2 * d) };
    let quantized = (cx.param_int8() == 1).then(|| unsafe {
        projection::quantized(cx.scratch_quantized(), 0, m, 2 * d, projection::tiled(m, &[&cx.arg_combine()]))
    });
    projection::project_staged_arithmetic(
        &cx.arg_combine(),
        rows.clone(),
        joined,
        quantized,
        |row, projected| {
            // SAFETY: each work item writes its own columns of every row.
            unsafe { result.span_mut([row / cx.dim_s() as usize, (row % cx.dim_s() as usize) * d + rows.start], rows.len()) }.store(projected);
        },
    );
}
