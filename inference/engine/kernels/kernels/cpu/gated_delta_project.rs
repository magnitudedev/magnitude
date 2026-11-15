// gated_delta_project on CPU (contract and portable body in
// recurrent.seismic). The operator's form launch forms each row's
// RMS-normalized hidden row (rounded to A) as F32 (`rows`) and its Q8 blocks.
// `gated_delta_project_rows` enumerates the segments
// qkv | z | alpha | beta in order, ROWS weight rows per work item, each
// projected against every normalized row and published in A at its segment's
// columns of the projection row.

use lib::core::activation;
use lib::projection::projection;

#[inline(always)]
fn gated_delta_project_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (m, h) = (cx.dim_m() as usize, cx.dim_h() as usize);
    let (nk, nv, w) = (
        cx.dim_nk() as usize,
        cx.dim_nv() as usize,
        cx.dim_w() as usize,
    );
    let totals = [(2 * nk + nv) * w, nv * w, nv, nv];
    let (segment, rows) = projection::segment_rows(group[0], cx.param_rows(), &totals);
    let weights = [
        cx.arg_qkv_weight(),
        cx.arg_gate_weight(),
        cx.arg_alpha_weight(),
        cx.arg_beta_weight(),
    ][segment];
    let column = totals[..segment].iter().sum::<usize>() + rows.start;
    let result = cx.result_0();
    // SAFETY: the form launch (or the hidden rows' producer) wrote every row before this launch.
    let normalized = unsafe { cx.scratch_rows().slice::<f32>(0, m * h) };
    // The alpha and beta segments project exactly under either arithmetic:
    // they set the recurrence's decay and write strength, where activation
    // rounding compounds over the whole sequence, and they are two of the
    // projection's (2 NK + 2 NV) W + 2 NV rows. The Q8 blocks are tiles
    // where the two weights that read them take tiles.
    let quantized = (cx.param_int8() == 1 && segment < 2).then(|| unsafe {
        projection::quantized(cx.scratch_quantized(), 0, m, h, projection::tiled(m, &[&cx.arg_qkv_weight(), &cx.arg_gate_weight()]))
    });
    projection::project_staged_arithmetic(
        &weights,
        rows.clone(),
        normalized,
        quantized,
        |row, projected| {
            // SAFETY: each work item writes its own columns of every row.
            activation::store::<E::A>(projected, unsafe {
                result.span_mut([row, column], rows.len())
            });
        },
    );
}
