// readout_selected_rows on CPU (contract and portable body in
// readout.seismic). The operator's form launch forms the RMS-normalized
// hidden row of each output row (rounded to A) as F32 (`rows`) and its Q8
// blocks. `readout_selected_rows_rows` gives each work
// item ROWS selected vocabulary rows; the selected weight rows are not
// adjacent, so each is projected alone against every normalized row.
// Logits are softcapped when `softcap` > 0.

use lib::core::functions;
use lib::projection::projection;

#[inline(always)]
fn readout_selected_rows_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (o, d, sv) = (
        cx.dim_o() as usize,
        cx.dim_d() as usize,
        cx.dim_sv() as usize,
    );
    let columns = projection::item_rows(group[0], cx.param_rows(), sv);
    let (weight, selected, result) = (cx.arg_weight(), cx.arg_selected(), cx.result_0());
    // SAFETY: the form launch (or the operand's producer) wrote every row before this launch.
    let normalized = unsafe { cx.scratch_rows().slice::<f32>(0, o * d) };
    let tiles = projection::tiled(o, &[&weight]);
    let quantized = (cx.param_int8() == 1).then(|| unsafe { projection::quantized(cx.scratch_quantized(), 0, o, d, tiles) });
    let cap = cx.arg_softcap();
    for column in columns {
        let vocabulary = selected.get([column]) as usize;
        projection::project_staged_arithmetic(&weight, vocabulary..vocabulary + 1, normalized, quantized, |row, logit| {
            let logit = if cap > 0.0 { functions::softcap(cap, logit[0]) } else { logit[0] };
            // SAFETY: each work item writes its own columns of every row.
            unsafe { result.set([row, column], logit) };
        });
    }
}
