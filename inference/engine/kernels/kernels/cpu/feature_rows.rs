// feature_rows on CPU (contract and portable body in dflash.seismic): one
// work item per output row, which rounds its `out_rows` fused row to A.

use lib::core::activation;

fn feature_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let source = cx.arg_out_rows().get([row]) as usize;
    // SAFETY: each work item writes its own row.
    activation::store::<E::A>(cx.arg_fused().row([source, 0]), unsafe { cx.result_0().row_mut([row, 0]) });
}
