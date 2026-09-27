// tap_rows on CPU (contract and portable body in dflash.seismic): one work
// item per row, which rounds its residual row to A into the row's column
// block `index[0]` of the draft input rows.

use lib::core::activation;

fn tap_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    let tap = cx.arg_index().get([0]) as usize;
    // SAFETY: each work item writes its own row.
    let taps = unsafe { cx.arg_taps().row_mut([row, 0]) };
    activation::store::<E::A>(cx.arg_residual().row([row, 0]), &mut taps[tap * d..(tap + 1) * d]);
}
