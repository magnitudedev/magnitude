// readout_features_rows on CPU (contract and portable body in
// readout.seismic): one work item per output row, which RMS-normalizes its
// `out_rows` hidden row and stores it to A.

use lib::core::activation;
use lib::projection::projection;

fn readout_features_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let d = cx.dim_d() as usize;
    let (features, norm) = shared.split_at_mut(4 * d);
    let features = seismic::cpu::tensor::floats(features, d);
    let source = cx.arg_out_rows().get([row]) as usize;
    projection::normalize::<E::A>(cx.arg_hidden().row([source, 0]), &cx.arg_norm(), cx.arg_epsilon(), norm, features);
    // SAFETY: each work item writes its own row.
    activation::store::<E::A>(features, unsafe { cx.result_0().row_mut([row, 0]) });
}
