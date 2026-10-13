// readout_features_rows on CPU (contract and portable body in
// readout.seismic): one work item per output row, which forms its `out_rows`
// hidden row (`form_input`: RMS-normalized, or the sum of a mix site's
// partials) from its PT parts and stores it to A.

use lib::core::activation;
use lib::projection::projection;

fn readout_features_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (d, parts) = (cx.dim_d() as usize, cx.dim_pt() as usize);
    let (features, rest) = shared.split_at_mut(4 * d);
    let (norm, gathered) = rest.split_at_mut(4 * d);
    let features = seismic::cpu::tensor::floats(features, d);
    let gathered = seismic::cpu::tensor::floats(gathered, parts * d);
    let source = cx.arg_out_rows().get([row]) as usize;
    for (part, values) in gathered.chunks_exact_mut(d).enumerate() {
        values.copy_from_slice(cx.arg_hidden().row([source, part, 0]));
    }
    let weights = cx.arg_norm();
    projection::form_input::<E::A>(
        gathered,
        (cx.dim_nm() != 0).then_some(&weights),
        cx.arg_epsilon(),
        norm,
        features,
    );
    // SAFETY: each work item writes its own row.
    activation::store::<E::A>(features, unsafe { cx.result_0().row_mut([row, 0]) });
}
