// embedding_rows on CPU (contract and portable body in target.seismic): one
// work item per token row, which decodes its table row, multiplies it by
// `scale`, with `normalize` applies its weightless RMS, rounds it to A and
// stores it both as A and, widened again, as F32.

use lib::core::{activation, reduce};

fn embedding_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    // A failed selection (token -1) embeds token 0.
    let token = cx.arg_tokens().get([row, 0]).max(0) as usize;
    // SAFETY: each work item writes its own row of both results.
    let (embedded, widened) = unsafe { (cx.result_0().row_mut([row, 0]), cx.result_1().row_mut([row, 0])) };
    cx.arg_table().decode_row(token, widened);
    let scale = cx.arg_scale();
    for value in widened.iter_mut() {
        *value *= scale;
    }
    let inverse = (cx.arg_normalize() != 0).then(|| reduce::rms_inverse(widened, cx.arg_epsilon()));
    for value in widened.iter_mut() {
        let decoded = *value;
        *value = activation::publish::<E::A>(inverse.map_or(decoded, |inverse| decoded * inverse));
    }
    activation::store::<E::A>(widened, embedded);
}
