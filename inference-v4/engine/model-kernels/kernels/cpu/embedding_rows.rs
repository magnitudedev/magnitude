// embedding_rows on CPU (contract and portable body in target.seismic): one
// work item per token row, which decodes its table row, rounds it to A and
// stores it both as A and, widened again, as F32.

use lib::core::activation;

fn embedding_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    // A failed selection (token -1) embeds token 0.
    let token = cx.arg_tokens().get([row, 0]).max(0) as usize;
    // SAFETY: each work item writes its own row of both results.
    let (embedded, widened) = unsafe { (cx.result_0().row_mut([row, 0]), cx.result_1().row_mut([row, 0])) };
    cx.arg_table().decode_row(token, widened);
    for value in widened.iter_mut() {
        *value = activation::publish::<E::A>(*value);
    }
    activation::store::<E::A>(widened, embedded);
}
