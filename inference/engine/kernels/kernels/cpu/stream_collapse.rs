// stream_collapse on CPU (contract and portable body in streams.seismic):
// one work item per row, which sums its streams by its pre in stream order.

use lib::core::functions;

#[inline(always)]
fn stream_collapse<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (hidden, pre) = (cx.arg_hidden(), cx.arg_pre().row([row, 0]));
    // SAFETY: each work item writes its own row.
    let out = unsafe { cx.result_0().row_mut([row, 0]) };
    functions::collapse(pre, |stream| hidden.row([row, stream, 0]), out);
}
