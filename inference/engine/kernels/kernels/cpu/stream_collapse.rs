// stream_collapse on CPU (contract and portable body in streams.seismic):
// one work item per row, which sums its streams in stream order, by the last
// site's pre (P = 1) or at weight 1 and divided by S (P = 0, the mean).

use lib::core::functions;

#[inline(always)]
fn stream_collapse<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (hidden, s) = (cx.arg_hidden(), cx.dim_s() as usize);
    let stream = |stream| hidden.row([row, stream, 0]);
    // SAFETY: each work item writes its own row.
    let out = unsafe { cx.result_0().row_mut([row, 0]) };
    if cx.dim_p() > 0 {
        let pre = cx.arg_pre().row([row, 0]);
        functions::collapse(s, |index| pre[index], stream, out);
    } else {
        functions::collapse(s, |_| 1.0, stream, out);
        let streams = s as f32;
        for value in out.iter_mut() {
            *value /= streams;
        }
    }
}
