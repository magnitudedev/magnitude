// stream_join on CPU (contract and portable body in streams.seismic): one
// work item per row, which joins its output into each stream through the
// projection epilogues' `scatter`.

use lib::core::functions;

fn stream_join<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (h, streams, inject_width) = (cx.dim_h() as usize, cx.dim_sn() as usize, cx.dim_in() as usize);
    let (residual, y, inject) = (cx.arg_residual(), cx.arg_y().row([row, 0]), cx.arg_inject());
    // SAFETY: each work item writes its own row.
    let out = unsafe { cx.result_0().row_mut([row, 0]) };
    for (stream, joined) in out.chunks_exact_mut(h).enumerate() {
        for (column, target) in joined.iter_mut().enumerate() {
            *target = functions::scatter(
                streams,
                inject_width,
                stream,
                y[column],
                |at| inject.get([row, at]),
                |from| residual.get([row, from * h + column]),
            );
        }
    }
}
