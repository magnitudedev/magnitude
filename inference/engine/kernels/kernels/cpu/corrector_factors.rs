// corrector_factors on CPU (contract and portable body in corrector.seismic):
// one work item per node, which joins its row x and its block's anchor a as
// [x ; a ; x · a], each rounded once to A.

use lib::core::activation;

fn corrector_factors<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (nodes, width) = (cx.dim_n() as usize, cx.dim_h() as usize);
    let (block, node) = (row / nodes, row % nodes);
    let (x, anchor) = seismic::cpu::tensor::floats(shared, 2 * width).split_at_mut(width);
    activation::widen::<E::A>(cx.arg_nodes().row([block, node, 0]), x);
    activation::widen::<E::A>(cx.arg_anchor().row([block, 0]), anchor);
    let result = cx.result_0();
    // SAFETY: each work item writes its own node's three rows.
    unsafe {
        activation::store::<E::A>(x, result.row_mut([block, node, 0, 0]));
        activation::store::<E::A>(anchor, result.row_mut([block, node, 1, 0]));
    }
    for (value, anchor) in x.iter_mut().zip(anchor.iter()) {
        *value *= anchor;
    }
    // SAFETY: as above.
    activation::store::<E::A>(x, unsafe { result.row_mut([block, node, 2, 0]) });
}
