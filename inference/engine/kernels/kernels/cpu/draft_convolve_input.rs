// draft_convolve_input on CPU (contract and portable body in dflash.seismic):
// one work item per row, which widens the row and its block predecessors to
// F32 and sums each column's taps in offset order.

use lib::core::activation;

fn draft_convolve_input<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (groups, channels, taps) = (cx.dim_g() as usize, cx.dim_c() as usize, cx.dim_k() as usize);
    let width = groups * channels;
    let (sums, source) = seismic::cpu::tensor::floats(shared, 2 * width).split_at_mut(width);
    sums.fill(0.0);
    let position = row % cx.arg_block() as usize;
    let (dynamic, base) = (cx.arg_dynamic(), cx.arg_base());
    for offset in 0..taps.min(position + 1) {
        activation::widen::<E::A>(cx.arg_input().row([row - offset, 0]), source);
        for (column, sum) in sums.iter_mut().enumerate() {
            let coefficient = base.get([0, offset, column]) + dynamic.get([row, 0, offset, column / channels]);
            *sum = coefficient.mul_add(source[column], *sum);
        }
    }
    // SAFETY: each work item writes its own row.
    activation::store::<E::A>(sums, unsafe { cx.result_0().row_mut([row, 0]) });
}
