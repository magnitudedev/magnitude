// draft_gated_rows on CPU (contract and portable body in dflash.seismic): one
// work item per row, silu(gate) * up rounded once to A.

use lib::core::activation;

fn draft_gated_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let width = cx.dim_f() as usize;
    let (gate, up) = seismic::cpu::tensor::floats(shared, 2 * width).split_at_mut(width);
    activation::widen::<E::A>(cx.arg_gate().row([row, 0]), gate);
    activation::widen::<E::A>(cx.arg_up().row([row, 0]), up);
    for (value, up) in gate.iter_mut().zip(up.iter()) {
        *value = activation::silu(*value) * up;
    }
    // SAFETY: each work item writes its own row.
    activation::store::<E::A>(gate, unsafe { cx.result_0().row_mut([row, 0]) });
}
