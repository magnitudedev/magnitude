// draft_convolve_residual on CPU (contract and portable body in
// dflash.seismic): one work item per row, which sums each column's taps over
// the row and its block predecessors in offset order and adds the residual.

fn draft_convolve_residual<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (channels, taps) = (cx.dim_c() as usize, cx.dim_k() as usize);
    let position = row % cx.arg_block() as usize;
    let (dynamic, base, output) = (cx.arg_dynamic(), cx.arg_base(), cx.arg_output());
    let residual = cx.arg_residual().row([row, 0]);
    // SAFETY: each work item writes its own row.
    let result = unsafe { cx.result_0().row_mut([row, 0]) };
    for (column, value) in result.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for offset in 0..taps.min(position + 1) {
            let coefficient = base.get([1, offset, column]) + dynamic.get([row, 1, offset, column / channels]);
            sum = coefficient.mul_add(output.get([row - offset, column]), sum);
        }
        *value = residual[column] + sum;
    }
}
