// draft_convolve_residual on CPU (contract and portable body in
// dflash.seismic): one work item per row, which sums each column's taps over
// the row and its block predecessors in offset order and adds the residual.

#[inline(always)]
fn draft_convolve_residual<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let (channels, taps) = (cx.dim_c() as usize, cx.dim_k() as usize);
    let position = row % cx.arg_block() as usize;
    let (dynamic, base, output) = (cx.arg_dynamic(), cx.arg_base(), cx.arg_output());
    let residual = cx.arg_residual().row([row, 0]);
    // SAFETY: each work item writes its own row.
    let mut result = unsafe { cx.result_0().row_mut([row, 0]) };
    result.compute(|sums| {
        if channels == 0 || sums.is_empty() {
            return;
        }
        sums.fill(0.0);
        for offset in 0..taps.min(position + 1) {
            let base_row = base.row([1, offset, 0]);
            let out_row = output.row([row - offset, 0]);
            for (g, ((sum_group, base_group), out_group)) in sums
                .chunks_exact_mut(channels)
                .zip(base_row.chunks_exact(channels))
                .zip(out_row.chunks_exact(channels))
                .enumerate()
            {
                let dyn_val = dynamic.get([row, 1, offset, g]);
                for ((sum, &base_val), &out_val) in sum_group.iter_mut().zip(base_group).zip(out_group) {
                    *sum = (base_val + dyn_val).mul_add(out_val, *sum);
                }
            }
        }
        for (sum, &res) in sums.iter_mut().zip(residual) {
            *sum = res + *sum;
        }
    });
}
