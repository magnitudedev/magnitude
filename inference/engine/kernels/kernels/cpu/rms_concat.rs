// rms_concat on CPU (contract and portable body in attn_res.seismic):
// one work item per row, writing `[RMS(a, eps) || RMS(b, eps)]` in A.

use lib::core::reduce;

#[inline(always)]
fn rms_concat<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let h = cx.dim_h() as usize;
    let eps = cx.arg_epsilon();
    let a = cx.arg_a().row([row, 0]);
    let b = cx.arg_b().row([row, 0]);
    let inv_a = reduce::rms_inverse(a, eps);
    let inv_b = reduce::rms_inverse(b, eps);
    // SAFETY: each work item writes its own row.
    let mut out = unsafe { cx.result_0().row_mut([row, 0]) };
    for (i, target) in out.iter_mut().enumerate() {
        if i < h {
            target.set(a[i] * inv_a);
        } else {
            target.set(b[i - h] * inv_b);
        }
    }
}
