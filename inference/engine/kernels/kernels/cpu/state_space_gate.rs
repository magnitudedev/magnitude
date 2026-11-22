// state_space_gate on CPU (contract and portable body in state_space.seismic):
// the state-space output rows, y * SiLU(z) RMS-normalized over each group of
// U heads and scaled by the state norm. A work item owns (row, group),
// computes the group's gated values once into a scratch buffer, reduces their
// squares via defined-order SIMD sum_squares, and writes each head's
// normalized row.

use lib::core::{activation, reduce};

const STACK_CAP: usize = 2048;

#[inline(always)]
fn state_space_gate<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (row, g) = (group[0] as usize, group[1] as usize);
    let (heads, width) = (cx.dim_u() as usize, cx.dim_p() as usize);
    let total = heads * width;
    if total == 0 {
        return;
    }
    let (mixed, projection, norm) = (cx.arg_mixed(), cx.arg_projection(), cx.arg_state_norm());
    let normalized = cx.result_0();

    let mut stack = [0.0f32; STACK_CAP];
    let mut heap = Vec::new();
    let gated = if total <= STACK_CAP {
        &mut stack[..total]
    } else {
        heap.resize(total, 0.0);
        &mut heap[..]
    };

    let proj_group = projection.span([row, g * total], total);
    for local in 0..heads {
        let head = g * heads + local;
        let offset = local * width;
        let mixed_row = mixed.row([row, head, 0]);
        let gate_row = &proj_group[offset..offset + width];
        let target = &mut gated[offset..offset + width];
        for ((dst, &m), &z) in target.iter_mut().zip(mixed_row).zip(gate_row) {
            *dst = E::A::widen(m) * activation::silu(E::A::widen(z));
        }
    }

    let inverse = reduce::rms_inverse(gated, cx.arg_epsilon());

    for local in 0..heads {
        let offset = local * width;
        let norm_row = norm.row([g, local, 0]);
        let slice = &gated[offset..offset + width];
        // SAFETY: each work item writes its own group's heads of its row.
        let mut out = unsafe { normalized.row_mut([row, g * heads + local, 0]) };
        for ((dst, &value), &scale) in out.iter_mut().zip(slice).zip(norm_row) {
            dst.set(value * inverse * scale);
        }
    }
}
