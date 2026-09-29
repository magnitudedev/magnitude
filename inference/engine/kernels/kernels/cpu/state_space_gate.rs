// state_space_gate on CPU (contract and portable body in state_space.seismic):
// the state-space output rows, y * SiLU(z) RMS-normalized over each group of
// U heads and scaled by the state norm. A work item owns (row, group) and
// sums the group's squared gated values in the body's order.

use lib::core::activation;
use seismic::cpu::math;

fn state_space_gate<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (row, g) = (group[0] as usize, group[1] as usize);
    let (heads, width) = (cx.dim_u() as usize, cx.dim_p() as usize);
    let (mixed, projection, norm) = (cx.arg_mixed(), cx.arg_projection(), cx.arg_state_norm());
    let normalized = cx.result_0();
    let gated = |local: usize, channel: usize| {
        let head = g * heads + local;
        let gate = projection.get([row, head * width + channel]);
        mixed.get([row, head, channel]) * activation::silu(gate)
    };
    let mut squares = 0.0f32;
    for local in 0..heads {
        for channel in 0..width {
            let value = gated(local, channel);
            squares = value.mul_add(value, squares);
        }
    }
    let inverse = math::rsqrt(squares / (heads * width) as f32 + cx.arg_epsilon());
    for local in 0..heads {
        // SAFETY: each work item writes its own group's heads of its row.
        let out = unsafe { normalized.row_mut([row, g * heads + local, 0]) };
        for (channel, out) in out.iter_mut().enumerate() {
            *out = E::A::narrow(gated(local, channel) * inverse * norm.get([g, local, channel]));
        }
    }
}
