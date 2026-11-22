// vision_gated_convolution on CPU (contract and portable body in
// vision.seismic): one work item per row. Active taps are hoisted once from
// the row's neighbour list; channels are tiled in L1-resident blocks so
// weights decode sequentially into a contiguous tap-major tile, active
// neighbour slices accumulate with SIMD FMA in F32, and the gated row is
// bulk-stored to A.

use lib::core::activation;

#[inline(always)]
fn vision_gated_convolution<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    const BLOCK: usize = 64;
    const STACK_TAPS: usize = 64;

    let row = group[0] as usize;
    let (h, t) = (cx.dim_h() as usize, cx.dim_t() as usize);
    let (value, weight, neighbors) = (cx.arg_value(), cx.arg_weight(), cx.arg_neighbors());
    let shared = seismic::cpu::tensor::floats(shared, (t + 1) * h + t);
    let (w_block_storage, rest) = shared.split_at_mut(t * h);
    let (acc, taps) = rest.split_at_mut(h);
    acc.fill(0.0);

    let neighbor_row = neighbors.row([row, 0]);
    let mut active_buf = [(0usize, 0usize); STACK_TAPS];
    let mut src_buf = [0.0f32; BLOCK];

    for (batch_idx, batch_neighbors) in neighbor_row.chunks(STACK_TAPS).enumerate() {
        let tap_base = batch_idx * STACK_TAPS;
        let mut active_len = 0usize;
        for (offset, &neighbor) in batch_neighbors.iter().enumerate() {
            if neighbor >= 0 {
                active_buf[active_len] = (tap_base + offset, neighbor as usize);
                active_len += 1;
            }
        }
        if active_len == 0 {
            continue;
        }
        let active = &active_buf[..active_len];

        for (block_idx, acc_chunk) in acc.chunks_mut(BLOCK).enumerate() {
            let c0 = block_idx * BLOCK;
            let len = acc_chunk.len();
            let w_block = &mut w_block_storage[..active_len * len];

            for local_c in 0..len {
                weight.decode_row(c0 + local_c, taps);
                for (slot, &(tap, _)) in active.iter().enumerate() {
                    w_block[slot * len + local_c] = taps[tap];
                }
            }

            let src_slice = &mut src_buf[..len];
            for (slot, &(_, source_row)) in active.iter().enumerate() {
                activation::widen::<E::A>(&value.row([source_row, 0])[c0..c0 + len], src_slice);
                let w_row = &w_block[slot * len..][..len];
                for ((a, &s), &w) in acc_chunk.iter_mut().zip(src_slice.iter()).zip(w_row) {
                    *a = s.mul_add(w, *a);
                }
            }
        }
    }

    let gate_row = cx.arg_gate().row([row, 0]);
    for (block_idx, acc_chunk) in acc.chunks_mut(BLOCK).enumerate() {
        let c0 = block_idx * BLOCK;
        let len = acc_chunk.len();
        let gate_slice = &mut src_buf[..len];
        activation::widen::<E::A>(&gate_row[c0..c0 + len], gate_slice);
        for (a, &g) in acc_chunk.iter_mut().zip(gate_slice.iter()) {
            *a = activation::publish::<E::A>(*a) * g;
        }
    }

    // SAFETY: each work item writes its own row.
    activation::store::<E::A>(acc, unsafe { cx.result_0().row_mut([row, 0]) });
}
