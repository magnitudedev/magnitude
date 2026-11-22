// pool_rows on CPU (contract and portable body in pooled.seismic).
// `append`: one work item per row stores its u (normed per token when TN =
// 1; then g) and coordinates in the window planes. `complete`: one work item
// per row publishes its latent, block row and origin coordinates (zero, -1
// and zeros for a row that completes no group); a completing row pools its
// group (the last RATE - 1 rows of its window spans, then its own inputs,
// its gates plus their place terms), norms the pooled row (TN = 0), rotates
// it at the coordinates stored with the group's first token, rounds it and
// stores it in the block plane in the row element Y (a rounded row held in A
// after its pooling, its norm and its rotation).

use lib::attention::attention::{self, Layout};
use lib::core::{norm, reduce, rounding::Rounding, spans};
use seismic::cpu::{math, slab::SlabTensor};

const MAX_STACK_RATE: usize = 64;
const MAX_STACK_WIDTH: usize = 512;

#[inline(always)]
fn pool_rows_append<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let destination = cx.arg_window_destinations().get([row]);
    if destination < 0 {
        return;
    }
    let (w, gated) = ((2 * cx.dim_p() + cx.dim_s()) as usize, cx.dim_gt() == 1);
    let slab_rows = cx.arg_window_slab_rows() as usize;
    let window = SlabTensor::from_bound(cx.arg_window_rows(), slab_rows);
    // SAFETY: window destinations are distinct freshly reserved rows no
    // work item of this launch reads.
    let mut target = unsafe { window.row_mut([destination as usize, 0]) };
    target.part_mut(0, w).compute(|u| {
        token_norm(cx, cx.arg_u().row([row, 0]), u);
    });
    if gated {
        let rest = target.len() - w;
        target.part_mut(w, rest).move_stored(cx.arg_g().stored([row, 0]));
    }
    // SAFETY: as above.
    let target = unsafe { coordinate_row(cx.arg_window_coordinates(), slab_rows, destination as usize) };
    target.copy_from_slice(cx.arg_coordinates().row([row, 0]));
}

/// Row `at` of the window coordinates plane (`[TW, 4]` i32 slabs of
/// `slab_rows` rows).
///
/// # Safety
/// The caller gives each work item distinct writes and no concurrent readers.
#[allow(clippy::mut_from_ref)]
#[inline(always)]
unsafe fn coordinate_row<'a>(plane: seismic::cpu::Scalars<'a, i32, 2>, slab_rows: usize, at: usize) -> &'a mut [i32] {
    // SAFETY: native launch validation owns every named slab and row; a row
    // is four contiguous words.
    unsafe {
        let region = seismic::cpu::slab::region::<i32>(plane.pointer().cast(), at / slab_rows);
        std::slice::from_raw_parts_mut(region.add((at % slab_rows) * 4), 4)
    }
}

/// `PoolNorm::Token` (TN = 1): `norm::row`'s layer norm of `src` into `dst`
/// with `norm` and `bias` (the variance biased); TN = 0 copies `src` into `dst`.
#[inline(always)]
fn token_norm<E: Elements>(cx: &Context<'_, E>, src: &[f32], dst: &mut [f32]) {
    if cx.dim_tn() == 0 {
        dst.copy_from_slice(src);
        return;
    }
    norm::row(src, true, Some(cx.arg_norm().row([0, 0])), Some(cx.arg_bias().row([0, 0])), cx.arg_epsilon(), dst);
}

#[inline(always)]
fn pool_rows_complete<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let destination = cx.arg_block_destinations().get([row]);
    // SAFETY: each work item writes its own row of every result.
    let (mut latent, origin_out) = unsafe { (cx.result_0().row_mut([row, 0]), cx.result_2().row_mut([row, 0])) };
    if destination < 0 {
        latent.fill(0.0);
        origin_out.fill(0);
        // SAFETY: as above.
        unsafe { cx.result_1().set([row], -1) };
        return;
    }
    let (p, w) = (cx.dim_p() as usize, (2 * cx.dim_p() + cx.dim_s()) as usize);
    let (rate, gated) = (cx.dim_rate() as usize, cx.dim_gt() == 1);
    let storage = seismic::cpu::tensor::floats(shared, 2 * p + w);
    let (angles, finished) = storage.split_at_mut(2 * p);
    let (cosines, sines) = angles.split_at_mut(p);
    let slab_rows = cx.arg_window_slab_rows() as usize;
    // The group's earlier tokens: the last RATE - 1 rows of the window
    // spans, in order. Walk spans in reverse so no heap collection is needed.
    let window_visible = cx.arg_window_visible();
    let need = rate - 1;
    let mut members_stack = [0usize; MAX_STACK_RATE];
    let mut members_heap = Vec::new();
    let members: &mut [usize] = if need <= MAX_STACK_RATE {
        &mut members_stack[..need]
    } else {
        members_heap.resize(need, 0);
        &mut members_heap[..]
    };
    let mut remaining = need;
    for span in (0..cx.dim_r() as usize).rev() {
        let range = spans::span(window_visible.get([row, span, 0]), window_visible.get([row, span, 1]));
        let len = (range.end - range.start) as usize;
        if len == 0 {
            continue;
        }
        let take = len.min(remaining);
        let start = range.end - take as i32;
        remaining -= take;
        for (slot, window_row) in members[remaining..remaining + take].iter_mut().zip(start..range.end) {
            *slot = window_row as usize;
        }
        if remaining == 0 {
            break;
        }
    }
    assert_eq!(remaining, 0, "pool_rows_complete: window spans held fewer than RATE - 1 rows");
    let window = SlabTensor::from_bound(cx.arg_window_rows(), slab_rows);
    // Each member's (u, g): the window's rows, then the row's own inputs.
    let (u, g) = (cx.arg_u(), cx.arg_g());
    let mut own_stack = [0.0f32; MAX_STACK_WIDTH];
    let mut own_heap = Vec::new();
    let own_u: &[f32] = if cx.dim_tn() == 0 {
        u.row([row, 0])
    } else {
        let dst = if w <= MAX_STACK_WIDTH {
            &mut own_stack[..w]
        } else {
            own_heap.resize(w, 0.0);
            &mut own_heap[..w]
        };
        token_norm(cx, u.row([row, 0]), dst);
        dst
    };
    let mut rows_stack: [(&[f32], &[f32]); MAX_STACK_RATE] = [(&[], &[]); MAX_STACK_RATE];
    let mut rows_heap: Vec<(&[f32], &[f32])> = Vec::new();
    let rows: &mut [(&[f32], &[f32])] = if rate <= MAX_STACK_RATE {
        &mut rows_stack[..rate]
    } else {
        rows_heap.resize(rate, (&[], &[]));
        &mut rows_heap[..]
    };
    for (slot, &at) in rows[..need].iter_mut().zip(members.iter()) {
        *slot = window.row([at, 0]).split_at(w);
    }
    rows[need] = (own_u, if gated { g.row([row, 0]) } else { &[] });
    // The group's first token's (RATE >= 2: a window member). SAFETY: only
    // read.
    let origin = &*unsafe { coordinate_row(cx.arg_window_coordinates(), slab_rows, members[0]) };
    origin_out.copy_from_slice(origin);
    attention::scaled_angles(
        origin,
        cx.arg_rotary_components().row([0]),
        cx.arg_rotary_frequencies().row([0]),
        cx.arg_rotary_amplitudes().row([0]),
        cosines,
        sines,
    );
    // A rounded row is rounded from values held in A (`pooled.seismic`).
    let rounding = Rounding::of(cx.arg_block(), cx.arg_element(), cx.arg_scale(), cx.arg_amax_floor());
    let hold = rounding.map(|_| lib::core::activation::publish::<E::A>);
    latent.compute(|latent| {
        if gated {
            // Each member's gate scores plus its place's term (PR = RATE; none at 0).
            let places = cx.dim_pr() as usize;
            let position = cx.arg_position();
            let mut weights_stack = [0.0f32; MAX_STACK_RATE];
            let mut weights_heap = Vec::new();
            let weights: &mut [f32] = if rate <= MAX_STACK_RATE {
                &mut weights_stack[..rate]
            } else {
                weights_heap.resize(rate, 0.0);
                &mut weights_heap[..]
            };
            for (column, pooled) in latent.iter_mut().enumerate() {
                let mut top = f32::NEG_INFINITY;
                for (member, (weight, (_, g_row))) in weights.iter_mut().zip(rows.iter()).enumerate() {
                    let term = if member < places { position.row([member, 0])[column] } else { 0.0 };
                    let score = g_row[column] + term;
                    *weight = score;
                    top = top.max(score);
                }
                let mut total = 0.0f32;
                for weight in weights.iter_mut() {
                    let e = math::exp(*weight - top);
                    *weight = e;
                    total += e;
                }
                let sum = weights
                    .iter()
                    .zip(rows.iter())
                    .fold(0.0f32, |sum, (&weight, (u_row, _))| sum + weight * u_row[column]);
                *pooled = sum / total;
            }
        } else {
            for (column, pooled) in latent.iter_mut().enumerate() {
                *pooled = rows.iter().fold(0.0f32, |sum, (u_row, _)| sum + u_row[column]) / rate as f32;
            }
        }
        attention::held(hold, latent);
        // The pooled row's norm (TN = 0); a token-normed row is its latent.
        if cx.dim_tn() == 0 {
            let inverse = reduce::rms_inverse(latent, cx.arg_epsilon());
            for (value, weight) in latent.iter_mut().zip(cx.arg_norm().row([0, 0])) {
                *value = *value * inverse * weight;
            }
            attention::held(hold, latent);
        }
        attention::prepare_rounded_head::<seismic::cpu::F32>(
            latent,
            None,
            Layout::new(cx.dim_ra(), cx.dim_rt()),
            cosines,
            sines,
            cx.arg_epsilon(),
            rounding,
            hold,
            finished,
        );
    });
    let blocks = SlabTensor::from_bound(cx.arg_block_rows(), cx.arg_block_slab_rows() as usize);
    // SAFETY: block destinations are distinct freshly reserved rows no work
    // item of this launch reads.
    lib::core::activation::store::<E::Y>(finished, unsafe { blocks.row_mut([destination as usize, 0]) });
    // SAFETY: each work item writes its own row.
    unsafe { cx.result_1().set([row], destination) };
}
