// pool_rows on CPU (contract and portable body in pooled.seismic).
// `append`: one work item per row stores its u (then g) and coordinates in
// the window planes. `complete`: one work item per row publishes its latent,
// block row and origin coordinates (zero, -1 and zeros for a row that
// completes no group); a
// completing row pools its group (the last RATE - 1 rows of its window
// spans, then its own inputs), norms it, rotates it at the coordinates stored
// with the group's first token, rounds it and stores it in the block plane
// (a rounded row held in A after its pooling, its norm and its rotation).

use lib::attention::attention::{self, Layout};
use lib::core::{reduce, rounding::Rounding, spans};
use seismic::cpu::{math, slab::SlabTensor};

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
    let target = unsafe { window.row_mut([destination as usize, 0]) };
    target[..w].copy_from_slice(cx.arg_u().row([row, 0]));
    if gated {
        target[w..].copy_from_slice(cx.arg_g().row([row, 0]));
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

#[inline(always)]
fn pool_rows_complete<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let destination = cx.arg_block_destinations().get([row]);
    // SAFETY: each work item writes its own row of every result.
    let (latent, origin_out) = unsafe { (cx.result_0().row_mut([row, 0]), cx.result_2().row_mut([row, 0])) };
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
    // spans, in order.
    let window_visible = cx.arg_window_visible();
    let earlier = spans::rows(&|span, bound| window_visible.get([row, span, bound]), cx.dim_r() as usize)
        .collect::<Vec<_>>();
    let members = &earlier[earlier.len() - (rate - 1)..];
    let window = SlabTensor::from_bound(cx.arg_window_rows(), slab_rows);
    // Each member's (u, g): the window's rows, then the row's own inputs.
    let (u, g) = (cx.arg_u(), cx.arg_g());
    let rows = members
        .iter()
        .map(|&at| window.row([at as usize, 0]).split_at(w))
        .chain(std::iter::once((u.row([row, 0]), if gated { g.row([row, 0]) } else { &[][..] })))
        .collect::<Vec<_>>();
    // The group's first token's (RATE >= 2: a window member). SAFETY: only
    // read.
    let origin = &*unsafe { coordinate_row(cx.arg_window_coordinates(), slab_rows, members[0] as usize) };
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
    for (column, pooled) in latent.iter_mut().enumerate() {
        *pooled = if gated {
            let top = rows.iter().fold(f32::NEG_INFINITY, |top, (_, g)| top.max(g[column]));
            let total = rows.iter().fold(0.0, |total, (_, g)| total + math::exp(g[column] - top));
            rows.iter().fold(0.0, |sum, (u, g)| sum + math::exp(g[column] - top) / total * u[column])
        } else {
            rows.iter().fold(0.0, |sum, (u, _)| sum + u[column]) / rate as f32
        };
    }
    attention::held(hold, latent);
    let inverse = reduce::rms_inverse(latent, cx.arg_epsilon());
    for (value, weight) in latent.iter_mut().zip(cx.arg_norm().row([0, 0])) {
        *value = *value * inverse * weight;
    }
    attention::held(hold, latent);
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
    let blocks = SlabTensor::from_bound(cx.arg_block_rows(), cx.arg_block_slab_rows() as usize);
    // SAFETY: block destinations are distinct freshly reserved rows no work
    // item of this launch reads.
    lib::core::activation::store::<E::A>(finished, unsafe { blocks.row_mut([destination as usize, 0]) });
    // SAFETY: each work item writes its own row.
    unsafe { cx.result_1().set([row], destination) };
}
