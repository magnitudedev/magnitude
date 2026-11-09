// indexer_keys on CPU (contract and portable body in indexer.seismic): one
// work item per row that completes a unit widens its projected key, norms it
// with the key norm, rotates it at its unit's first coordinates (in the
// attention's layout), rounds it and stores it at its key-plane row (F32).

use lib::attention::attention::{self, Layout};
use lib::core::rounding::Rounding;
use seismic::cpu::slab::SlabTensor;

fn indexer_keys<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let destination = cx.arg_latent_rows().get([row]);
    if destination < 0 {
        return;
    }
    let (p, w) = (cx.dim_p() as usize, (2 * cx.dim_p() + cx.dim_s()) as usize);
    let storage = seismic::cpu::tensor::floats(shared, 2 * p + 2 * w);
    let (angles, vectors) = storage.split_at_mut(2 * p);
    let (cosines, sines) = angles.split_at_mut(p);
    let (raw, prepared) = vectors.split_at_mut(w);
    attention::scaled_angles(
        cx.arg_coordinates().row([row, 0]),
        cx.arg_rotary_components().row([0]),
        cx.arg_rotary_frequencies().row([0]),
        cx.arg_rotary_amplitudes().row([0]),
        cosines,
        sines,
    );
    cx.arg_key().decode_row(row, raw);
    attention::prepare_head::<seismic::cpu::F32>(
        raw,
        Some(cx.arg_key_norm().row([0, 0])),
        Layout::new(cx.dim_ra(), cx.dim_rt()),
        cosines,
        sines,
        cx.arg_epsilon(),
        prepared,
    );
    if let Some(rounding) = Rounding::of(cx.arg_block(), cx.arg_element(), cx.arg_scale(), cx.arg_amax_floor()) {
        rounding.apply(prepared);
    }
    let keys = SlabTensor::from_bound(cx.arg_keys(), cx.arg_slab_rows() as usize);
    // SAFETY: key-plane rows are distinct freshly completed rows no work
    // item of this launch reads.
    unsafe { keys.row_mut([destination as usize, 0]) }.copy_from_slice(prepared);
}
