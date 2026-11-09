// Ordered cache publication before causal target attention.
use lib::attention::attention::{self, Form, Layout};
use seismic::cpu::slab::SlabTensor;

fn attention_append_dense<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    // The keys' and values' form: no queries, so no query norm or output
    // rotary.
    let layout = Layout::new(cx.dim_ra(), cx.dim_rt());
    let form = Form::new(cx.dim_kv(), 0, cx.dim_p(), cx.dim_s(), 0, 0, cx.dim_f(), 0, cx.dim_n(), cx.dim_nv(), layout, 0, cx.dim_vk())
        .with_key_rounding(cx.dim_kr(), cx.arg_key_block(), cx.arg_key_element(), cx.arg_key_scale(), cx.arg_key_amax_floor());
    let destination = cx.arg_destinations().get([row]);
    if !form.fresh || destination < 0 { return; }
    let (w, kv) = (form.heads.w, form.heads.kv);
    let storage = seismic::cpu::tensor::floats(shared, 2 * form.heads.p + 2 * kv * w);
    let (angles, vectors) = storage.split_at_mut(2 * form.heads.p);
    let (cosines, sines) = angles.split_at_mut(form.heads.p);
    attention::scaled_angles(
        cx.arg_coordinates().row([row, 0]), cx.arg_rotary_components().row([0]),
        cx.arg_rotary_frequencies().row([0]), cx.arg_rotary_amplitudes().row([0]), cosines, sines);
    let (keys, values) = vectors.split_at_mut(kv * w);
    let queries = &mut [];
    let (key_norm, value_norm) = (cx.arg_key_norm(), cx.arg_value_norm());
    let norms = (
        None,
        form.key_norm.then(|| key_norm.row([0, 0])),
        form.value_norm.then(|| value_norm.row([0, 0])),
    );
    let (key, value) = (cx.arg_key(), cx.arg_value());
    let value = if form.own_values() { value.row([0, row, 0]) } else { &[][..] };
    attention::prepare_form_row::<E::A>(
        form,
        |_| &[],
        key.row([0, row, 0]),
        value,
        norms,
        cosines,
        sines,
        cx.arg_epsilon(),
        queries,
        keys,
        values,
    );
    let slab_rows = cx.arg_slab_rows() as usize;
    // SAFETY: every destination is a distinct freshly reserved row.
    unsafe {
        attention::store_dense_row::<E::A>(
            form,
            keys,
            values,
            &SlabTensor::from_bound(cx.arg_history_key(), slab_rows),
            &SlabTensor::from_bound(cx.arg_history_value(), slab_rows),
            destination as usize,
        );
    }
}
