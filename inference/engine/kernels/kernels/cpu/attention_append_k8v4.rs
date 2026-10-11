// Ordered cache publication before causal target attention.
use lib::attention::attention::{self, Form, KEY_BITS, VALUE_BITS};
use seismic::cpu::slab::SlabTensor;

fn attention_append_k8v4<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let form = Form::new(cx.dim_kv(), 0, cx.dim_p(), cx.dim_s(), 0, 0, cx.dim_f(), cx.dim_n(), cx.dim_nv()).affine();
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
        form.norm.then(|| key_norm.row([0, 0])),
        form.value_norm.then(|| value_norm.row([0, 0])),
    );
    let (key, value) = (cx.arg_key(), cx.arg_value());
    let (key, value) = if form.fresh { (key.row([0, row, 0]), value.row([0, row, 0])) } else { (&[][..], &[][..]) };
    attention::prepare_form_row::<E::A>(
        form,
        |_| &[],
        key,
        value,
        norms,
        cosines,
        sines,
        cx.arg_epsilon(),
        queries,
        keys,
        values,
    );
    if form.fresh && destination >= 0 {
        let destination = destination as usize;
        let slab_rows = cx.arg_slab_rows() as usize;
        let (key_codes, key_coefficients) = (
            SlabTensor::from_scalars(cx.arg_history_key_codes(), slab_rows),
            SlabTensor::from_bound(cx.arg_history_key_coefficients(), slab_rows),
        );
        let (value_codes, value_coefficients) = (
            SlabTensor::from_scalars(cx.arg_history_value_codes(), slab_rows),
            SlabTensor::from_bound(cx.arg_history_value_coefficients(), slab_rows),
        );
        for head in 0..kv {
            // SAFETY: destinations are distinct, freshly reserved history rows
            // the subsequent attention phase reads after this launch completes.
            unsafe {
                attention::affine_encode(
                    &keys[head * w..][..w],
                    KEY_BITS,
                    key_codes.row_mut([destination, head, 0]),
                    key_coefficients.row_mut([destination, head, 0]),
                );
                attention::affine_encode(
                    &values[head * w..][..w],
                    VALUE_BITS,
                    value_codes.row_mut([destination, head, 0]),
                    value_coefficients.row_mut([destination, head, 0]),
                );
            }
        }
    }
}
