// attention_prefill on CPU (contract and portable body in attention.seismic).
// `prepare`: one work item per row prepares its queries and, for a layer with
// fresh rows, its keys and values (activation values, as F32) into scratch
// and appends its key and value at its destination. `attend`: one work item
// per (row, kv head) absorbs the row's keys (spans in order, then its fresh
// span) for the kv head's G query heads and stores their gated outputs.

use lib::attention::attention::{self, Form, Online, BLOCK};
use lib::core::activation;
use seismic::cpu::slab::SlabTensor;

fn form<E: Elements>(cx: &Context<'_, E>) -> Form {
    Form::new(cx.dim_kv(), cx.dim_g(), cx.dim_p(), cx.dim_s(), cx.dim_i(), cx.dim_u(), cx.dim_f(), cx.dim_n(), cx.dim_nv())
}

fn attention_prefill_prepare<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let form = form(cx);
    let (w, q, kv) = (form.heads.w, form.heads.queries(), form.heads.kv);
    let angles = seismic::cpu::tensor::floats(shared, 2 * form.heads.p);
    let (cosines, sines) = angles.split_at_mut(form.heads.p);
    attention::scaled_angles(
        cx.arg_coordinates().row([row, 0]),
        cx.arg_rotary_components().row([0]),
        cx.arg_rotary_frequencies().row([0]),
        cx.arg_rotary_amplitudes().row([0]),
        cosines,
        sines,
    );
    // SAFETY: each work item writes its own row of the scratch.
    let queries = unsafe { cx.scratch_queries().slice_mut::<f32>(4 * row * q * w, q * w) };
    let keys = unsafe { cx.scratch_keys().slice_mut::<f32>(4 * row * kv * w, kv * w) };
    let values = unsafe { cx.scratch_values().slice_mut::<f32>(4 * row * kv * w, kv * w) };
    let (query_norm, key_norm, value_norm) = (cx.arg_query_norm(), cx.arg_key_norm(), cx.arg_value_norm());
    let norms = (
        form.norm.then(|| query_norm.row([0, 0])),
        form.norm.then(|| key_norm.row([0, 0])),
        form.value_norm.then(|| value_norm.row([0, 0])),
    );
    let (query, key, value) = (cx.arg_query(), cx.arg_key(), cx.arg_value());
    let (key, value) = if form.fresh { (key.row([0, row, 0]), value.row([0, row, 0])) } else { (&[][..], &[][..]) };
    attention::prepare_form_row::<E::A>(
        form,
        |head| query.row([row, head, 0]),
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
    let destination = cx.arg_destinations().get([row]);
    if form.fresh && destination >= 0 {
        let slab_rows = cx.arg_slab_rows() as usize;
        let (history_key, history_value) = (
            SlabTensor::from_bound(cx.arg_history_key(), slab_rows),
            SlabTensor::from_bound(cx.arg_history_value(), slab_rows),
        );
        for head in 0..kv {
            // SAFETY: destinations are distinct, freshly reserved history rows
            // no row of the call reads.
            let (key_row, value_row) = unsafe {
                (history_key.row_mut([destination as usize, head, 0]), history_value.row_mut([destination as usize, head, 0]))
            };
            activation::store::<E::A>(&keys[head * w..][..w], key_row);
            activation::store::<E::A>(&values[head * w..][..w], value_row);
        }
    }
}

fn attention_prefill_attend<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (row, kv_head) = (group[0] as usize, group[1] as usize);
    let form = form(cx);
    let (g, w, kv) = (form.heads.g, form.heads.w, form.heads.kv);
    let (m, r) = (cx.dim_m() as usize, cx.dim_r() as usize);
    let (visible, fresh) = (cx.arg_visible(), cx.arg_fresh());
    let spans = (0..r)
        .map(move |span| (visible.get([row, span, 0]), visible.get([row, span, 1])))
        .filter(|(lo, hi)| hi > lo);
    // A layer without fresh rows sees none (the portable `for layer in 0..F`).
    let fresh = if form.fresh { (fresh.get([row, 0]), fresh.get([row, 1])) } else { (0, 0) };
    let range = 0..attention::total(spans.clone(), fresh);
    let work = seismic::cpu::tensor::floats(shared, 2 * g + g * w + g * BLOCK + w);
    let (maximum, work) = work.split_at_mut(g);
    let (denominator, work) = work.split_at_mut(g);
    let (accumulator, work) = work.split_at_mut(g * w);
    let (scores, key_row) = work.split_at_mut(g * BLOCK);
    let mut state = Online { maximum, denominator, accumulator };
    state.reset();
    // SAFETY: the prepare launch wrote every row before this launch.
    let (queries, keys, values) = unsafe {
        (
            cx.scratch_queries().slice::<f32>(4 * (row * kv + kv_head) * g * w, g * w),
            cx.scratch_keys().slice::<f32>(0, m * kv * w),
            cx.scratch_values().slice::<f32>(0, m * kv * w),
        )
    };
    let slab_rows = cx.arg_slab_rows() as usize;
    let (history_key, history_value) = (
        SlabTensor::from_bound(cx.arg_history_key(), slab_rows),
        SlabTensor::from_bound(cx.arg_history_value(), slab_rows),
    );
    attention::attend_rows(
        &mut state,
        queries,
        cx.arg_scale(),
        scores,
        key_row,
        spans,
        fresh,
        range,
        |token, out| activation::widen::<E::A>(history_key.row([token, kv_head, 0]), out),
        |token, out| activation::widen::<E::A>(history_value.row([token, kv_head, 0]), out),
        |token, out| out.copy_from_slice(&keys[(token * kv + kv_head) * w..][..w]),
        |token, out| out.copy_from_slice(&values[(token * kv + kv_head) * w..][..w]),
    );
    let (query, gate, result) = (cx.arg_query(), cx.arg_gate(), cx.result_0());
    let softplus = cx.arg_gate_function() != 0;
    for local in 0..g {
        let head = kv_head * g + local;
        let gates = if form.interleaved > 0 {
            query.span([row, head, w], form.interleaved)
        } else {
            gate.row([row, head, 0])
        };
        // SAFETY: each work item writes its own query heads of its row.
        let out = unsafe { result.row_mut([row, head, 0]) };
        attention::gate_row::<E::A>(&state.accumulator[local * w..][..w], state.denominator[local], gates, softplus, out);
    }
}
