// attention_prefill_k8v4 on CPU (contract and portable body in
// attention.seismic): `attention_prefill`'s launches over affine K8/V4
// history. `prepare` appends the row's prepared key and value encoded, then
// leaves them in the fresh rows' scratch as history will hold them (encoded
// and decoded back), so a row's result does not depend on how many rows share
// the launch; `attend`, per (block of rows, kv head) as the dense entry's,
// decodes each history key and value (code * scale + zero, F32) as it
// absorbs them. Head widths are whole affine groups
// (the declaration's `where`), so every vector has at least one group.

use lib::attention::attention::{self, Form, Layout, Online, BLOCK};
use lib::core::spans;
use seismic::cpu::slab::SlabTensor;

#[inline(always)]
fn form<E: Elements>(cx: &Context<'_, E>) -> Form {
    Form::new(
        cx.dim_kv(),
        cx.dim_g(),
        cx.dim_p(),
        cx.dim_s(),
        cx.dim_i(),
        cx.dim_u(),
        cx.dim_f(),
        cx.dim_nq(),
        cx.dim_n(),
        cx.dim_nv(),
        Layout::new(cx.dim_ra(), cx.dim_rt()),
        cx.dim_ro(),
        cx.dim_vk(),
    )
    .affine()
}

#[inline(always)]
fn attention_prefill_k8v4_prepare<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let form = form(cx);
    let (g, w, kv, m) = (form.heads.g, form.heads.w, form.heads.kv, cx.dim_m() as usize);
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
    let (query_norm, key_norm, value_norm) = (cx.arg_query_norm(), cx.arg_key_norm(), cx.arg_value_norm());
    let (query, query_norm) = (cx.arg_query(), form.query_norm.then(|| query_norm.row([0, 0])));
    for kv_head in 0..kv {
        // SAFETY: each work item writes its own row of each kv head's queries.
        let queries = unsafe { cx.scratch_queries().slice_mut::<f32>(4 * (kv_head * m + row) * g * w, g * w) };
        attention::prepare_queries::<E::A>(
            form,
            kv_head * g,
            g,
            |head| query.row([row, head, 0]),
            query_norm,
            cosines,
            sines,
            cx.arg_epsilon(),
            queries,
        );
    }
    if !form.fresh {
        return;
    }
    // SAFETY: each work item writes its own row of the scratch.
    let keys = unsafe { cx.scratch_keys().slice_mut::<f32>(4 * row * kv * w, kv * w) };
    let values = unsafe { cx.scratch_values().slice_mut::<f32>(4 * row * kv * w, kv * w) };
    let norms = (form.key_norm.then(|| key_norm.row([0, 0])), form.value_norm.then(|| value_norm.row([0, 0])));
    let (key, value) = (cx.arg_key(), cx.arg_value());
    attention::prepare_form_row::<E::A>(
        form,
        key.row([0, row, 0]),
        value.row([0, row, 0]),
        norms,
        cosines,
        sines,
        cx.arg_epsilon(),
        keys,
        values,
    );
    let bits = |words: u64| (32 * words / (2 * cx.dim_p() + cx.dim_s())) as u32;
    let (key_bits, value_bits) = (bits(cx.dim_kc()), bits(cx.dim_vc()));
    let destination = cx.arg_destinations().get([row]);
    if destination >= 0 {
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
            // no row of the call reads.
            unsafe {
                attention::store_affine_head(
                    &keys[head * w..][..w],
                    &values[head * w..][..w],
                    (key_bits, value_bits),
                    (&key_codes, &key_coefficients),
                    (&value_codes, &value_coefficients),
                    destination,
                    head,
                );
            }
        }
    }
    // The row's keys and values as the history row it becomes reads: each
    // encoded as its append encodes it, decoded back.
    for head in 0..kv {
        attention::affine_round_trip(&mut keys[head * w..][..w], key_bits, cx.dim_kp() as usize);
        attention::affine_round_trip(&mut values[head * w..][..w], value_bits, cx.dim_vp() as usize);
    }
}

#[inline(always)]
fn attention_prefill_k8v4_attend<L: Isa, E: Elements>(l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (block, kv_head) = (group[0] as usize, group[1] as usize);
    let form = form(cx);
    let (g, w, kv) = (form.heads.g, form.heads.w, form.heads.kv);
    let (m, r) = (cx.dim_m() as usize, cx.dim_r() as usize);
    let size = m.div_ceil(cx.groups()[0] as usize);
    let first = block * size;
    assert!(first < m, "row block {block} of {} past the {m} rows", cx.groups()[0]);
    let count = size.min(m - first);
    let (visible, fresh, selection) = (cx.arg_visible(), cx.arg_fresh(), cx.arg_selection());
    // Row `local` of the block's key sequence: its spans, its fresh span (none
    // for a layer without fresh rows, the portable `for layer in 0..F`) and
    // its WorkList bitmap (none in the Visible form).
    let sequence = |local: usize| {
        let row = first + local;
        let spans = (0..r)
            .map(move |span| spans::span(visible.get([row, span, 0]), visible.get([row, span, 1])))
            .filter(|rows| !rows.is_empty())
            .map(|rows| (rows.start, rows.end));
        let fresh = if form.fresh { (fresh.get([row, 0]), fresh.get([row, 1])) } else { (0, 0) };
        let bitmap: &[u32] = if cx.dim_b() == 0 { &[] } else { selection.row([row, 0]) };
        (spans, fresh, bitmap)
    };
    let turning = 2 * form.heads.p * usize::from(form.output_rotary);
    let heads = count * g;
    let work = seismic::cpu::tensor::floats(shared, heads * (2 + w + BLOCK) + attention::work(w, true) + turning);
    let (maximum, work) = work.split_at_mut(heads);
    let (denominator, work) = work.split_at_mut(heads);
    let (accumulator, work) = work.split_at_mut(heads * w);
    let (scores, work) = work.split_at_mut(heads * BLOCK);
    let (work, angles) = work.split_at_mut(attention::work(w, true));
    let mut state = Online { maximum, denominator, accumulator };
    state.reset();
    // SAFETY: the prepare launch wrote every row before this launch.
    let (queries, keys, values) = unsafe {
        (
            cx.scratch_queries().slice::<f32>(4 * (kv_head * m + first) * g * w, heads * w),
            cx.scratch_keys().slice::<f32>(0, m * kv * w),
            cx.scratch_values().slice::<f32>(0, m * kv * w),
        )
    };
    let slab_rows = cx.arg_slab_rows() as usize;
    let (key_codes, key_coefficients) = (
        SlabTensor::from_scalars(cx.arg_history_key_codes(), slab_rows),
        SlabTensor::from_bound(cx.arg_history_key_coefficients(), slab_rows),
    );
    let (value_codes, value_coefficients) = (
        SlabTensor::from_scalars(cx.arg_history_value_codes(), slab_rows),
        SlabTensor::from_bound(cx.arg_history_value_coefficients(), slab_rows),
    );
    let history = (
        attention::affine_rows::<8, _>(&key_codes, &key_coefficients, kv_head),
        attention::affine_rows::<4, _>(&value_codes, &value_coefficients, kv_head),
    );
    let batch = (attention::batch_rows(keys, kv, kv_head, w), attention::batch_rows(values, kv, kv_head, w));
    attention::attend_rows(
        l,
        &mut state,
        queries,
        cx.arg_scale(),
        scores,
        work,
        count,
        sequence,
        cx.dim_br() as i32,
        (&history.0, &history.1),
        (&batch.0, &batch.1),
    );
    let (query, gate, result, sinks) = (cx.arg_query(), cx.arg_gate(), cx.result_0(), cx.arg_sinks());
    let softplus = cx.arg_gate_function() != 0;
    for local in 0..count {
        let row = first + local;
        let state = state.heads(local * g, g);
        let turned = attention::output_angles(
            form,
            cx.arg_coordinates().row([row, 0]),
            cx.arg_rotary_components().row([0]),
            cx.arg_rotary_frequencies().row([0]),
            cx.arg_rotary_amplitudes().row([0]),
            angles,
        );
        for head in 0..g {
            let column = kv_head * g + head;
            let gates = if form.interleaved > 0 {
                query.span([row, column, w], form.interleaved)
            } else {
                gate.row([row, column, 0])
            };
            let mut denominator = state.denominator[head];
            if cx.dim_sk() == 1 {
                denominator += attention::sink_weight(sinks.get([0, column]), state.maximum[head]);
            }
            // SAFETY: each work item writes its own query heads of its rows.
            let mut out = unsafe { result.row_mut([row, column, 0]) };
            attention::gate_row::<E::A>(&state.accumulator[head * w..][..w], denominator, gates, softplus, &mut out);
            if let Some((cosines, sines)) = turned {
                attention::output_rotary::<E::A>(&mut out, form.layout, cosines, sines);
            }
        }
    }
}
