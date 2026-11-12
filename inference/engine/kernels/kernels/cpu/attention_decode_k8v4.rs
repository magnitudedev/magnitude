// attention_decode_k8v4 on CPU (contract and portable body in
// attention.seismic): `attention_decode`'s one launch over affine K8/V4
// history. Each (kv head, partition, row) item prepares its own queries, keys
// and values, part 0 appends the row's key and value encoded; the item
// decodes each history key and value (code * scale + zero, F32) as it absorbs
// them; fresh rows attend as history will hold them (encoded and decoded
// back), so a row's result does not depend on how many rows verify beside
// it. The last item of a (kv head, row) to arrive
// merges its partitions. Head widths are whole affine groups (the
// declaration's `where`), so every vector has at least one group.

use lib::attention::attention::{self, Form, Layout, Online, BLOCK};
use lib::core::{arrive, spans};
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
fn attention_decode_k8v4_attend<L: Isa, E: Elements>(l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (kv_head, part, row) = (group[0] as usize, group[1] as usize, group[2] as usize);
    let form = form(cx);
    let (g, w, kv) = (form.heads.g, form.heads.w, form.heads.kv);
    let (m, r, parts) = (cx.dim_m() as usize, cx.dim_r() as usize, cx.param_parts() as usize);
    let (visible, fresh) = (cx.arg_visible(), cx.arg_fresh());
    let spans = (0..r)
        .map(move |span| spans::span(visible.get([row, span, 0]), visible.get([row, span, 1])))
        .filter(|rows| !rows.is_empty())
        .map(|rows| (rows.start, rows.end));
    // A layer without fresh rows sees none (the portable `for layer in 0..F`).
    let fresh = if form.fresh { (fresh.get([row, 0]), fresh.get([row, 1])) } else { (0, 0) };
    let range = attention::partition(attention::total(spans.clone(), fresh), parts, part);
    let item = (row * kv + kv_head) * parts + part;
    // SAFETY: each work item writes its own partial state.
    let accumulator = unsafe { cx.scratch_partials().slice_mut::<f32>(4 * item * g * w, g * w) };
    let statistics = unsafe { cx.scratch_statistics().slice_mut::<f32>(4 * item * 2 * g, 2 * g) };
    let (maximum, denominator) = statistics.split_at_mut(g);
    let mut state = Online { maximum, denominator, accumulator };
    state.reset();
    let (p, ro) = (form.heads.p, usize::from(form.output_rotary));
    let shared = seismic::cpu::tensor::floats(
        shared,
        g * BLOCK + attention::work(w, true) + 2 * p + g * w + 2 * m * w + w + 2 * p * ro,
    );
    let (scores, shared) = shared.split_at_mut(g * BLOCK);
    let (work, shared) = shared.split_at_mut(attention::work(w, true));
    let (angles, shared) = shared.split_at_mut(2 * p);
    let (queries, shared) = shared.split_at_mut(g * w);
    let (keys, shared) = shared.split_at_mut(m * w);
    let (values, merged) = shared.split_at_mut(m * w);
    let (query_norm, key_norm, value_norm) = (cx.arg_query_norm(), cx.arg_key_norm(), cx.arg_value_norm());
    let norms = (
        form.query_norm.then(|| query_norm.row([0, 0])),
        form.key_norm.then(|| key_norm.row([0, 0])),
        form.value_norm.then(|| value_norm.row([0, 0])),
    );
    let (query, key_rows, value_rows) = (cx.arg_query(), cx.arg_key(), cx.arg_value());
    let destination = cx.arg_destinations().get([row]);
    let append = form.fresh && destination >= 0 && part == 0;
    attention::prepare_group::<E::A>(
        form,
        kv_head,
        row,
        append,
        spans::span(fresh.0, fresh.1),
        |r, cosines, sines| {
            attention::scaled_angles(
                cx.arg_coordinates().row([r, 0]),
                cx.arg_rotary_components().row([0]),
                cx.arg_rotary_frequencies().row([0]),
                cx.arg_rotary_amplitudes().row([0]),
                cosines,
                sines,
            )
        },
        |head| query.row([row, head, 0]),
        |r| key_rows.row([0, r, 0]),
        |r| value_rows.row([0, r, 0]),
        norms,
        cx.arg_epsilon(),
        angles.split_at_mut(p),
        queries,
        keys,
        values,
    );
    let slab_rows = cx.arg_slab_rows() as usize;
    let (key_codes, key_coefficients) = (
        SlabTensor::from_scalars(cx.arg_history_key_codes(), slab_rows),
        SlabTensor::from_bound(cx.arg_history_key_coefficients(), slab_rows),
    );
    let (value_codes, value_coefficients) = (
        SlabTensor::from_scalars(cx.arg_history_value_codes(), slab_rows),
        SlabTensor::from_bound(cx.arg_history_value_coefficients(), slab_rows),
    );
    let bits = |words: u64| (32 * words / (2 * cx.dim_p() + cx.dim_s())) as u32;
    let (key_bits, value_bits) = (bits(cx.dim_kc()), bits(cx.dim_vc()));
    if append {
        // SAFETY: destinations are distinct, freshly reserved history rows
        // no row of the call reads.
        unsafe {
            attention::store_affine_head(
                &keys[row * w..][..w],
                &values[row * w..][..w],
                (key_bits, value_bits),
                (&key_codes, &key_coefficients),
                (&value_codes, &value_coefficients),
                destination as usize,
                kv_head,
            );
        }
    }
    // The fresh keys and values as the history rows they become read: each
    // encoded as its append encodes it, decoded back.
    for fresh_row in spans::span(fresh.0, fresh.1) {
        let at = fresh_row as usize * w;
        attention::affine_round_trip(&mut keys[at..][..w], key_bits, cx.dim_kp() as usize);
        attention::affine_round_trip(&mut values[at..][..w], value_bits, cx.dim_vp() as usize);
    }
    let history = (
        attention::affine_rows::<8, _>(&key_codes, &key_coefficients, kv_head),
        attention::affine_rows::<4, _>(&value_codes, &value_coefficients, kv_head),
    );
    let batch = (attention::batch_rows(keys, 1, 0, w), attention::batch_rows(values, 1, 0, w));
    attention::attend(
        l,
        &mut state,
        queries,
        cx.arg_scale(),
        scores,
        work,
        spans,
        fresh,
        range,
        |_| true,
        (&history.0, &history.1),
        (&batch.0, &batch.1),
    );
    // SAFETY: one counter a (row, kv head) of the launch's `arrivals` scratch,
    // each of the group's PARTS items arriving once, after its partial state.
    let last = unsafe { arrive::last(cx.scratch_arrivals().pointer::<u32>(4 * (row * kv + kv_head)), parts as u32) };
    if last {
        merge(cx, form, kv_head, row, merged);
    }
}

/// The merge of one (kv head, row)'s partitions, run by the last item of the
/// group to finish: `work` holds the output row and the output rotary's angles.
#[inline(always)]
fn merge<E: Elements>(cx: &Context<'_, E>, form: Form, kv_head: usize, row: usize, work: &mut [f32]) {
    let (g, w, kv) = (form.heads.g, form.heads.w, form.heads.kv);
    let (m, parts) = (cx.dim_m() as usize, cx.param_parts() as usize);
    // SAFETY: every item of the group stored its partial state before it arrived.
    let (partials, statistics) = unsafe {
        (
            cx.scratch_partials().slice::<f32>(0, m * kv * parts * g * w),
            cx.scratch_statistics().slice::<f32>(0, m * kv * parts * 2 * g),
        )
    };
    let (output, angles) = work.split_at_mut(w);
    let turned = attention::output_angles(
        form,
        cx.arg_coordinates().row([row, 0]),
        cx.arg_rotary_components().row([0]),
        cx.arg_rotary_frequencies().row([0]),
        cx.arg_rotary_amplitudes().row([0]),
        angles,
    );
    let (query, gate, result) = (cx.arg_query(), cx.arg_gate(), cx.result_0());
    let softplus = cx.arg_gate_function() != 0;
    for local in 0..g {
        let head = kv_head * g + local;
        let item = |part: usize| (row * kv + kv_head) * parts + part;
        let maximum = |part: usize| statistics[item(part) * 2 * g + local];
        let mut denominator = attention::merge(
            parts,
            maximum,
            |part| statistics[item(part) * 2 * g + g + local],
            |part| &partials[(item(part) * g + local) * w..][..w],
            output,
        );
        if cx.dim_sk() == 1 {
            let overall = (0..parts).map(maximum).fold(f32::NEG_INFINITY, f32::max);
            denominator += attention::sink_weight(cx.arg_sinks().get([0, head]), overall);
        }
        let gates = if form.interleaved > 0 {
            query.span([row, head, w], form.interleaved)
        } else {
            gate.row([row, head, 0])
        };
        // SAFETY: each work item writes its own row of the result.
        let out = unsafe { result.row_mut([row, head, 0]) };
        attention::gate_row::<E::A>(output, denominator, gates, softplus, out);
        if let Some((cosines, sines)) = turned {
            attention::output_rotary::<E::A>(out, form.layout, cosines, sines);
        }
    }
}
