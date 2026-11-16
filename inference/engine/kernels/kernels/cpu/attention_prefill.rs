// attention_prefill on CPU (contract and portable body in attention.seismic).
// `prepare`: one work item per row prepares its queries (into the scratch by
// kv head, [KV][M][G][W], so a block of rows' queries of one kv head are
// contiguous) and, for a layer with fresh rows, its keys and values
// (activation values, as F32) into scratch and appends its key and value at
// its destination. `attend`: one work item per (block of consecutive rows,
// kv head) absorbs the rows' keys (spans in order, then each row's fresh
// span) for the kv head's G query heads of each row (`attention::attend_rows`,
// which reads the keys the rows share once for all of them) and stores their
// gated outputs. The block is the grid's: ceil(M / blocks) rows, which never
// changes result bits.

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
    .with_key_rounding(cx.dim_kr(), cx.arg_key_block(), cx.arg_key_element(), cx.arg_key_scale(), cx.arg_key_amax_floor())
}

#[inline(always)]
fn attention_prefill_prepare<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
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
    let value = cx.arg_value();
    let value = if form.own_values() { value.row([0, row, 0]) } else { &[][..] };
    attention::prepare_form_row::<E::A>(form, cx.arg_key().row([0, row, 0]), value, norms, cosines, sines, cx.arg_epsilon(), keys, values);
    let destination = cx.arg_destinations().get([row]);
    if destination >= 0 {
        let slab_rows = cx.arg_slab_rows() as usize;
        // SAFETY: destinations are distinct, freshly reserved history rows
        // no row of the call reads.
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
}

#[inline(always)]
fn attention_prefill_attend<L: Isa, E: Elements>(l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
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
    let work = seismic::cpu::tensor::floats(shared, heads * (2 + w + BLOCK) + attention::work(w, false) + turning);
    let (maximum, work) = work.split_at_mut(heads);
    let (denominator, work) = work.split_at_mut(heads);
    let (accumulator, work) = work.split_at_mut(heads * w);
    let (scores, work) = work.split_at_mut(heads * BLOCK);
    let (work, angles) = work.split_at_mut(attention::work(w, false));
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
    let history_key = SlabTensor::from_bound(cx.arg_history_key(), slab_rows);
    let history_value = SlabTensor::from_bound(cx.arg_history_value(), slab_rows);
    // Stored-key values are read from the key plane.
    let history_value = if form.stored_key { &history_key } else { &history_value };
    let history = (attention::history_rows(&history_key, kv_head), attention::history_rows(history_value, kv_head));
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
    // The index-list source (picks, after each row's keys).
    let picks = (cx.dim_ks() > 0).then(|| SlabTensor::from_bound(cx.arg_selected(), cx.arg_selected_slab_rows() as usize));
    for local in 0..count {
        let row = first + local;
        let mut state = state.heads(local * g, g);
        let queries = &queries[local * g * w..][..g * w];
        if let Some(selected) = &picks {
            attention::attend_picks::<_, E::A>(
                l,
                &mut state,
                queries,
                cx.arg_scale(),
                scores,
                work,
                cx.arg_selected_rows().row([row, 0]),
                |pick| selected.row([pick, kv_head, 0]),
            );
        }
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
