// gated_attention_prefill on CPU (contract and portable body in
// attention.seismic). `prepare`: one work item per row prepares its queries
// and keys (activation values, as F32) into scratch and appends its key and
// value at its destination. `attend`: one work item per (row, kv head)
// absorbs the row's keys (spans in order, then its fresh span) for the kv
// head's G query heads and stores their gated outputs.

use lib::attention::attention::{self, Fresh, Heads, Online, BLOCK};
use lib::core::activation;

fn heads<E: Elements>(cx: &Context<'_, E>) -> Heads {
    Heads::new(cx.dim_kv(), cx.dim_g(), cx.dim_p(), cx.dim_s())
}

fn gated_attention_prefill_prepare<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let heads = heads(cx);
    let (w, q, kv) = (heads.w, heads.queries(), heads.kv);
    let angles = seismic::cpu::tensor::floats(shared, 2 * heads.p);
    let (cosines, sines) = angles.split_at_mut(heads.p);
    attention::angles(
        cx.arg_coordinates().row([row, 0]),
        cx.arg_rotary_components().row([0]),
        cx.arg_rotary_frequencies().row([0]),
        cosines,
        sines,
    );
    // SAFETY: each work item writes its own row of the scratch.
    let queries = unsafe { cx.scratch_queries().slice_mut::<f32>(4 * row * q * w, q * w) };
    let keys = unsafe { cx.scratch_keys().slice_mut::<f32>(4 * row * kv * w, kv * w) };
    attention::prepare_row::<E::A>(
        heads,
        cx.arg_query_gate().row([row, 0]),
        cx.arg_key().row([row, 0]),
        cx.arg_query_norm().row([0]),
        cx.arg_key_norm().row([0]),
        cosines,
        sines,
        cx.arg_epsilon(),
        queries,
        keys,
    );
    let destination = cx.arg_destinations().get([row]);
    if destination >= 0 {
        let (history_key, history_value, value) = (cx.arg_history_key(), cx.arg_history_value(), cx.arg_value());
        for head in 0..kv {
            // SAFETY: destinations are distinct, freshly reserved history rows
            // no row of the call reads.
            let (key_row, value_row) = unsafe {
                (history_key.row_mut([destination as usize, head, 0]), history_value.row_mut([destination as usize, head, 0]))
            };
            activation::store::<E::A>(&keys[head * w..][..w], key_row);
            value_row.copy_from_slice(value.span([row, head * w], w));
        }
    }
}

fn gated_attention_prefill_attend<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (row, kv_head) = (group[0] as usize, group[1] as usize);
    let heads = heads(cx);
    let (g, w, kv) = (heads.g, heads.w, heads.kv);
    let (m, r) = (cx.dim_m() as usize, cx.dim_r() as usize);
    let (visible, fresh) = (cx.arg_visible(), cx.arg_fresh());
    let spans = (0..r)
        .map(move |span| (visible.get([row, span, 0]), visible.get([row, span, 1])))
        .filter(|(lo, hi)| hi > lo);
    let fresh = (fresh.get([row, 0]), fresh.get([row, 1]));
    let range = 0..attention::total(spans.clone(), fresh);
    let work = seismic::cpu::tensor::floats(shared, 2 * g + g * w + g * BLOCK + w);
    let (maximum, work) = work.split_at_mut(g);
    let (denominator, work) = work.split_at_mut(g);
    let (accumulator, work) = work.split_at_mut(g * w);
    let (scores, key_row) = work.split_at_mut(g * BLOCK);
    let mut state = Online { maximum, denominator, accumulator };
    state.reset();
    // SAFETY: the prepare launch wrote every row before this launch.
    let (queries, keys) = unsafe {
        (
            cx.scratch_queries().slice::<f32>(4 * (row * kv + kv_head) * g * w, g * w),
            cx.scratch_keys().slice::<f32>(0, m * kv * w),
        )
    };
    let rows = Fresh { keys, values: cx.arg_value() };
    let (history_key, history_value) = (cx.arg_history_key(), cx.arg_history_value());
    attention::attend(
        &mut state,
        heads,
        kv_head,
        queries,
        cx.arg_scale(),
        scores,
        key_row,
        spans,
        fresh,
        range,
        &rows,
        |token, out| activation::widen::<E::A>(history_key.row([token, kv_head, 0]), out),
        |token, out| activation::widen::<E::A>(history_value.row([token, kv_head, 0]), out),
    );
    let (query_gate, result) = (cx.arg_query_gate(), cx.result_0());
    for local in 0..g {
        let head = kv_head * g + local;
        // SAFETY: each work item writes its own query heads of its row.
        let out = unsafe { result.row_mut([row, head, 0]) };
        attention::gate::<E::A>(
            &state.accumulator[local * w..][..w],
            state.denominator[local],
            query_gate.span([row, head * 2 * w + w], w),
            out,
        );
    }
}
