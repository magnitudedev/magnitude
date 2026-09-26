// gated_attention_prefill_k8v4 on CPU (contract and portable body in
// attention.seismic): `gated_attention_prefill`'s launches over affine K8/V4
// history. `prepare` appends the row's prepared key and projected value
// encoded; `attend` decodes each history key and value (code * scale + zero,
// F32) as it absorbs them; fresh rows attend dense. Head widths are whole
// affine groups (the declaration's `where`), so every vector has at least one
// group.

use lib::attention::attention::{self, Fresh, Heads, Online, BLOCK, KEY_BITS, VALUE_BITS};
use lib::core::activation;

fn heads<E: Elements>(cx: &Context<'_, E>) -> Heads {
    Heads::new(cx.dim_kv(), cx.dim_g(), cx.dim_p(), cx.dim_s()).affine()
}

fn gated_attention_prefill_k8v4_prepare<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let row = group[0] as usize;
    let heads = heads(cx);
    let (w, q, kv) = (heads.w, heads.queries(), heads.kv);
    let work = seismic::cpu::tensor::floats(shared, 2 * heads.p + w);
    let (angles, value_row) = work.split_at_mut(2 * heads.p);
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
        let destination = destination as usize;
        let (key_codes, key_coefficients) = (cx.arg_history_key_codes(), cx.arg_history_key_coefficients());
        let (value_codes, value_coefficients) = (cx.arg_history_value_codes(), cx.arg_history_value_coefficients());
        let value = cx.arg_value();
        for head in 0..kv {
            activation::widen::<E::A>(value.span([row, head * w], w), value_row);
            // SAFETY: destinations are distinct, freshly reserved history rows
            // no row of the call reads.
            unsafe {
                attention::affine_encode(
                    &keys[head * w..][..w],
                    KEY_BITS,
                    key_codes.row_mut([destination, head, 0]),
                    key_coefficients.row_mut([destination, head, 0]),
                );
                attention::affine_encode(
                    value_row,
                    VALUE_BITS,
                    value_codes.row_mut([destination, head, 0]),
                    value_coefficients.row_mut([destination, head, 0]),
                );
            }
        }
    }
}

fn gated_attention_prefill_k8v4_attend<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    shared: &mut [u8],
) {
    let (row, kv_head) = (group[0] as usize, group[1] as usize);
    let heads = heads(cx);
    let (g, w, kv) = (heads.g, heads.w, heads.kv);
    let (m, r) = (cx.dim_m() as usize, cx.dim_r() as usize);
    let (visible, fresh) = (cx.arg_visible(), cx.arg_fresh());
    let spans = (0..r)
        .map(move |span| (visible.get([row, span, 0]), visible.get([row, span, 1])))
        .filter(|(lo, hi)| *lo >= 0 && hi > lo);
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
    let (key_codes, key_coefficients) = (cx.arg_history_key_codes(), cx.arg_history_key_coefficients());
    let (value_codes, value_coefficients) = (cx.arg_history_value_codes(), cx.arg_history_value_coefficients());
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
        |token, out| {
            attention::affine_decode(
                key_codes.row([token, kv_head, 0]),
                key_coefficients.row([token, kv_head, 0]),
                KEY_BITS,
                out,
            )
        },
        |token, out| {
            attention::affine_decode(
                value_codes.row([token, kv_head, 0]),
                value_coefficients.row([token, kv_head, 0]),
                VALUE_BITS,
                out,
            )
        },
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
