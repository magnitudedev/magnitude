// vision_attention on CPU (contract and portable body in vision.seismic).
// `vision_attention_prepare`, one work item per row, forms the row's
// operands in F32 scratch [M][H + 2KV][4P] (H = KV * G): the query heads,
// then the key heads, then the value heads, each query and key head
// normalized with its weight row (when bound) and rotated, each value head
// normalized (when bound), all published to A. `vision_attention_attend` runs
// one query row of one head over the keys of its span (every row, or its
// window), query head h reading key and value head h / G.

use lib::core::activation;
use lib::vision::vision;

fn vision_attention_prepare<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (kv_heads, p) = (cx.dim_kv() as usize, cx.dim_p() as usize);
    let heads = kv_heads * cx.dim_g() as usize;
    let (w, slots) = (4 * p, heads + 2 * kv_heads);
    let epsilon = cx.arg_epsilon();
    let coordinates = cx.arg_coordinates();
    let coordinates = [coordinates.get([row, 0]), coordinates.get([row, 1])];
    let source = seismic::cpu::tensor::floats(shared, w);
    // SAFETY: each work item writes its own row of the scratch.
    let operands = unsafe { cx.scratch_operands().slice_mut::<f32>(4 * row * slots * w, slots * w) };
    let (queries, rest) = operands.split_at_mut(heads * w);
    let (keys, values) = rest.split_at_mut(kv_heads * w);
    for (operand, norm, targets) in
        [(cx.arg_query(), cx.arg_query_norm(), queries), (cx.arg_key(), cx.arg_key_norm(), keys)]
    {
        for (head, target) in targets.chunks_exact_mut(w).enumerate() {
            seismic::cpu::tensor::widen_row::<E::A>(operand.row([row, head, 0]), source);
            if cx.dim_nq() == 1 {
                vision::norm(source, false, Some(norm.row([0, 0])), None, epsilon, target);
            } else {
                target.copy_from_slice(source);
            }
            if cx.arg_log_base() != 0.0 {
                vision::rotate::<E::A>(target, coordinates, p, cx.arg_log_base());
            } else {
                for element in target.iter_mut() {
                    *element = activation::publish::<E::A>(*element);
                }
            }
        }
    }
    for (head, value) in values.chunks_exact_mut(w).enumerate() {
        seismic::cpu::tensor::widen_row::<E::A>(cx.arg_value().row([row, head, 0]), source);
        if cx.dim_nv() == 1 {
            vision::norm(source, false, Some(cx.arg_value_norm().row([0, 0])), None, epsilon, value);
            for element in value.iter_mut() {
                *element = activation::publish::<E::A>(*element);
            }
        } else {
            value.copy_from_slice(source);
        }
    }
}

fn vision_attention_attend<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (row, head) = (group[0] as usize, group[1] as usize);
    let (m, kv_heads, grouped, p) =
        (cx.dim_m() as usize, cx.dim_kv() as usize, cx.dim_g() as usize, cx.dim_p() as usize);
    let heads = kv_heads * grouped;
    let (w, slots) = (4 * p, heads + 2 * kv_heads);
    // SAFETY: the prepare launch wrote every row before this launch.
    let operands = unsafe { cx.scratch_operands().slice::<f32>(0, m * slots * w) };
    let operand = |row: usize, slot: usize| &operands[(row * slots + slot) * w..][..w];
    let (key, value) = (heads + head / grouped, heads + kv_heads + head / grouped);
    let span = if cx.dim_ws() == 1 {
        let spans = cx.arg_spans();
        spans.get([0, row, 0]) as usize..spans.get([0, row, 1]) as usize
    } else {
        0..m
    };
    let shared = seismic::cpu::tensor::floats(shared, m + w);
    let (scores, attended) = shared.split_at_mut(m);
    vision::attend::<E::A>(
        operand(row, head),
        span,
        |row| operand(row, key),
        |row| operand(row, value),
        cx.arg_unit_scale() != 0,
        scores,
        attended,
    );
    let result = cx.result_0();
    // SAFETY: each work item writes its own head of its own row.
    let out = unsafe { result.span_mut([row, head, 0], w) };
    for (target, value) in out.iter_mut().zip(attended.iter()) {
        *target = E::A::narrow(*value);
    }
}
