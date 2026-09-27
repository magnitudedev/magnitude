// vision_attention on CPU (contract and portable body in vision.seismic).
// `vision_attention_prepare`, one work item per row, forms the row's
// operands in F32 scratch [M][3][H][4P]: each query and key head normalized
// with its weight row (when bound) and rotated, each value head normalized
// (when bound), all published to A. `vision_attention_attend` runs one query
// row of one head over the keys of its span (every row, or its window).

use lib::core::activation;
use lib::vision::vision;

fn vision_attention_prepare<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (heads, p) = (cx.dim_h() as usize, cx.dim_p() as usize);
    let w = 4 * p;
    let epsilon = cx.arg_epsilon();
    let coordinates = cx.arg_coordinates();
    let coordinates = [coordinates.get([row, 0]), coordinates.get([row, 1])];
    let source = seismic::cpu::tensor::floats(shared, w);
    // SAFETY: each work item writes its own row of the scratch.
    let operands = unsafe { cx.scratch_operands().slice_mut::<f32>(4 * row * 3 * heads * w, 3 * heads * w) };
    let (queries, rest) = operands.split_at_mut(heads * w);
    let (keys, values) = rest.split_at_mut(heads * w);
    for head in 0..heads {
        for (part, target) in [&mut queries[head * w..][..w], &mut keys[head * w..][..w]]
            .into_iter()
            .enumerate()
        {
            let operand = if part == 0 { cx.arg_query() } else { cx.arg_key() };
            seismic::cpu::tensor::widen_row::<E::A>(operand.row([row, head, 0]), source);
            if cx.dim_nq() == 1 {
                let norm = if part == 0 { cx.arg_query_norm() } else { cx.arg_key_norm() };
                vision::norm(source, false, Some(norm.row([0, 0])), None, epsilon, target);
            } else {
                target.copy_from_slice(source);
            }
            vision::rotate::<E::A>(target, coordinates, p, cx.arg_log_base());
        }
        let value = &mut values[head * w..][..w];
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
    let (m, heads, p) = (cx.dim_m() as usize, cx.dim_h() as usize, cx.dim_p() as usize);
    let w = 4 * p;
    // SAFETY: the prepare launch wrote every row before this launch.
    let operands = unsafe { cx.scratch_operands().slice::<f32>(0, m * 3 * heads * w) };
    let operand = |row: usize, part: usize| &operands[((row * 3 + part) * heads + head) * w..][..w];
    let span = if cx.dim_ws() == 1 {
        let spans = cx.arg_spans();
        spans.get([0, row, 0]) as usize..spans.get([0, row, 1]) as usize
    } else {
        0..m
    };
    let shared = seismic::cpu::tensor::floats(shared, m + w);
    let (scores, attended) = shared.split_at_mut(m);
    vision::attend::<E::A>(
        operand(row, 0),
        span,
        |key| operand(key, 1),
        |key| operand(key, 2),
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
