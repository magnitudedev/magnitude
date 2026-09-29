// vision_norm on CPU (contract and portable body in vision.seismic). One work
// item per result row r (member s of cell r / G): the norm of its source row
// (source row order[r] when gathered), with the decoded weight and bias,
// stored at member s of its cell side by side or interleaved.

use lib::vision::vision;

fn vision_norm<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let row = group[0] as usize;
    let (g, h) = (cx.dim_g() as usize, cx.dim_h() as usize);
    let from = if cx.dim_no() == 1 { cx.arg_order().get([0, row]) as usize } else { row };
    let shared = seismic::cpu::tensor::floats(shared, 3 * h);
    let (weight, rest) = shared.split_at_mut(h);
    let (bias, normalized) = rest.split_at_mut(h);
    let weight = (cx.dim_nw() == 1).then(|| vision::decode_vector(&cx.arg_weight(), weight));
    let bias = (cx.dim_nb() == 1).then(|| vision::decode_vector(&cx.arg_bias(), bias));
    let source = cx.arg_source().row([from / g, from % g, 0]);
    vision::norm(source, cx.arg_centered() != 0, weight, bias, cx.arg_epsilon(), normalized);
    let (cell, member) = (row / g, row % g);
    let result = cx.result_0();
    if cx.arg_interleave() != 0 {
        for (column, value) in normalized.iter().enumerate() {
            // SAFETY: each work item writes its own member's columns.
            let slot = unsafe { result.span_mut([cell, column * g + member], 1) };
            slot[0] = E::Y::narrow(*value);
        }
    } else {
        // SAFETY: each work item writes its own member's columns.
        let out = unsafe { result.span_mut([cell, member * h], h) };
        for (target, value) in out.iter_mut().zip(normalized.iter()) {
            *target = E::Y::narrow(*value);
        }
    }
}
