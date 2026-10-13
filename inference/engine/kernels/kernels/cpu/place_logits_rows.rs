fn place_logits_rows<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let row = group[0] as usize;
    let vocabulary = cx.dim_v() as usize;
    // SAFETY: each work item owns one complete output row. Indexed stores
    // preserve the result's actual strides rather than assuming a contiguous row.
    for token in 0..vocabulary {
        unsafe {
            cx.arg_logits().set([row, token], f32::NEG_INFINITY);
        }
    }
    for index in 0..cx.dim_s() as usize {
        let token = cx.arg_token_rows().get([row, index]);
        if token >= 0 && (token as usize) < vocabulary {
            unsafe {
                cx.arg_logits()
                    .set([row, token as usize], cx.arg_scores().get([row, index]));
            }
        }
    }
}
