// draft_top_k on CPU (contract and portable body in dflash.seismic): one work
// item per row. A single pass over the vocabulary maintains the K highest
// entries in (value descending, token ascending) order, scanning in 16-float
// chunks so the common below-threshold case vectorizes without per-element
// branches.

const SMALL_K: usize = 64;
const CHUNK: usize = 16;

#[inline(always)]
fn insert(values: &mut [f32], tokens: &mut [i64], threshold: &mut f32, value: f32, token: i64) {
    let last = values.len() - 1;
    if value > *threshold || (value == *threshold && token < tokens[last]) {
        let mut pos = last;
        while pos > 0
            && (value > values[pos - 1]
                || (value == values[pos - 1] && token < tokens[pos - 1]))
        {
            values[pos] = values[pos - 1];
            tokens[pos] = tokens[pos - 1];
            pos -= 1;
        }
        values[pos] = value;
        tokens[pos] = token;
        *threshold = values[last];
    }
}

#[inline(always)]
fn select_top_k(line: &[f32], values: &mut [f32], tokens: &mut [i64]) {
    let mut threshold = f32::NEG_INFINITY;
    let mut base_token = 0usize;
    let chunks = line.chunks_exact(CHUNK);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let mut any = false;
        for &v in chunk {
            any |= v >= threshold;
        }
        if any {
            for (offset, &value) in chunk.iter().enumerate() {
                if value >= threshold {
                    insert(values, tokens, &mut threshold, value, (base_token + offset) as i64);
                }
            }
        }
        base_token += CHUNK;
    }
    for (offset, &value) in remainder.iter().enumerate() {
        if value >= threshold {
            insert(values, tokens, &mut threshold, value, (base_token + offset) as i64);
        }
    }
}

#[inline(always)]
fn draft_top_k<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let count = cx.dim_k() as usize;
    if count == 0 {
        return;
    }
    let line = cx.arg_logits().row([row, 0]);
    let write = |values: &[f32], tokens: &[i64]| {
        for (rank, (&value, &token)) in values.iter().zip(tokens).enumerate() {
            let at = row * count + rank;
            // SAFETY: each work item writes its own rows.
            unsafe {
                cx.arg_candidates().set([at, 0], token as i32);
                cx.arg_candidates().set([at, 1], 0);
                cx.arg_unary().set([row, rank], value);
            }
        }
    };
    if count <= SMALL_K {
        let mut values = [f32::NEG_INFINITY; SMALL_K];
        let mut tokens = [i64::MAX; SMALL_K];
        select_top_k(line, &mut values[..count], &mut tokens[..count]);
        write(&values[..count], &tokens[..count]);
    } else {
        let mut values = vec![f32::NEG_INFINITY; count];
        let mut tokens = vec![i64::MAX; count];
        select_top_k(line, &mut values, &mut tokens);
        write(&values, &tokens);
    }
}
