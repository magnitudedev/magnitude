// select_compact_rows on CPU (contract and portable body in
// clustered_readout.seismic): one work item per row selects the winning
// canonical token and its softmax confidence directly from the row's compact
// `[S]` shortlist.

const NO_TOKEN: u32 = u32::MAX;

#[inline(always)]
fn gumbel(value: f32, token: u32, draw: &[u32]) -> f32 {
    let mut counter = [token, draw[3], draw[4], draw[5]];
    let mut key = [draw[1], draw[2]];
    for _ in 0..10 {
        let p0 = u64::from(3_528_531_795u32) * u64::from(counter[0]);
        let p1 = u64::from(3_449_720_151u32) * u64::from(counter[2]);
        counter = [
            (p1 >> 32) as u32 ^ counter[1] ^ key[0],
            p1 as u32,
            (p0 >> 32) as u32 ^ counter[3] ^ key[1],
            p0 as u32,
        ];
        key[0] = key[0].wrapping_add(2_654_435_769);
        key[1] = key[1].wrapping_add(3_144_134_277);
    }
    let uniform = ((counter[0] >> 9) as f32 + 0.5) * 0.000_000_119_209_289_550_781_25;
    value - (-uniform.ln()).ln()
}

#[inline(always)]
fn better(score: f32, token: u32, best: f32, best_token: u32) -> bool {
    best_token == NO_TOKEN || (token != NO_TOKEN && (score > best || (score == best && token < best_token)))
}

#[inline(always)]
fn select_compact_rows<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let mask_words = cx.dim_w() as usize;
    let count = cx.dim_s() as usize;
    let scores = cx.arg_scores().row([row, 0]);
    let tokens = cx.arg_token_rows().row([row, 0]);
    let mask = cx.arg_mask();
    let draw = cx.arg_draws().row([row, 0]);
    let masked = cx.arg_constrained().get([row]) != 0;
    let noisy = draw[0] == 1;

    let mut best_score = f32::NEG_INFINITY;
    let mut best_logit = f32::NEG_INFINITY;
    let mut best_token = NO_TOKEN;
    let mut bad = false;
    let mut max_logit = f32::NEG_INFINITY;

    for col in 0..count {
        let raw_token = tokens[col];
        if raw_token < 0 || ((raw_token as usize) >> 5) >= mask_words {
            continue;
        }
        let token = raw_token as u32;
        let value = scores[col];
        if value.is_nan() || value == f32::INFINITY {
            bad = true;
        }
        if value > max_logit {
            max_logit = value;
        }
        if !(value > f32::NEG_INFINITY && value < f32::INFINITY) {
            continue;
        }
        if masked && ((mask.get([row, (token >> 5) as usize]) >> (token & 31)) & 1) == 0 {
            continue;
        }
        let candidate = if noisy { gumbel(value, token, draw) } else { value };
        if better(candidate, token, best_score, best_token) {
            best_score = candidate;
            best_logit = value;
            best_token = token;
        }
    }

    let status = if bad {
        2
    } else if best_token == NO_TOKEN {
        1
    } else {
        0
    };
    let (out_token, probability) = if status == 0 {
        let mut total = 0.0f32;
        for col in 0..count {
            let raw_token = tokens[col];
            if raw_token >= 0 && ((raw_token as usize) >> 5) < mask_words {
                total += (scores[col] - max_logit).exp();
            }
        }
        let prob = if total != 0.0 {
            (best_logit - max_logit).exp() / total
        } else {
            0.0
        };
        (best_token as i32, prob)
    } else {
        (-1, 0.0)
    };
    // SAFETY: each work item owns its own row.
    unsafe {
        cx.arg_result().set([row, 0], out_token);
        cx.arg_result().set([row, 1], status);
        cx.arg_confidence().set([row], probability);
    }
}
