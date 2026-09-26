// sample_rows on CPU (contract in sampling.seismic, portable body in
// seismic-std). `sample_rows_partition` gives each work item one of PARTS
// contiguous vocabulary slices of a row and keeps its winner (score bits,
// token, invalid flag) in scratch; `sample_rows_merge` combines a row's slice
// winners in slice order. The maximum score wins, ties go to the lowest token
// and the winning score is carried unchanged, so the result does not depend
// on PARTS.
//
// Counter (token, draws[3], draws[4], draws[5]), key (draws[1], draws[2]);
// draws[0] == 1 enables the Gumbel perturbation v - log(-log u) with
// u = ((x >> 9) + 0.5) * 2^-23. A token competes when its logit is finite
// and its row is unconstrained or its mask bit is set. Status 0 success,
// 1 no competing token, 2 some logit of the row is NaN or +inf.

const NO_TOKEN: u32 = u32::MAX;

/// The Philox4x32-10 Gumbel score of `value` at `token` (the portable body's
/// perturbation), or `value` when the row draws greedily.
#[inline(always)]
fn score(value: f32, token: u32, draw: &[u32]) -> f32 {
    if draw[0] != 1 {
        return value;
    }
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

/// Whether candidate (score, token) beats (best, best_token): a higher
/// score, then a lower token.
#[inline(always)]
fn better(score: f32, token: u32, best: f32, best_token: u32) -> bool {
    best_token == NO_TOKEN || (token != NO_TOKEN && (score > best || (score == best && token < best_token)))
}

fn sample_rows_partition<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (part, row) = (group[0] as usize, group[1] as usize);
    let parts = cx.param_parts() as usize;
    let vocabulary = cx.dim_v() as usize;
    let span = vocabulary.div_ceil(parts);
    let begin = (part * span).min(vocabulary);
    let end = (begin + span).min(vocabulary);
    let values = cx.arg_logits().row([row, 0]);
    let draw = cx.arg_draws().row([row, 0]);
    let mask = cx.arg_mask();
    // An unconstrained row admits every token; its mask row is never read.
    let masked = cx.arg_constrained().get([row]) != 0;
    let mut best = f32::NEG_INFINITY;
    let mut best_token = NO_TOKEN;
    let mut bad = 0u32;
    for (token, value) in values.iter().enumerate().take(end).skip(begin) {
        let value = *value;
        bad |= u32::from(value.is_nan() || value == f32::INFINITY);
        if !value.is_finite() {
            continue;
        }
        if masked && (mask.get([row, token / 32]) >> (token % 32)) & 1 == 0 {
            continue;
        }
        let token = token as u32;
        let score = score(value, token, draw);
        if better(score, token, best, best_token) {
            best = score;
            best_token = token;
        }
    }
    // SAFETY: each work item writes its own (row, part) triple.
    let out = unsafe { cx.scratch_partials().slice_mut::<u32>(12 * (row * parts + part), 3) };
    out.copy_from_slice(&[best.to_bits(), best_token, bad]);
}

fn sample_rows_merge<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let parts = cx.param_parts() as usize;
    // SAFETY: the partition launch wrote every triple before this launch.
    let partials = unsafe { cx.scratch_partials().slice::<u32>(12 * row * parts, 3 * parts) };
    let mut best = f32::NEG_INFINITY;
    let mut best_token = NO_TOKEN;
    let mut bad = 0u32;
    for partial in partials.chunks_exact(3) {
        let (score, token) = (f32::from_bits(partial[0]), partial[1]);
        bad |= partial[2];
        if better(score, token, best, best_token) {
            best = score;
            best_token = token;
        }
    }
    let status = if bad != 0 { 2 } else if best_token == NO_TOKEN { 1 } else { 0 };
    let result = cx.arg_result();
    // SAFETY: each work item writes its own row of the result.
    let out = unsafe { result.row_mut([row, 0]) };
    out[0] = if status == 0 { best_token as i32 } else { -1 };
    out[1] = status;
}
