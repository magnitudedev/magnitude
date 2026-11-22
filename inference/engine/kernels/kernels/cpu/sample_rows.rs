// sample_rows on CPU (contract in sampling.seismic, portable body in
// seismic-std). One launch, `sample_rows_partition`, of one work item per
// pool participant: the rows' logits, flattened, split into equal spans
// (`lib/core/flat`), each item keeping the winner (score bits, token, invalid
// flag) of each row's piece of its span in that piece's slot. The last item to
// finish a row's pieces (`lib/core/arrive`) combines them in span order and
// writes the row's result. The maximum score wins, ties go to the lowest
// token and the winning score is carried unchanged, so the result does not
// depend on the split.
//
// Counter (token, draws[3], draws[4], draws[5]), key (draws[1], draws[2]);
// draws[0] == 1 enables the Gumbel perturbation v - log(-log u) with
// u = ((x >> 9) + 0.5) * 2^-23. A token competes when its logit is finite
// and its row is unconstrained or its mask bit is set. Status 0 success,
// 1 no competing token, 2 some logit of the row is NaN or +inf.

use lib::core::{arrive, flat};
use seismic::cpu::reduce::LANES;

const NO_TOKEN: u32 = u32::MAX;

/// The Philox4x32-10 Gumbel score of `value` at `token` (the portable body's
/// perturbation).
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

/// Whether candidate (score, token) beats (best, best_token): a higher
/// score, then a lower token.
#[inline(always)]
fn better(score: f32, token: u32, best: f32, best_token: u32) -> bool {
    best_token == NO_TOKEN || (token != NO_TOKEN && (score > best || (score == best && token < best_token)))
}

/// Branch-light greedy scan over a contiguous slice of logits starting at
/// token `start`: tracks the maximum finite score (lowest token on ties) and
/// whether any value is NaN or +inf.
#[inline(always)]
fn greedy_unconstrained(values: &[f32], start: usize) -> (f32, u32, u32) {
    let mut best_val = [f32::NEG_INFINITY; LANES];
    let mut best_idx = [NO_TOKEN; LANES];
    let mut bad_lanes = [0u32; LANES];
    let mut chunks = values.chunks_exact(LANES);
    let mut base = start as u32;
    for chunk in &mut chunks {
        for lane in 0..LANES {
            let value = chunk[lane];
            bad_lanes[lane] |= u32::from(value.is_nan() || value == f32::INFINITY);
            if value > best_val[lane] {
                best_val[lane] = value;
                best_idx[lane] = base + lane as u32;
            }
        }
        base += LANES as u32;
    }
    let mut best = f32::NEG_INFINITY;
    let mut best_token = NO_TOKEN;
    let mut bad = 0u32;
    for lane in 0..LANES {
        bad |= bad_lanes[lane];
        if better(best_val[lane], best_idx[lane], best, best_token) {
            best = best_val[lane];
            best_token = best_idx[lane];
        }
    }
    for (offset, &value) in chunks.remainder().iter().enumerate() {
        bad |= u32::from(value.is_nan() || value == f32::INFINITY);
        if value > best {
            best = value;
            best_token = base + offset as u32;
        }
    }
    (best, best_token, bad)
}

#[inline(always)]
fn sample_rows_partition<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let (span, spans) = (group[0] as usize, cx.groups()[0] as usize);
    let (rows, vocabulary) = (cx.dim_m() as usize, cx.dim_v() as usize);
    let Some(flat) = flat::Flat::new(rows, vocabulary, spans) else {
        // Rows without a token have none competing: the first item reports
        // each.
        if span == 0 {
            (0..rows).for_each(|row| publish(cx, row, NO_TOKEN, 0));
        }
        return;
    };
    let (mask, constrained) = (cx.arg_mask(), cx.arg_constrained());
    for piece in flat.pieces(span) {
        let row = piece.row;
        let values = cx.arg_logits().row([row, 0]);
        let draw = cx.arg_draws().row([row, 0]);
        // An unconstrained row admits every token; its mask row is never read.
        let noisy = draw[0] == 1;
        let masked = constrained.get([row]) != 0;
        let (best, best_token, bad) = if !noisy && !masked {
            greedy_unconstrained(&values[piece.columns.clone()], piece.columns.start)
        } else {
            let mut best = f32::NEG_INFINITY;
            let mut best_token = NO_TOKEN;
            let mut bad = 0u32;
            for token in piece.columns {
                let value = values[token];
                bad |= u32::from(value.is_nan() || value == f32::INFINITY);
                if !value.is_finite() {
                    continue;
                }
                if masked && (mask.get([row, token / 32]) >> (token % 32)) & 1 == 0 {
                    continue;
                }
                let token = token as u32;
                let score = if noisy { gumbel(value, token, draw) } else { value };
                if better(score, token, best, best_token) {
                    best = score;
                    best_token = token;
                }
            }
            (best, best_token, bad)
        };
        // SAFETY: each piece's slot is its own (`Flat`).
        unsafe { cx.scratch_partials().slice_mut::<u32>(12 * piece.slot, 3) }.copy_from_slice(&[best.to_bits(), best_token, bad]);
        let pieces = flat.spans_of(row);
        // SAFETY: one counter a row of the launch's `arrivals` scratch, each
        // span holding the row's values arriving once, after its partial.
        let last = unsafe { arrive::last(cx.scratch_arrivals().pointer::<u32>(4 * row), pieces.len() as u32) };
        if last {
            // SAFETY: every piece of the row stored its partial before it
            // arrived.
            let partials = unsafe { cx.scratch_partials().slice::<u32>(12 * (pieces.start + row), 3 * pieces.len()) };
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
            publish(cx, row, best_token, bad);
        }
    }
}

/// Row `row`'s result from its winning token (`NO_TOKEN`: none competed)
/// and invalid flag `bad`.
#[inline(always)]
fn publish<E: Elements>(cx: &Context<'_, E>, row: usize, best_token: u32, bad: u32) {
    let status = if bad != 0 { 2 } else if best_token == NO_TOKEN { 1 } else { 0 };
    let result = cx.arg_result();
    // SAFETY: one work item, the last of its pieces, writes a row's result.
    let out = unsafe { result.row_mut([row, 0]) };
    out[0] = if status == 0 { best_token as i32 } else { -1 };
    out[1] = status;
}
