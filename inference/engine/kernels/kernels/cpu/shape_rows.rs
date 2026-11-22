// shape_rows on CPU (contract and portable body in sampling.seismic).
//
// `shape_rows_prepare` has one work item per pool participant: the rows'
// logits, flattened, split into equal spans (`lib/core/flat`). For each row's
// piece of its span, an item rewrites the piece's logits in place as their
// penalized and tempered values (-inf where a constrained row's mask rejects
// a finite value), reading each logit once, and keeps the piece's maximum and
// a non-finite flag in the piece's slot; the last item to finish a row's
// pieces (`lib/core/arrive`) combines them into the row's `maxima` record.
// `shape_rows_filter` gives each work item one row: it finds the top-k
// threshold (the k-th largest value; every tie of it is kept) by selection,
// applies min-p, and ranks the top-k and min-p survivors by descending value
// and ascending token for top-p, writing -inf over every removed token.
//
// Decisions are those of the Metal, CUDA and Vulkan forms bit for bit: values
// are compared through the order-preserving key of `v + 0` (one tie group for
// -0 and +0), top-p masses are exp(v - max) quantized to 2^-32 and summed as
// integers, and a token survives top-p while the mass before it is below
// ceil(top_p * total). A row whose tempered values contain NaN or +inf, or
// whose maximum is -inf, is left unfiltered (sampling reports it).

use lib::core::functions::finite_ordered_key;
use lib::core::{arrive, flat};

/// Quantizes a precomputed `exp(value - maximum)` weight to 2^-32 fixed-point mass.
#[inline(always)]
fn mass_from_weight(weight: f32) -> u64 {
    if weight >= 1.0 { 1 << 32 } else { (weight * 4_294_967_296.0) as u64 }
}

/// exp(diff) as a 64-bit 2^-32 fixed-point mass.
///
/// For `diff < -25.0`, `exp(-25.0) * 2^32 < 0.06 < 1.0`, so the quantized
/// `u64` mass is provably `0` without evaluating `exp`.
#[inline(always)]
fn mass_from_diff(diff: f32) -> u64 {
    if diff < -25.0 {
        return 0;
    }
    mass_from_weight(seismic::cpu::math::exp(diff))
}

/// ceil(top_p * total) exactly: top_p = m * 2^-shift with m a 24-bit
/// integer, so the product is an exact 128-bit integer.
#[inline(always)]
fn top_p_target(top_p: f32, total: u64) -> u64 {
    if !(top_p > 0.0) || total == 0 {
        return 0;
    }
    let bits = top_p.to_bits();
    let biased = bits >> 23;
    let (m, shift) = if biased == 0 {
        (u128::from(bits & 0x7f_ffff), 149i32)
    } else {
        (u128::from((bits & 0x7f_ffff) | 0x80_0000), 150 - biased as i32)
    };
    let product = m * u128::from(total);
    if shift <= 0 {
        return (product << -shift) as u64;
    }
    if shift >= 128 {
        return 1;
    }
    let quotient = product >> shift;
    let remainder = product & ((1u128 << shift) - 1) != 0;
    (quotient + u128::from(remainder)) as u64
}

/// The row's sampling parameters.
struct Shaping {
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    repetition: f32,
    presence: f32,
    frequency: f32,
}

#[inline(always)]
fn shaping<E: Elements>(cx: &Context<'_, E>, row: usize) -> Shaping {
    let params = cx.arg_params();
    let p = |column: usize| params.get([row, column]);
    Shaping {
        temperature: p(0),
        top_k: p(1) as i32,
        top_p: p(2),
        min_p: p(3),
        repetition: p(4),
        presence: p(5),
        frequency: p(6),
    }
}

#[inline(always)]
fn shape_rows_prepare<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (span, spans) = (group[0] as usize, cx.groups()[0] as usize);
    let (rows, vocabulary) = (cx.dim_sx() as usize, cx.dim_v() as usize);
    // Rows without a token have nothing to shape (`shape_rows_filter`).
    let Some(flat) = flat::Flat::new(rows, vocabulary, spans) else { return };
    for piece in flat.pieces(span) {
        let row = piece.row;
        let (maximum, nonfinite) = shape_piece(cx, row, piece.columns, shared);
        // SAFETY: each piece's slot is its own (`Flat`).
        unsafe { cx.scratch_partials().slice_mut::<u32>(8 * piece.slot, 2) }
            .copy_from_slice(&[maximum.to_bits(), u32::from(nonfinite)]);
        let pieces = flat.spans_of(row);
        // SAFETY: one counter a row of the launch's `arrivals` scratch, each
        // span holding the row's values arriving once, after its partial.
        let last = unsafe { arrive::last(cx.scratch_arrivals().pointer::<u32>(4 * row), pieces.len() as u32) };
        if last {
            // SAFETY: every piece of the row stored its partial before it
            // arrived; the last writes the row's record alone.
            let (partials, record) = unsafe {
                (
                    cx.scratch_partials().slice::<u32>(8 * (pieces.start + row), 2 * pieces.len()),
                    cx.scratch_maxima().slice_mut::<u32>(8 * row, 2),
                )
            };
            let mut maximum = f32::NEG_INFINITY;
            let mut nonfinite = 0;
            for partial in partials.chunks_exact(2) {
                maximum = maximum.max(f32::from_bits(partial[0]));
                nonfinite |= partial[1];
            }
            record.copy_from_slice(&[maximum.to_bits(), nonfinite]);
        }
    }
}

/// Shapes `columns` of row `row` in place (the history tokens among them
/// staged in `shared`): their maximum, and whether any is NaN or +inf.
#[inline(always)]
fn shape_piece<E: Elements>(cx: &Context<'_, E>, row: usize, columns: std::ops::Range<usize>, shared: &mut [u8]) -> (f32, bool) {
    let (begin, end) = (columns.start, columns.end);
    let s = shaping(cx, row);
    // SAFETY: a piece is its span's alone, so each work item rewrites its
    // own columns of the row.
    let mut values = unsafe { cx.arg_logits().span_mut([row, begin], end - begin) };
    values.compute(|values| {
        // The history tokens of this slice, sorted, so each penalized token
        // counts its occurrences once. Penalties apply to the raw logits before
        // every value of the slice is tempered.
        let history = cx.arg_history().row([row, 0]);
        // SAFETY: every bit pattern is a `u32`; the pool aligns private bytes
        // for any scalar, so the middle part starts at the first byte.
        let (head, recent, _) = unsafe { shared.align_to_mut::<u32>() };
        assert!(head.is_empty() && recent.len() >= history.len(), "the launch declares Hn words of private bytes");
        let mut marked = 0;
        for token in history {
            if *token >= begin as i32 && (*token as usize) < end {
                recent[marked] = *token as u32;
                marked += 1;
            }
        }
        let recent = &mut recent[..marked];
        recent.sort_unstable();
        let mut first = 0;
        while first < recent.len() {
            let token = recent[first] as usize;
            let count = recent[first..].iter().take_while(|other| **other as usize == token).count();
            let value = &mut values[token - begin];
            *value = if *value < 0.0 { *value * s.repetition } else { *value / s.repetition };
            *value = *value - s.presence - s.frequency * count as f32;
            first += count;
        }
        // Divide by temperature, apply any constrained mask, and collect the
        // piece's maximum and non-finite flag in a single pass. A constrained
        // row removes every rejected value but NaN and +inf, which keep the
        // row unfiltered for sampling to report.
        let mut maximum = f32::NEG_INFINITY;
        let mut nonfinite = false;
        let tempered = s.temperature != 0.0;
        if cx.arg_constrained().get([row]) != 0 {
            let mask = cx.arg_mask();
            for (token, slot) in (begin..end).zip(values.iter_mut()) {
                let mut v = *slot;
                if tempered {
                    v /= s.temperature;
                }
                if v < f32::INFINITY {
                    if (mask.get([row, token / 32]) >> (token % 32)) & 1 == 0 {
                        v = f32::NEG_INFINITY;
                    }
                } else {
                    nonfinite = true;
                }
                *slot = v;
                maximum = maximum.max(v);
            }
        } else if tempered {
            let temperature = s.temperature;
            for slot in values.iter_mut() {
                let v = *slot / temperature;
                *slot = v;
                maximum = maximum.max(v);
                nonfinite |= !(v < f32::INFINITY);
            }
        } else {
            for &v in values.iter() {
                maximum = maximum.max(v);
                nonfinite |= !(v < f32::INFINITY);
            }
        }
        (maximum, nonfinite)
    })
}

#[inline(always)]
fn shape_rows_filter<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let vocabulary = cx.dim_v() as usize;
    let s = shaping(cx, row);
    let (top_k, min_p, top_p) = (s.top_k > 0, s.min_p > 0.0, s.top_p < 1.0);
    if vocabulary == 0 || s.temperature == 0.0 || !(top_k || min_p || top_p) {
        return;
    }
    // SAFETY: the prepare launch wrote every row's record before this
    // launch.
    let record = unsafe { cx.scratch_maxima().slice::<u32>(8 * row, 2) };
    let (maximum, nonfinite) = (f32::from_bits(record[0]), record[1]);
    if nonfinite != 0 || maximum == f32::NEG_INFINITY {
        return;
    }
    // SAFETY: each work item writes its own row.
    let mut values = unsafe { cx.arg_logits().row_mut([row, 0]) };
    values.compute(|values| {
        // SAFETY: each work item owns its row's region of the scratch.
        let order = unsafe { cx.scratch_order().slice_mut::<u64>(8 * row * vocabulary, vocabulary) };

        // Top-k: keep every value at least the k-th largest.
        let need = s.top_k.max(0) as usize;
        let threshold = if top_k && need <= vocabulary {
            for (slot, value) in order.iter_mut().zip(values.iter()) {
                *slot = u64::from(finite_ordered_key(*value));
            }
            let (_, kth, _) = order.select_nth_unstable_by(need - 1, |a, b| b.cmp(a));
            *kth as u32
        } else {
            0
        };
        // Precompute a conservative guard band around `ln(min_p)` in
        // `value - maximum` space so tokens clearly below or above the
        // threshold avoid evaluating `exp`.
        let (min_p_lo, min_p_hi) = if min_p {
            let ln_min_p = s.min_p.ln();
            let margin = 1e-3 * ln_min_p.abs().max(1.0);
            (ln_min_p - margin, ln_min_p + margin)
        } else {
            (f32::NEG_INFINITY, f32::NEG_INFINITY)
        };
        // Returns `None` if rejected by `min_p`, `Some(None)` if accepted
        // without evaluating `exp`, or `Some(Some(weight))` if `exp(diff)` was
        // evaluated inside the guard band and accepted.
        let check_min_p = |diff: f32| -> Option<Option<f32>> {
            if !min_p {
                return Some(None);
            }
            if diff < min_p_lo {
                return None;
            }
            if diff > min_p_hi {
                return Some(None);
            }
            let weight = seismic::cpu::math::exp(diff);
            if weight < s.min_p { None } else { Some(Some(weight)) }
        };
        if !top_p {
            for value in values.iter_mut() {
                if finite_ordered_key(*value) < threshold
                    || (min_p && check_min_p(*value - maximum).is_none())
                {
                    *value = f32::NEG_INFINITY;
                }
            }
            return;
        }

        // Top-p over the survivors, ranked by descending key then ascending
        // token: (!key, token) ascending.
        let mut survivors = 0;
        let mut total = 0u64;
        for (token, value) in values.iter_mut().enumerate() {
            let key = finite_ordered_key(*value);
            if key < threshold {
                *value = f32::NEG_INFINITY;
                continue;
            }
            let diff = *value - maximum;
            let Some(weight) = check_min_p(diff) else {
                *value = f32::NEG_INFINITY;
                continue;
            };
            order[survivors] = (u64::from(!key) << 32) | token as u64;
            survivors += 1;
            total += match weight {
                Some(w) => mass_from_weight(w),
                None => mass_from_diff(diff),
            };
        }
        let target = top_p_target(s.top_p, total);
        let ranked = &mut order[..survivors];
        ranked.sort_unstable();
        let mut before = 0u64;
        for entry in ranked.iter() {
            let token = (*entry & 0xffff_ffff) as usize;
            if before < target {
                before += mass_from_diff(values[token] - maximum);
            } else {
                values[token] = f32::NEG_INFINITY;
            }
        }
    });
}
