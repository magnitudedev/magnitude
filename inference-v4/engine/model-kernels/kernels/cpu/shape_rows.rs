// shape_rows on CPU (contract and portable body in sampling.seismic).
//
// `shape_rows_prepare` gives each work item one of PARTS contiguous
// vocabulary slices of a row: it writes the penalized and tempered values into
// `out` and keeps the slice maximum and a non-finite flag in scratch.
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

/// The order-preserving key of `value` (-0 and +0 share one key).
#[inline(always)]
fn key(value: f32) -> u32 {
    let bits = (value + 0.0).to_bits();
    if bits & 0x8000_0000 != 0 { !bits } else { bits | 0x8000_0000 }
}

/// exp(value - maximum) as a 64-bit 2^-32 fixed-point mass.
#[inline(always)]
fn mass(value: f32, maximum: f32) -> u64 {
    let weight = seismic::cpu::math::exp(value - maximum);
    if weight >= 1.0 { 1 << 32 } else { (weight * 4_294_967_296.0) as u64 }
}

/// ceil(top_p * total) exactly: top_p = m * 2^-shift with m a 24-bit
/// integer, so the product is an exact 128-bit integer.
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

fn shape_rows_prepare<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], shared: &mut [u8]) {
    let (part, row) = (group[0] as usize, group[1] as usize);
    let parts = cx.param_parts() as usize;
    let vocabulary = cx.dim_v() as usize;
    let span = vocabulary.div_ceil(parts);
    let begin = (part * span).min(vocabulary);
    let end = (begin + span).min(vocabulary);
    // SAFETY: each work item writes its own (row, part) pair.
    let partial = unsafe { cx.scratch_partials().slice_mut::<u32>(8 * (row * parts + part), 2) };
    if begin == end {
        partial.copy_from_slice(&[f32::NEG_INFINITY.to_bits(), 0]);
        return;
    }
    let s = shaping(cx, row);
    let logits = cx.arg_logits().row([row, 0]);
    let temper = |value: f32| if s.temperature != 0.0 { value / s.temperature } else { value };
    // SAFETY: each work item writes its own slice of its row.
    let values = unsafe { cx.arg_out().span_mut([row, begin], end - begin) };
    for (target, logit) in values.iter_mut().zip(&logits[begin..end]) {
        *target = temper(*logit);
    }
    // The history tokens of this slice, sorted, so each penalized token
    // counts its occurrences once.
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
        let mut value = logits[token];
        value = if value < 0.0 { value * s.repetition } else { value / s.repetition };
        value = value - s.presence - s.frequency * count as f32;
        values[token - begin] = temper(value);
        first += count;
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let nonfinite = values.iter().any(|value| value.is_nan() || *value == f32::INFINITY);
    partial.copy_from_slice(&[maximum.to_bits(), u32::from(nonfinite)]);
}

fn shape_rows_filter<L: Isa, E: Elements>(_l: L, cx: &Context<'_, E>, group: [u64; 3], _shared: &mut [u8]) {
    let row = group[0] as usize;
    let parts = cx.param_parts() as usize;
    let vocabulary = cx.dim_v() as usize;
    let s = shaping(cx, row);
    let (top_k, min_p, top_p) = (s.top_k > 0, s.min_p > 0.0, s.top_p < 1.0);
    if s.temperature == 0.0 || !(top_k || min_p || top_p) {
        return;
    }
    // SAFETY: the prepare launch wrote every pair before this launch.
    let partials = unsafe { cx.scratch_partials().slice::<u32>(8 * row * parts, 2 * parts) };
    let mut maximum = f32::NEG_INFINITY;
    let mut nonfinite = 0;
    for partial in partials.chunks_exact(2) {
        maximum = maximum.max(f32::from_bits(partial[0]));
        nonfinite |= partial[1];
    }
    if nonfinite != 0 || maximum == f32::NEG_INFINITY {
        return;
    }
    // SAFETY: each work item writes its own row.
    let values = unsafe { cx.arg_out().row_mut([row, 0]) };
    // SAFETY: each work item owns its row's region of the scratch.
    let order = unsafe { cx.scratch_order().slice_mut::<u64>(8 * row * vocabulary, vocabulary) };

    // Top-k: keep every value at least the k-th largest.
    let need = s.top_k.max(0) as usize;
    let threshold = if top_k && need <= vocabulary {
        for (slot, value) in order.iter_mut().zip(values.iter()) {
            *slot = u64::from(key(*value));
        }
        let (_, kth, _) = order.select_nth_unstable_by(need - 1, |a, b| b.cmp(a));
        *kth as u32
    } else {
        0
    };
    let survives = |value: f32| {
        key(value) >= threshold && !(min_p && seismic::cpu::math::exp(value - maximum) < s.min_p)
    };
    if !top_p {
        for value in values.iter_mut() {
            if !survives(*value) {
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
        if survives(*value) {
            order[survivors] = (u64::from(!key(*value)) << 32) | token as u64;
            survivors += 1;
            total += mass(*value, maximum);
        } else {
            *value = f32::NEG_INFINITY;
        }
    }
    let target = top_p_target(s.top_p, total);
    let ranked = &mut order[..survivors];
    ranked.sort_unstable();
    let mut before = 0u64;
    for entry in ranked.iter() {
        let token = (*entry & 0xffff_ffff) as usize;
        if before < target {
            before += mass(values[token], maximum);
        } else {
            values[token] = f32::NEG_INFINITY;
        }
    }
}
