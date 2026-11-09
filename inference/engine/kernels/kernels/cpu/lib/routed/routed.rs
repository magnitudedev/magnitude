// Shared pieces of the routed (mixture-of-experts) CPU entries, built on the
// projection library: expert row offsets in expert-stacked weights, the
// paired gate/up expansion with the routed SiLU . mul epilogue, the live-row
// count of a grouped block and the router's top-K selection (`ranked`). The
// counterpart of `metal/lib/routed/routed.h`, `cuda/lib/routed/routed.cuh`
// and `vulkan/lib/routed/routed.glsl`.
//
// Expert-stacked weights: an [E, N, K] tensor is a `Weights` operand of E * N
// rows, so expert e's row n is row e * N + n.
//
// Numerics (routed.seismic): the expansion accumulates gate and up in F32 and
// publishes their `glu` (functions.seismic); down projections are published
// to the activation element before they are weighted.

use super::super::core::{activation, functions, reduce};
use super::super::projection::projection;
use super::ranked;
use seismic::cpu::quant::Q8Block;
use seismic::cpu::{Dense, Scratch, Weights};

/// First row of expert `expert` in an [E, N, K] tensor of `rows` = N rows per
/// expert.
#[inline(always)]
pub fn expert_row(expert: i32, rows: usize) -> usize {
    usize::try_from(expert).expect("a routed expert index is non-negative") * rows
}

/// The expansion of weight rows `first..first + out.len()` (at most
/// `projection::MAX_BLOCK`) of a paired gate/up operand against the F32
/// activation row `x` (or its quantized form `q8`), as published F32: the
/// `glu` of `gate_scale * gate` and `up_scale * up` (functions.seismic;
/// `functions` code `function`, [`functions::clamped_glu`] at `limit` where
/// the expansion has one; the scales: the expert's second-level scales on
/// the accumulators, 1 where a port is absent, which leaves every bit as it
/// is).
#[inline(always)]
pub fn expand<A: Dense>(
    gate: &Weights<'_>,
    up: &Weights<'_>,
    first: usize,
    x: &[f32],
    q8: Option<&[Q8Block]>,
    function: i32,
    limit: Option<f32>,
    (gate_scale, up_scale): (f32, f32),
    out: &mut [f32],
) {
    let mut up_sums = [0.0f32; projection::MAX_BLOCK];
    let up_sums = &mut up_sums[..out.len()];
    projection::project_arithmetic(gate, first, x, q8, out);
    projection::project_arithmetic(up, first, x, q8, up_sums);
    let pairs = out.iter_mut().zip(up_sums.iter());
    match limit {
        None => {
            for (target, up) in pairs {
                *target = functions::glu::<A>(function, gate_scale * *target, up_scale * up);
            }
        }
        Some(limit) => {
            for (target, up) in pairs {
                *target = functions::clamped_glu::<A>(function, limit, gate_scale * *target, up_scale * up);
            }
        }
    }
}

/// [`expand`] stored to a row span of `A` elements.
#[inline(always)]
pub fn expand_into<A: Dense>(
    gate: &Weights<'_>,
    up: &Weights<'_>,
    first: usize,
    x: &[f32],
    q8: Option<&[Q8Block]>,
    function: i32,
    limit: Option<f32>,
    scales: (f32, f32),
    out: &mut [A::Storage],
) {
    let mut values = [0.0f32; projection::MAX_ROWS];
    let values = &mut values[..out.len()];
    expand::<A>(gate, up, first, x, q8, function, limit, scales, values);
    activation::store::<A>(values, out);
}

/// A per-expert scale port's factor for `expert`: 1 where the port is
/// absent (`extent` 0), which leaves the accumulator's bits as they are.
#[inline(always)]
pub fn scale_factor(extent: u64, scale: impl Fn(usize) -> f32, expert: i32) -> f32 {
    match extent {
        0 => 1.0,
        _ => scale(usize::try_from(expert).expect("a routed expert index is non-negative")),
    }
}

/// The activated projection of weight rows `first..first + out.len()` of an
/// up-only operand against the F32 activation row `x` (or `q8`), stored to a
/// row span of `A` elements: `A(act(A(scale * up)))` (`scale`: the weight's
/// second-level scale on the accumulator).
#[inline(always)]
pub fn activated_into<A: Dense>(
    up: &Weights<'_>,
    first: usize,
    x: &[f32],
    q8: Option<&[Q8Block]>,
    function: i32,
    scale: f32,
    out: &mut [A::Storage],
) {
    let mut ups = [0.0f32; projection::MAX_ROWS];
    let ups = &mut ups[..out.len()];
    projection::project_arithmetic(up, first, x, q8, ups);
    for (target, up) in out.iter_mut().zip(ups.iter()) {
        *target = functions::activated::<A>(function, scale * *up);
    }
}

/// The live rows of a grouped block: its `order` row is one expert's source
/// rows followed by -1 padding, so the count is the index of the first -1.
#[inline(always)]
pub fn block_rows(order: &[i32]) -> usize {
    order.partition_point(|row| *row >= 0)
}

/// The record a map row holds for the step (`held`, routed.seismic): `map(w)`
/// its word w (MAP_RECORD, the MAP_FROM and MAP_UNTIL pairs, each low word
/// first), `step(w)` the step pair's word w; -1 unless it names a record
/// and MAP_FROM <= q < MAP_UNTIL.
#[inline(always)]
pub fn held(map: impl Fn(usize) -> i32, step: impl Fn(usize) -> i32) -> i32 {
    let pair = |word: &dyn Fn(usize) -> i32, at: usize| u64::from(word(at) as u32) | (u64::from(word(at + 1) as u32) << 32);
    let q = pair(&step, crate::constants::STEP_LOW as usize);
    let record = map(crate::constants::MAP_RECORD as usize);
    let held = record >= 0
        && pair(&map, crate::constants::MAP_FROM as usize) <= q
        && q < pair(&map, crate::constants::MAP_UNTIL as usize);
    if held { record } else { -1 }
}

/// The weights holding expert `expert`, and its index in them: the record
/// `held(expert)` of `records` while the layer's map (of `mapped` rows)
/// holds one for the step (>= 0), else its place in `experts` (the stack,
/// or the scratch tier), which is every expert's without a map.
#[inline(always)]
pub fn expert_weights<'a, 'w>(
    experts: &'a Weights<'w>,
    records: &'a Weights<'w>,
    mapped: u64,
    held: impl Fn(usize) -> i32,
    expert: i32,
) -> (&'a Weights<'w>, i32) {
    let slot = match mapped {
        0 => -1,
        _ => held(usize::try_from(expert).expect("a routed expert index is non-negative")),
    };
    match slot {
        0.. => (records, slot),
        _ => (experts, expert),
    }
}

/// Where a decode choice reads its weights (`decode_source`): a record of
/// the bank's stacks, or of the host tier's, read in place.
#[derive(Clone, Copy)]
pub enum Place {
    Bank(i32),
    Tier(i32),
}

impl Place {
    /// The stack of `bank` and `tier` this place reads, and its record.
    #[inline(always)]
    pub fn of<'a, T>(self, bank: &'a T, tier: &'a T) -> (&'a T, i32) {
        match self {
            Place::Bank(record) => (bank, record),
            Place::Tier(record) => (tier, record),
        }
    }
}

/// Where a decode expert entry reads a choice's weights from, of expert
/// `expert` (`decode_source`, routed.seismic): the expert itself without
/// sources (`rows` 0, a resident layer), else the step's source
/// (`source()`, the choice's SOURCE_RECORD word): a bank record (>= 0), a
/// tier record t as -2 - t; `None` where it names none (-1, a choice that
/// reads no record and contributes nothing; when is stated there).
#[inline(always)]
pub fn decode_source(rows: u64, source: impl Fn() -> i32, expert: i32) -> Option<Place> {
    match rows {
        0 => Some(Place::Bank(expert)),
        _ => match source() {
            -1 => None,
            record @ 0.. => Some(Place::Bank(record)),
            tier => Some(Place::Tier(-2 - tier)),
        },
    }
}

/// Writes the grids of an expert pass over `rows` grouped rows and `used`
/// blocks of `tile_rows` rows (`routed_experts`) into `grids`, in launch
/// order (stage, expand, quantize, down): a stage item per row,
/// `expand_items` items of each used block, a quantize item per block row,
/// `down_items` items of each used block. The one work item of its launch
/// writes them.
pub fn expert_grids(
    grids: [Scratch<'_>; 4],
    rows: i32,
    used: i32,
    tile_rows: usize,
    expand_items: usize,
    down_items: usize,
) {
    let values =
        [[rows, 1, 1], [expand_items as i32, used, 1], [used * tile_rows as i32, 1, 1], [down_items as i32, used, 1]];
    for (grid, value) in grids.iter().zip(values) {
        // SAFETY: each grid is a 12-byte scratch buffer the caller alone writes.
        unsafe { grid.slice_mut::<i32>(0, 3) }.copy_from_slice(&value);
    }
}

/// The route of one row from its router logits: the softmax over every
/// expert, then its K winners by `ranked::rank`, rank r stored at slot
/// K - 1 - r with its ranked probability; with `normalize` the scores are
/// divided by their slot-order sum; then each score is multiplied by its
/// expert's `expert_scale` factor ([`scale_factor`] over a port of `scales`
/// values). `probabilities` is E values and `order` E indices of work space.
#[inline(always)]
pub fn select(
    logits: &[f32],
    probabilities: &mut [f32],
    order: &mut [u32],
    normalize: bool,
    (scales, expert_scale): (u64, impl Fn(usize) -> f32),
    routes: &mut [i32],
    scores: &mut [f32],
) {
    let maximum = reduce::max(logits);
    let probabilities = &mut probabilities[..logits.len()];
    for (probability, logit) in probabilities.iter_mut().zip(logits) {
        *probability = (logit - maximum).exp();
    }
    let total = reduce::sum(probabilities);
    for probability in probabilities.iter_mut() {
        *probability /= total;
    }
    ranked::rank(order, |expert| probabilities[expert]);
    let k = routes.len();
    for (rank, winner) in order[..k].iter().enumerate() {
        routes[k - 1 - rank] = *winner as i32;
        scores[k - 1 - rank] = ranked::ranked(probabilities[*winner as usize]);
    }
    if normalize {
        let denominator = scores.iter().fold(0.0f32, |sum, score| sum + score);
        for score in scores.iter_mut() {
            *score /= denominator;
        }
    }
    for (score, expert) in scores.iter_mut().zip(routes.iter()) {
        *score *= scale_factor(scales, &expert_scale, *expert);
    }
}
