// Shared pieces of the routed (mixture-of-experts) CPU entries, built on the
// projection library: the record an expert's weights are read from, the
// paired gate/up expansion with the routed SiLU . mul epilogue, the live-row
// count of a grouped block and the router's top-K selection (`ranked`). The
// counterpart of `metal/lib/routed/routed.h`, `cuda/lib/routed/routed.cuh`
// and `vulkan/lib/routed/routed.glsl`.
//
// Expert-stacked weights: an [E, N, K] tensor is a record port, bound as its
// records' address table (`Records`), and an expert's weights are the N rows
// of its record, wherever that record lies.
//
// Numerics (routed.seismic): the expansion accumulates gate and up in F32 and
// publishes their `glu` (functions.seismic); down projections are published
// to the activation element before they are weighted.

use super::super::core::{activation, functions, reduce};
use super::super::projection::projection;
use super::ranked;
use seismic::cpu::quant::Q8Block;
use seismic::cpu::slab::Records;
use seismic::cpu::{Dense, Scratch, Store, Weights};

/// An expert-stacked weight's records, each an expert's weights.
pub type Stack<'w> = Records<Weights<'w>>;

/// The weights of record `record` of `stack`.
#[inline(always)]
pub fn record<'w>(stack: &Stack<'w>, record: i32) -> Weights<'w> {
    stack.record(usize::try_from(record).expect("a routed record index is non-negative"))
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
    scales: (f32, f32),
    out: &mut [f32],
) {
    let mut up_sums = [0.0f32; projection::MAX_BLOCK];
    let up_sums = &mut up_sums[..out.len()];
    projection::project_arithmetic(gate, first, x, q8, out);
    projection::project_arithmetic(up, first, x, q8, up_sums);
    let pairs = out.iter_mut().zip(up_sums.iter()).map(|(target, up)| {
        let gate = *target;
        (target, gate, *up)
    });
    functions::glu_row::<A, _>(function, limit, scales, pairs, |target, value| *target = value);
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
    out: Store<'_, A>,
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
    mut out: Store<'_, A>,
) {
    let mut ups = [0.0f32; projection::MAX_ROWS];
    let ups = &mut ups[..out.len()];
    projection::project_arithmetic(up, first, x, q8, ups);
    for (target, up) in out.iter_mut().zip(ups.iter()) {
        target.set(functions::activated_value::<A>(function, scale * *up));
    }
}

/// The live rows of a grouped block: its `order` row is one expert's source
/// rows followed by -1 padding, so the count is the index of the first -1.
#[inline(always)]
pub fn block_rows(order: &[i32]) -> usize {
    order.partition_point(|row| *row >= 0)
}

/// The entry a map row holds for the step (`held`, routed.seismic): `map(w)`
/// its word w (MAP_RECORD, the MAP_FROM and MAP_UNTIL pairs, each low word
/// first), `step(w)` the step pair's word w; -1 unless MAP_FROM <= q <
/// MAP_UNTIL.
#[inline(always)]
pub fn held(map: impl Fn(usize) -> i32, step: impl Fn(usize) -> i32) -> i32 {
    let pair = |word: &dyn Fn(usize) -> i32, at: usize| u64::from(word(at) as u32) | (u64::from(word(at + 1) as u32) << 32);
    let q = pair(&step, crate::constants::STEP_LOW as usize);
    let record = map(crate::constants::MAP_RECORD as usize);
    let held = pair(&map, crate::constants::MAP_FROM as usize) <= q
        && q < pair(&map, crate::constants::MAP_UNTIL as usize);
    if held { record } else { -1 }
}

/// The resolved source of expert `expert`: `held(expert)` through the layer's
/// map (`mapped` > 0), or -2 - `expert` without a map (`mapped` == 0).
#[inline(always)]
pub fn expert_slot(mapped: u64, held: impl Fn(usize) -> i32, expert: i32) -> i32 {
    match mapped {
        0 => -2 - expert,
        _ => held(usize::try_from(expert).expect("a routed expert index is non-negative")),
    }
}

/// The weights at resolved `slot`: `None` when skipped (-1), record `slot` of
/// `records` (`slot` >= 0), or record -2 - `slot` of `experts` (`slot` <= -2).
#[inline(always)]
pub fn expert_weights<'w>(
    experts: &Stack<'w>,
    records: &Stack<'w>,
    slot: i32,
) -> Option<Weights<'w>> {
    match slot {
        -1 => None,
        0.. => Some(record(records, slot)),
        _ => Some(record(experts, -2 - slot)),
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
    /// The weights this place reads: its record of `bank` or of `tier`.
    #[inline(always)]
    pub fn of<'w>(self, bank: &Stack<'w>, tier: &Stack<'w>) -> Weights<'w> {
        match self {
            Place::Bank(index) => record(bank, index),
            Place::Tier(index) => record(tier, index),
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

/// Writes the grids of an expert pass over `used` blocks of `tile_rows`
/// rows (`routed_experts`) into `grids`, in launch order (expand, down):
/// `expand_items` items of each used block, `down_items` items of each used
/// block; and clears the used rows' arrival counters (`arrivals`, a word a
/// product row), which the expand launch's epilogue counts its stored
/// features on (`projection::emit`). The one work item of its launch writes
/// them.
#[inline(always)]
pub fn expert_grids(
    grids: [Scratch<'_>; 2],
    arrivals: Scratch<'_>,
    used: i32,
    tile_rows: usize,
    expand_items: usize,
    down_items: usize,
) {
    let values = [[expand_items as i32, used, 1], [down_items as i32, used, 1]];
    for (grid, value) in grids.iter().zip(values) {
        // SAFETY: each grid is a 12-byte scratch buffer the caller alone writes.
        unsafe { grid.slice_mut::<i32>(0, 3) }.copy_from_slice(&value);
    }
    let rows = usize::try_from(used).expect("a used block count is not negative") * tile_rows;
    // SAFETY: a word a product row of the used blocks, this item's alone.
    unsafe { arrivals.slice_mut::<u32>(0, rows) }.fill(0);
}

/// The arrival of `arrived` features an expand work item stored into
/// product row `row` (of `width` values, in `rows`): the row's last arrival
/// forms the down projection's operand of it in place, its Q8 blocks into
/// `quantized` where the INT8 form reads them and the NVFP4 transform where
/// `nvfp4` (the down weight takes NVFP4 operands), as
/// `projection::emit` forms an operand; a row that needs neither arrives
/// nowhere, nor does an item that stored none of it (a counter it would
/// read at its completed value).
///
/// # Safety
/// As `projection::emit`: the buffers are the entry's product rows, their
/// Q8 blocks and their cleared counters, and each feature of the row
/// arrives once, after its store.
#[inline(always)]
pub unsafe fn product_arrival(
    rows: Scratch<'_>,
    quantized: Option<Scratch<'_>>,
    nvfp4: bool,
    arrivals: Scratch<'_>,
    width: usize,
    row: usize,
    arrived: usize,
) {
    if (quantized.is_some() || nvfp4) && arrived > 0 {
        // SAFETY: the caller's.
        unsafe { projection::emit(rows, quantized, nvfp4, arrivals, width, row, arrived, |_| {}) };
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
