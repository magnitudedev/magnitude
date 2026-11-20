//! What one kernel call costs on a device, from what it reads and computes
//! and the device's measured rates: the per-call formula every launch price
//! is built from.
//!
//! A call costs the slower of its bytes and its arithmetic, plus its
//! attention's fixed cost and its launches ([`CallSeconds`]):
//!
//! - bytes: weights (a routed call's distinct experts its rows read, as its
//!   caller states them, [`ExpertRead`]; of independent rows
//!   E · (1 − (1 − k/E)^N) slabs, [`expert_union`]), each format's at its own
//!   weight rate; other depth-independent bytes (rows, sequences' state,
//!   uploaded rows) at the activation format's rate; history (each
//!   sequence's read at its depth) along its layout's line
//!   ([`HistoryFigure`]), or at the activation rate where no attention entry
//!   reads it;
//! - arithmetic: the rows' matmul FLOPs over their format's throughput, in
//!   a decode form no faster than its row line ([`DecodeLine`]), and
//!   attention's over the activation format's;
//! - launches: the call's native launches times the launch cost.
//!
//! A launch of several calls ([`compose`]) costs its calls, each its
//! multiplicity of times, scaled by the measured composition ratio, plus the
//! host's step cost under its lookahead policy and each submission boundary
//! its calls end beyond its own at the round trip ([`LaunchSeconds`]).
//!
//! Formats and history layouts are keyed by their stable names
//! ([`Format`], [`Layout`]); what a rate is and how it was measured is the
//! caller's.

use std::fmt;

/// A weight format, by its element's unique storage name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Format(pub &'static str);

/// A history layout, by its codec's identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Layout(pub &'static str);

/// The storage kinds operands are allocated in: a graph run's inputs lie in
/// upload storage; host storage holds shared words and an expert bank's host
/// tier, which a streamed layer's device reads in place.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Storage {
    Device,
    Upload,
    Host,
}

impl Storage {
    /// Every kind, in declaration order.
    pub const ALL: [Self; 3] = [Self::Device, Self::Upload, Self::Host];
}

/// History one call reads for each token of context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryRead {
    /// The layout an attention entry reads it in; none for an indexer's
    /// F32 block keys and scores, which a selection call streams.
    pub layout: Option<Layout>,
    pub bytes_per_token: u64,
    /// FLOPs of one row per token read: attention's score and value
    /// products.
    pub flops_per_token: u64,
    /// The most tokens read: a window domain's window, or the blocks a
    /// selected read reads and its tail. `None` reads every token.
    pub window: Option<u64>,
}

impl HistoryRead {
    /// The tokens read at `depth` tokens of context.
    pub fn tokens_at(&self, depth: u32) -> u64 {
        self.window.map_or(u64::from(depth), |window| window.min(u64::from(depth)))
    }
}

/// Matmul FLOPs over weights resident in one format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flops {
    pub format: Format,
    pub flops: u64,
}

/// Weight bytes resident in one format, which its own weight rate prices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bytes {
    pub format: Format,
    pub bytes: u64,
}

/// The expert slabs a routed call reads: `selected` of `experts` equal
/// slabs per row, the union of the rows' choices once per call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Experts {
    pub experts: u64,
    pub selected: u64,
    /// One expert's bytes over every expert tensor the call reads, by
    /// format.
    pub slab_bytes: Vec<Bytes>,
    /// One expert's FLOPs per row over those tensors, by format.
    pub slab_flops: Vec<Flops>,
}

/// What one call reads and computes. Weights and per-sequence state are in
/// device storage; the host writes upload rows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Work {
    /// Weights read whole, once per call, by format.
    pub weights: Vec<Bytes>,
    pub experts: Option<Experts>,
    /// Device bytes each row reads and writes.
    pub row_bytes: u64,
    /// Upload bytes each row reads.
    pub upload_row_bytes: u64,
    /// Each row's matmul FLOPs, by weight format.
    pub row_flops: Vec<Flops>,
    /// Device state each sequence reads and writes, whatever its depth.
    pub sequence_bytes: u64,
    pub history: Option<HistoryRead>,
}

impl Work {
    /// The formats the call reads or computes in, whole weights' and
    /// experts' alike, repeated where several of its amounts are of one: the
    /// formats whose rates its price reads ([`Rates::covers`]).
    pub fn formats(&self) -> impl Iterator<Item = Format> + '_ {
        let experts = self.experts.iter().flat_map(|experts| {
            experts.slab_bytes.iter().map(|bytes| bytes.format).chain(experts.slab_flops.iter().map(|flops| flops.format))
        });
        self.weights.iter().map(|bytes| bytes.format).chain(self.row_flops.iter().map(|flops| flops.format)).chain(experts)
    }

    /// The history layout the call reads, if it reads one of a layout.
    pub fn layout(&self) -> Option<Layout> {
        self.history.and_then(|history| history.layout)
    }
}

/// Why a call or a launch has no price.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PriceError {
    /// A decode step of rows outside the decode row classes, beyond which
    /// prefill forms run.
    DecodeRows { rows: u64, decode_rows: u64 },
    /// A call's launches at its rows, which its declaration counts no one
    /// way: why.
    Launches(String),
    /// History no attention entry reads: why.
    History(String),
    /// The composition ratio: why it has no figure.
    Composition(String),
}

impl fmt::Display for PriceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DecodeRows { rows, decode_rows } => write!(
                formatter,
                "a decode step of {rows} rows is not one the decode demand describes (1 to {decode_rows} rows; beyond, prefill forms run)"
            ),
            Self::Launches(reason) | Self::History(reason) | Self::Composition(reason) => formatter.write_str(reason),
        }
    }
}

impl std::error::Error for PriceError {}

/// The cost of a history call at one depth: the segment of a layout's
/// ladder the depth lies in (the nearest where it lies outside), as a fixed
/// cost per call and the seconds each history byte read beyond it takes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoryLine {
    pub fixed_seconds: f64,
    pub seconds_per_byte: f64,
}

/// A history layout's ladder: one token's bytes, and each rung read, its
/// depth in tokens and one call's seconds there, each at least that of
/// every rung below it, as a call over more history takes no less time.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryLadder {
    per_token: u64,
    rungs: Vec<(f64, f64)>,
}

impl HistoryLadder {
    /// The ladder of `per_token` bytes a token whose rungs read `rungs`,
    /// `(depth, seconds)` strictly ascending in positive depth.
    pub fn new(per_token: u64, rungs: impl IntoIterator<Item = (u64, f64)>) -> Self {
        assert!(
            per_token > 0,
            "invariant violated: history ladder per_token must be positive"
        );
        let mut prev_depth = 0u64;
        let rungs = rungs
            .into_iter()
            .scan(0f64, |slowest, (depth, seconds)| {
                assert!(
                    depth > prev_depth,
                    "invariant violated: history ladder depths must be positive and strictly ascending ({depth} after {prev_depth})"
                );
                prev_depth = depth;
                *slowest = slowest.max(seconds);
                Some((depth as f64, *slowest))
            })
            .collect();
        Self { per_token, rungs }
    }

    /// The line of the segment `tokens` lies in. Its seconds per byte are
    /// the far rung's time beyond the near one's, none on a flat segment;
    /// its fixed cost the near time less its bytes' time, none where the
    /// line meets zero tokens below zero seconds. Where one rung was read
    /// the line runs through it from no fixed cost, and where none was, at
    /// `stream` bytes per second, which a call reads its history no faster
    /// than.
    pub fn line(&self, tokens: f64, stream: f64) -> HistoryLine {
        let per_token = self.per_token as f64;
        match self.rungs[..] {
            [] => HistoryLine { fixed_seconds: 0., seconds_per_byte: 1. / stream },
            [(depth, seconds)] => HistoryLine { fixed_seconds: 0., seconds_per_byte: seconds / (depth * per_token) },
            _ => {
                let segment = self.rungs.windows(2).position(|pair| tokens <= pair[1].0).unwrap_or(self.rungs.len() - 2);
                let [(near_depth, near), (far_depth, far)] = [self.rungs[segment], self.rungs[segment + 1]];
                let seconds_per_byte = (far - near) / ((far_depth - near_depth) * per_token);
                HistoryLine { fixed_seconds: (near - near_depth * per_token * seconds_per_byte).max(0.), seconds_per_byte }
            }
        }
    }
}

/// A history layout's figure: its ladder, measured along device storage's
/// first route at `along` bytes a second, its seconds a byte scaled up by
/// `slowdown`, the slowest route of the placement against that one, since
/// a kernel reads no faster than the path its history lies along.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryFigure {
    pub ladder: HistoryLadder,
    pub along: f64,
    pub slowdown: f64,
}

impl HistoryFigure {
    /// A call's line over `tokens` of the layout's history.
    pub fn line(&self, tokens: f64) -> HistoryLine {
        assert!(
            self.slowdown.is_finite() && self.slowdown > 0.0,
            "invariant violated: history figure slowdown must be finite and positive, got {}",
            self.slowdown
        );
        let line = self.ladder.line(tokens, self.along);
        HistoryLine { seconds_per_byte: line.seconds_per_byte / self.slowdown, ..line }
    }
}

/// The device's rates a call is priced at: each format's weight bytes a
/// second in each storage, each format's matrix FLOPs a second, the format
/// activations take (`activation`), each history layout's figure (or why
/// none reads it), a launch's seconds, and a decode form's row line where
/// it was measured (`decode`).
#[derive(Clone, Debug, PartialEq)]
pub struct Rates {
    pub activation: Format,
    pub weights: Vec<(Storage, Format, f64)>,
    pub flops: Vec<(Format, f64)>,
    pub history: Vec<(Layout, Result<HistoryFigure, PriceError>)>,
    pub launch: f64,
    pub decode: Option<DecodeLine>,
}

/// What a decode form's rows cost in arithmetic, whatever the weights'
/// format: a decode form (a GEMV over a few rows) issues each row's
/// activation words apart, so its arithmetic grows by a fixed time a FLOP
/// per row from its intercept (`fixed + row · rows` seconds a FLOP of one
/// row), not at the matrix units' throughput. Measured on a projection's
/// slope over its decode rows; none where it was not, a decode form's rows
/// then priced at the matrix throughput alone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeLine {
    pub fixed: f64,
    pub row: f64,
}

impl Rates {
    /// The bytes a second of `format` in `storage`.
    ///
    /// # Panics
    /// Where the rates hold none: a call is priced only at rates of its
    /// formats ([`Self::covers`]).
    pub fn weight_rate(&self, storage: Storage, format: Format) -> f64 {
        self.weights
            .iter()
            .find(|(at, read, _)| *at == storage && *read == format)
            .map(|&(_, _, rate)| rate)
            .unwrap_or_else(|| panic!("invariant violated: no rate of {format:?} in {storage:?}"))
    }

    /// The matrix FLOPs a second of `format`.
    ///
    /// # Panics
    /// Where the rates hold none ([`Self::covers`]).
    pub fn matrix_flops(&self, format: Format) -> f64 {
        self.flops
            .iter()
            .find(|(read, _)| *read == format)
            .map(|&(_, flops)| flops)
            .unwrap_or_else(|| panic!("invariant violated: no matrix throughput of {format:?}"))
    }

    /// The activation format's bytes a second in `storage`.
    pub fn activations(&self, storage: Storage) -> f64 {
        self.weight_rate(storage, self.activation)
    }

    /// `layout`'s history figure, or why no entry reads it.
    ///
    /// # Panics
    /// Where the rates hold no entry of it ([`Self::covers`]).
    pub fn history(&self, layout: Layout) -> Result<&HistoryFigure, PriceError> {
        let (_, figure) = self
            .history
            .iter()
            .find(|(read, _)| *read == layout)
            .unwrap_or_else(|| panic!("invariant violated: no history figure of {layout:?}"));
        figure.as_ref().map_err(Clone::clone)
    }

    /// Whether the rates price every format of `formats` in every storage
    /// and every layout of `layouts`.
    pub fn covers(&self, formats: &[Format], layouts: &[Layout]) -> bool {
        formats.iter().chain([&self.activation]).all(|&format| {
            Storage::ALL.iter().all(|&storage| self.weights.iter().any(|(at, read, _)| *at == storage && *read == format))
                && self.flops.iter().any(|(read, _)| *read == format)
        }) && layouts.iter().all(|layout| self.history.iter().any(|(read, _)| read == layout))
    }

    /// The seconds of `bytes` read from `storage`, each format's at its
    /// weight rate there, `scale` times over.
    pub fn weights_in(&self, storage: Storage, bytes: &[Bytes], scale: f64) -> f64 {
        bytes.iter().fold(0.0, |seconds, bytes| seconds + scale * bytes.bytes as f64 / self.weight_rate(storage, bytes.format))
    }

    /// The seconds of `bytes` in device storage, `scale` times over.
    pub fn weights(&self, bytes: &[Bytes], scale: f64) -> f64 {
        self.weights_in(Storage::Device, bytes, scale)
    }

    /// The seconds of `flops` over their format's throughput, `scale` times
    /// over.
    pub fn compute(&self, flops: &[Flops], scale: f64) -> f64 {
        flops.iter().fold(0.0, |seconds, flops| seconds + scale * flops.flops as f64 / self.matrix_flops(flops.format))
    }

    /// The seconds of `rows` rows of `flops` each in a decode form: no
    /// faster than the matrix throughput, nor than the decode row line
    /// where it was measured ([`DecodeLine`]).
    pub fn decode_compute(&self, flops: &[Flops], rows: f64) -> f64 {
        let matrix = self.compute(flops, rows);
        self.decode.map_or(matrix, |line| {
            let one = flops.iter().map(|flops| flops.flops as f64).sum::<f64>();
            matrix.max(one * (line.fixed + line.row * rows))
        })
    }

    /// `work`'s history figure where an attention entry reads its history;
    /// none where it reads none or no entry does, or why its layout has
    /// none.
    pub fn history_of(&self, work: &Work) -> Result<Option<&HistoryFigure>, PriceError> {
        match work.history.and_then(|history| history.layout) {
            Some(layout) => self.history(layout).map(Some),
            None => Ok(None),
        }
    }
}

/// What a launch is priced with beside its calls' rates: the composition
/// ratio its calls' sum is scaled by (or why it has none), and the host's
/// seconds a launch beyond its device time, pipelined (runs queued back to
/// back) and as one submit and wait.
#[derive(Clone, Debug, PartialEq)]
pub struct Figures {
    pub rates: Rates,
    pub composition: Result<f64, PriceError>,
    pub step_pipelined: f64,
    pub step_round_trip: f64,
}

impl Figures {
    /// The step cost a launch under `lookahead` pays.
    pub fn step_seconds(&self, lookahead: bool) -> f64 {
        match lookahead {
            true => self.step_pipelined,
            false => self.step_round_trip,
        }
    }
}

/// What a launch is priced with: the figures, and whether its decode runs
/// with lookahead (steps queued ahead) or each submitted and waited for.
#[derive(Clone, Copy, Debug)]
pub struct Pricing<'a> {
    pub figures: &'a Figures,
    pub lookahead: bool,
}

/// One call's time and its parts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallSeconds {
    /// Its bytes' time.
    pub bytes: f64,
    /// Its arithmetic's time.
    pub compute: f64,
    /// Its attention's fixed cost per call.
    pub fixed: f64,
    /// The native launches it dispatches.
    pub launches: u64,
    /// Its launches' time.
    pub launch: f64,
}

impl CallSeconds {
    /// The call's time: the slower of its bytes and its arithmetic, plus
    /// its fixed cost and its launches.
    pub fn total(&self) -> f64 {
        self.bytes.max(self.compute) + self.fixed + self.launch
    }

    /// Whether its arithmetic, not its bytes, bounds it.
    pub fn compute_bound(&self) -> bool {
        self.compute > self.bytes
    }
}

/// The expected slabs `rows` rows read of `experts` slabs, each row
/// choosing `selected` of them: every slab not chosen by any row is not
/// read, E · (1 − (1 − k/E)^rows), the union of their `rows · k` picks
/// (`picked_union`).
pub fn expert_union(experts: u64, selected: u64, rows: u64) -> f64 {
    picked_union(experts, selected, rows * selected)
}

/// The expected slabs `picks` picks read of `experts` slabs, picked `selected`
/// to a row: E · (1 − (1 − k/E)^(picks/k)), the union of `picks/k` rows'
/// choices, whole rows' or not.
fn picked_union(experts: u64, selected: u64, picks: u64) -> f64 {
    assert!(
        selected > 0 && selected <= experts,
        "invariant violated: selected experts ({selected}) must satisfy 0 < selected <= experts ({experts})"
    );
    let experts_f = experts as f64;
    let missed = 1.0 - selected as f64 / experts_f;
    experts_f * (1.0 - missed.powf(picks as f64 / selected as f64))
}

/// The distinct experts a routed call's rows read and the picks they
/// compute: what its caller knows of how its rows route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertRead {
    pub experts: f64,
    pub picks: f64,
}

impl ExpertRead {
    /// `rows` rows routing independently, each computing every pick
    /// ([`expert_union`]).
    pub fn independent(experts: &Experts, rows: u64) -> Self {
        Self::picked(experts, rows * experts.selected)
    }

    /// `picks` picks routing independently, `selected` to a row, each
    /// computed.
    pub fn picked(experts: &Experts, picks: u64) -> Self {
        Self { experts: picked_union(experts.experts, experts.selected, picks), picks: picks as f64 }
    }
}

/// The time of `work` at `rows` rows of `sequences` sequences and `depth`
/// tokens of context, dispatched in `launches` launches of its batch
/// (matrix) form, at `rates`, its history along `history` (its layout's
/// figure, [`Rates::history_of`]); a routed call's rows routing
/// independently, each computing every pick.
pub fn call_seconds(work: &Work, history: Option<&HistoryFigure>, rates: &Rates, rows: u64, sequences: u64, depth: u32, launches: u64) -> CallSeconds {
    let read = work.experts.as_ref().map(|experts| ExpertRead::independent(experts, rows));
    let compute = rates.compute(&work.row_flops, rows as f64);
    seconds(work, history, rates, rows, sequences, depth, launches, compute, read)
}

/// The time of `work` as [`call_seconds`] prices it, in a decode form
/// ([`Rates::decode_compute`]), a routed call's rows reading and computing
/// `read` (none exactly where it routes nothing).
#[allow(clippy::too_many_arguments)]
pub fn decode_call_seconds(work: &Work, history: Option<&HistoryFigure>, rates: &Rates, rows: u64, sequences: u64, depth: u32, launches: u64, read: Option<ExpertRead>) -> CallSeconds {
    let compute = rates.decode_compute(&work.row_flops, rows as f64);
    seconds(work, history, rates, rows, sequences, depth, launches, compute, read)
}

/// The time of `work` as [`call_seconds`] prices it, its rows' arithmetic
/// `compute` seconds, its routed experts read and computed as `read` (none
/// exactly where it routes nothing).
#[allow(clippy::too_many_arguments)]
fn seconds(work: &Work, history: Option<&HistoryFigure>, rates: &Rates, rows: u64, sequences: u64, depth: u32, launches: u64, mut compute: f64, read: Option<ExpertRead>) -> CallSeconds {
    let (rows_f, sequences) = (rows as f64, sequences as f64);
    let mut bytes = rates.weights(&work.weights, 1.0);
    match (&work.experts, read) {
        (Some(experts), Some(read)) => {
            bytes += rates.weights(&experts.slab_bytes, read.experts);
            compute += rates.compute(&experts.slab_flops, read.picks);
        }
        (None, None) => {}
        _ => panic!("invariant violated: a call's experts are read exactly where it routes"),
    }
    let activations = rows_f * work.row_bytes as f64 + sequences * work.sequence_bytes as f64;
    if activations > 0.0 {
        bytes += activations / rates.activations(Storage::Device);
    }
    let upload = rows_f * work.upload_row_bytes as f64;
    if upload > 0.0 {
        bytes += upload / rates.activations(Storage::Upload);
    }
    let mut fixed = 0.0;
    if let Some(read) = &work.history {
        let tokens = read.tokens_at(depth) as f64;
        let read_bytes = sequences * read.bytes_per_token as f64 * tokens;
        bytes += match history {
            Some(figure) => {
                let line = figure.line(tokens);
                fixed = line.fixed_seconds;
                read_bytes * line.seconds_per_byte
            }
            None => read_bytes / rates.activations(Storage::Device),
        };
        compute += rows_f * read.flops_per_token as f64 * tokens / rates.matrix_flops(rates.activation);
    }
    CallSeconds { bytes, compute, fixed, launches, launch: launches as f64 * rates.launch }
}

/// The calls of `works`' time at one row and `depth` tokens alone, each a
/// launch: what a composition chain's calls are priced at alone.
pub fn calls_seconds(works: &[Work], rates: &Rates, depth: u32) -> Result<f64, PriceError> {
    works.iter().try_fold(0.0, |seconds, work| Ok(seconds + call_seconds(work, rates.history_of(work)?, rates, 1, 1, depth, 1).total()))
}

/// A call's time in a launch and the calls of it the launch makes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallTerm {
    pub seconds: CallSeconds,
    pub multiplicity: u64,
}

/// One launch's time and its terms: each call kind's time and multiplicity,
/// the ratio their sum is scaled by, composed in a launch, the host's step
/// cost under the launch's lookahead policy, and its submission boundaries'
/// round trips.
#[derive(Clone, Debug, PartialEq)]
pub struct LaunchSeconds {
    pub calls: Vec<CallTerm>,
    pub composition: f64,
    pub step: f64,
    pub boundaries: f64,
}

impl LaunchSeconds {
    /// The launch's time, derived from its terms: its calls composed, each
    /// its multiplicity of times, plus the step cost and its boundaries'
    /// round trips.
    pub fn total(&self) -> f64 {
        self.composition * self.calls.iter().map(|call| call.multiplicity as f64 * call.seconds.total()).sum::<f64>() + self.step + self.boundaries
    }
}

/// A launch of `calls` ending `boundaries` submissions beyond its own,
/// priced at `pricing`: none where the composition ratio has no figure.
pub fn compose(calls: Vec<CallTerm>, boundaries: u64, pricing: &Pricing<'_>) -> Result<LaunchSeconds, PriceError> {
    Ok(composed(calls, boundaries, pricing, pricing.figures.composition.clone()?))
}

/// [`compose`] at the composition ratio `composition`, read already.
fn composed(calls: Vec<CallTerm>, boundaries: u64, pricing: &Pricing<'_>, composition: f64) -> LaunchSeconds {
    let figures = pricing.figures;
    LaunchSeconds { calls, composition, step: figures.step_seconds(pricing.lookahead), boundaries: boundaries as f64 * figures.step_round_trip }
}
