//! What the engine's own kernels achieve on an executor (spec §2.14): for
//! each weight format a load can hold resident, the weight bytes its decode
//! projection streams per second along each allocation route of its
//! placements and the FLOPs per second of its projection over many rows; what its
//! decode attention costs per call and per history byte over each history
//! layout, how much slower its calls run composed in a step than alone,
//! what a launch costs inside a sealed graph, and what a step costs on the
//! host beyond its device time.
//!
//! The costs are made of measured points (spec §2.6): a kernel's calls, or a
//! sealed chain of them. One deterministic walk ([`walk`]) names them, in
//! order, and derives the costs from their timings: the measurement job
//! measures it ([`ExecutorCosts::measure`]), loading each point's record or
//! timing and storing it, and every start retraces it from the records alone
//! ([`ExecutorCosts::stored`]), so what a start derives from is exactly what
//! the job measured. The costs themselves are never stored.
//!
//! Every kernel is formed at its declaration's default specialization (the
//! fallback every load runs until tuning confirms something faster, so what
//! loads run is at least as fast) at the backend's resident layout, bf16
//! activations and H = 4096; the attention geometry is the catalog's
//! H = 4096 model's (Qwen3.5-9B: 4 key/value heads of 4 query heads, 256
//! wide, 64 of them rotated). Weights and activations are zeros: the
//! kernels' time does not depend on their values, and zeros hold no
//! subnormal a CPU would slow on.
//!
//! The kernels are few, and each is formed once per job ([`Programs`]): two
//! `dense_expand` programs per format at H = 4096 and the executor's F
//! ([`Sizes`]), one held to the decode projection's one row and one to the
//! matrix throughput's rows ([`Rows`]: each forms only the launches its
//! rows select), one `dense_output` at the
//! composition chain's shape, the decode attention over each history
//! layout, and `conditioning_overlay`. Every route, size and depth reuses
//! them. Forming a kernel is a compile (0.9 s on an M4 Pro's Metal, 1.5 to
//! 2.9 s through NVRTC on an RTX 3090 and a GB10, each about 0.01 s from
//! the kernel cache), so the job forms them all in parallel on the host's
//! threads, through the kernel cache, from when the first point no record
//! holds is reached, in the order the walk reads them, while the walk goes
//! on: each point waits only for its own kernels. The points are timed one
//! after another on the device, which a compile does not share; those that
//! touch the host (a route of host storage, the launch graph's host times)
//! come last, once every compile has ended.
//!
//! The history's depth is searched from the depth the one-token call and
//! the activation dtype's weight rate predict meets the rule (one call
//! takes at least 100 launch costs and its history at least its fixed
//! part), each depth its own point of one argument set (unrotated),
//! doubling until a measured call meets it. That search only finds the
//! depth. A streaming kernel's figure is timed with its calls rotating
//! through as many argument sets as span four times the largest cache that
//! can serve them ([`CACHE_MULTIPLE`]), so the bytes come from memory: a
//! projection's sets hold megabytes to tens of megabytes each, and a
//! history's, at a depth meeting the 100-launch rule, at least the bytes it
//! reads in 50 launch costs, so a rotation's sets stay few. A rotation at
//! one token of dense history (4 KiB) would take 262,144 sets over a 256
//! MiB cache, more than a device keeps resident (m4-pro's Metal driver
//! refused that rotation's views to its residency set, about 370,000
//! resources). Every rotated operand of every point is a view of one
//! zero-filled backing per route ([`Backings`]), allocated once a job, so a
//! point's making allocates and fills none of its rotation; the history
//! rows a point appends are zeroed again once it is timed.
//!
//! A weight format, history layout or upload-only route whose points fail,
//! or that the job records unobtainable, is unmeasured with its reason, and
//! what it would price is unavailable; the other figures stand.

use super::AssessmentError;
use crate::{resident_element, resident_elements, resident_layout, ExecutionPath};
use magnitude_artifacts::gguf::Encoding;
use magnitude_batching::{ClassLimits, Demand, PackedRowTables, Row, RowHistory, Slot};
use magnitude_family_contracts::Rotary;
use magnitude_kernels::{
    attention_decode, attention_decode_k8v4, conditioning_overlay, dense_expand, dense_output,
};
use magnitude_state::{HistoryEncoding, KvCodec};
use seismic::measure::{Attempt, Measure, Outcome, Reason, Record, RecordStore, Timing, Unformed};
use seismic::{
    AccessPathId, AccessRoute, BoundEntry, CallError, DType, Device, DeviceTopology, Element,
    Entry, Executor, GraphTiming, Identity, KernelIdentity, Measurer, NativeKernel,
    NativeSpecialization, Point, PointKey, Reported, ReportedBy, SlabLayout, SlabRegion,
    SlabTensor, StorageKind, Tensor,
};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// The format the composition chain's projections are resident in: the
/// reference model's (Qwen3.5-9B at Q4).
const CHAIN_FORMAT: Encoding = Encoding::Q4K;

/// How large the walk's projections are on an executor: the F of every
/// `dense_expand` program, where the executor's largest allocation holds a
/// weight of it (else the largest power of two below that does), and the
/// rows of the matrix throughput's.
///
/// On a GPU, F is 16384: at one row a call reads two Q4_K weights of 16384
/// × 4096 (75 MB), 128 µs at the 588 GB/s an RTX 3090 streams and 321 µs at
/// an M4 Pro's 235 GB/s, against their launch costs of 1.06 and 1.53 µs, so
/// launching is under 1% of the call (0.8% and 0.5%). Its arithmetic reaches
/// its throughput only over many rows, which reuse each weight: 512, the
/// prefill chunk loads run (`programs::invocation_demand`, the largest
/// tuning row class within the standard service's 512 launch rows).
///
/// The CPU's kernels are arithmetic where a GPU's stream: on a 4-vCPU EPYC
/// 7763 (2026-10-21) a 512-row call of the default specialization took
/// about 512 one-row calls (Q4G32S 33 s), 4 to 52 s each, so the 17 formats'
/// throughputs took most of an hour. Its throughput is reached at the 8
/// rows a decode step is priced at most (every format within 5% of its 512
/// × 16384 figure), and F 4096, the reference H, is as far from its 46 ns
/// launch: a one-row call reads 19 MB, at least 2.4 ms. Every projection
/// point is then a quarter as long, and a throughput's a 256th.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sizes {
    features: u64,
    matrix_rows: u64,
}

impl Sizes {
    /// The sizes of `backend`'s points.
    fn of(backend: seismic::BackendName) -> Self {
        match backend {
            seismic::BackendName::Cpu => Self {
                features: HIDDEN,
                matrix_rows: crate::operators::attention::graph::DECODE_ROWS,
            },
            _ => Self {
                features: 16384,
                matrix_rows: 512,
            },
        }
    }
}

/// The representation `backend` keeps [`CHAIN_FORMAT`] weights resident as.
fn chain_element(backend: seismic::BackendName) -> Option<Element> {
    resident_element(
        CHAIN_FORMAT,
        HIDDEN,
        DType::BF16,
        resident_layout(ExecutionPath::Native, backend),
    )
}

/// The measured formats: every element a load on `backend` holds weights
/// resident as ([`resident_elements`]), whatever any model reads, so the
/// job's points are fixed for the executor and every model's speed is
/// arithmetic over them. The activation dtype's (bf16, which also prices
/// the bytes no weight format holds: rows, state and uploads) and the
/// chain's ([`CHAIN_FORMAT`]) are among them.
fn measured_formats(backend: seismic::BackendName) -> Vec<Element> {
    resident_elements(resident_layout(ExecutionPath::Native, backend))
}

/// The history layouts whose decode attention is measured: every codec
/// whose history a native attention entry reads (`HistoryEncoding::Dense`
/// and `Affine`); rotated K4/V4 has none, so no load stores it.
pub const HISTORY_LAYOUTS: [KvCodec; 2] = [KvCodec::Dense, KvCodec::AffineK8V4];

/// The hidden width of every reference kernel: the catalog's H = 4096
/// model's (Qwen3.5-9B).
const HIDDEN: u64 = 4096;

/// The calls of the launch graph and of the composition chain: the regime a
/// step is priced in (a decode step makes 100 to 300 calls).
const GRAPH_CALLS: usize = 256;

/// The width of the launch graph's rows: 64 F32 values, one call's least
/// work.
const GRAPH_WIDTH: u64 = 64;

/// The calls of one layer of the composition chain: the gate/up
/// projection, the attention over its first query-wide rows, the down
/// projection onto the residual, and the copy of the residual a row op
/// makes.
const CHAIN_LAYER_CALLS: usize = 4;

/// How many launch costs one sized call takes at least.
const LAUNCH_MULTIPLE: f64 = 100.;

/// The part of a history call's time below which its bytes' time from
/// memory leaves it unrotated: half the tolerance a median is resolved to
/// ([`seismic::measure::MEDIAN_TOLERANCE`]), so reading every byte from
/// memory, with nothing of it overlapping the call's other work, would move
/// the call by less than its measurement can tell. A GPU's attention reads
/// near its memory's rate, far above it; a CPU's 0.3% to 1% of its call
/// (a 4-vCPU EPYC 7763, 2026-10-21).
const UNROTATED: f64 = seismic::measure::MEDIAN_TOLERANCE / 2.;

/// The device seconds a point's samples cover at most beyond the one that
/// reaches it, unless its median is resolved first. A GPU's points resolve
/// in milliseconds, and a CPU's projection points in under 2 s (eight
/// samples, the fewest that bound a median at one in twenty, of calls of 2
/// to 220 ms on a 4-vCPU EPYC 7763): they are resolved as without it. A
/// CPU's history rotation or composition chain takes 0.5 to 1.5 s a sample
/// there, which eight samples and their leads would make 15 s a point; two
/// such samples of hundreds of calls each agree within a few percent, inside
/// the 10% a median is resolved to. A point then takes at most this, one
/// more sample and the two leads around them ([`seismic::measure::Stop`]).
pub const SAMPLING_SECONDS: f64 = 2.0;

/// How many times the largest cache serving it a rotation spans (sized
/// beyond the cache, spec §2.10). Under the rotation's cyclic access an LRU
/// cache, or an approximation of one, keeps none of the bytes when they
/// come round again; a cache that kept a whole cache's worth would still
/// serve at most a quarter of them.
const CACHE_MULTIPLE: u64 = 4;

/// The reference attention geometry (Qwen3.5-9B's): key/value heads, query
/// heads per key/value head, rotated pairs and unrotated columns of a head.
const KV_HEADS: u64 = 4;
const GROUP: u64 = 4;
const ROTARY_PAIRS: u64 = 32;
const STATIC_WIDTH: u64 = 192;
const HEAD_WIDTH: u64 = 2 * ROTARY_PAIRS + STATIC_WIDTH;

/// The query width of the reference attention, which the composition
/// chain's attention reads of its projection's product.
const QUERY_WIDTH: u64 = KV_HEADS * GROUP * HEAD_WIDTH;

/// One measured cost: its median, the interval its samples reached (none
/// when too few bound one) and their count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasuredCost {
    pub value: f64,
    pub interval: Option<(f64, f64)>,
    pub samples: u32,
}

impl MeasuredCost {
    /// `units` over each of `timing`'s times: a rate, its interval from the
    /// times' (the slowest time bounds it below).
    fn rate(units: f64, timing: &Timing) -> Self {
        Self {
            value: units / timing.median,
            interval: timing
                .interval
                .map(|(fastest, slowest)| (units / slowest, units / fastest)),
            samples: timing.samples,
        }
    }
}

/// The weight rate of one format along one allocation route of the
/// executor's placements. Every access path along a route allocates the
/// same memory (`seismic::AccessRoute`), so each path along it has this
/// rate.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightRate {
    /// The route the weights were allocated along.
    pub route: AccessRoute,
    /// The resident representation the weights take.
    pub format: Element,
    /// Weight bytes per second its `dense_expand` streams at one row (its
    /// decode form) with its weights along the route.
    pub bytes_per_second: MeasuredCost,
    /// The program's F.
    pub features: u64,
}

/// The matrix throughput of one format.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixCosts {
    /// The resident representation the weights take on the executor.
    pub format: Element,
    /// FLOPs per second of `dense_expand` at the executor's matrix rows.
    pub flops: MeasuredCost,
    /// The program's F.
    pub features: u64,
}

/// What the decode attention entry costs over one history layout, its
/// history in device storage, from a call at `depths[0]` and one at
/// `depths[1]` tokens: a fixed cost per call and the history bytes read per
/// second beyond it.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryCosts {
    /// The codec the history is stored in.
    pub layout: KvCodec,
    /// The seconds of a call beyond its history's bytes at the rate.
    pub fixed_seconds: MeasuredCost,
    /// History bytes read per second along device storage's first route.
    pub bytes_per_second: MeasuredCost,
    /// The two depths the line runs through.
    pub depths: [u64; 2],
}

/// One storage kind's placement as the routes of its paths, in order: at
/// least one, as an executor that places none of a kind has no costs.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    pub first: AccessRoute,
    pub rest: Vec<AccessRoute>,
}

impl Placement {
    /// A placement along `route` alone.
    pub fn along(route: AccessRoute) -> Self {
        Self {
            first: route,
            rest: Vec::new(),
        }
    }
}

/// The placement of each storage kind.
#[derive(Clone, Debug, PartialEq)]
pub struct Placements {
    pub device: Placement,
    pub upload: Placement,
}

impl Placements {
    /// The placement of `storage`.
    pub fn of(&self, storage: StorageKind) -> &Placement {
        match storage {
            StorageKind::Device => &self.device,
            StorageKind::Upload => &self.upload,
        }
    }
}

/// A figure of executor costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Figure {
    /// The weight rate of one format along one allocation route.
    WeightRate(AccessRoute, Element),
    /// The matrix throughput of weights resident as one representation.
    MatrixThroughput(Element),
    /// The decode attention cost over one history layout.
    HistoryCost(KvCodec),
}

impl fmt::Display for Figure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WeightRate(route, format) => {
                write!(
                    formatter,
                    "the {} weight rate along {route:?}",
                    format.name()
                )
            }
            Self::MatrixThroughput(format) => {
                write!(formatter, "the {} matrix throughput", format.name())
            }
            Self::HistoryCost(layout) => {
                write!(formatter, "the {} attention cost", layout.identity())
            }
        }
    }
}

/// A figure the executor's kernels could not produce, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unmeasured {
    /// The figure left out.
    pub figure: Figure,
    /// The failure its points ended with, or the reason recorded with one.
    pub reason: String,
}

/// What the engine's kernels achieve on one executor.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutorCosts {
    /// One per measured format and route of the executor's placements.
    pub weight_rates: Vec<WeightRate>,
    /// Each storage kind's placement as the routes of its paths, in order:
    /// its weight rate is the slowest of theirs.
    pub placements: Placements,
    /// One per measured format ([`measured_formats`]), in its order.
    pub matrices: Vec<MatrixCosts>,
    /// One per measured [`HISTORY_LAYOUTS`] entry, in its order.
    pub histories: Vec<HistoryCosts>,
    /// The device time of the composition chain's sealed graph over the sum
    /// of its calls' times alone.
    pub composition: MeasuredCost,
    /// Device seconds of one call inside a sealed graph.
    pub launch_seconds: MeasuredCost,
    /// Host seconds per run beyond its device time, runs queued back to back
    /// (decode with lookahead).
    pub step_pipelined_seconds: MeasuredCost,
    /// Host seconds of one submit and wait beyond its device time (decode
    /// without lookahead).
    pub step_round_trip_seconds: MeasuredCost,
    /// The figures not measured, with why.
    pub unmeasured: Vec<Unmeasured>,
}

impl ExecutorCosts {
    /// The step cost a decode under `lookahead` pays.
    pub fn step_seconds(&self, lookahead: bool) -> f64 {
        if lookahead {
            self.step_pipelined_seconds.value
        } else {
            self.step_round_trip_seconds.value
        }
    }

    /// Why `figure` is not one of the costs: the reason it was left
    /// unmeasured, or that the walk measures no such figure.
    fn missing(&self, figure: Figure) -> AssessmentError {
        let reason = self
            .unmeasured
            .iter()
            .find(|unmeasured| unmeasured.figure == figure)
            .map_or("the costs measure no such figure", |unmeasured| {
                &unmeasured.reason
            });
        AssessmentError::Unmeasured(format!("{figure} on this executor: {reason}"))
    }

    /// The bytes per second of `format` along `route`.
    fn route_rate(&self, route: AccessRoute, format: Element) -> Result<f64, AssessmentError> {
        self.weight_rates
            .iter()
            .find(|rate| rate.route == route && rate.format == format)
            .map(|rate| rate.bytes_per_second.value)
            .ok_or_else(|| self.missing(Figure::WeightRate(route, format)))
    }

    /// The bytes per second of `format` in `storage`: the slowest rate of
    /// its placement's paths, since a load's bytes may lie along any of
    /// them.
    pub fn weight_rate(
        &self,
        storage: StorageKind,
        format: Element,
    ) -> Result<f64, AssessmentError> {
        let placement = self.placements.of(storage);
        placement.rest.iter().try_fold(
            self.route_rate(placement.first, format)?,
            |slowest, route| Ok(slowest.min(self.route_rate(*route, format)?)),
        )
    }

    /// The matrix throughput of weights resident as `format`.
    pub fn matrix_flops(&self, format: Element) -> Result<f64, AssessmentError> {
        self.matrices
            .iter()
            .find(|matrix| matrix.format == format)
            .map(|matrix| matrix.flops.value)
            .ok_or_else(|| self.missing(Figure::MatrixThroughput(format)))
    }

    /// The decode attention costs of history stored as `layout`.
    pub fn history(&self, layout: KvCodec) -> Result<&HistoryCosts, AssessmentError> {
        self.histories
            .iter()
            .find(|history| history.layout == layout)
            .ok_or_else(|| self.missing(Figure::HistoryCost(layout)))
    }

    /// The history bytes per second of `layout`'s history in device
    /// storage: its layout's rate, measured along device storage's first
    /// route, scaled down by the slowest route of the placement against
    /// that one (in the activation dtype the history holds), since the
    /// kernel reads no faster than the path its history lies along (spec
    /// §2.14).
    pub fn history_rate(&self, layout: KvCodec) -> Result<f64, AssessmentError> {
        let measured_along = self.placements.device.first;
        let activation = Element::bf16();
        let slowdown = self.weight_rate(StorageKind::Device, activation)?
            / self.route_rate(measured_along, activation)?;
        Ok(self.history(layout)?.bytes_per_second.value * slowdown)
    }
}

/// Why executor costs could not be measured or derived.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorCostsError(pub String);

impl fmt::Display for ExecutorCostsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ExecutorCostsError {}

fn failed(what: &str) -> impl Fn(&dyn fmt::Display) -> ExecutorCostsError + '_ {
    move |error| ExecutorCostsError(format!("{what}: {error}"))
}

/// What the stored records give of an executor's costs.
#[derive(Clone, Debug, PartialEq)]
pub enum StoredCosts {
    Complete {
        costs: ExecutorCosts,
        /// The records the costs were derived from, in the walk's order:
        /// the evidence they are a function of.
        records: Vec<(PointKey, Record)>,
    },
    /// The first point of the walk no record holds: never attempted, or
    /// not to its end.
    Missing(PointKey),
}

/// What a measurement job's walk did.
#[derive(Debug)]
pub struct Measurement {
    /// Every point visited, in order, with what became of it.
    pub points: Vec<(PointKey, Outcome<CallError, ExecutorCostsError>)>,
    /// Why the walk ended before its end where no visited point's outcome
    /// says: a kernel's identity or a layout's planes that cannot be
    /// derived, a placement the executor lacks, or the dense attention a
    /// later figure needs left unmeasured. None when it reached its end or
    /// a point every estimate needs ended it.
    pub ended: Option<ExecutorCostsError>,
}

/// The least cache a path is bounded by: 256 MiB, the largest memory-side
/// cache shipping (AMD MI300X's Infinity Cache). It is a floor even where a
/// device reports a cache: no API reports a memory-side cache (Infinity
/// Cache, Apple's SLC), which can sit behind a reported L2, nor any Metal or
/// Vulkan device's cache, so no smaller bound is known to clear one.
const UNREPORTED_CACHE: u64 = 256 << 20;

/// How the walk ended without its costs.
enum Stop<M> {
    /// A point no record holds, which nothing measures here: its key where
    /// times come from records alone, and never where every point is timed
    /// (`M` is uninhabited).
    Missing(M),
    /// A point cannot be timed: it failed, or is recorded unobtainable. The
    /// job's visited points report it.
    Point(ExecutorCostsError),
    /// The walk failed outside any point's timing: a kernel's identity, a
    /// layout's planes, a placement, a record that cannot be read.
    Failed(ExecutorCostsError),
}

impl<M> From<ExecutorCostsError> for Stop<M> {
    fn from(error: ExecutorCostsError) -> Self {
        Self::Failed(error)
    }
}

/// What a point is: its kernels, the configuration of each and what is
/// timed, which its key names and which the job forms and times.
struct PointSpec {
    kernels: Vec<KernelIdentity>,
    /// The shape every size of the point shares.
    configuration: BTreeMap<String, u64>,
    /// Its sized dimensions and argument sets.
    point: BTreeMap<String, u64>,
}

fn named(values: &[(&str, u64)]) -> BTreeMap<String, u64> {
    values
        .iter()
        .map(|(name, value)| ((*name).to_owned(), *value))
        .collect()
}

impl PointSpec {
    fn new(
        kernels: Vec<KernelIdentity>,
        configuration: &[(&str, u64)],
        point: &[(&str, u64)],
    ) -> Self {
        Self {
            kernels,
            configuration: named(configuration),
            point: named(point),
        }
    }
}

/// The kernels a point forms, by entry, element bindings and the values its
/// statics take.
struct Formation<E: Entry> {
    entry: BoundEntry<E>,
    values: Vec<(&'static str, u64)>,
}

impl<E: Entry> Clone for Formation<E> {
    fn clone(&self) -> Self {
        Self {
            entry: self.entry.clone(),
            values: self.values.clone(),
        }
    }
}

/// The rows a formed projection serves: one, a decode step's, or the
/// matrix throughput's ([`Sizes`]). A kernel formed for one forms only the
/// launches those rows select, and is never called at others.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rows {
    Decode,
    Matrix,
}

impl Rows {
    fn count(self, sizes: Sizes) -> u64 {
        match self {
            Self::Decode => 1,
            Self::Matrix => sizes.matrix_rows,
        }
    }
}

/// The selected-rows dimension of `dense_expand` and `dense_output`, by
/// which their launches choose between GEMV and GEMM.
const SELECTED_ROWS: &str = "O";

impl<E: Entry> Formation<E> {
    fn new(entry: BoundEntry<E>, values: &[(&'static str, u64)]) -> Self {
        Self {
            entry,
            values: values.to_vec(),
        }
    }

    /// The statics its declaration on `backend` names, from its values.
    fn statics(
        &self,
        backend: seismic::BackendName,
    ) -> Result<NativeSpecialization, ExecutorCostsError> {
        let implementation = seismic::generated::native_implementation_for_backend::<E>(backend)
            .map_err(|error| failed(E::NAME)(&error))?
            .ok_or_else(|| {
                ExecutorCostsError(format!(
                    "`{}` has no {} implementation",
                    E::NAME,
                    backend.as_str()
                ))
            })?;
        let mut statics = NativeSpecialization::new();
        for name in &implementation.statics {
            let value = self
                .values
                .iter()
                .find(|(candidate, _)| candidate == name)
                .map(|(_, value)| *value)
                .ok_or_else(|| {
                    ExecutorCostsError(format!("`{}` declares `{name}` static, not given", E::NAME))
                })?;
            statics = statics.with_static(name.clone(), value);
        }
        Ok(statics)
    }

    /// Its source identity on `backend`, formable without a device.
    fn identity(
        &self,
        backend: seismic::BackendName,
    ) -> Result<KernelIdentity, ExecutorCostsError> {
        KernelIdentity::of(backend, &self.entry, &self.statics(backend)?)
            .map_err(|error| failed(E::NAME)(&error))
    }

    /// Formed on `device` at its default specialization, its calls held to
    /// `rows` selected rows where given.
    fn form(
        &self,
        device: &Device,
        rows: Option<u64>,
    ) -> Result<NativeKernel<E>, ExecutorCostsError> {
        let statics = self.statics(device.backend())?;
        let specialization = self
            .entry
            .default_specialization(device, &statics)
            .map_err(|error| failed(E::NAME)(&error))?
            .ok_or_else(|| {
                ExecutorCostsError(format!("`{}` admits nothing at {statics:?}", E::NAME))
            })?;
        let specialization = match rows {
            Some(rows) => specialization.with_range(SELECTED_ROWS, rows, rows),
            None => specialization,
        };
        self.entry
            .prepare(device, &specialization)
            .map_err(|error| failed(E::NAME)(&error))
    }
}

/// Every kernel the walk times, as forms of them: one `dense_expand`
/// program per measured format, the chain's `dense_output`, the decode
/// attention over each layout and `conditioning_overlay`.
#[derive(Clone)]
struct ProgramSet {
    sizes: Sizes,
    /// Each measured format with its F and program.
    expands: Vec<(Element, u64, Formation<dense_expand::Entry>)>,
    /// The chain's down projection, at its format's F.
    output: Formation<dense_output::Entry>,
    dense: Formation<attention_decode::Entry>,
    affine: Formation<attention_decode_k8v4::Entry>,
    copy: Formation<conditioning_overlay::Entry>,
}

impl ProgramSet {
    /// The programs of `formats` ([`measured_formats`]) at `sizes` on an
    /// executor whose largest allocation is `largest`, the chain in `chain`.
    fn new(
        sizes: Sizes,
        formats: &[Element],
        chain: Element,
        largest: u64,
    ) -> Result<Self, ExecutorCostsError> {
        let expands = formats
            .iter()
            .map(|&format| {
                let features = features(sizes, format, largest)?;
                Ok((format, features, expand(format, features)))
            })
            .collect::<Result<Vec<_>, ExecutorCostsError>>()?;
        Ok(Self {
            sizes,
            output: output(chain, features(sizes, chain, largest)?),
            expands,
            dense: attend::<attention_decode::Entry>(KvCodec::Dense)?,
            affine: attend::<attention_decode_k8v4::Entry>(KvCodec::AffineK8V4)?,
            copy: copy(),
        })
    }

    /// `format`'s F and program.
    fn expand(&self, format: Element) -> (u64, &Formation<dense_expand::Entry>) {
        let (_, features, formation) = self
            .expands
            .iter()
            .find(|(measured, _, _)| *measured == format)
            .expect("every measured format has a program");
        (*features, formation)
    }

    /// Every program formed on `device` on the host's threads
    /// (`seismic::Workers`, which tuning forms its programs on), each kernel
    /// the cache holds loaded, not compiled. They form on a thread of their
    /// own, the walk's early reads first, while the walk goes on:
    /// a point waits only for its own kernels, and a compile is host work
    /// while most points are timed on the device ([`Programs::settled`]).
    fn form(&self, device: &Device) -> Programs {
        let programs = Programs {
            expands: self
                .expands
                .iter()
                .map(|(format, _, _)| (*format, Pending::new(), Pending::new()))
                .collect(),
            output: Pending::new(),
            dense: Pending::new(),
            affine: Pending::new(),
            copy: Pending::new(),
        };
        let (set, device, forming) = (self.clone(), device.clone(), programs.clone());
        std::thread::Builder::new()
            .name("measurement-formation".into())
            .spawn(move || {
                let decodes = (0..set.expands.len()).map(|index| Job::Expand(index, Rows::Decode));
                let matrices = (0..set.expands.len()).map(|index| Job::Expand(index, Rows::Matrix));
                // The attention programs are one long compile each (the
                // affine one's 1.2 s alone on an RTX 3090's Vulkan, against
                // 0.7 s for a decode projection): they start first, beside
                // the short ones the weight rates read, so each has formed
                // by its history's points.
                let jobs = [Job::Affine, Job::Dense]
                    .into_iter()
                    .chain(decodes)
                    .chain([Job::Copy, Job::Output])
                    .chain(matrices)
                    .collect::<Vec<_>>();
                let formed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    seismic::Workers::of_the_host().each(&jobs, |job| match *job {
                        Job::Expand(index, rows) => {
                            let (_, decode, matrix) = &forming.expands[index];
                            let formed = set.expands[index]
                                .2
                                .form(&device, Some(rows.count(set.sizes)));
                            match rows {
                                Rows::Decode => decode.set(formed),
                                Rows::Matrix => matrix.set(formed),
                            }
                        }
                        Job::Copy => forming.copy.set(set.copy.form(&device, None)),
                        Job::Dense => forming.dense.set(set.dense.form(&device, None)),
                        Job::Affine => forming.affine.set(set.affine.form(&device, None)),
                        Job::Output => forming.output.set(set.output.form(&device, Some(1))),
                    })
                }));
                if formed.is_err() {
                    forming.abandon();
                }
            })
            .expect("the host starts the formation thread");
        programs
    }
}

/// One program [`ProgramSet::form`] forms.
#[derive(Clone, Copy)]
enum Job {
    Expand(usize, Rows),
    Copy,
    Dense,
    Affine,
    Output,
}

/// The F of `format`'s program: the executor's ([`Sizes`]), halved while a
/// weight of it exceeds the executor's largest allocation.
fn features(sizes: Sizes, format: Element, largest: u64) -> Result<u64, ExecutorCostsError> {
    let mut features = sizes.features;
    while expand_weight_bytes(format, features)? / 2 > largest {
        features /= 2;
        if features < QUERY_WIDTH {
            return Err(ExecutorCostsError(format!(
                "the largest allocation holds no {} weight of {QUERY_WIDTH} rows",
                format.name()
            )));
        }
    }
    Ok(features)
}

/// The walk's kernels as formed on the job's device, each with why it could
/// not be where it failed: the points of a kernel that failed record its
/// failure as their own.
#[derive(Clone)]
struct Programs {
    /// Each measured format's program for decode, then for the matrix
    /// throughput.
    expands: Vec<(
        Element,
        Forming<dense_expand::Entry>,
        Forming<dense_expand::Entry>,
    )>,
    /// The chain's down projection, for decode.
    output: Forming<dense_output::Entry>,
    dense: Forming<attention_decode::Entry>,
    affine: Forming<attention_decode_k8v4::Entry>,
    copy: Forming<conditioning_overlay::Entry>,
}

/// A kernel forming on the formation thread.
type Forming<E> = Arc<Pending<Result<NativeKernel<E>, ExecutorCostsError>>>;

/// A value one thread sets once and others wait for.
struct Pending<T> {
    value: std::sync::OnceLock<T>,
    /// The thread setting it panicked first.
    abandoned: std::sync::atomic::AtomicBool,
    lock: std::sync::Mutex<()>,
    ready: std::sync::Condvar,
}

impl<T> Pending<T> {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            value: std::sync::OnceLock::new(),
            abandoned: std::sync::atomic::AtomicBool::new(false),
            lock: std::sync::Mutex::new(()),
            ready: std::sync::Condvar::new(),
        })
    }

    fn set(&self, value: T) {
        let _ = self.value.set(value);
        let _held = self.lock.lock().expect("pending lock poisoned");
        self.ready.notify_all();
    }

    fn abandon(&self) {
        self.abandoned
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let _held = self.lock.lock().expect("pending lock poisoned");
        self.ready.notify_all();
    }

    /// The value, once set. A defect of the thread setting it (a panic) is
    /// this thread's too.
    fn wait(&self) -> &T {
        let mut held = self.lock.lock().expect("pending lock poisoned");
        loop {
            if let Some(value) = self.value.get() {
                return value;
            }
            if self.abandoned.load(std::sync::atomic::Ordering::SeqCst) {
                panic!("the formation thread panicked");
            }
            held = self.ready.wait(held).expect("pending lock poisoned");
        }
    }
}

impl Programs {
    /// `format`'s program formed for `rows`.
    fn expand(
        &self,
        format: Element,
        rows: Rows,
    ) -> Result<&NativeKernel<dense_expand::Entry>, ExecutorCostsError> {
        let (_, decode, matrix) = self
            .expands
            .iter()
            .find(|(measured, _, _)| *measured == format)
            .expect("every measured format has a program");
        match rows {
            Rows::Decode => kernel(decode),
            Rows::Matrix => kernel(matrix),
        }
    }

    /// Every program formed: no compile runs beside a point that touches
    /// the host.
    fn settled(&self) {
        for (_, decode, matrix) in &self.expands {
            decode.wait();
            matrix.wait();
        }
        self.output.wait();
        self.dense.wait();
        self.affine.wait();
        self.copy.wait();
    }

    /// Every program not formed yet is never formed: the formation thread
    /// panicked.
    fn abandon(&self) {
        for (_, decode, matrix) in &self.expands {
            decode.abandon();
            matrix.abandon();
        }
        self.output.abandon();
        self.dense.abandon();
        self.affine.abandon();
        self.copy.abandon();
    }
}

/// A formed kernel, or why it was not, once it is formed.
fn kernel<E: Entry>(forming: &Forming<E>) -> Result<&NativeKernel<E>, ExecutorCostsError> {
    forming.wait().as_ref().map_err(Clone::clone)
}

/// How a point is made on a device from the walk's formed kernels, its
/// rotated operands views of the job's backings, when no record holds it:
/// a kernel that did not form can never make it ([`Unformed::Never`]), and
/// anything else that fails (its operands' allocation first) fails it now
/// ([`Unformed::Now`]).
type Build<'b> = dyn FnMut(&Device, &Programs, &mut Backings) -> Result<Point, Unformed<ExecutorCostsError>>
    + 'b;

/// Where the walk's timings come from.
trait Times {
    /// What a point no record holds is to these timings ([`Stop::Missing`]).
    type Missing;

    /// The timing of `spec`'s point: its record, or `build`'s point, made
    /// from `programs` formed on a device, measured there and its record
    /// stored.
    fn time(
        &mut self,
        spec: PointSpec,
        programs: &ProgramSet,
        build: &mut Build<'_>,
    ) -> Result<Timing, Stop<Self::Missing>>;
}

/// The records of one executor, keyed by its identity.
struct Records<'a> {
    executor: Identity,
    store: &'a dyn RecordStore<PointKey>,
}

impl Records<'_> {
    fn key(&self, spec: PointSpec) -> PointKey {
        PointKey {
            executor: self.executor.clone(),
            kernels: spec.kernels,
            configuration: spec.configuration,
            point: spec.point,
        }
    }

    /// The point's recorded outcome: its timing, or why it is unobtainable;
    /// none when it was never attempted to its end.
    fn load<M>(&self, key: &PointKey) -> Result<Option<Record>, Stop<M>> {
        self.store
            .load(key)
            .map_err(|error| Stop::Failed(failed("a measurement record")(&error)))
    }
}

/// What a record gives the walk: its timing, or the point unobtainable.
fn outcome<M>(record: &Record) -> Result<Timing, Stop<M>> {
    match record {
        Record::Measured(timing) => Ok(*timing),
        Record::Unobtainable { reason } => Err(unobtainable(reason)),
    }
}

fn unobtainable<M>(reason: &Reason) -> Stop<M> {
    Stop::Point(ExecutorCostsError(reason.as_str().to_owned()))
}

/// The records alone, and every one the walk read: a start's derivation,
/// which measures nothing.
struct Retraced<'a> {
    records: Records<'a>,
    read: Vec<(PointKey, Record)>,
}

impl Times for Retraced<'_> {
    type Missing = PointKey;

    fn time(
        &mut self,
        spec: PointSpec,
        _: &ProgramSet,
        _: &mut Build<'_>,
    ) -> Result<Timing, Stop<PointKey>> {
        let key = self.records.key(spec);
        match self.records.load(&key)? {
            Some(record) => {
                let timing = outcome(&record);
                self.read.push((key, record));
                timing
            }
            None => Err(Stop::Missing(key)),
        }
    }
}

/// The job's timings: each point's record, or measured through the
/// measurer, which stores its record.
struct Measured<'a, 'd> {
    records: Records<'a>,
    measurer: &'a mut Measurer,
    device: &'d Device,
    attempting: &'a mut dyn FnMut(&PointKey) -> Attempt,
    /// The walk's kernels, formed before the first point no record holds: a
    /// job whose points are all stored forms nothing.
    programs: std::cell::OnceCell<Programs>,
    /// The storage every point's rotated operands view, allocated as the
    /// first points that need it are made.
    backings: Backings,
    /// Every point visited, in order, with what became of it.
    visited: Vec<(PointKey, Outcome<CallError, ExecutorCostsError>)>,
}

impl Times for Measured<'_, '_> {
    /// Every point is timed where no record holds it.
    type Missing = std::convert::Infallible;

    fn time(
        &mut self,
        spec: PointSpec,
        programs: &ProgramSet,
        build: &mut Build<'_>,
    ) -> Result<Timing, Stop<Self::Missing>> {
        let key = self.records.key(spec);
        let how = Measure {
            stop: seismic::measure::Stop::Within {
                error_rate: crate::SEARCH_SETTINGS.error_rate,
                seconds: SAMPLING_SECONDS,
            },
            // The device time a sample covers at least, which sets its
            // repetitions: tuning's, the one owner.
            min_sample_seconds: crate::MIN_SAMPLE_SECONDS,
        };
        // The point is made only when no record holds it and the host
        // attempts it, so a stored or skipped point is never made. Every
        // program is formed, all at once, when the first point no record
        // holds is reached, before the host records its attempt: a compile
        // that fails or ends the job is no point's.
        let device = self.device;
        let formed = &self.programs;
        let backings = &mut self.backings;
        let host = &mut *self.attempting;
        let mut outcomes = self.measurer.measure_all(
            vec![(key, || {
                build(
                    device,
                    formed.get_or_init(|| programs.form(device)),
                    backings,
                )
            })],
            &how,
            self.records.store,
            &mut |key| {
                formed.get_or_init(|| programs.form(device));
                host(key)
            },
        );
        let (key, outcome) = outcomes.pop().expect("one point is measured");
        let result = match &outcome {
            Outcome::Loaded(record) => self::outcome(record),
            Outcome::Measured(timing) => Ok(*timing),
            Outcome::Skipped(reason) => Err(unobtainable(reason)),
            Outcome::Unformed(error) | Outcome::Unmade(error) => Err(Stop::Point(error.clone())),
            Outcome::Failed(error) => Err(Stop::Point(failed("timing a point")(error))),
            Outcome::Store(error) => Err(Stop::Point(failed("a measurement record")(error))),
        };
        self.visited.push((key, outcome));
        // The next point reads zeros where this one wrote.
        self.backings.restore().map_err(Stop::Failed)?;
        result
    }
}

/// The largest cache that can serve the executor's storage: its device's
/// reported cache, at least [`UNREPORTED_CACHE`], the floor for the
/// memory-side caches no API reports. A host CPU's last-level caches as
/// Linux or Windows reports them (every instance the executor's cores share,
/// summed) are its largest, and stand alone: a rotation over four times them
/// reads what one over 256 MiB does (on a 4-vCPU EPYC 7763 of 32 MiB, every
/// weight rate within run-to-run spread over 128 MiB and over 1 GiB,
/// 2026-10-21), in an eighth of the time, which on a CPU, whose kernels read
/// a gigabyte in up to 1.5 s, is most of its job. macOS reports an Apple
/// CPU's L2, behind which its system-level cache sits.
fn largest_cache(topology: &DeviceTopology, executor: &Executor) -> u64 {
    match topology
        .physical_device(executor.physical())
        .and_then(|physical| physical.cache())
    {
        Some(Reported {
            value,
            by: ReportedBy::LinuxSysfsCache | ReportedBy::WindowsProcessorCache,
        }) => value,
        Some(cache) => cache.value.max(UNREPORTED_CACHE),
        None => UNREPORTED_CACHE,
    }
}

/// The argument sets a rotation of calls each reading `bytes` takes so that
/// its bytes span [`CACHE_MULTIPLE`] times `cache`: at least two, so no call
/// reads the bytes the call before it read (a CPU's f32 projection, one set
/// of 134 MB over a 128 MiB span, read 15% faster than over two).
fn rotation_sets(cache: u64, bytes: u64) -> u64 {
    (CACHE_MULTIPLE * cache).div_ceil(bytes.max(1)).max(2)
}

/// One allocation route of the executor's placements, and how an operand
/// is allocated along it: in the first storage kind whose placement reaches
/// it, passing over the paths before it there. Every path along a route
/// allocates the same memory (`seismic::AccessRoute`), so one rate is
/// measured per route.
struct Route {
    route: AccessRoute,
    storage: StorageKind,
    /// The placement's paths before the route's first: each is held at what
    /// it has charged while an operand allocates, so the allocation moves
    /// on along the placement (spec §2.3).
    passed: Vec<AccessPathId>,
    /// The placement's paths along the route, where an operand must land.
    along: Vec<AccessPathId>,
}

/// The route `path` of `topology` allocates along.
fn route_of(topology: &DeviceTopology, path: AccessPathId) -> AccessRoute {
    topology
        .access_path(path)
        .expect("a placement's paths belong to its topology")
        .route()
}

/// A route's figure in a point's key: device storage 0, upload storage 1,
/// Vulkan heap `h` 2 + `h`.
fn route_ordinal(route: AccessRoute) -> u64 {
    match route {
        AccessRoute::Device => 0,
        AccessRoute::Upload => 1,
        AccessRoute::VulkanHeap(heap) => 2 + u64::from(heap),
    }
}

/// Every route of an executor's placements, each with the storage kind
/// whose placement reaches it first: device storage's first route, where
/// the figures not of a route are measured, and the others.
struct Routes {
    device: Route,
    others: Vec<Route>,
}

impl Routes {
    /// Every route, device storage's first.
    fn iter(&self) -> impl Iterator<Item = &Route> {
        std::iter::once(&self.device).chain(&self.others)
    }
}

impl Route {
    /// The route of `path`, the `index`th of `storage`'s `placement`: in
    /// `storage`, passing over the paths before it.
    fn of(
        topology: &DeviceTopology,
        storage: StorageKind,
        placement: &[AccessPathId],
        index: usize,
    ) -> Self {
        let route = route_of(topology, placement[index]);
        Self {
            route,
            storage,
            passed: placement[..index].to_vec(),
            along: placement
                .iter()
                .filter(|path| route_of(topology, **path) == route)
                .copied()
                .collect(),
        }
    }

    /// Every route of `executor`'s placements, and each kind's placement as
    /// its routes.
    fn all(
        topology: &DeviceTopology,
        executor: &Executor,
    ) -> Result<(Routes, Placements), ExecutorCostsError> {
        let placement = |storage: StorageKind| {
            let paths = executor.placement(storage);
            let (first, rest) = paths.split_first().ok_or_else(|| {
                ExecutorCostsError(format!(
                    "executor {} places no {storage:?} storage",
                    executor.selector()
                ))
            })?;
            Ok::<_, ExecutorCostsError>(Placement {
                first: route_of(topology, *first),
                rest: rest.iter().map(|path| route_of(topology, *path)).collect(),
            })
        };
        let placements = Placements {
            device: placement(StorageKind::Device)?,
            upload: placement(StorageKind::Upload)?,
        };
        let mut routes = Routes {
            device: Self::of(
                topology,
                StorageKind::Device,
                executor.placement(StorageKind::Device),
                0,
            ),
            others: Vec::new(),
        };
        for storage in StorageKind::ALL {
            let placement = executor.placement(storage);
            for (index, path) in placement.iter().enumerate() {
                let route = route_of(topology, *path);
                if !routes.iter().any(|known| known.route == route) {
                    routes
                        .others
                        .push(Self::of(topology, storage, placement, index));
                }
            }
        }
        Ok((routes, placements))
    }

    /// The point entries naming where a point's operands lie.
    fn point(&self) -> [(&'static str, u64); 2] {
        [
            ("storage", self.storage as u64),
            ("route", route_ordinal(self.route)),
        ]
    }

    /// `allocate`'s tensors, allocated passing over the route's earlier
    /// paths, each landed along the route. The job's device holds no path
    /// limit of its own (the measurement job opens it without a ceiling),
    /// so the passed paths are released to none after.
    fn place(
        &self,
        device: &Device,
        allocate: impl FnOnce() -> Result<Vec<Tensor>, ExecutorCostsError>,
    ) -> Result<Vec<Tensor>, ExecutorCostsError> {
        let error = failed("placing an operand");
        let charges = device
            .path_charges()
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        // Every path held is released whatever fails: the limits set before
        // a failure to set one, and after the allocation.
        let mut held = Vec::with_capacity(self.passed.len());
        let mut holding = Ok(());
        for path in &self.passed {
            let charged = charges.get(path).copied().unwrap_or(0);
            holding = device.set_path_limit(*path, Some(charged));
            if holding.is_err() {
                break;
            }
            held.push(*path);
        }
        let allocated = holding
            .map_err(|cause| error(&cause))
            .and_then(|()| allocate());
        let released = held
            .iter()
            .fold(Ok(()), |released, path| {
                released.and(device.set_path_limit(*path, None))
            })
            .map_err(|cause| error(&cause));
        let tensors = allocated?;
        released?;
        for tensor in &tensors {
            let charges = tensor
                .observe_storage()
                .path_charges()
                .expect("a held tensor's storage is charged");
            if let Some((path, _)) = charges.iter().find(|(path, _)| !self.along.contains(path)) {
                return Err(ExecutorCostsError(format!(
                    "an operand for the {:?} route landed on path {path:?}, off it",
                    self.route
                )));
            }
        }
        Ok(tensors)
    }

    /// A zero-filled tensor of the route's storage kind.
    fn zeros(
        &self,
        device: &Device,
        element: Element,
        extents: &[u64],
    ) -> Result<Tensor, seismic::TensorError> {
        match self.storage {
            StorageKind::Device => Tensor::zeros(device, element, extents),
            StorageKind::Upload => Tensor::shared(device, element, extents),
        }
    }
}

/// One rotated operand of a point: `count` tensors (one per argument set)
/// of `element` and `extents`, each starting at a multiple of `alignment`
/// in its allocation.
#[derive(Clone)]
struct Operand {
    element: Element,
    extents: Vec<u64>,
    count: u64,
    alignment: u64,
}

impl Operand {
    /// `count` tensors of `element` and `extents` at their canonical
    /// layout's alignment.
    fn canonical(
        element: Element,
        extents: &[u64],
        count: u64,
    ) -> Result<Self, ExecutorCostsError> {
        let (_, alignment) = Tensor::canonical_layout(element, extents)
            .map_err(|cause| failed("rotated operand")(&cause))?;
        Ok(Self {
            element,
            extents: extents.to_vec(),
            count,
            alignment,
        })
    }

    /// The bytes of one of its tensors.
    fn bytes(&self) -> Result<u64, ExecutorCostsError> {
        Tensor::canonical_layout(self.element, &self.extents)
            .map(|(bytes, _)| bytes)
            .map_err(|cause| failed("rotated operand")(&cause))
    }
}

/// The zero-filled storage every rotated operand of the job's points views,
/// per route: allocations of at most the executor's largest, each placed
/// along its route, appended only where a point lays out more than they
/// hold. They hold the most any point has laid out, each byte zero-filled
/// once a job, and a job whose points are all stored allocates none. A
/// point's operands lie one after another from their start, so its
/// argument sets span distinct bytes, as many as its rotation counts. A
/// decode attention call appends its row's key and value into its history,
/// whose encoding of the zero row is not all zero bytes (its rotated key is
/// -0 at angles of negative cosine, which the dense key plane holds and an
/// affine group's zero point can take; read as an f32 weight, a -0 bf16 can
/// be a subnormal's low half), so the rows a point writes are restored to
/// zeros once it is timed ([`Self::restore`]) and every point reads zeros.
#[derive(Default)]
struct Backings {
    routes: Vec<(AccessRoute, Vec<Tensor>)>,
    /// The views of the backings the points made since the last restore
    /// write.
    written: Vec<Tensor>,
}

impl Backings {
    /// Note that the point being made writes `views`.
    fn writes(&mut self, views: impl IntoIterator<Item = Tensor>) {
        self.written.extend(views);
    }

    /// Every view written since the last restore filled with zeros again.
    fn restore(&mut self) -> Result<(), ExecutorCostsError> {
        let error = failed("restoring the rotated operands");
        for mut view in self.written.drain(..) {
            let (bytes, _) = Tensor::canonical_layout(view.element(), view.extents())
                .map_err(|cause| error(&cause))?;
            view.write_from_host(&vec![0; bytes as usize])
                .map_err(|cause| error(&cause))?;
        }
        Ok(())
    }

    /// `operands`' tensors, in order, laid out from the start of `route`'s
    /// backing, which is first extended along it by what they need beyond
    /// it.
    fn lay_out<const N: usize>(
        &mut self,
        device: &Device,
        route: &Route,
        operands: [Operand; N],
    ) -> Result<[Vec<Tensor>; N], ExecutorCostsError> {
        let error = failed("rotated operands");
        let largest = device.info().limits().max_allocation.value;
        let index = match self
            .routes
            .iter()
            .position(|(known, _)| *known == route.route)
        {
            Some(index) => index,
            None => {
                self.routes.push((route.route, Vec::new()));
                self.routes.len() - 1
            }
        };
        let backing = &mut self.routes[index].1;
        let mut tensors = Vec::new();
        for operand in &operands {
            let bytes = operand.bytes()?;
            if bytes > largest {
                return Err(ExecutorCostsError(format!(
                    "a rotated operand of {bytes} bytes exceeds the largest allocation"
                )));
            }
            tensors.extend((0..operand.count).map(|_| (bytes, operand.alignment)));
        }
        // Each tensor at the next multiple of its alignment in the current
        // allocation, else at the start of the next; past the backing's
        // last, a new one holds what is left, as far as the largest admits.
        let mut sizes = backing.iter().map(Tensor::byte_len).collect::<Vec<_>>();
        let mut placed = Vec::with_capacity(tensors.len());
        let (mut allocation, mut end) = (0, 0u64);
        for (position, &(bytes, alignment)) in tensors.iter().enumerate() {
            loop {
                if allocation == sizes.len() {
                    let rest = tensors[position..]
                        .iter()
                        .map(|(bytes, alignment)| bytes + alignment - 1)
                        .sum::<u64>();
                    sizes.push(rest.min(largest));
                }
                let start = end.next_multiple_of(alignment);
                if start + bytes <= sizes[allocation] {
                    placed.push((allocation, start));
                    end = start + bytes;
                    break;
                }
                allocation += 1;
                end = 0;
            }
        }
        let added = sizes[backing.len()..].to_vec();
        if !added.is_empty() {
            backing.extend(route.place(device, || {
                added
                    .iter()
                    .map(|&bytes| {
                        route
                            .zeros(device, Element::bool(), &[bytes])
                            .map_err(|cause| error(&cause))
                    })
                    .collect()
            })?);
        }
        let mut placed = placed.into_iter();
        let views = operands
            .iter()
            .map(|operand| {
                (0..operand.count)
                    .map(|_| {
                        let (allocation, offset) =
                            placed.next().expect("every tensor laid out is placed");
                        backing[allocation]
                            .region(operand.element, &operand.extents, offset)
                            .map_err(|cause| error(&cause))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(views
            .try_into()
            .unwrap_or_else(|_| unreachable!("one view list per operand")))
    }
}

/// Whether one call of `timing` takes at least [`LAUNCH_MULTIPLE`] launch
/// costs.
fn sized(timing: &Timing, launch: f64) -> bool {
    timing.median >= LAUNCH_MULTIPLE * launch
}

/// The arguments of one `dense_expand` call over `rows` rows of F
/// `features`, its weights along one route.
struct ExpandCase {
    residual: Tensor,
    norm: Tensor,
    gate: Tensor,
    up: Tensor,
    out_rows: Tensor,
    scale: Tensor,
    bound: Tensor,
}

impl ExpandCase {
    /// The rotated operands of `sets` argument sets at F `features`: each
    /// set's gate weights, then each set's up weights.
    fn operands(
        weight: Element,
        features: u64,
        sets: u64,
    ) -> Result<[Operand; 2], ExecutorCostsError> {
        let weights = Operand::canonical(weight, &[features, HIDDEN], sets)?;
        Ok([weights.clone(), weights])
    }

    /// `sets` argument sets, each with weights of its own along `route`;
    /// the rows and the constants are one set's for all.
    fn new(
        device: &Device,
        backings: &mut Backings,
        weight: Element,
        route: &Route,
        rows: u64,
        features: u64,
        sets: u64,
    ) -> Result<Vec<Self>, ExecutorCostsError> {
        let [gates, ups] =
            backings.lay_out(device, route, Self::operands(weight, features, sets)?)?;
        Self::over(device, weight, rows, gates, ups)
    }

    /// One argument set per gate and up weight of `gates` and `ups`.
    fn over(
        device: &Device,
        weight: Element,
        rows: u64,
        gates: Vec<Tensor>,
        ups: Vec<Tensor>,
    ) -> Result<Vec<Self>, ExecutorCostsError> {
        let error = failed("dense_expand arguments");
        let zeros = |element, extents: &[u64]| {
            Tensor::zeros(device, element, extents).map_err(|cause| error(&cause))
        };
        let residual = zeros(Element::f32(), &[rows, HIDDEN])?;
        let norm = zeros(Element::f32(), &[HIDDEN])?;
        let out_rows = every_row(device, rows)?;
        let scale = zeros(Element::f32(), &[0])?;
        let bound = zeros(Element::f32(), &[bound_extent(weight)])?;
        Ok(gates
            .into_iter()
            .zip(ups)
            .map(|(gate, up)| Self {
                residual: residual.clone(),
                norm: norm.clone(),
                gate,
                up,
                out_rows: out_rows.clone(),
                scale: scale.clone(),
                bound: bound.clone(),
            })
            .collect())
    }

    fn args(&mut self) -> dense_expand::Args<'_> {
        dense_expand::Args {
            residual: &self.residual,
            norm: &self.norm,
            gate_weight: &self.gate,
            up_weight: &self.up,
            out_rows: &self.out_rows,
            eps: 1e-6,
            activation: 0,
            gate_scale: &self.scale,
            up_scale: &self.scale,
            gate_bound: &self.bound,
            up_bound: &self.bound,
        }
    }
}

/// The identity row table of `rows` rows: every row advances.
fn every_row(device: &Device, rows: u64) -> Result<Tensor, ExecutorCostsError> {
    let table = (0..rows as i32)
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::i32(), &[rows], &table)
        .map_err(|cause| failed("row table")(&cause))
}

/// A weight's packing bound extent as a load plans it
/// (`planning::weights::packing_bound`): one value per run of the columns
/// sharing a scale, in the layout the packed forms read; else none.
fn bound_extent(weight: Element) -> u64 {
    crate::planning::packing_run(weight).map_or(0, |run| HIDDEN / run)
}

/// `dense_expand` over `weight` weights at F `features`.
fn expand(weight: Element, features: u64) -> Formation<dense_expand::Entry> {
    let bound = bound_extent(weight);
    Formation::new(
        dense_expand::native_entry_with(dense_expand::Elements {
            NW: Element::f32(),
            GW: weight,
            UW: weight,
            A: Element::bf16(),
        }),
        &[
            ("H", HIDDEN),
            ("F", features),
            ("GS", 0),
            ("US", 0),
            ("GB", bound),
            ("UB", bound),
        ],
    )
}

/// The bytes of both of one `dense_expand` call's weights at F `features`.
fn expand_weight_bytes(weight: Element, features: u64) -> Result<u64, ExecutorCostsError> {
    weight
        .canonical_byte_len(&[features, HIDDEN])
        .map(|bytes| 2 * bytes)
        .map_err(|error| failed("dense_expand weights")(&error))
}

/// The arguments of one decode attention call of one sequence at `depth`
/// tokens of history in one layout: one row that appends its key and value
/// and reads every token before it.
struct AttendCase {
    query: Tensor,
    gate: Tensor,
    key: Tensor,
    value: Tensor,
    norms: Tensor,
    rotary_components: Tensor,
    rotary_frequencies: Tensor,
    rotary_amplitudes: Tensor,
    coordinates: Tensor,
    visible: Tensor,
    fresh: Tensor,
    destinations: Tensor,
    /// The slab store the history lives in, as the engine's history store
    /// holds it: the kernels address history through its slab tables.
    _history: SlabTensor,
    /// The layout's history planes, in its plane order: the store's
    /// logical regions.
    planes: Vec<Tensor>,
    slab_rows: u32,
}

/// The rotation of the reference attention (Qwen3.5-9B's interleaved
/// rotary over 64 of its columns).
fn rotary() -> Rotary {
    Rotary::Interleaved {
        width: 2 * ROTARY_PAIRS,
        base: 10_000_000.0,
        sections: vec![ROTARY_PAIRS],
        axis_pattern: vec![0],
    }
}

/// The plane elements and widths per head of `layout`'s history: dense
/// keys and values in the activation dtype, or each affine codec plane.
fn planes(layout: KvCodec) -> Result<Vec<(Element, u64)>, ExecutorCostsError> {
    let spec = layout.spec(DType::BF16, HEAD_WIDTH as usize, HEAD_WIDTH as usize);
    match spec.encoding() {
        HistoryEncoding::Dense => Ok(vec![(Element::bf16(), HEAD_WIDTH); 2]),
        HistoryEncoding::Affine => {
            let geometry = spec
                .affine_geometry()
                .map_err(|cause| failed("affine history")(&cause))?;
            Ok(vec![
                (Element::u32(), geometry.key_codes as u64),
                (Element::f16(), geometry.key_coefficients as u64),
                (Element::u32(), geometry.value_codes as u64),
                (Element::f16(), geometry.value_coefficients as u64),
            ])
        }
        HistoryEncoding::Unsupported => Err(ExecutorCostsError(format!(
            "{} history has no native attention entry",
            layout.identity()
        ))),
    }
}

/// The history bytes one token of `layout` holds over the reference
/// key/value heads.
fn token_bytes(layout: KvCodec) -> Result<u64, ExecutorCostsError> {
    planes(layout)?
        .iter()
        .try_fold(0u64, |total, (element, width)| {
            element
                .canonical_byte_len(&[KV_HEADS, *width])
                .map(|bytes| total + bytes)
                .map_err(|cause| failed("history planes")(&cause))
        })
}

/// The bytes of `layout`'s largest history plane of one sequence at
/// `depth` tokens: its slab of `depth` + 1 rows, which one allocation
/// holds.
fn largest_plane(layout: KvCodec, depth: u64) -> Result<u64, ExecutorCostsError> {
    planes(layout)?
        .iter()
        .try_fold(0u64, |largest, (element, width)| {
            element
                .canonical_byte_len(&[depth + 1, KV_HEADS, *width])
                .map(|bytes| largest.max(bytes))
                .map_err(|cause| failed("history planes")(&cause))
        })
}

/// The slab store of one sequence's `layout` history at `depth` tokens, as
/// the engine's geometry rule lays out a store of its `depth` + 1 rows: its
/// rows per slab, regions, slab layout and slabs. A slab holds no more
/// pages than the rows take: the rule's slabs fill a byte target that a
/// shallow history reads a small part of, and every argument set of a
/// rotation holds a store of its own, so a rotation of many shallow sets
/// would allocate many times the bytes it spans (625 sets of 64 MiB, 41 GB,
/// for the 1.7 MB a 4-vCPU EPYC's dense history reads at depth 420).
fn history_store(
    layout: KvCodec,
    depth: u64,
) -> Result<(u64, Vec<SlabRegion>, SlabLayout, u64), ExecutorCostsError> {
    let rows = depth + 1;
    let geometry = magnitude_state::history_geometry(
        token_bytes(layout)?,
        usize::try_from(rows).map_err(|_| ExecutorCostsError("depth exceeds usize".into()))?,
        0,
    )
    .map_err(ExecutorCostsError)?;
    let page_rows = geometry.page_rows as u64;
    let rows_per_slab = (geometry.slab_rows as u64).min(rows.next_multiple_of(page_rows));
    let regions = planes(layout)?
        .into_iter()
        .map(|(element, width)| SlabRegion {
            element,
            row_shape: vec![KV_HEADS, width],
        })
        .collect::<Vec<_>>();
    let slab = SlabLayout::for_regions(rows_per_slab, rows, &regions)
        .map_err(|cause| failed("history slabs")(&cause))?;
    Ok((rows_per_slab, regions, slab, rows.div_ceil(rows_per_slab)))
}

/// The bytes from one slab's start to the next's in a run of them: its
/// bytes, to the next multiple of its alignment.
fn slab_stride(slab: &SlabLayout) -> u64 {
    slab.slab_bytes.next_multiple_of(slab.slab_alignment)
}

impl AttendCase {
    /// The rotated operand of `sets` argument sets: each set's history, one
    /// run of its store's slabs.
    fn operand(layout: KvCodec, depth: u64, sets: u64) -> Result<Operand, ExecutorCostsError> {
        let (_, _, slab, slabs) = history_store(layout, depth)?;
        Ok(Operand {
            element: Element::bool(),
            extents: vec![slabs * slab_stride(&slab)],
            count: sets,
            alignment: slab.slab_alignment,
        })
    }

    /// `sets` argument sets, each with history planes of its own along
    /// `route`; the row's operands and tables are one set's for all.
    fn new(
        device: &Device,
        backings: &mut Backings,
        layout: KvCodec,
        route: &Route,
        depth: u64,
        sets: u64,
    ) -> Result<Vec<Self>, ExecutorCostsError> {
        let [histories] = backings.lay_out(device, route, [Self::operand(layout, depth, sets)?])?;
        Self::over(device, backings, layout, depth, histories)
    }

    /// One argument set per history of `histories`, each [`Self::operand`]'s
    /// run of slabs of `backings`, which restore the row each set appends.
    fn over(
        device: &Device,
        backings: &mut Backings,
        layout: KvCodec,
        depth: u64,
        histories: Vec<Tensor>,
    ) -> Result<Vec<Self>, ExecutorCostsError> {
        let error = failed("attention arguments");
        let zeros = |element, extents: &[u64]| {
            Tensor::zeros(device, element, extents).map_err(|cause| error(&cause))
        };
        let host = |element, extents: &[u64], bytes: Vec<u8>| {
            Tensor::from_host(device, element, extents, &bytes).map_err(|cause| error(&cause))
        };
        let words = |values: &[i32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        let floats = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        let position =
            i32::try_from(depth).map_err(|_| ExecutorCostsError("depth exceeds i32".into()))?;
        // One row of one sequence appending at `depth` and seeing every
        // token before it, as the batch builder lays it out.
        let slot = Slot {
            rows: vec![Row {
                token: 0,
                coordinates: [position, position, position, 0],
                histories: vec![RowHistory {
                    visible: vec![[0, position]],
                    fresh_start: 0,
                    bidirectional_end: None,
                    destination: position,
                }],
                demand: Demand::NONE,
                select: None,
            }],
            bank: 1,
            previous_tape: 0,
            following_bank: 2,
            stop: 1,
        };
        let tables = PackedRowTables::pack(
            &[slot],
            1,
            ClassLimits {
                rows: 1,
                segments: 1,
            },
        )
        .map_err(|cause| error(&cause))?;
        let history = &tables.histories[0];
        let rotary = rotary();
        let components = crate::operators::attention::graph::rotary_components(&rotary)
            .map_err(|cause| error(&cause))?;
        let rows = depth + 1;
        let heads = KV_HEADS * GROUP;
        let query = zeros(Element::bf16(), &[1, heads, HEAD_WIDTH])?;
        let gate = zeros(Element::bf16(), &[1, heads, 0])?;
        let key = zeros(Element::bf16(), &[1, 1, KV_HEADS * HEAD_WIDTH])?;
        let value = zeros(Element::bf16(), &[1, 1, KV_HEADS * HEAD_WIDTH])?;
        let norms = zeros(Element::f32(), &[0, HEAD_WIDTH])?;
        let rotary_components = host(Element::i32(), &[ROTARY_PAIRS], words(&components))?;
        let rotary_frequencies = host(
            Element::f32(),
            &[ROTARY_PAIRS],
            floats(&crate::operators::attention::graph::rotary_frequencies(
                &rotary,
            )),
        )?;
        let rotary_amplitudes = host(
            Element::f32(),
            &[ROTARY_PAIRS],
            floats(&crate::operators::attention::graph::rotary_amplitudes(
                &rotary,
            )),
        )?;
        let coordinates = host(Element::i32(), &[1, 4], words(&tables.coordinates.concat()))?;
        let visible = host(
            Element::i32(),
            &[1, 1, 2],
            words(&history.visible.concat().concat()),
        )?;
        let fresh = host(Element::i32(), &[1, 2], words(&history.fresh.concat()))?;
        let destinations = host(Element::i32(), &[1], words(&history.destinations))?;
        // Each set's history in a store of its own over its run of slabs.
        let (rows_per_slab, regions, slab, slabs) = history_store(layout, depth)?;
        let slab_rows = u32::try_from(rows_per_slab)
            .map_err(|_| ExecutorCostsError("history slab rows exceed u32".into()))?;
        histories
            .into_iter()
            .map(|run| {
                let views = (0..slabs)
                    .map(|index| {
                        run.region(
                            Element::bool(),
                            &[slab.slab_bytes],
                            index * slab_stride(&slab),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|cause| error(&cause))?;
                let store = SlabTensor::over(device, rows_per_slab, rows, regions.clone(), views)
                    .map_err(|cause| error(&cause))?;
                // The row the call appends, at `depth`, in every plane.
                backings.writes(
                    (0..regions.len())
                        .map(|region| store.region_rows(region, depth, 1))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|cause| error(&cause))?,
                );
                let planes = (0..regions.len())
                    .map(|region| store.logical_region(region).map_err(|cause| error(&cause)))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self {
                    query: query.clone(),
                    gate: gate.clone(),
                    key: key.clone(),
                    value: value.clone(),
                    norms: norms.clone(),
                    rotary_components: rotary_components.clone(),
                    rotary_frequencies: rotary_frequencies.clone(),
                    rotary_amplitudes: rotary_amplitudes.clone(),
                    coordinates: coordinates.clone(),
                    visible: visible.clone(),
                    fresh: fresh.clone(),
                    destinations: destinations.clone(),
                    _history: store,
                    planes,
                    slab_rows,
                })
            })
            .collect()
    }

    fn dense(&mut self) -> attention_decode::Args<'_> {
        let [key, value] = &mut self.planes[..] else {
            unreachable!("dense history has two planes");
        };
        attention_decode::Args {
            query: &self.query,
            gate: &self.gate,
            key: &self.key,
            value: &self.value,
            query_norm: &self.norms,
            key_norm: &self.norms,
            value_norm: &self.norms,
            rotary_components: &self.rotary_components,
            rotary_frequencies: &self.rotary_frequencies,
            rotary_amplitudes: &self.rotary_amplitudes,
            coordinates: &self.coordinates,
            visible: &self.visible,
            fresh: &self.fresh,
            destinations: &self.destinations,
            history_key: key,
            history_value: value,
            epsilon: 1e-6,
            scale: 1.0 / (HEAD_WIDTH as f32).sqrt(),
            gate_function: 0,
            slab_rows: self.slab_rows,
        }
    }

    fn affine(&mut self) -> attention_decode_k8v4::Args<'_> {
        let [key_codes, key_coefficients, value_codes, value_coefficients] = &mut self.planes[..]
        else {
            unreachable!("affine history has four planes");
        };
        attention_decode_k8v4::Args {
            query: &self.query,
            gate: &self.gate,
            key: &self.key,
            value: &self.value,
            query_norm: &self.norms,
            key_norm: &self.norms,
            value_norm: &self.norms,
            rotary_components: &self.rotary_components,
            rotary_frequencies: &self.rotary_frequencies,
            rotary_amplitudes: &self.rotary_amplitudes,
            coordinates: &self.coordinates,
            visible: &self.visible,
            fresh: &self.fresh,
            destinations: &self.destinations,
            history_key_codes: key_codes,
            history_key_coefficients: key_coefficients,
            history_value_codes: value_codes,
            history_value_coefficients: value_coefficients,
            epsilon: 1e-6,
            scale: 1.0 / (HEAD_WIDTH as f32).sqrt(),
            gate_function: 0,
            slab_rows: self.slab_rows,
        }
    }
}

/// The statics of the reference attention: one row that appends its own
/// key and value, no gate and no head norms, and `layout`'s planes.
fn attention_values(layout: KvCodec) -> Result<Vec<(&'static str, u64)>, ExecutorCostsError> {
    let mut values = vec![
        ("KV", KV_HEADS),
        ("G", GROUP),
        ("P", ROTARY_PAIRS),
        ("S", STATIC_WIDTH),
        ("I", 0),
        ("U", 0),
        ("F", 1),
        ("N", 0),
        ("NV", 0),
    ];
    // An affine layout's statics are its planes' widths.
    if let [key_codes, key_coefficients, value_codes, value_coefficients] = planes(layout)?[..] {
        values.extend([
            ("KC", key_codes.1),
            ("KP", key_coefficients.1),
            ("VC", value_codes.1),
            ("VP", value_coefficients.1),
        ]);
    }
    Ok(values)
}

/// A decode attention entry and the arguments of its calls.
trait Attention: Entry + Sized {
    fn entry() -> BoundEntry<Self>;
    fn args(case: &mut AttendCase) -> Self::Args<'_>;
    /// Its program in the walk's set.
    fn program(programs: &ProgramSet) -> &Formation<Self>;
    /// Its kernel as formed.
    fn formed(programs: &Programs) -> &Forming<Self>;
}

impl Attention for attention_decode::Entry {
    fn entry() -> BoundEntry<Self> {
        attention_decode::native_entry_with(attention_decode::Elements { A: Element::bf16() })
    }
    fn args(case: &mut AttendCase) -> Self::Args<'_> {
        case.dense()
    }
    fn program(programs: &ProgramSet) -> &Formation<Self> {
        &programs.dense
    }
    fn formed(programs: &Programs) -> &Forming<Self> {
        &programs.dense
    }
}

impl Attention for attention_decode_k8v4::Entry {
    fn entry() -> BoundEntry<Self> {
        attention_decode_k8v4::native_entry_with(attention_decode_k8v4::Elements {
            A: Element::bf16(),
        })
    }
    fn args(case: &mut AttendCase) -> Self::Args<'_> {
        case.affine()
    }
    fn program(programs: &ProgramSet) -> &Formation<Self> {
        &programs.affine
    }
    fn formed(programs: &Programs) -> &Forming<Self> {
        &programs.affine
    }
}

/// A sealed graph with every port bound to a tensor of its own, run as a
/// decode runs its steps.
struct SealedGraph {
    plan: seismic::NativeGraphPlan,
    bound: Vec<(seismic::NativePort, Tensor)>,
}

/// What of a graph's runs a point's key names it times: the device time
/// of a run (0), or its host time, each run waited for (1) or queued behind
/// the last (2).
fn timing_ordinal(timing: GraphTiming) -> u64 {
    match timing {
        GraphTiming::Device => 0,
        GraphTiming::Waited => 1,
        GraphTiming::Queued => 2,
    }
}

impl SealedGraph {
    /// The graph's runs over its bound tensors, timed as `timing` says.
    fn point(&self, timing: GraphTiming) -> Result<Point, ExecutorCostsError> {
        let error = failed("graph point");
        let mut bindings = self.plan.bindings();
        for (port, tensor) in &self.bound {
            bindings.set(port, tensor).map_err(|cause| error(&cause))?;
        }
        self.plan
            .point(bindings, timing)
            .map(Point::Graph)
            .map_err(|cause| error(&cause))
    }
}

/// The launch graph: [`GRAPH_CALLS`] dependent `conditioning_overlay` calls
/// of one row, each reading the last one's output.
fn launch_graph(
    device: &Device,
    kernel: &NativeKernel<conditioning_overlay::Entry>,
) -> Result<SealedGraph, ExecutorCostsError> {
    let error = failed("launch graph");
    let mut graph = device.native_graph();
    let mut port = || {
        graph
            .port(Element::f32(), &[1, GRAPH_WIDTH])
            .map_err(|cause| error(&cause))
    };
    let mut ports = [port()?, port()?];
    for call in 0..GRAPH_CALLS {
        let [first, second] = &mut ports;
        let (input, out) = if call % 2 == 0 {
            (&*first, second)
        } else {
            (&*second, first)
        };
        graph
            .enqueue(
                kernel,
                conditioning_overlay::WorkflowArgs {
                    input: input.tensor().into(),
                    out: out.tensor_mut().into(),
                },
            )
            .map_err(|cause| error(&cause))?;
    }
    let plan = graph.seal().map_err(|cause| error(&cause))?;
    let bound = ports
        .into_iter()
        .map(|port| {
            Tensor::zeros(device, Element::f32(), &[1, GRAPH_WIDTH])
                .map(|tensor| (port, tensor))
                .map_err(|cause| error(&cause))
        })
        .collect::<Result<_, _>>()?;
    Ok(SealedGraph { plan, bound })
}

/// The arguments of one `dense_output` call at one row.
struct OutputCase {
    residual: Tensor,
    product: Tensor,
    down: Tensor,
    out_rows: Tensor,
    scale: Tensor,
    bound: Tensor,
}

impl OutputCase {
    /// One call at one row onto a residual of H = [`HIDDEN`] from a product
    /// of F `features`.
    fn new(device: &Device, weight: Element, features: u64) -> Result<Self, ExecutorCostsError> {
        let error = failed("dense_output arguments");
        let zeros = |element, extents: &[u64]| {
            Tensor::zeros(device, element, extents).map_err(|cause| error(&cause))
        };
        Ok(Self {
            residual: zeros(Element::f32(), &[1, HIDDEN])?,
            product: zeros(Element::bf16(), &[1, features])?,
            down: zeros(weight, &[HIDDEN, features])?,
            out_rows: every_row(device, 1)?,
            scale: zeros(Element::f32(), &[0])?,
            bound: zeros(Element::f32(), &[bound_extent(weight)])?,
        })
    }

    fn args(&mut self) -> dense_output::Args<'_> {
        dense_output::Args {
            residual: &self.residual,
            product: &self.product,
            down_weight: &self.down,
            out_rows: &self.out_rows,
            down_scale: &self.scale,
            down_bound: &self.bound,
        }
    }
}

/// The arguments of one `conditioning_overlay` call of one row of `width`.
struct CopyCase {
    input: Tensor,
    out: Tensor,
}

impl CopyCase {
    fn new(device: &Device, width: u64) -> Result<Self, ExecutorCostsError> {
        let zeros = || {
            Tensor::zeros(device, Element::f32(), &[1, width])
                .map_err(|cause| failed("conditioning_overlay arguments")(&cause))
        };
        Ok(Self {
            input: zeros()?,
            out: zeros()?,
        })
    }

    fn args(&mut self) -> conditioning_overlay::Args<'_> {
        conditioning_overlay::Args {
            input: &self.input,
            out: &mut self.out,
        }
    }
}

/// `dense_output` onto the residual from the chain's query-wide product.
fn output(weight: Element, features: u64) -> Formation<dense_output::Entry> {
    Formation::new(
        dense_output::native_entry_with(dense_output::Elements {
            A: Element::bf16(),
            DW: weight,
        }),
        &[
            ("H", HIDDEN),
            ("F", features),
            ("DS", 0),
            ("DB", bound_extent(weight)),
        ],
    )
}

fn copy() -> Formation<conditioning_overlay::Entry> {
    Formation::new(conditioning_overlay::native_entry(), &[])
}

fn attend<E: Attention>(layout: KvCodec) -> Result<Formation<E>, ExecutorCostsError> {
    Ok(Formation::new(E::entry(), &attention_values(layout)?))
}

/// The formed kernels of the composition chain's layer.
struct ChainKernels<'p> {
    expand: &'p NativeKernel<dense_expand::Entry>,
    attend: &'p NativeKernel<attention_decode::Entry>,
    output: &'p NativeKernel<dense_output::Entry>,
    copy: &'p NativeKernel<conditioning_overlay::Entry>,
}

impl<'p> ChainKernels<'p> {
    /// The chain's kernels of `programs`, its projection at decode rows over
    /// `weight`.
    fn of(programs: &'p Programs, weight: Element) -> Result<Self, ExecutorCostsError> {
        Ok(Self {
            expand: programs.expand(weight, Rows::Decode)?,
            attend: kernel(&programs.dense)?,
            output: kernel(&programs.output)?,
            copy: kernel(&programs.copy)?,
        })
    }
}

/// The composition chain: [`GRAPH_CALLS`] / [`CHAIN_LAYER_CALLS`] layers of
/// the engine's decode kernels at one row, from the walk's formed kernels:
/// each layer's gate/up projection (`dense_expand`, F `features`), whose
/// product's first query-wide values are the dense attention's query at
/// `depth` tokens and whose whole product the down projection
/// (`dense_output`) takes onto the residual, which a row op's copy
/// (`conditioning_overlay`) hands to the next layer. Every layer binds the
/// same weights and history, as the calls timed alone do, so the two differ
/// only in their composition.
fn composition_chain(
    device: &Device,
    kernels: ChainKernels<'_>,
    backings: &mut Backings,
    weight: Element,
    features: u64,
    route: &Route,
    depth: u64,
) -> Result<SealedGraph, ExecutorCostsError> {
    let error = failed("composition chain");
    let ChainKernels {
        expand: expand_kernel,
        attend: attend_kernel,
        output: output_kernel,
        copy: copy_kernel,
    } = kernels;
    // The weights and the history in one layout, so they span bytes of
    // their own.
    let [gate_weights, up_weights] = ExpandCase::operands(weight, features, 1)?;
    let [gates, ups, histories] = backings.lay_out(
        device,
        route,
        [
            gate_weights,
            up_weights,
            AttendCase::operand(KvCodec::Dense, depth, 1)?,
        ],
    )?;
    let expands = ExpandCase::over(device, weight, 1, gates, ups)?;
    let expand = &expands[0];
    let attends = AttendCase::over(device, backings, KvCodec::Dense, depth, histories)?;
    let attend = &attends[0];
    let output = OutputCase::new(device, weight, features)?;
    let residual_tensor = CopyCase::new(device, HIDDEN)?.out;
    let mut graph = device.native_graph();
    let mut bound = Vec::new();
    let mut bind = |graph: &mut seismic::NativeGraph, tensor: &Tensor| {
        let port = graph
            .port(tensor.element(), tensor.extents())
            .map_err(|cause| error(&cause))?;
        bound.push((port.clone(), tensor.clone()));
        Ok::<_, ExecutorCostsError>(port)
    };
    let mut residual = bind(&mut graph, &residual_tensor)?;
    let norm = bind(&mut graph, &expand.norm)?;
    let gate = bind(&mut graph, &expand.gate)?;
    let up = bind(&mut graph, &expand.up)?;
    let rows = bind(&mut graph, &expand.out_rows)?;
    let scale = bind(&mut graph, &expand.scale)?;
    let expand_bound = bind(&mut graph, &expand.bound)?;
    let down = bind(&mut graph, &output.down)?;
    let down_bound = bind(&mut graph, &output.bound)?;
    let attention_gate = bind(&mut graph, &attend.gate)?;
    let key = bind(&mut graph, &attend.key)?;
    let value = bind(&mut graph, &attend.value)?;
    let norms = bind(&mut graph, &attend.norms)?;
    let components = bind(&mut graph, &attend.rotary_components)?;
    let frequencies = bind(&mut graph, &attend.rotary_frequencies)?;
    let amplitudes = bind(&mut graph, &attend.rotary_amplitudes)?;
    let coordinates = bind(&mut graph, &attend.coordinates)?;
    let visible = bind(&mut graph, &attend.visible)?;
    let fresh = bind(&mut graph, &attend.fresh)?;
    let destinations = bind(&mut graph, &attend.destinations)?;
    let mut history_key = bind(&mut graph, &attend.planes[0])?;
    let mut history_value = bind(&mut graph, &attend.planes[1])?;
    for _ in 0..GRAPH_CALLS / CHAIN_LAYER_CALLS {
        let product = graph
            .enqueue(
                expand_kernel,
                dense_expand::WorkflowArgs {
                    residual: residual.tensor().into(),
                    norm: norm.tensor().into(),
                    gate_weight: gate.tensor().into(),
                    up_weight: up.tensor().into(),
                    out_rows: rows.tensor().into(),
                    eps: 1e-6,
                    activation: 0,
                    gate_scale: scale.tensor().into(),
                    up_scale: scale.tensor().into(),
                    gate_bound: expand_bound.tensor().into(),
                    up_bound: expand_bound.tensor().into(),
                },
            )
            .map_err(|cause| error(&cause))?
            .value;
        // The attention's query is the product's first query-wide values.
        let query = product
            .reshape(&[features / QUERY_WIDTH, QUERY_WIDTH])
            .slice_leading(0, 1)
            .reshape(&[1, KV_HEADS * GROUP, HEAD_WIDTH]);
        graph
            .enqueue(
                attend_kernel,
                attention_decode::WorkflowArgs {
                    query: (&query).into(),
                    gate: attention_gate.tensor().into(),
                    key: key.tensor().into(),
                    value: value.tensor().into(),
                    query_norm: norms.tensor().into(),
                    key_norm: norms.tensor().into(),
                    value_norm: norms.tensor().into(),
                    rotary_components: components.tensor().into(),
                    rotary_frequencies: frequencies.tensor().into(),
                    rotary_amplitudes: amplitudes.tensor().into(),
                    coordinates: coordinates.tensor().into(),
                    visible: visible.tensor().into(),
                    fresh: fresh.tensor().into(),
                    destinations: destinations.tensor().into(),
                    history_key: history_key.tensor_mut().into(),
                    history_value: history_value.tensor_mut().into(),
                    epsilon: 1e-6,
                    scale: 1.0 / (HEAD_WIDTH as f32).sqrt(),
                    gate_function: 0,
                    slab_rows: attend.slab_rows,
                },
            )
            .map_err(|cause| error(&cause))?;
        let summed = graph
            .enqueue(
                output_kernel,
                dense_output::WorkflowArgs {
                    residual: residual.tensor().into(),
                    product: (&product).into(),
                    down_weight: down.tensor().into(),
                    out_rows: rows.tensor().into(),
                    down_scale: scale.tensor().into(),
                    down_bound: down_bound.tensor().into(),
                },
            )
            .map_err(|cause| error(&cause))?
            .value;
        graph
            .enqueue(
                copy_kernel,
                conditioning_overlay::WorkflowArgs {
                    input: (&summed).into(),
                    out: residual.tensor_mut().into(),
                },
            )
            .map_err(|cause| error(&cause))?;
    }
    let plan = graph.seal().map_err(|cause| error(&cause))?;
    Ok(SealedGraph { plan, bound })
}

/// One formed kernel's calls cycling through the argument sets `cases`
/// makes, its rotated operands views of the job's backings; `kernel` picks
/// it from the walk's programs.
fn calls<'f, E: Entry, C: 'f>(
    kernel: impl for<'p> Fn(&'p Programs) -> Result<&'p NativeKernel<E>, ExecutorCostsError> + 'f,
    cases: impl Fn(&Device, &mut Backings) -> Result<Vec<C>, ExecutorCostsError> + 'f,
    args: for<'a> fn(&'a mut C) -> E::Args<'a>,
) -> impl FnMut(&Device, &Programs, &mut Backings) -> Result<Point, Unformed<ExecutorCostsError>> + 'f
{
    move |device, programs, backings| {
        let kernel = kernel(programs).map_err(Unformed::Never)?;
        let mut cases = cases(device, backings).map_err(Unformed::Now)?;
        kernel
            .point(cases.iter_mut().map(args).collect())
            .map(Point::Kernel)
            .map_err(|error| Unformed::Now(failed(E::NAME)(&error)))
    }
}

/// The formats measured along `route`: every measured format where it is a
/// route of device storage's placement, which holds a load's weights;
/// otherwise the activation dtype alone, as only rows are uploaded.
fn route_formats(route: &Route, formats: &[Element]) -> Vec<Element> {
    if route.storage == StorageKind::Device {
        formats.to_vec()
    } else {
        vec![Element::bf16()]
    }
}

/// One timing of the launch graph. A host timing is made only once every
/// program is formed ([`Programs::settled`]), so no compile shares the host
/// with it.
fn launch_graph_point<T: Times>(
    times: &mut T,
    programs: &ProgramSet,
    copy: &KernelIdentity,
    timing: GraphTiming,
) -> Result<Timing, Stop<T::Missing>> {
    let point = PointSpec::new(
        vec![copy.clone()],
        &[("calls", GRAPH_CALLS as u64), ("width", GRAPH_WIDTH)],
        &[("timing", timing_ordinal(timing))],
    );
    times.time(point, programs, &mut |device, formed, _| {
        if timing != GraphTiming::Device {
            formed.settled();
        }
        let copy = kernel(&formed.copy).map_err(Unformed::Never)?;
        launch_graph(device, copy)
            .and_then(|graph| graph.point(timing))
            .map_err(Unformed::Now)
    })
}

/// The walk: every point executor costs are made of, in order, and the
/// costs derived from their timings.
fn walk<T: Times>(
    topology: &DeviceTopology,
    executor: &Executor,
    times: &mut T,
) -> Result<ExecutorCosts, Stop<T::Missing>> {
    let backend = executor.backend();
    let formats = &measured_formats(backend);
    let cache = largest_cache(topology, executor);
    let largest = executor.limits().max_allocation.value;
    let chain_format = chain_element(backend).ok_or_else(|| {
        Stop::Failed(ExecutorCostsError(format!(
            "{CHAIN_FORMAT:?} has no resident form on {}",
            backend.as_str()
        )))
    })?;
    let programs = ProgramSet::new(Sizes::of(backend), formats, chain_format, largest)?;
    let copy_identity = programs.copy.identity(backend)?;
    let (routes, placements) = Route::all(topology, executor)?;
    // Device storage's first route: where the figures not of a route (the
    // matrix throughputs, the history costs, the composition chain) are
    // measured, as a load's weights and history are allocated first.
    let device_route = &routes.device;

    let mut unmeasured = Vec::new();
    let mut optional = |figure: Figure, stop: Stop<T::Missing>| match stop {
        Stop::Point(error) | Stop::Failed(error) => {
            unmeasured.push(Unmeasured {
                figure,
                reason: error.0,
            });
            Ok(())
        }
        missing @ Stop::Missing(_) => Err(missing),
    };

    // Each format's weight rate along each route of device storage's
    // placement. A route of another storage reads host memory, so it is
    // timed last, once no compile shares the host (`weight_rate`).
    let mut weight_rates = Vec::new();
    let (device_storage, host_storage): (Vec<_>, Vec<_>) = routes
        .iter()
        .partition(|route| route.storage == StorageKind::Device);
    for route in device_storage {
        for format in route_formats(route, formats) {
            match weight_rate(backend, &programs, format, route, cache, times) {
                Ok(rate) => weight_rates.push(rate),
                // The activation dtype's rate along a route of device
                // storage's placement prices every load's rows and state:
                // it is no option.
                Err(stop) if format == Element::bf16() => {
                    return Err(stop);
                }
                Err(stop) => optional(Figure::WeightRate(route.route, format), stop)?,
            }
        }
    }

    // The launch graph's runs' device time. It follows the weight rates: the
    // first point measured on a device establishes its sustained clock with
    // its own work, and the graph's trivial calls leave a GPU at a clock it
    // does not sustain under a step's load (an RTX 3090's Vulkan launch read
    // 13.6 µs so, 3.1 µs after streaming work). Its host times are the last
    // points (`launch_graph_point`).
    let device = launch_graph_point(times, &programs, &copy_identity, GraphTiming::Device)?;
    // A run's device time is its calls' (the launch cost is its share of
    // one).
    let graph_calls = GRAPH_CALLS as f64;
    let launch = device.median / graph_calls;
    let launch_seconds = MeasuredCost {
        value: launch,
        interval: device
            .interval
            .map(|(low, high)| (low / graph_calls, high / graph_calls)),
        samples: device.samples,
    };

    // Each history layout at two depths in its bytes' regime, searched from
    // the depth the activation dtype's rate along the route predicts.
    let activation_rate = weight_rates
        .iter()
        .find(|rate| rate.route == device_route.route && rate.format == Element::bf16())
        .expect("the walk measures the activation dtype's rate along device storage or ends")
        .bytes_per_second
        .value;
    // The dense history first: the composition chain reads its depth. Every
    // other layout follows the matrix throughputs, so its program's compile
    // (the affine attention's is the job's longest) ends meanwhile.
    let layout_history =
        |layout: KvCodec, times: &mut T| match layout.spec(DType::BF16, 1, 1).encoding() {
            HistoryEncoding::Dense => history::<attention_decode::Entry, _>(
                backend,
                &programs,
                layout,
                device_route,
                cache,
                largest,
                launch,
                activation_rate,
                times,
            ),
            HistoryEncoding::Affine => history::<attention_decode_k8v4::Entry, _>(
                backend,
                &programs,
                layout,
                device_route,
                cache,
                largest,
                launch,
                activation_rate,
                times,
            ),
            HistoryEncoding::Unsupported => Err(Stop::Failed(ExecutorCostsError(
                "the layout has no native attention entry".into(),
            ))),
        };
    let (dense_layouts, later_layouts): (Vec<KvCodec>, Vec<KvCodec>) = HISTORY_LAYOUTS
        .into_iter()
        .partition(|layout| *layout == KvCodec::Dense);
    let mut histories = Vec::new();
    for layout in dense_layouts {
        match layout_history(layout, times) {
            Ok(history) => histories.push(history),
            Err(stop) => optional(Figure::HistoryCost(layout), stop)?,
        }
    }

    // The composition chain, over the dense history at the depth its bytes'
    // regime was measured at, and its calls alone.
    let depth = histories
        .iter()
        .find(|history| history.layout == KvCodec::Dense)
        .map(|history| history.depths[1])
        .ok_or_else(|| {
            Stop::Failed(ExecutorCostsError(
                "the dense attention was not measured".into(),
            ))
        })?;
    let (features, expand_chain) = programs.expand(chain_format);
    let kernels = vec![
        expand_chain.identity(backend)?,
        programs.dense.identity(backend)?,
        programs.output.identity(backend)?,
        copy_identity.clone(),
    ];
    let chain = times.time(
        PointSpec::new(
            kernels.clone(),
            &[("calls", GRAPH_CALLS as u64), ("rows", 1)],
            &[("depth", depth)],
        ),
        &programs,
        &mut |device, formed, backings| {
            let kernels = ChainKernels::of(formed, chain_format).map_err(Unformed::Never)?;
            composition_chain(
                device,
                kernels,
                backings,
                chain_format,
                features,
                device_route,
                depth,
            )
            .and_then(|chain| chain.point(GraphTiming::Device))
            .map_err(Unformed::Now)
        },
    )?;
    // Each call alone is the point of its kind at the chain's shape and one
    // argument set: where the history's depth search timed that point, its
    // record serves both.
    let alone = |index: usize, point: &[(&str, u64)]| {
        PointSpec::new(vec![kernels[index].clone()], &[("rows", 1)], point)
    };
    let alone = [
        times.time(
            alone(0, &weight_point(features, 1, device_route)),
            &programs,
            &mut calls::<dense_expand::Entry, _>(
                move |formed: &Programs| formed.expand(chain_format, Rows::Decode),
                |device, backings| {
                    ExpandCase::new(device, backings, chain_format, device_route, 1, features, 1)
                },
                ExpandCase::args,
            ),
        )?,
        times.time(
            alone(1, &[("depth", depth), ("sets", 1)]),
            &programs,
            &mut calls::<attention_decode::Entry, _>(
                |formed: &Programs| kernel(&formed.dense),
                |device, backings| {
                    AttendCase::new(device, backings, KvCodec::Dense, device_route, depth, 1)
                },
                AttendCase::dense,
            ),
        )?,
        times.time(
            alone(2, &[("features", features), ("sets", 1)]),
            &programs,
            &mut calls::<dense_output::Entry, _>(
                |formed: &Programs| kernel(&formed.output),
                |device, _| Ok(vec![OutputCase::new(device, chain_format, features)?]),
                OutputCase::args,
            ),
        )?,
        times.time(
            alone(3, &[("width", HIDDEN), ("sets", 1)]),
            &programs,
            &mut calls::<conditioning_overlay::Entry, _>(
                |formed: &Programs| kernel(&formed.copy),
                |device, _| Ok(vec![CopyCase::new(device, HIDDEN)?]),
                CopyCase::args,
            ),
        )?,
    ];
    let layer = alone.iter().map(|timing| timing.median).sum::<f64>();
    let separate = layer * (GRAPH_CALLS / CHAIN_LAYER_CALLS) as f64;
    let composition = MeasuredCost {
        value: chain.median / separate,
        interval: chain
            .interval
            .map(|(low, high)| (low / separate, high / separate)),
        samples: chain.samples,
    };

    // Each format's throughput at the executor's matrix rows.
    let mut matrices = Vec::new();
    for &format in formats {
        match matrix(backend, &programs, format, device_route, times) {
            Ok(matrix) => matrices.push(matrix),
            Err(stop) => optional(Figure::MatrixThroughput(format), stop)?,
        }
    }
    for layout in later_layouts {
        match layout_history(layout, times) {
            Ok(history) => histories.push(history),
            Err(stop) => optional(Figure::HistoryCost(layout), stop)?,
        }
    }

    for route in host_storage {
        for format in route_formats(route, formats) {
            match weight_rate(backend, &programs, format, route, cache, times) {
                Ok(rate) => weight_rates.push(rate),
                Err(stop) => optional(Figure::WeightRate(route.route, format), stop)?,
            }
        }
    }

    // The launch graph's host time waited for and queued, once no compile
    // runs beside it. A run's host time beyond its device time is the step
    // cost, whose interval is the host time's beyond the median device time.
    let waited = launch_graph_point(times, &programs, &copy_identity, GraphTiming::Waited)?;
    let queued = launch_graph_point(times, &programs, &copy_identity, GraphTiming::Queued)?;
    let beyond = |timing: Timing| MeasuredCost {
        value: timing.median - device.median,
        interval: timing
            .interval
            .map(|(low, high)| (low - device.median, high - device.median)),
        samples: timing.samples,
    };
    let step_round_trip_seconds = beyond(waited);
    let step_pipelined_seconds = beyond(queued);

    Ok(ExecutorCosts {
        weight_rates,
        placements,
        matrices,
        histories,
        composition,
        launch_seconds,
        step_pipelined_seconds,
        step_round_trip_seconds,
        unmeasured,
    })
}

/// The weight rate of `format` along `route`: its `dense_expand` program at
/// one row, its calls rotating through the sets that span the cache.
fn weight_rate<T: Times>(
    backend: seismic::BackendName,
    programs: &ProgramSet,
    format: Element,
    route: &Route,
    cache: u64,
    times: &mut T,
) -> Result<WeightRate, Stop<T::Missing>> {
    let (features, formation) = programs.expand(format);
    let bytes = expand_weight_bytes(format, features)?;
    let sets = rotation_sets(cache, bytes);
    let timing = times.time(
        PointSpec::new(
            vec![formation.identity(backend)?],
            &[("rows", 1)],
            &weight_point(features, sets, route),
        ),
        programs,
        &mut calls::<dense_expand::Entry, _>(
            move |formed: &Programs| {
                // Host memory is timed with no compile sharing the host.
                if route.storage != StorageKind::Device {
                    formed.settled();
                }
                formed.expand(format, Rows::Decode)
            },
            move |device, backings| {
                ExpandCase::new(device, backings, format, route, 1, features, sets)
            },
            ExpandCase::args,
        ),
    )?;
    Ok(WeightRate {
        route: route.route,
        format,
        bytes_per_second: MeasuredCost::rate(bytes as f64, &timing),
        features,
    })
}

/// The point entries of a `dense_expand` call at F `features` over `sets`
/// argument sets, its weights along `route`.
fn weight_point(features: u64, sets: u64, route: &Route) -> [(&'static str, u64); 4] {
    let [storage, along] = route.point();
    [("features", features), ("sets", sets), storage, along]
}

/// The matrix throughput of weights resident as `format`: its
/// `dense_expand` program at the executor's matrix rows ([`Sizes`]) over
/// one argument set (arithmetic, which a cache does not serve).
fn matrix<T: Times>(
    backend: seismic::BackendName,
    programs: &ProgramSet,
    format: Element,
    route: &Route,
    times: &mut T,
) -> Result<MatrixCosts, Stop<T::Missing>> {
    let (features, formation) = programs.expand(format);
    let rows = Rows::Matrix.count(programs.sizes);
    let timing = times.time(
        PointSpec::new(
            vec![formation.identity(backend)?],
            &[("rows", rows)],
            &weight_point(features, 1, route),
        ),
        programs,
        &mut calls::<dense_expand::Entry, _>(
            move |formed: &Programs| formed.expand(format, Rows::Matrix),
            move |device, backings| {
                ExpandCase::new(device, backings, format, route, rows, features, 1)
            },
            ExpandCase::args,
        ),
    )?;
    // A multiply and an add per weight, per row, of the gate and the up
    // projection.
    let flops = 2.0 * 2.0 * (rows * features * HIDDEN) as f64;
    Ok(MatrixCosts {
        format,
        flops: MeasuredCost::rate(flops, &timing),
        features,
    })
}

/// What `E` costs over `layout` at one sequence, its history along
/// `route`: `t = fixed + bytes / rate` through two depths in the bytes'
/// regime. A call over one argument set is sized when it takes at least
/// [`LAUNCH_MULTIPLE`] launch costs and its history at least its fixed part
/// (twice the one-token time). The search starts at the depth the one-token
/// call predicts sized with its history read at `activation_rate` (the
/// activation dtype's weight rate along `route`), halved while its plane
/// exceeds the largest allocation, and doubles until a timed call is sized
/// or no deeper plane fits; the line runs through that depth and twice it,
/// or, where the allocation admits no deeper, through half of it and it,
/// each timed rotating through the sets that span the cache, or over one
/// set where the searched call's bytes from memory are too small a part of
/// it to move it ([`UNROTATED`]).
fn history<E: Attention, T: Times>(
    backend: seismic::BackendName,
    programs: &ProgramSet,
    layout: KvCodec,
    route: &Route,
    cache: u64,
    largest: u64,
    launch: f64,
    activation_rate: f64,
    times: &mut T,
) -> Result<HistoryCosts, Stop<T::Missing>> {
    let identity = E::program(programs).identity(backend)?;
    let per_token = token_bytes(layout)?;
    let time = |depth: u64, sets: u64, times: &mut T| {
        times.time(
            PointSpec::new(
                vec![identity.clone()],
                &[("rows", 1)],
                &[("depth", depth), ("sets", sets)],
            ),
            programs,
            &mut calls::<E, _>(
                |formed: &Programs| kernel(E::formed(formed)),
                move |device, backings| {
                    AttendCase::new(device, backings, layout, route, depth, sets)
                },
                E::args,
            ),
        )
    };
    let fits = |depth: u64| -> Result<bool, Stop<T::Missing>> {
        Ok(largest_plane(layout, depth)? <= largest)
    };
    let one_token = time(1, 1, times)?;
    // A call at depth d takes the one-token time and d - 1 more tokens'
    // bytes at the rate: the least d whose call takes the larger of the two
    // bounds.
    let bound = (LAUNCH_MULTIPLE * launch).max(2.0 * one_token.median);
    let predicted = ((bound - one_token.median) * activation_rate / per_token as f64).ceil();
    let mut depth = (predicted as u64).saturating_add(1).max(2);
    while depth > 1 && !fits(depth)? {
        depth /= 2;
    }
    let searched = loop {
        let timing = time(depth, 1, times)?;
        if (sized(&timing, launch) && timing.median >= 2.0 * one_token.median) || !fits(2 * depth)?
        {
            break timing;
        }
        depth *= 2;
    };
    let (near_depth, far_depth) = if fits(2 * depth)? {
        (depth, 2 * depth)
    } else {
        (depth / 2, depth)
    };
    if near_depth == 0 {
        return Err(Stop::Failed(ExecutorCostsError(
            "the largest allocation holds one token of history: no second depth to separate \
             the fixed cost from"
                .into(),
        )));
    }
    // Where the searched call's bytes, read from memory at the activation
    // rate, are a small part of its time, where they come from cannot move
    // it: its kernel is that much slower than streaming them. Each depth is
    // then timed over one argument set, a call of milliseconds rather than a
    // rotation of a second (a CPU's attention, its history read at 110 to 310
    // MB/s against 30 GB/s streaming), and resolved; else over a rotation.
    let rotated = (depth * per_token) as f64 / activation_rate > UNROTATED * searched.median;
    let sets = |depth: u64| {
        if rotated {
            rotation_sets(cache, depth * per_token)
        } else {
            1
        }
    };
    let near = time(near_depth, sets(near_depth), times)?;
    let far = time(far_depth, sets(far_depth), times)?;
    if !(far.median > near.median) {
        return Err(Stop::Failed(ExecutorCostsError(format!(
            "{far_depth} tokens took no longer than {near_depth}"
        ))));
    }
    let bytes = |depth: u64| (depth * per_token) as f64;
    // The bytes' rate is the far depth's time beyond the near one's, its
    // interval from the far time's; the fixed cost is the near time less
    // its bytes at that rate.
    let between = bytes(far_depth) - bytes(near_depth);
    let rate = between / (far.median - near.median);
    let interval = far.interval.map(|(fastest, slowest)| {
        let rate = |seconds: f64| between / (seconds - near.median).max(f64::MIN_POSITIVE);
        (rate(slowest), rate(fastest))
    });
    // A call takes at least its bytes' time: where the line meets zero
    // tokens below zero seconds (an arithmetic-bound attention's fixed part,
    // well under a millisecond, read at depths whose calls take tens of
    // them), the fixed cost is none.
    let near_bytes = bytes(near_depth) / rate;
    let fixed = |seconds: f64| (seconds - near_bytes).max(0.);
    Ok(HistoryCosts {
        layout,
        fixed_seconds: MeasuredCost {
            value: fixed(near.median),
            interval: near
                .interval
                .map(|(fastest, slowest)| (fixed(fastest), fixed(slowest))),
            samples: near.samples,
        },
        bytes_per_second: MeasuredCost {
            value: rate,
            interval,
            samples: far.samples,
        },
        depths: [near_depth, far_depth],
    })
}

/// The executor `executor` names in `topology`, and its identity.
fn executor_identity(
    topology: &DeviceTopology,
    executor: &Executor,
) -> Result<Identity, ExecutorCostsError> {
    topology.executor_identity(executor.id()).ok_or_else(|| {
        ExecutorCostsError(format!(
            "executor {} is not of the topology's revision",
            executor.selector()
        ))
    })
}

impl ExecutorCosts {
    /// Measure `device`'s cost points through `measurer`: each point of the
    /// walk goes to `measure_all`, which loads it from `store` or reports it to
    /// `attempting`, makes it only when attempted (the walk's kernels formed
    /// with the first, [`ProgramSet::form`]), times it at the
    /// sustained clock until its median is resolved, and stores its record.
    /// `device` holds no path limit: the walk passes over a placement's paths
    /// by holding them. Fails when `device`'s executor is not of its own
    /// topology's revision; otherwise returns every point visited, with what
    /// ended the walk early.
    pub fn measure(
        measurer: &mut Measurer,
        device: &Device,
        store: &dyn RecordStore<PointKey>,
        attempting: &mut dyn FnMut(&PointKey) -> Attempt,
    ) -> Result<Measurement, ExecutorCostsError> {
        let topology = device.topology().clone();
        let executor = device.info().clone();
        let mut measured = Measured {
            records: Records {
                executor: executor_identity(&topology, &executor)?,
                store,
            },
            measurer,
            device,
            attempting,
            programs: std::cell::OnceCell::new(),
            backings: Backings::default(),
            visited: Vec::new(),
        };
        // What the walk derives is the start's to derive from the records;
        // the job reports the records, and why the walk ended early where no
        // visited point says.
        let ended = match walk(&topology, &executor, &mut measured) {
            Ok(_) | Err(Stop::Point(_)) => None,
            Err(Stop::Failed(error)) => Some(error),
            Err(Stop::Missing(never)) => match never {},
        };
        Ok(Measurement {
            points: measured.visited,
            ended,
        })
    }

    /// The costs `store`'s records give `executor`, retracing the walk the job
    /// measures and measuring nothing (forming nothing): complete, or the first
    /// point missing (never attempted, or not to its end: pending). A point
    /// recorded unobtainable leaves its format, layout or upload-only route
    /// unmeasured with the recorded reason, or fails the costs when it is one
    /// every estimate needs (the launch graph, the weight rate of each route of
    /// device storage's placement, the dense attention and the composition
    /// chain), as does a record that cannot be read.
    pub fn stored(
        topology: &DeviceTopology,
        executor: &Executor,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<StoredCosts, ExecutorCostsError> {
        let mut stored = Retraced {
            records: Records {
                executor: executor_identity(topology, executor)?,
                store,
            },
            read: Vec::new(),
        };
        match walk(topology, executor, &mut stored) {
            Ok(costs) => Ok(StoredCosts::Complete {
                costs,
                records: stored.read,
            }),
            Err(Stop::Missing(key)) => Ok(StoredCosts::Missing(key)),
            Err(Stop::Point(error) | Stop::Failed(error)) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every codec a native attention entry reads is a measured layout.
    #[test]
    fn every_stored_history_layout_is_measured() {
        for name in ["dense", "affine-k8v4", "rotated-k4v4"] {
            let codec: KvCodec = name.parse().unwrap();
            let read = codec.spec(DType::BF16, 64, 64).encoding() != HistoryEncoding::Unsupported;
            assert_eq!(HISTORY_LAYOUTS.contains(&codec), read, "{name}");
        }
    }

    /// A CPU's projections are sized to its arithmetic, every other
    /// backend's to a GPU's: a CPU measures F 4096 and its throughput at the
    /// 8 rows a decode step is priced at most, so no point's call takes
    /// seconds.
    #[test]
    fn a_cpus_projections_are_sized_to_its_arithmetic() {
        for backend in seismic::BackendName::ALL {
            let sizes = Sizes::of(backend);
            if backend == seismic::BackendName::Cpu {
                assert_eq!(sizes.features, HIDDEN);
                assert_eq!(
                    sizes.matrix_rows,
                    crate::operators::attention::graph::DECODE_ROWS
                );
            } else {
                assert_eq!((sizes.features, sizes.matrix_rows), (16384, 512));
            }
        }
    }

    /// A rotation spans four times the cache in at least two sets: a call
    /// never reads the bytes the call before it read.
    #[test]
    fn a_rotation_takes_at_least_two_sets() {
        assert_eq!(rotation_sets(32 << 20, 134 << 20), 2);
        assert_eq!(rotation_sets(32 << 20, 64 << 20), 2);
        assert_eq!(rotation_sets(32 << 20, 1 << 20), 128);
        assert_eq!(rotation_sets(UNREPORTED_CACHE, 75 << 20), 14);
    }

    /// The measured formats hold the activation dtype's and the chain's,
    /// and every element a weight can be resident as, each once.
    #[test]
    fn measured_formats_hold_every_resident_element_once() {
        for backend in seismic::BackendName::ALL {
            let formats = measured_formats(backend);
            let layout = resident_layout(ExecutionPath::Native, backend);
            for dense in [DType::BF16, DType::F16, DType::F32] {
                for encoding in Encoding::ALL {
                    for width in [HIDDEN, HIDDEN + 32] {
                        if let Some(element) = resident_element(encoding, width, dense, layout) {
                            assert!(formats.contains(&element), "{backend:?} {element:?}");
                        }
                    }
                }
            }
            assert!(formats.contains(&Element::bf16()), "{backend:?}");
            assert!(
                formats.contains(&chain_element(backend).unwrap()),
                "{backend:?}"
            );
            for (at, format) in formats.iter().enumerate() {
                assert!(!formats[..at].contains(format), "{backend:?} {format:?}");
            }
        }
    }
}
