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
//! touch the host (a route of host storage, the launch graph's host times,
//! and every point of an executor whose device is the host's CPU) are timed
//! only once every compile has ended.
//!
//! The costs need one point: the probe, the activation dtype's rate along
//! device storage's first route, the walk's first. Every other figure is
//! held where its point was read and priced from those that were until
//! then ([`ExecutorCosts`]), so the costs exist, and price every model, once
//! the probe is stored. The walk's steps after it are one plan ([`plan`]),
//! each figure before those whose prices without it are less far off, so
//! what a job's time leaves unread is what is priced closest. A job takes
//! [`MEASUREMENT_SECONDS`] of wall time from its start at most, whatever
//! the machine: each point samples for its part of the time left, or until
//! its median is resolved; nothing is submitted that is forecast to end past
//! it, compiles' waits included, and no point is begun whose making and
//! fewest passes, where records forecast them, would. A point it ends
//! before any sample is not recorded, and the next job measures it.
//!
//! A history layout's costs are read on a ladder of depths
//! ([`HISTORY_DEPTHS`], the depths estimates read), as far as a rung fits
//! its storage and its call a point's share of the job, each rung's calls
//! rotating from memory where memory can move them ([`UNROTATED`]), in at
//! most an eighth of its storage; an estimate prices a depth on the segment
//! of the ladder it lies in. The composition chain's layers read their
//! weights and history from memory as a model's do, and its ratio is its
//! time over its calls' as the other figures price them
//! ([`ChainRead::composition`]), so it scales a step's price to its time. A
//! streaming kernel's figure is timed with its calls rotating through as
//! many argument sets as span four times the largest cache that can serve
//! them ([`CACHE_MULTIPLE`]), so the bytes come from memory: a projection's
//! sets hold megabytes to tens of megabytes each, and a history rung's
//! thirty-two megabytes or more, so a rotation's sets stay few (Metal
//! refuses a rotation of hundreds of thousands of views to its residency
//! set). Every rotated operand of every point is a view of one
//! zero-filled backing per route ([`Backings`]), allocated once a job, so a
//! point's making allocates and fills none of its rotation; the history
//! rows a point appends are zeroed again once it is timed.
//!
//! A point that fails, or that the job records unobtainable, leaves its
//! figure priced from those read; a probe that does means the executor
//! cannot stream its own weights, and it has no costs.

use super::demand::{Bytes, Flops, HistoryRead, Work};
use super::AssessmentError;
use crate::{resident_element, resident_elements, resident_layout, ExecutionPath};
use magnitude_artifacts::gguf::Encoding;
use magnitude_batching::{ClassLimits, Demand, PackedRowTables, Row, RowHistory, Slot};
use magnitude_family_contracts::{Rotary, ScaledActivations};
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
        ScaledActivations::Rounded,
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

/// The calls of the launch graph, and the most of the composition chain:
/// the regime a step is priced in (a decode step makes 100 to 300 calls).
const GRAPH_CALLS: usize = 256;

/// The width of the launch graph's rows: 64 F32 values, one call's least
/// work.
const GRAPH_WIDTH: u64 = 64;

/// The calls of one layer of the composition chain: the gate/up
/// projection, the attention over its first query-wide rows, the down
/// projection onto the residual, and the copy of the residual a row op
/// makes.
const CHAIN_LAYER_CALLS: usize = 4;

/// The wall seconds a measurement job takes at most, from its start,
/// whatever the machine. Its points share them equally ([`plan`]): each
/// samples for its share, or until its median is resolved, so a slower
/// machine takes fewer samples of each point rather than longer, and a
/// faster one ends sooner.
pub const MEASUREMENT_SECONDS: f64 = 10.0;

/// The depths a history layout's ladder is read at, in tokens, as far as
/// its rungs fit ([`Walk::rung`]): from below the shortest context a model
/// serves to 256K, doubling. Estimates read 8K to 1M tokens (25K, 50K and
/// 75K, and each model's context); past the last rung measured, the
/// deepest segment prices them. A straight line through two depths does
/// not run through calls at the others: a history call's time per token
/// falls with depth where more of the device's parallelism fills (a GB10's
/// 20-core CPU: a line through 6K and 12K tokens priced 262K 36% slow), and
/// rises where a GPU's attention splits more (a GB10's CUDA and Vulkan: a
/// line through 64K and 128K priced 16K 25% to 78% slow; 2026-10-22).
pub const HISTORY_DEPTHS: [u64; 6] = [8192, 16384, 32768, 65536, 131072, 262144];

/// The part of the smallest store a history rung's rotation reaches that
/// the rotation may take: a rotation that would exceed it reads fewer sets,
/// and a rung one set of which exceeds it is not read. The job runs while no
/// model is loaded, but the OS and other processes hold some of each store,
/// which a store's capacity does not say (an 8 GB M1's Metal had 2.4 GB to
/// give, 2026-10-21).
const ROTATION_SHARE: u64 = 8;

/// The part of a history call's time below which its bytes' time from
/// memory leaves it timed over one argument set: half the tolerance a
/// median is resolved to ([`seismic::measure::MEDIAN_TOLERANCE`]), so
/// reading every byte from memory, with nothing of it overlapping the
/// call's other work, would move the call by less than its measurement can
/// tell. A GPU's attention reads near its memory's rate, far above it; a
/// CPU's 0.3% to 1% of its call (a 4-vCPU EPYC 7763, 2026-10-21), where a
/// rotation's pass of ten 95 ms calls made one sample a second (a 2-core
/// Ryzen 5600X, 2026-10-22).
const UNROTATED: f64 = seismic::measure::MEDIAN_TOLERANCE / 2.;

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
    /// `timing`'s seconds.
    fn of(timing: &Timing) -> Self {
        Self {
            value: timing.median,
            interval: timing.interval,
            samples: timing.samples,
        }
    }

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
    /// Weight bytes per second its `dense_expand` streams at one row (its
    /// decode form) with its weights along the route.
    pub bytes_per_second: MeasuredCost,
    /// The program's F.
    pub features: u64,
}

/// One allocation route's weight rates: the activation dtype's `A`, which
/// every estimate reads (rows, state and uploads along it), and each other
/// format's read along it. A route of device storage's placement holds a
/// load's weights, so every measured format is read along it; any other
/// holds rows alone.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteCosts<A = Option<WeightRate>> {
    pub route: AccessRoute,
    /// The activation dtype's (bf16) rate, its weights streaming from
    /// memory: along device storage's first route the probe, which the
    /// costs require ([`ExecutorCosts::first`]); along another, where read.
    pub activation: A,
    /// Each other format's rates, where its pilot was read.
    pub formats: Vec<(Element, FormatRates)>,
}

/// A format's rates along one route: its pilot's, its calls over one
/// argument set, and its rate streaming from memory, read over the sets
/// that span the cache (the pilot's own where memory cannot move its calls,
/// [`UNROTATED`]) once a job has read it.
#[derive(Clone, Debug, PartialEq)]
pub struct FormatRates {
    pub pilot: WeightRate,
    pub streamed: Option<WeightRate>,
}

/// The matrix throughput of one format.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixCosts {
    /// FLOPs per second of `dense_expand` at the executor's matrix rows.
    pub flops: MeasuredCost,
    /// The program's F.
    pub features: u64,
}

/// What the decode attention entry costs over one history layout, its
/// history in device storage along its first route: one sequence's call at
/// each rung of its ladder read, the `i`th at [`HISTORY_DEPTHS`]`[i]`
/// tokens, as far as the ladder was read.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryCosts {
    layout: KvCodec,
    /// The history bytes one token of the layout holds.
    per_token: u64,
    rungs: Vec<MeasuredCost>,
}

/// The cost of a history call at one depth: the segment of a layout's
/// ladder the depth lies in (the nearest where it lies outside), as a fixed
/// cost per call and the seconds each history byte read beyond it takes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoryLine {
    pub fixed_seconds: f64,
    pub seconds_per_byte: f64,
}

impl HistoryCosts {
    /// `layout`'s costs from the `rungs` its ladder read, at most one at
    /// each of [`HISTORY_DEPTHS`].
    pub fn new(layout: KvCodec, rungs: Vec<MeasuredCost>) -> Result<Self, ExecutorCostsError> {
        assert!(
            rungs.len() <= HISTORY_DEPTHS.len(),
            "a ladder has a rung at each of its depths at most"
        );
        Ok(Self {
            layout,
            per_token: token_bytes(layout)?,
            rungs,
        })
    }

    /// The codec the history is stored in.
    pub fn layout(&self) -> KvCodec {
        self.layout
    }

    /// Each rung read: its depth in tokens and one call's seconds there.
    pub fn rungs(&self) -> impl Iterator<Item = (u64, &MeasuredCost)> {
        HISTORY_DEPTHS.into_iter().zip(&self.rungs)
    }

    /// The line of the segment `tokens` lies in, on each rung's time at
    /// least that of every rung below it, as a call over more history takes
    /// no less time (a GB10's 20-core CPU read 16K tokens of affine history
    /// no slower than 8K, 2026-10-22). Its seconds per byte are the far
    /// rung's time beyond the near one's, none on a flat segment; its fixed
    /// cost the near time less its bytes' time, none where the line meets
    /// zero tokens below zero seconds (a call takes at least its bytes'
    /// time). Where one rung was read the line runs through it from no
    /// fixed cost, and where none was, at `stream` bytes per second, the
    /// rate memory streams them at, which a call reads its history no
    /// faster than.
    pub fn line(&self, tokens: f64, stream: f64) -> HistoryLine {
        let per_token = self.per_token as f64;
        let rungs = self
            .rungs()
            .scan(0f64, |slowest, (depth, cost)| {
                *slowest = slowest.max(cost.value);
                Some((depth as f64, *slowest))
            })
            .collect::<Vec<_>>();
        match rungs[..] {
            [] => HistoryLine {
                fixed_seconds: 0.,
                seconds_per_byte: 1. / stream,
            },
            [(depth, seconds)] => HistoryLine {
                fixed_seconds: 0.,
                seconds_per_byte: seconds / (depth * per_token),
            },
            _ => {
                let segment = rungs
                    .windows(2)
                    .position(|pair| tokens <= pair[1].0)
                    .unwrap_or(rungs.len() - 2);
                let [(near_depth, near), (far_depth, far)] = [rungs[segment], rungs[segment + 1]];
                let seconds_per_byte = (far - near) / ((far_depth - near_depth) * per_token);
                HistoryLine {
                    fixed_seconds: (near - near_depth * per_token * seconds_per_byte).max(0.),
                    seconds_per_byte,
                }
            }
        }
    }
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

/// A figure no point of the costs measured, which the costs price from the
/// figures that were ([`ExecutorCosts`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derived {
    /// The activation dtype's rate along a route other than device
    /// storage's first.
    ActivationRate(AccessRoute),
    /// A format's rate streaming along a route of device storage's
    /// placement.
    WeightRate(AccessRoute, Element),
    /// The activation dtype's matrix throughput.
    ActivationMatrix,
    /// A format's matrix throughput.
    MatrixThroughput(Element),
    /// A history layout's costs at the depths past its last rung read.
    History(KvCodec),
    /// The composition ratio.
    Composition,
    /// The launch cost.
    Launch,
    /// The step costs.
    Step,
}

/// What the engine's kernels achieve on one executor: every figure an
/// estimate reads. One is required: the probe, the activation dtype's rate
/// along device storage's first route, the walk's first point. Every other
/// is held where it was read, and until then priced from what was by its
/// accessor ([`Derived`]), so the costs give an estimate once the probe is
/// read, and a closer one with every figure read after it.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutorCosts {
    /// Each storage kind's placement as the routes of its paths, in order:
    /// its weight rate is the slowest of theirs.
    pub placements: Placements,
    /// Device storage's first route, the activation dtype's rate along it
    /// the probe.
    pub first: RouteCosts<WeightRate>,
    /// Every other route of the placements.
    pub routes: Vec<RouteCosts>,
    /// The activation dtype's (bf16) matrix throughput, where read.
    pub activation_matrix: Option<MatrixCosts>,
    /// Each other format's matrix throughput, where it was measured.
    pub matrices: Vec<(Element, MatrixCosts)>,
    /// The dense history's costs.
    pub dense: HistoryCosts,
    /// The affine K8/V4 history's costs.
    pub affine: HistoryCosts,
    /// The device time of the composition chain's sealed graph over its
    /// calls' as the other figures price them alone, where read.
    pub composition: Option<MeasuredCost>,
    /// Device seconds of one call inside a sealed graph, where read.
    pub launch_seconds: Option<MeasuredCost>,
    /// Host seconds per run beyond its device time, runs queued back to back
    /// (decode with lookahead), where read.
    pub step_pipelined_seconds: Option<MeasuredCost>,
    /// Host seconds of one submit and wait beyond its device time (decode
    /// without lookahead), where read.
    pub step_round_trip_seconds: Option<MeasuredCost>,
}

impl ExecutorCosts {
    /// The step cost a decode under `lookahead` pays; none until read.
    pub fn step_seconds(&self, lookahead: bool) -> f64 {
        let step = if lookahead {
            self.step_pipelined_seconds
        } else {
            self.step_round_trip_seconds
        };
        step.map_or(0., |step| step.value)
    }

    /// The device seconds of one call inside a sealed graph; none until
    /// read.
    pub fn launch(&self) -> f64 {
        self.launch_seconds.map_or(0., |launch| launch.value)
    }

    /// The composition ratio; until read, 1, as the calls run alone.
    pub fn composition(&self) -> f64 {
        self.composition.map_or(1., |composition| composition.value)
    }

    /// The costs of `route`, a route of the placements: the activation
    /// dtype's bytes per second along it (the probe along device storage's
    /// first route, the rate read along another, and until read there, the
    /// probe), and each other format's rates.
    fn route(&self, route: AccessRoute) -> (f64, &[(Element, FormatRates)]) {
        let probe = &self.first.activation;
        if route == self.first.route {
            return (probe.bytes_per_second.value, &self.first.formats);
        }
        let costs = self
            .routes
            .iter()
            .find(|costs| costs.route == route)
            .expect("every route of the placements has its costs");
        let activation = costs.activation.as_ref().unwrap_or(probe);
        (activation.bytes_per_second.value, &costs.formats)
    }

    /// The bytes per second of `format` along `route`: its rate streaming
    /// from memory; until a job reads it, the slower of its pilot's rate and
    /// the activation dtype's along the route, as a call takes at least its
    /// arithmetic (which its pilot reads from a cache) and at least its
    /// bytes' time from memory. Where its pilot could not be read, the
    /// activation dtype's alone, which streams the bytes at least as fast as
    /// a format that decodes them.
    pub fn route_rate(&self, route: AccessRoute, format: Element) -> f64 {
        let (activation, formats) = self.route(route);
        if format == Element::bf16() {
            return activation;
        }
        match formats.iter().find(|(read, _)| *read == format) {
            Some((_, rates)) => rates.streamed.as_ref().map_or_else(
                || rates.pilot.bytes_per_second.value.min(activation),
                |streamed| streamed.bytes_per_second.value,
            ),
            None => activation,
        }
    }

    /// The bytes per second of `format` in `storage`: the slowest rate of
    /// its placement's paths, since a load's bytes may lie along any of
    /// them.
    pub fn weight_rate(&self, storage: StorageKind, format: Element) -> f64 {
        let placement = self.placements.of(storage);
        placement.rest.iter().fold(
            self.route_rate(placement.first, format),
            |slowest, route| slowest.min(self.route_rate(*route, format)),
        )
    }

    /// The matrix throughput of weights resident as `format`: its measured
    /// throughput; until a job measures it, its pilot's arithmetic rate at
    /// one row along device storage's first route, as a CPU's rows take
    /// about one row's time each (a 12-core M6's CPU ran 8 rows in 7 to 9.4
    /// times one row's time, 2026-10-22), and a GPU, whose rows share its
    /// weights' bytes, reads every throughput in its first job. Where its
    /// pilot could not be read, the activation dtype's, which its arithmetic
    /// runs in once its weights are decoded (an M4 Max's Metal, 2026-10-21:
    /// every format within 2.5% of bf16); the activation dtype's own, until
    /// read, at the probe's one-row rate, by the same bound.
    pub fn matrix_flops(&self, format: Element) -> f64 {
        let activation = self.activation_matrix.as_ref().map_or_else(
            || one_row_flops(Element::bf16(), &self.first.activation),
            |matrix| matrix.flops.value,
        );
        if format == Element::bf16() {
            return activation;
        }
        if let Some((_, matrix)) = self.matrices.iter().find(|(read, _)| *read == format) {
            return matrix.flops.value;
        }
        match self.first.formats.iter().find(|(read, _)| *read == format) {
            Some((_, rates)) => one_row_flops(format, &rates.pilot),
            None => activation,
        }
    }

    /// The figures no point measured, of every format `backend` measures
    /// ([`measured_formats`]): priced where an estimate reads them.
    pub fn derived(&self, backend: seismic::BackendName) -> Vec<Derived> {
        let others = measured_formats(backend)
            .into_iter()
            .filter(|format| *format != Element::bf16())
            .collect::<Vec<_>>();
        let device = &self.placements.device;
        let rates = std::iter::once(device.first)
            .chain(device.rest.iter().copied())
            .flat_map(|route| {
                let (_, formats) = self.route(route);
                others
                    .iter()
                    .filter(|format| {
                        !formats
                            .iter()
                            .any(|(read, rates)| read == *format && rates.streamed.is_some())
                    })
                    .map(move |format| Derived::WeightRate(route, *format))
            });
        let matrices = others
            .iter()
            .filter(|format| {
                self.matrices
                    .iter()
                    .all(|(measured, _)| measured != *format)
            })
            .map(|format| Derived::MatrixThroughput(*format));
        let activations = self
            .routes
            .iter()
            .filter(|costs| costs.activation.is_none())
            .map(|costs| Derived::ActivationRate(costs.route));
        let histories = [&self.dense, &self.affine]
            .into_iter()
            .filter(|history| history.rungs.len() < HISTORY_DEPTHS.len())
            .map(|history| Derived::History(history.layout));
        let unread = [
            (self.activation_matrix.is_none(), Derived::ActivationMatrix),
            (self.composition.is_none(), Derived::Composition),
            (self.launch_seconds.is_none(), Derived::Launch),
            (
                self.step_pipelined_seconds.is_none() || self.step_round_trip_seconds.is_none(),
                Derived::Step,
            ),
        ]
        .into_iter()
        .filter_map(|(unread, figure)| unread.then_some(figure));
        activations
            .chain(rates)
            .chain(unread)
            .chain(histories)
            .chain(matrices)
            .collect()
    }

    /// The decode attention costs of history stored as `layout`; none for
    /// a layout no native attention entry reads, which no load stores.
    pub fn history(&self, layout: KvCodec) -> Result<&HistoryCosts, AssessmentError> {
        match layout.spec(DType::BF16, 1, 1).encoding() {
            HistoryEncoding::Dense => Ok(&self.dense),
            HistoryEncoding::Affine => Ok(&self.affine),
            HistoryEncoding::Unsupported => Err(AssessmentError::Demand(format!(
                "no native attention entry reads {} history",
                layout.identity()
            ))),
        }
    }

    /// A call's cost over `tokens` of `layout`'s history in device storage:
    /// its ladder's line there ([`HistoryCosts::line`]), measured along
    /// device storage's first route, its seconds per byte scaled up by the
    /// slowest route of the placement against that one (in the activation
    /// dtype the history holds), since the kernel reads no faster than the
    /// path its history lies along (spec §2.14).
    pub fn history_line(
        &self,
        layout: KvCodec,
        tokens: f64,
    ) -> Result<HistoryLine, AssessmentError> {
        let measured_along = self.placements.device.first;
        let activation = Element::bf16();
        let along = self.route_rate(measured_along, activation);
        let slowdown = self.weight_rate(StorageKind::Device, activation) / along;
        let line = self.history(layout)?.line(tokens, along);
        Ok(HistoryLine {
            seconds_per_byte: line.seconds_per_byte / slowdown,
            ..line
        })
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
    /// The probe is stored.
    Complete {
        costs: ExecutorCosts,
        /// The records the costs were derived from, in the walk's order:
        /// the evidence they are a function of.
        records: Vec<(PointKey, Record)>,
        /// Whether a later point no record holds is still to be measured:
        /// its figure is priced until it is.
        pending: bool,
    },
    /// The probe, which no record holds: never attempted, or not to its
    /// end.
    Missing(PointKey),
}

/// What a measurement job's walk did.
#[derive(Debug)]
pub struct Measurement {
    /// Why the walk ended before its end where no visited point's outcome
    /// says: the probe's kernel identity that cannot be derived, or a
    /// placement the executor lacks. None when it reached its end or the
    /// probe's own outcome ended it.
    pub ended: Option<ExecutorCostsError>,
    /// Why a step after the probe could not be read outside its points'
    /// timings: its figure stays priced.
    pub failures: Vec<ExecutorCostsError>,
    /// What the job's points were made in and from, which its caller
    /// releases once it has reported the job: every record is stored by
    /// then, and releasing it is no part of the job's time (a GB10's CPU
    /// took 54 ms over a job's rotations and kernels, 2026-10-22).
    pub held: Held,
}

/// The storage a job's points' rotations viewed and the kernels it formed.
pub struct Held {
    _backings: Backings,
    _programs: Option<Programs>,
}

impl fmt::Debug for Held {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Held")
    }
}

/// The least cache a path is bounded by: 256 MiB, the largest memory-side
/// cache shipping (AMD MI300X's Infinity Cache). It is a floor even where a
/// device reports a cache: no API reports a memory-side cache (Infinity
/// Cache, Apple's SLC), which can sit behind a reported L2, nor any Metal or
/// Vulkan device's cache, so no smaller bound is known to clear one.
const UNREPORTED_CACHE: u64 = 256 << 20;

/// Why a step has no timing: for the probe, how the walk ended without its
/// costs ([`walk`]).
enum Stop<M> {
    /// A point no record holds, which nothing measures here: its key where
    /// times come from records alone, and never where every point is timed
    /// (`M` is uninhabited).
    Missing(M),
    /// A point cannot be timed: it failed, or is recorded unobtainable. The
    /// job's visited points report it.
    Point(ExecutorCostsError),
    /// The step failed outside any point's timing: a kernel's identity, a
    /// layout's planes, a placement, a record store that cannot be read.
    Failed(ExecutorCostsError),
    /// The job's deadline passed before a point read a sample: the next job
    /// measures it.
    Expired,
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
    /// The format the composition chain's projections are resident in.
    chain: Element,
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
            chain,
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
    /// while most points are timed on the device ([`Programs::settled`]),
    /// no point waiting past `deadline`.
    fn form(&self, device: &Device, deadline: std::time::Instant) -> Programs {
        let host = matches!(
            device
                .topology()
                .physical_device(device.info().physical())
                .map(|physical| physical.identity()),
            Some(seismic::PhysicalIdentity::Host { .. })
        );
        let programs = Programs {
            deadline,
            host,
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
                let index = |format: Element| {
                    set.expands
                        .iter()
                        .position(|(measured, _, _)| *measured == format)
                        .expect("every measured format has a program")
                };
                let (activation, chain) = (index(Element::bf16()), index(set.chain));
                let others = (0..set.expands.len()).filter(|&other| other != activation);
                let decodes = others
                    .clone()
                    .filter(|&other| other != chain)
                    .map(|other| Job::Expand(other, Rows::Decode));
                let matrices = others.map(|other| Job::Expand(other, Rows::Matrix));
                // The probe's program comes first, so the probe waits on no
                // other compile however few the host's threads; then the
                // programs of the plan's early points ([`plan`]), so those
                // wait on no other format's. The attention programs are one
                // long compile each (the affine one's 1.2 s alone on an RTX
                // 3090's Vulkan, against 0.7 s for a decode projection):
                // they start next, beside the short ones the weight rates
                // read, so each has formed by its history's points.
                let mut jobs = vec![
                    Job::Expand(activation, Rows::Decode),
                    Job::Affine,
                    Job::Dense,
                    Job::Expand(chain, Rows::Decode),
                    Job::Copy,
                    Job::Output,
                    Job::Expand(activation, Rows::Matrix),
                ];
                // The chain's format is a backend's resident form of Q4_K,
                // which may be the activation dtype's.
                jobs.dedup();
                jobs.extend(decodes.chain(matrices));
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
#[derive(Clone, Copy, PartialEq)]
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
    /// When the job's [`MEASUREMENT_SECONDS`] end: a point waits for its
    /// kernels no longer.
    deadline: std::time::Instant,
    /// Whether the executor's device is the host's CPU, whose cores the
    /// compiles run on.
    host: bool,
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

    /// The value, once set; none where `deadline` passes first. A defect of
    /// the thread setting it (a panic) is this thread's too.
    fn wait(&self, deadline: std::time::Instant) -> Option<&T> {
        let mut held = self.lock.lock().expect("pending lock poisoned");
        loop {
            if let Some(value) = self.value.get() {
                return Some(value);
            }
            if self.abandoned.load(std::sync::atomic::Ordering::SeqCst) {
                panic!("the formation thread panicked");
            }
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                return None;
            }
            held = self
                .ready
                .wait_timeout(held, left)
                .expect("pending lock poisoned")
                .0;
        }
    }
}

impl Programs {
    /// `format`'s program formed for `rows`.
    fn expand(
        &self,
        format: Element,
        rows: Rows,
    ) -> Result<&NativeKernel<dense_expand::Entry>, Unformed<ExecutorCostsError>> {
        let (_, decode, matrix) = self
            .expands
            .iter()
            .find(|(measured, _, _)| *measured == format)
            .expect("every measured format has a program");
        match rows {
            Rows::Decode => self.kernel(decode),
            Rows::Matrix => self.kernel(matrix),
        }
    }

    /// A formed kernel for a point, or why it was not: it failed to form
    /// ([`Unformed::Never`]), or the job's deadline passed first
    /// ([`Unformed::Expired`]). A point on the host's CPU waits for every
    /// program, so no compile shares the cores it is timed on.
    fn kernel<'p, E: Entry>(
        &'p self,
        forming: &'p Forming<E>,
    ) -> Result<&'p NativeKernel<E>, Unformed<ExecutorCostsError>> {
        if self.host {
            self.settled()?;
        }
        forming
            .wait(self.deadline)
            .ok_or(Unformed::Expired)?
            .as_ref()
            .map_err(|error| Unformed::Never(error.clone()))
    }

    /// Every program formed, or the job's deadline passed first: no compile
    /// runs beside a point that touches the host.
    fn settled(&self) -> Result<(), Unformed<ExecutorCostsError>> {
        let formed = self
            .expands
            .iter()
            .flat_map(|(_, decode, matrix)| {
                [decode.wait(self.deadline), matrix.wait(self.deadline)]
            })
            .all(|formed| formed.is_some())
            && self.output.wait(self.deadline).is_some()
            && self.dense.wait(self.deadline).is_some()
            && self.affine.wait(self.deadline).is_some()
            && self.copy.wait(self.deadline).is_some();
        if formed {
            Ok(())
        } else {
            Err(Unformed::Expired)
        }
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
    /// from `programs` formed on a device, measured there for at most its
    /// `share` of the job or until its median is resolved, and its record
    /// stored.
    fn time(
        &mut self,
        spec: PointSpec,
        programs: &ProgramSet,
        build: &mut Build<'_>,
        share: Share,
    ) -> Result<Timing, Stop<Self::Missing>>;

    /// The timing of `spec`'s point as [`Times::time`] gives it, what its
    /// making adds to the job's backings released once it is timed: a deep
    /// history rung's rotation, which no later point reads.
    fn time_apart(
        &mut self,
        spec: PointSpec,
        programs: &ProgramSet,
        build: &mut Build<'_>,
        share: Share,
    ) -> Result<Timing, Stop<Self::Missing>>;
}

/// A step's part of the job: one point of the `left` the plan reads from
/// it on, its own among them, each sampling for an equal part of the time
/// the job has left. A point that ends before its part passes the rest
/// to the points after it, and one that takes longer (its fewest samples
/// longer than its part) takes it from them, so the job keeps its time.
#[derive(Clone, Copy, Debug)]
struct Share {
    left: usize,
    /// What reading its point is forecast to take, where records forecast
    /// it (a deeper rung from the rung below, a format's streaming rate and
    /// throughput from its pilot): the job starts no point whose making and
    /// fewest passes, a lead and a sample, end past its time, as the first a
    /// point submits has no reading of its own to forecast it by (a GB10's
    /// CPU started a 262K rung 9.6 s into a job and ended it at 10.85 s,
    /// 2026-10-22).
    forecast: Option<Forecast>,
}

/// What reading a point is forecast to take: the seconds of one pass of its
/// calls, and the bytes its making lays out.
#[derive(Clone, Copy, Debug)]
struct Forecast {
    pass: f64,
    bytes: u64,
}

impl Share {
    /// This share where a pass of its point's calls is forecast to take
    /// `pass` seconds and its making to lay out `bytes`.
    fn forecast(self, pass: f64, bytes: u64) -> Self {
        Self {
            forecast: Some(Forecast { pass, bytes }),
            ..self
        }
    }
}

/// What a point's outcome gives the walk: its timing, or why it has none.
fn timed<M>(outcome: &Outcome<CallError, ExecutorCostsError>) -> Result<Timing, Stop<M>> {
    match outcome {
        Outcome::Loaded(record) => self::outcome(record),
        Outcome::Measured(timing) => Ok(*timing),
        Outcome::Skipped(reason) => Err(unobtainable(reason)),
        Outcome::Unformed(error) | Outcome::Unmade(error) => Err(Stop::Point(error.clone())),
        Outcome::Failed(error) => Err(Stop::Point(failed("timing a point")(error))),
        Outcome::Expired => Err(Stop::Expired),
        Outcome::Store(error) => Err(Stop::Point(failed("a measurement record")(error))),
    }
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
        _: Share,
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

    fn time_apart(
        &mut self,
        spec: PointSpec,
        programs: &ProgramSet,
        build: &mut Build<'_>,
        share: Share,
    ) -> Result<Timing, Stop<PointKey>> {
        self.time(spec, programs, build, share)
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
    /// Told of every point visited, in order, with what became of it, as
    /// soon as it is known.
    visit: &'a mut dyn FnMut(PointKey, Outcome<CallError, ExecutorCostsError>),
    /// When the job's [`MEASUREMENT_SECONDS`] end.
    deadline: std::time::Instant,
}

impl Measured<'_, '_> {
    /// Whether a point of `share` can be read before the job's time ends:
    /// its forecast making, lead and one sample end before it
    /// ([`Share::forecast`]), or nothing forecasts it.
    fn starts(&self, share: Share) -> bool {
        share.forecast.is_none_or(|Forecast { pass, bytes }| {
            let seconds = self.backings.filling(bytes) + 2. * pass;
            std::time::Instant::now() + std::time::Duration::from_secs_f64(seconds)
                <= self.deadline
        })
    }

    /// How a point is sampled in its `share` of the time the job has left:
    /// until its median is resolved or its share is spent, each sample
    /// covering at least tuning's least device time (which sets its
    /// repetitions: tuning's, the one owner).
    fn within(&self, share: Share) -> Measure {
        let remaining = self
            .deadline
            .saturating_duration_since(std::time::Instant::now())
            .as_secs_f64();
        Measure {
            stop: seismic::measure::Stop::Within {
                error_rate: crate::SEARCH_SETTINGS.error_rate,
                seconds: remaining / share.left as f64,
            },
            min_sample_seconds: crate::MIN_SAMPLE_SECONDS,
        }
    }
}

impl Times for Measured<'_, '_> {
    /// Every point is timed where no record holds it.
    type Missing = std::convert::Infallible;

    fn time(
        &mut self,
        spec: PointSpec,
        programs: &ProgramSet,
        build: &mut Build<'_>,
        share: Share,
    ) -> Result<Timing, Stop<Self::Missing>> {
        let key = self.records.key(spec);
        // One no record holds that cannot be read in the time left is not
        // attempted, so not recorded: the next job reads it.
        if !self.starts(share) && self.records.load::<Self::Missing>(&key)?.is_none() {
            return Err(Stop::Expired);
        }
        let how = self.within(share);
        let deadline = self.deadline;
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
                    formed.get_or_init(|| programs.form(device, deadline)),
                    backings,
                )
            })],
            &how,
            seismic::measure::Deadline::new(Some(deadline)),
            self.records.store,
            &mut |key| {
                formed.get_or_init(|| programs.form(device, deadline));
                host(key)
            },
        );
        let (key, outcome) = outcomes.pop().expect("one point is measured");
        let result = timed(&outcome);
        (self.visit)(key, outcome);
        // The next point reads zeros where this one wrote.
        self.backings.restore().map_err(Stop::Failed)?;
        result
    }

    fn time_apart(
        &mut self,
        spec: PointSpec,
        programs: &ProgramSet,
        build: &mut Build<'_>,
        share: Share,
    ) -> Result<Timing, Stop<Self::Missing>> {
        let held = self.backings.held();
        let timed = self.time(spec, programs, build, share);
        self.backings.release_to(&held);
        timed
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
    /// The bytes the job's backings were allocated and zero-filled with, and
    /// the wall seconds that took: the rate a point's making lays out more
    /// at ([`Backings::filling`]).
    filled: (u64, f64),
}

impl Backings {
    /// The wall seconds allocating and zero-filling `bytes` more is
    /// forecast to take, at the rate the job's backings filled so far; none
    /// before any was, when the job has made no point to spend its time on.
    /// Filling is page faults as much as writes (a GB10's CPU took about a
    /// second over a 262K rung's two gigabytes, 2026-10-22).
    fn filling(&self, bytes: u64) -> f64 {
        let (filled, seconds) = self.filled;
        if filled == 0 {
            0.
        } else {
            bytes as f64 * seconds / filled as f64
        }
    }

    /// How many allocations each route's backing holds.
    fn held(&self) -> Vec<usize> {
        self.routes
            .iter()
            .map(|(_, allocations)| allocations.len())
            .collect()
    }

    /// Release every allocation beyond those `held` counted, once the views
    /// of them are restored and gone.
    fn release_to(&mut self, held: &[usize]) {
        self.routes.truncate(held.len());
        for ((_, allocations), &kept) in self.routes.iter_mut().zip(held) {
            allocations.truncate(kept);
        }
    }

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
            let began = std::time::Instant::now();
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
            self.filled.0 += added.iter().sum::<u64>();
            self.filled.1 += began.elapsed().as_secs_f64();
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

/// The FLOPs per second of `format`'s decode projection at one row at its
/// weight rate `rate`: a multiply and an add per weight of its gate and up
/// projections in one call's time at the rate.
fn one_row_flops(format: Element, rate: &WeightRate) -> f64 {
    2.0 * 2.0 * (rate.features * HIDDEN) as f64 / call_seconds(format, rate)
}

/// The seconds of one `dense_expand` call of `format` at one row at its
/// weight rate `rate`: its weights' bytes at the rate.
fn call_seconds(format: Element, rate: &WeightRate) -> f64 {
    let bytes = expand_weight_bytes(format, rate.features)
        .expect("a format whose rate was read has weights of its F");
    bytes as f64 / rate.bytes_per_second.value
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
    /// No sinks (`SK = 0`).
    sinks: Tensor,
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
pub(super) fn token_bytes(layout: KvCodec) -> Result<u64, ExecutorCostsError> {
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
        let sinks = zeros(Element::f32(), &[0, heads])?;
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
                    sinks: sinks.clone(),
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
            sinks: &self.sinks,
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
            sinks: &self.sinks,
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
/// key and value, no gate, head norms or sinks, and `layout`'s planes.
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
        ("SK", 0),
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
    fn of(programs: &'p Programs, weight: Element) -> Result<Self, Unformed<ExecutorCostsError>> {
        Ok(Self {
            expand: programs.expand(weight, Rows::Decode)?,
            attend: programs.kernel(&programs.dense)?,
            output: programs.kernel(&programs.output)?,
            copy: programs.kernel(&programs.copy)?,
        })
    }
}

/// The operands of the composition chain: `sets` layers' projection
/// weights, down weights and dense history in one layout, each set's own, so
/// a chain cycling through them reads them from memory as a model's layers
/// do; the row's operands, one set's for all; the down projection's bound,
/// and the residual each layer's copy writes.
struct ChainOperands {
    expand: Vec<ExpandCase>,
    attend: Vec<AttendCase>,
    downs: Vec<Tensor>,
    down_bound: Tensor,
    residual: Tensor,
    features: u64,
}

impl ChainOperands {
    /// The operands of `sets` layers over `weight` weights at F `features`
    /// and `depth` tokens of dense history, its weights and history along
    /// `route`.
    fn new(
        device: &Device,
        backings: &mut Backings,
        weight: Element,
        features: u64,
        route: &Route,
        depth: u64,
        sets: u64,
    ) -> Result<Self, ExecutorCostsError> {
        let [gate_weights, up_weights] = ExpandCase::operands(weight, features, sets)?;
        let [gates, ups, downs, histories] = backings.lay_out(
            device,
            route,
            [
                gate_weights,
                up_weights,
                Operand::canonical(weight, &[HIDDEN, features], sets)?,
                AttendCase::operand(KvCodec::Dense, depth, sets)?,
            ],
        )?;
        let zeros = |element, extents: &[u64]| {
            Tensor::zeros(device, element, extents)
                .map_err(|cause| failed("composition chain arguments")(&cause))
        };
        Ok(Self {
            expand: ExpandCase::over(device, weight, 1, gates, ups)?,
            attend: AttendCase::over(device, backings, KvCodec::Dense, depth, histories)?,
            downs,
            down_bound: zeros(Element::f32(), &[bound_extent(weight)])?,
            residual: zeros(Element::f32(), &[1, HIDDEN])?,
            features,
        })
    }
}

/// The composition chain: `layers` layers of [`CHAIN_LAYER_CALLS`] calls of
/// the engine's decode kernels at one row, from the walk's formed kernels:
/// each layer's gate/up projection (`dense_expand`, F `features`), whose
/// product's first query-wide values are the dense attention's query at
/// `depth` tokens and whose whole product the down projection
/// (`dense_output`) takes onto the residual, which a row op's copy
/// (`conditioning_overlay`) hands to the next layer. The `i`th layer binds
/// the weights and history of the operands' set `i` modulo their sets.
fn composition_chain(
    device: &Device,
    kernels: ChainKernels<'_>,
    operands: &ChainOperands,
    layers: usize,
) -> Result<SealedGraph, ExecutorCostsError> {
    let error = failed("composition chain");
    let ChainKernels {
        expand: expand_kernel,
        attend: attend_kernel,
        output: output_kernel,
        copy: copy_kernel,
    } = kernels;
    let ChainOperands {
        expand: expands,
        attend: attends,
        downs,
        down_bound,
        residual: residual_tensor,
        features,
    } = operands;
    let features = *features;
    // The row's operands, one set's for all.
    let (expand, attend) = (&expands[0], &attends[0]);
    let mut graph = device.native_graph();
    let mut bound = Vec::new();
    let mut bind = |graph: &mut seismic::NativeGraph, tensor: &Tensor| {
        let port = graph
            .port(tensor.element(), tensor.extents())
            .map_err(|cause| error(&cause))?;
        bound.push((port.clone(), tensor.clone()));
        Ok::<_, ExecutorCostsError>(port)
    };
    let mut residual = bind(&mut graph, residual_tensor)?;
    let norm = bind(&mut graph, &expand.norm)?;
    let rows = bind(&mut graph, &expand.out_rows)?;
    let scale = bind(&mut graph, &expand.scale)?;
    let expand_bound = bind(&mut graph, &expand.bound)?;
    let down_bound = bind(&mut graph, down_bound)?;
    let attention_gate = bind(&mut graph, &attend.gate)?;
    let key = bind(&mut graph, &attend.key)?;
    let value = bind(&mut graph, &attend.value)?;
    let norms = bind(&mut graph, &attend.norms)?;
    let sinks = bind(&mut graph, &attend.sinks)?;
    let components = bind(&mut graph, &attend.rotary_components)?;
    let frequencies = bind(&mut graph, &attend.rotary_frequencies)?;
    let amplitudes = bind(&mut graph, &attend.rotary_amplitudes)?;
    let coordinates = bind(&mut graph, &attend.coordinates)?;
    let visible = bind(&mut graph, &attend.visible)?;
    let fresh = bind(&mut graph, &attend.fresh)?;
    let destinations = bind(&mut graph, &attend.destinations)?;
    // Each set's weights and history.
    let mut sets = expands
        .iter()
        .zip(attends)
        .zip(downs)
        .map(|((expand, attend), down)| {
            Ok::<_, ExecutorCostsError>((
                bind(&mut graph, &expand.gate)?,
                bind(&mut graph, &expand.up)?,
                bind(&mut graph, down)?,
                bind(&mut graph, &attend.planes[0])?,
                bind(&mut graph, &attend.planes[1])?,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let count = sets.len();
    for layer in 0..layers {
        let (gate, up, down, history_key, history_value) = &mut sets[layer % count];
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
                    sinks: sinks.tensor().into(),
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
    kernel: impl for<'p> Fn(&'p Programs) -> Result<&'p NativeKernel<E>, Unformed<ExecutorCostsError>>
        + 'f,
    cases: impl Fn(&Device, &mut Backings) -> Result<Vec<C>, ExecutorCostsError> + 'f,
    args: for<'a> fn(&'a mut C) -> E::Args<'a>,
) -> impl FnMut(&Device, &Programs, &mut Backings) -> Result<Point, Unformed<ExecutorCostsError>> + 'f
{
    move |device, programs, backings| {
        let kernel = kernel(programs)?;
        let mut cases = cases(device, backings).map_err(Unformed::Now)?;
        kernel
            .point(cases.iter_mut().map(args).collect())
            .map(Point::Kernel)
            .map_err(|error| Unformed::Now(failed(E::NAME)(&error)))
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
    share: Share,
) -> Result<Timing, Stop<T::Missing>> {
    let point = PointSpec::new(
        vec![copy.clone()],
        &[("calls", GRAPH_CALLS as u64), ("width", GRAPH_WIDTH)],
        &[("timing", timing_ordinal(timing))],
    );
    let mut build = |device: &Device, formed: &Programs, _: &mut Backings| {
        if timing != GraphTiming::Device {
            formed.settled()?;
        }
        let copy = formed.kernel(&formed.copy)?;
        launch_graph(device, copy)
            .and_then(|graph| graph.point(timing))
            .map_err(Unformed::Now)
    };
    times.time(point, programs, &mut build, share)
}

/// What the walk reads, one step at a time, in its order ([`plan`]).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Step {
    /// A point of a format's weight rate along the route at an index of the
    /// walk's routes ([`Routes::iter`]).
    WeightRate(usize, Element, Pass),
    /// One timing of the launch graph.
    Launch(GraphTiming),
    /// One point of a history layout's ladder.
    Rung(KvCodec, Rung),
    /// The composition chain.
    Chain,
    /// A format's matrix throughput.
    Matrix(Element),
}

/// A point of a weight rate: its pilot, its calls over one argument set,
/// which decides whether they rotate ([`UNROTATED`]); or its rate, read
/// over the sets that span the cache, or the pilot's one where memory
/// cannot move its calls. The activation dtype's calls stream from memory
/// by what its rate is, and have no pilot: its rate prices the bytes every
/// other pilot weighs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    Pilot,
    Rate,
}

/// A point of a history ladder: its pilot, one argument set at the first
/// depth, which decides whether its rungs rotate ([`UNROTATED`]); or the
/// rung at an index of [`HISTORY_DEPTHS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rung {
    Pilot,
    Depth(usize),
}

/// The rungs of a ladder the plan reads early: its two shallowest, the
/// fewest a line with a fixed cost runs through.
const FIRST_RUNGS: usize = 2;

impl Step {
    /// The probe: the activation dtype's rate along device storage's first
    /// route, the walk's first step. The costs need it, and price every
    /// figure not read from it ([`ExecutorCosts`]); and the first point
    /// measured establishes the device's sustained clock with its own work.
    fn probe() -> Self {
        Self::WeightRate(0, Element::bf16(), Pass::Rate)
    }

    /// Whether it reads a format's rate streaming from memory, which its
    /// pilot prices until then.
    fn streams(self) -> bool {
        matches!(self, Self::WeightRate(_, format, Pass::Rate) if format != Element::bf16())
    }
}

impl fmt::Display for Step {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WeightRate(_, format, pass) => {
                write!(formatter, "the {} weight rate's {pass:?}", format.name())
            }
            Self::Launch(timing) => write!(formatter, "the launch graph's {timing:?} time"),
            Self::Rung(layout, rung) => {
                write!(formatter, "the {} attention's {rung:?}", layout.identity())
            }
            Self::Chain => formatter.write_str("the composition chain"),
            Self::Matrix(format) => write!(formatter, "the {} matrix throughput", format.name()),
        }
    }
}

/// Every step of the walk over `routes` and `formats` after its probe
/// ([`Step::probe`]), in its order, each figure before those priced closer
/// without it: first the activation dtype along device storage's other
/// routes. The launch graph's runs follow it, as the graph's trivial calls
/// leave a GPU at a clock it does not sustain under a step's load (an RTX
/// 3090's Vulkan launch read 13.6 µs so, 3.1 µs after streaming work); then
/// the dense history's first rungs, which the composition chain reads (a
/// CPU's attention reads its history at a hundredth of memory's rate, which
/// prices history unread: 110 to 310 MB/s against 30 GB/s on a 4-vCPU EPYC
/// 7763); the activation dtype's matrix throughput, during which the affine
/// attention's compile (the job's longest) ends; and what touches the host
/// (host storage's routes, which hold only rows, and the launch graph's
/// host times), once no compile shares it; then every other format's pilot
/// along each device storage route, without which it is priced at the
/// activation dtype's figures (a 12-core M6's CPU formats stream at 0.05 to
/// 0.58 of bf16 and multiply at 0.13 to 0.83 of it, 2026-10-22). Then the
/// rest, the figures priced furthest off until read first: the formats'
/// streaming rates, which a pilot may price up to twice as fast and which
/// the walk reads in the order their pilots price them furthest off
/// ([`Read::overlap`]); then each layout's deeper rungs (a GB10's CPU priced
/// 75K tokens 33% slow on its first rungs' line, 2026-10-22); then the
/// formats' matrix throughputs, which their pilots price within the 7 to
/// 9.4 eighths of a row's time eight rows took on a 12-core M6's CPU, and
/// which price a step only where arithmetic bounds it.
fn plan(routes: &[&Route], formats: &[Element]) -> Vec<Step> {
    let activation = Element::bf16();
    let others = formats
        .iter()
        .copied()
        .filter(|format| *format != activation)
        .collect::<Vec<_>>();
    let (device, host): (Vec<usize>, Vec<usize>) =
        (0..routes.len()).partition(|&index| routes[index].storage == StorageKind::Device);
    let first_rungs = |layout| {
        [Rung::Pilot, Rung::Depth(0), Rung::Depth(1)].map(move |rung| Step::Rung(layout, rung))
    };
    let mut steps = Vec::new();
    steps.extend(
        device
            .iter()
            .map(|&route| Step::WeightRate(route, activation, Pass::Rate))
            .filter(|step| *step != Step::probe()),
    );
    steps.push(Step::Launch(GraphTiming::Device));
    steps.extend(first_rungs(KvCodec::Dense));
    steps.push(Step::Chain);
    steps.push(Step::Matrix(activation));
    steps.extend(first_rungs(KvCodec::AffineK8V4));
    steps.extend(
        host.iter()
            .map(|&route| Step::WeightRate(route, activation, Pass::Rate)),
    );
    steps.extend([GraphTiming::Waited, GraphTiming::Queued].map(Step::Launch));
    for pass in [Pass::Pilot, Pass::Rate] {
        for &route in &device {
            steps.extend(
                others
                    .iter()
                    .map(|&format| Step::WeightRate(route, format, pass)),
            );
        }
    }
    steps.extend(HISTORY_LAYOUTS.into_iter().flat_map(|layout| {
        (FIRST_RUNGS..HISTORY_DEPTHS.len()).map(move |index| Step::Rung(layout, Rung::Depth(index)))
    }));
    steps.extend(others.iter().map(|&format| Step::Matrix(format)));
    steps
}

/// What a walk gives: the costs, whether a point it reached no record holds
/// (whose figure is priced until a job measures it), and why each step
/// whose figure is otherwise priced could not be read outside its points'
/// timings (a kernel's identity that cannot be derived), which the job
/// reports.
struct Walked {
    costs: ExecutorCosts,
    pending: bool,
    failures: Vec<ExecutorCostsError>,
}

/// The walk: every step of its [`plan`], in order, and the costs derived
/// from their timings. Where the probe, its first, has none, the walk ends
/// without costs. Any other step that has none leaves its figure priced
/// from those read ([`ExecutorCosts`]), and a ladder's step that has none
/// ends the ladder below it.
fn walk<T: Times>(
    topology: &DeviceTopology,
    executor: &Executor,
    times: &mut T,
) -> Result<Walked, Stop<T::Missing>> {
    let backend = executor.backend();
    let formats = &measured_formats(backend);
    let largest = executor.limits().max_allocation.value;
    let chain = chain_element(backend).ok_or_else(|| {
        Stop::Failed(ExecutorCostsError(format!(
            "{CHAIN_FORMAT:?} has no resident form on {}",
            backend.as_str()
        )))
    })?;
    let programs = ProgramSet::new(Sizes::of(backend), formats, chain, largest)?;
    let (routes, placements) = Route::all(topology, executor)?;
    let routes = routes.iter().collect::<Vec<_>>();
    let steps = plan(&routes, formats);
    // Each step is one point, after the probe.
    let points = 1 + steps.len();
    let walk = Walk {
        backend,
        copy: programs.copy.identity(backend)?,
        programs,
        cache: largest_cache(topology, executor),
        largest,
        room: rotation_room(topology, routes[0]),
        // A point's part of the job, which shapes are sized to from records
        // (a history rung, the composition chain): a start derives it from
        // the same plan, so it retraces the same points.
        point_seconds: MEASUREMENT_SECONDS / points as f64,
        routes,
    };
    let probe = walk.probe(
        Share {
            left: points,
            forecast: None,
        },
        times,
    )?;
    let mut read = Read::new(walk.routes.len(), probe);
    let mut left = points - 1;
    let mut pending = false;
    let mut failures = Vec::new();
    let mut steps = steps;
    for index in 0..steps.len() {
        // The formats' streaming rates are read in the order their pilots
        // price them furthest off ([`Read::overlap`]), once every pilot is
        // read: the order follows from records, so a start retraces it.
        let streamed = |step: &Step| step.streams();
        if streamed(&steps[index]) && (index == 0 || !streamed(&steps[index - 1])) {
            let end = index + steps[index..].iter().take_while(|step| streamed(step)).count();
            steps[index..end]
                .sort_by(|near, far| read.overlap(*far).total_cmp(&read.overlap(*near)));
        }
        let step = steps[index];
        let share = Share {
            left,
            forecast: None,
        };
        left -= 1;
        match walk.step(step, share, &mut read, times) {
            Ok(()) => {}
            Err(stop) => {
                if let Step::Rung(layout, _) = step {
                    read.ladder(layout).ended = true;
                }
                match stop {
                    Stop::Missing(_) => pending = true,
                    Stop::Failed(error) => failures.push(failed(&step.to_string())(&error)),
                    // The point's own outcome reports it.
                    Stop::Point(_) | Stop::Expired => {}
                }
            }
        }
    }
    Ok(Walked {
        costs: read.costs(&walk, placements)?,
        pending,
        failures,
    })
}

/// What every step of a walk reads from: the executor's programs, its
/// routes (device storage's first route first: where the figures not of a
/// route, the matrix throughputs, the history costs and the composition
/// chain, are measured, as a load's weights and history are allocated
/// first), the largest cache serving them and the largest allocation, what
/// a history rotation may take, and a point's share of the job.
struct Walk<'r> {
    backend: seismic::BackendName,
    programs: ProgramSet,
    copy: KernelIdentity,
    routes: Vec<&'r Route>,
    cache: u64,
    largest: u64,
    room: u64,
    point_seconds: f64,
}

/// What a history rotation along `route` may take: an eighth of the least
/// of the stores and grants along it ([`ROTATION_SHARE`]).
fn rotation_room(topology: &DeviceTopology, route: &Route) -> u64 {
    route
        .along
        .iter()
        .map(|path| {
            let path = topology
                .access_path(*path)
                .expect("a placement's paths belong to its topology");
            let store = topology
                .store(path.store())
                .expect("a path's store belongs to its topology")
                .capacity()
                .value;
            path.grant().map_or(store, |grant| grant.value.min(store))
        })
        .min()
        .expect("a route has a path")
        / ROTATION_SHARE
}

/// What a walk has read so far, step by step: its probe, and every other
/// figure where read.
struct Read {
    probe: WeightRate,
    /// The activation dtype's rate along each route after the probe's, the
    /// `i`th the route at index `i + 1`.
    activation: Vec<Option<WeightRate>>,
    /// The other formats' rates along each route, by index.
    formats: Vec<Vec<(Element, FormatRates)>>,
    launch: Option<Timing>,
    waited: Option<Timing>,
    queued: Option<Timing>,
    dense: Ladder,
    affine: Ladder,
    chain: Option<ChainRead>,
    activation_matrix: Option<MatrixCosts>,
    matrices: Vec<(Element, MatrixCosts)>,
}

/// A history layout's ladder as the walk has read it.
#[derive(Default)]
struct Ladder {
    /// Whether its calls rotate through the sets that span the cache, once
    /// its pilot is read ([`UNROTATED`]).
    rotated: Option<bool>,
    /// The rungs read, the `i`th at [`HISTORY_DEPTHS`]`[i]` tokens.
    rungs: Vec<MeasuredCost>,
    /// Whether no deeper rung is read.
    ended: bool,
}

impl Read {
    fn new(routes: usize, probe: WeightRate) -> Self {
        Self {
            probe,
            activation: vec![None; routes - 1],
            formats: vec![Vec::new(); routes],
            launch: None,
            waited: None,
            queued: None,
            dense: Ladder::default(),
            affine: Ladder::default(),
            chain: None,
            activation_matrix: None,
            matrices: Vec::new(),
        }
    }

    /// `layout`'s ladder, a measured layout's ([`HISTORY_LAYOUTS`]).
    fn ladder(&mut self, layout: KvCodec) -> &mut Ladder {
        match layout {
            KvCodec::Dense => &mut self.dense,
            KvCodec::AffineK8V4 => &mut self.affine,
            KvCodec::RotatedK4V4 => unreachable!("no step reads a layout no entry reads"),
        }
    }

    /// How far `step`'s format's pilot may price its streaming rate off: its
    /// arithmetic's time over its bytes' time from memory, the smaller over
    /// the larger. Its pilot prices a call at the larger of the two, as if
    /// they overlapped wholly, which a call taking both in turn reads up to
    /// twice as long as where they are alike (a GB10's CPU read Q8_0 models
    /// 42% slower than its pilot priced them, 2026-10-22), and within the
    /// smaller part of it where one dominates. A format whose pilot was not
    /// read has no rate to read.
    fn overlap(&self, step: Step) -> f64 {
        let Step::WeightRate(index, format, Pass::Rate) = step else {
            unreachable!("only streaming rates are ordered")
        };
        let stream = self.stream(index);
        self.formats[index]
            .iter()
            .find(|(read, _)| *read == format)
            .map_or(0., |(_, rates)| {
                let pilot = rates.pilot.bytes_per_second.value;
                pilot.min(stream) / pilot.max(stream)
            })
    }

    /// The activation dtype's rate along the route at `index`: the rate
    /// memory streams it at, the probe's along device storage's first route
    /// and wherever none was read ([`ExecutorCosts::route_rate`]).
    fn stream(&self, index: usize) -> f64 {
        index
            .checked_sub(1)
            .and_then(|other| self.activation[other].as_ref())
            .unwrap_or(&self.probe)
            .bytes_per_second
            .value
    }

    /// The costs of what the walk read.
    fn costs(
        self,
        walk: &Walk<'_>,
        placements: Placements,
    ) -> Result<ExecutorCosts, ExecutorCostsError> {
        // A run's device time is its calls' (the launch cost is its share
        // of one), none where unread.
        let device = self.launch.map_or(0., |device| device.median);
        // A run's host time beyond its device time is the step cost, whose
        // interval is the host time's beyond the median device time.
        let beyond = |timing: Option<Timing>| {
            timing.map(|timing| MeasuredCost {
                value: timing.median - device,
                interval: timing
                    .interval
                    .map(|(low, high)| (low - device, high - device)),
                samples: timing.samples,
            })
        };
        let mut formats = self.formats.into_iter();
        let mut costs = ExecutorCosts {
            first: RouteCosts {
                route: walk.routes[0].route,
                activation: self.probe,
                formats: formats.next().expect("the walk has device storage's first route"),
            },
            routes: walk.routes[1..]
                .iter()
                .zip(self.activation)
                .zip(formats)
                .map(|((route, activation), formats)| RouteCosts {
                    route: route.route,
                    activation,
                    formats,
                })
                .collect(),
            placements,
            activation_matrix: self.activation_matrix,
            matrices: self.matrices,
            dense: HistoryCosts::new(KvCodec::Dense, self.dense.rungs)?,
            affine: HistoryCosts::new(KvCodec::AffineK8V4, self.affine.rungs)?,
            composition: None,
            launch_seconds: self.launch.map(|device| MeasuredCost {
                value: device.median / GRAPH_CALLS as f64,
                interval: device
                    .interval
                    .map(|(low, high)| (low / GRAPH_CALLS as f64, high / GRAPH_CALLS as f64)),
                samples: device.samples,
            }),
            step_pipelined_seconds: beyond(self.queued),
            step_round_trip_seconds: beyond(self.waited),
        };
        // The chain's calls are priced from every other figure.
        costs.composition = self
            .chain
            .map(|chain| chain.composition(&walk.programs, &costs))
            .transpose()?;
        Ok(costs)
    }
}

impl Walk<'_> {
    /// The probe ([`Step::probe`]): the activation dtype's rate along device
    /// storage's first route, rotating through the sets that span the cache.
    fn probe<T: Times>(&self, share: Share, times: &mut T) -> Result<WeightRate, Stop<T::Missing>> {
        weight_rate(
            self.backend,
            &self.programs,
            Element::bf16(),
            self.routes[0],
            self.cache,
            true,
            times,
            share,
        )
    }

    /// Read `step` into `read`.
    fn step<T: Times>(
        &self,
        step: Step,
        share: Share,
        read: &mut Read,
        times: &mut T,
    ) -> Result<(), Stop<T::Missing>> {
        match step {
            Step::WeightRate(index, format, pass) => {
                let route = self.routes[index];
                let rate = |rotated, times: &mut T, share| {
                    weight_rate(
                        self.backend,
                        &self.programs,
                        format,
                        route,
                        self.cache,
                        rotated,
                        times,
                        share,
                    )
                };
                if format == Element::bf16() {
                    // The plan reads the probe's route's rate as the probe.
                    read.activation[index - 1] = Some(rate(true, times, share)?);
                    return Ok(());
                }
                match pass {
                    Pass::Pilot => {
                        let pilot = rate(false, times, share)?;
                        read.formats[index].push((
                            format,
                            FormatRates {
                                pilot,
                                streamed: None,
                            },
                        ));
                    }
                    Pass::Rate => {
                        let stream = read.stream(index);
                        // A format whose pilot was not read has no rates.
                        let Some((_, rates)) = read.formats[index]
                            .iter_mut()
                            .find(|(read, _)| *read == format)
                        else {
                            return Ok(());
                        };
                        // The part of a call its bytes' time from memory
                        // takes, at the rate memory streams them.
                        let pilot = rates.pilot.bytes_per_second.value;
                        let rotated = pilot / stream > UNROTATED;
                        // A call is forecast at the slower of its pilot's
                        // rate and memory's, a pass at its sets' calls.
                        let bytes = expand_weight_bytes(format, rates.pilot.features)?;
                        let sets = weight_sets(self.cache, bytes, rotated);
                        let call = bytes as f64 / pilot.min(stream);
                        let share = share.forecast(call * sets as f64, bytes * sets);
                        let streamed = rate(rotated, times, share)?;
                        rates.streamed = Some(streamed);
                    }
                }
            }
            Step::Launch(timing) => {
                let measured =
                    launch_graph_point(times, &self.programs, &self.copy, timing, share)?;
                *match timing {
                    GraphTiming::Device => &mut read.launch,
                    GraphTiming::Waited => &mut read.waited,
                    GraphTiming::Queued => &mut read.queued,
                } = Some(measured);
            }
            Step::Rung(layout, rung) => {
                let activation_rate = read.stream(0);
                let ladder = read.ladder(layout);
                match layout.spec(DType::BF16, 1, 1).encoding() {
                    HistoryEncoding::Dense => self.rung::<attention_decode::Entry, _>(
                        layout,
                        rung,
                        ladder,
                        activation_rate,
                        times,
                        share,
                    )?,
                    HistoryEncoding::Affine => self.rung::<attention_decode_k8v4::Entry, _>(
                        layout,
                        rung,
                        ladder,
                        activation_rate,
                        times,
                        share,
                    )?,
                    HistoryEncoding::Unsupported => {
                        unreachable!("every measured layout has a native attention entry")
                    }
                }
            }
            Step::Chain => read.chain = Some(self.chain(read, times, share)?),
            Step::Matrix(format) => {
                // Another format's call is forecast at its matrix rows' times
                // its pilot's one-row call, about what a CPU takes
                // ([`ExecutorCosts::matrix_flops`]) and more than a GPU.
                let share = match read.formats[0].iter().find(|(read, _)| *read == format) {
                    Some((_, rates)) => {
                        let rows = Rows::Matrix.count(self.programs.sizes) as f64;
                        let call = rows * call_seconds(format, &rates.pilot);
                        share.forecast(call, expand_weight_bytes(format, rates.pilot.features)?)
                    }
                    None => share,
                };
                let matrix = matrix(
                    self.backend,
                    &self.programs,
                    format,
                    self.routes[0],
                    times,
                    share,
                )?;
                if format == Element::bf16() {
                    read.activation_matrix = Some(matrix);
                } else {
                    read.matrices.push((format, matrix));
                }
            }
        }
        Ok(())
    }
}

impl Walk<'_> {
    /// The composition chain over the dense history's first rung: its
    /// device time, and its layers. Its layers cycle through as many sets of
    /// weights and history as span the cache ([`rotation_sets`]), as many as
    /// the rotation's room holds and a step's calls take at most, so each
    /// reads them from memory as a model's layers do: one set of weights and
    /// history the cache holds runs a chain faster than a model runs (a
    /// GB10's CPU composed one set's layers at 0.855 of their calls alone,
    /// where a 4B model decoded at 1.11 of its calls' price, 2026-10-23).
    fn chain<T: Times>(
        &self,
        read: &Read,
        times: &mut T,
        share: Share,
    ) -> Result<ChainRead, Stop<T::Missing>> {
        let backend = self.backend;
        let programs = &self.programs;
        let chain_format = programs.chain;
        let device_route = self.routes[0];
        let depth = HISTORY_DEPTHS[0];
        let (features, expand_chain) = programs.expand(chain_format);
        let kernels = vec![
            expand_chain.identity(backend)?,
            programs.dense.identity(backend)?,
            programs.output.identity(backend)?,
            self.copy.clone(),
        ];
        // A layer's time as the walk's records price it: its projections'
        // weights at the activation dtype's rate (a forecast, which an
        // M4 Max's Q4_K rate meets within 2%), its attention at the dense
        // history's line, and its copy a launch, each priced as the costs
        // price it where unread.
        let stream = read.stream(0);
        let dense = HistoryCosts::new(KvCodec::Dense, read.dense.rungs.clone())?;
        let launch = read
            .launch
            .map_or(0., |device| device.median / GRAPH_CALLS as f64);
        let weight_bytes = expand_weight_bytes(chain_format, features)?
            + chain_format
                .canonical_byte_len(&[HIDDEN, features])
                .map_err(|error| failed("dense_output weights")(&error))?;
        let history_bytes = AttendCase::operand(KvCodec::Dense, depth, 1)?.bytes()?;
        let line = dense.line(depth as f64, stream);
        let layer = weight_bytes as f64 / stream
            + line.fixed_seconds
            + (depth * token_bytes(KvCodec::Dense)?) as f64 * line.seconds_per_byte
            + launch;
        let most = (GRAPH_CALLS / CHAIN_LAYER_CALLS) as u64;
        let sets = rotation_sets(self.cache, weight_bytes + history_bytes)
            .min(self.room / history_bytes)
            .clamp(1, most);
        // As many layers as a run holds that leaves a point's share of the
        // job twice the fewest samples that bound a median, at least the
        // sets' and at most a step's calls: room for its leads and for the
        // readings past the fewest a wide spread takes.
        let run = self.point_seconds
            / (2 * seismic::measure::fewest_bounding(crate::SEARCH_SETTINGS.error_rate)) as f64;
        let layers = ((run / layer) as u64).clamp(sets, most);
        let timing = times.time(
            PointSpec::new(
                kernels,
                &[("calls", layers * CHAIN_LAYER_CALLS as u64), ("rows", 1)],
                &[("depth", depth), ("sets", sets)],
            ),
            programs,
            &mut |device: &Device, formed: &Programs, backings: &mut Backings| {
                let kernels = ChainKernels::of(formed, chain_format)?;
                let operands = ChainOperands::new(
                    device,
                    backings,
                    chain_format,
                    features,
                    device_route,
                    depth,
                    sets,
                )
                .map_err(Unformed::Now)?;
                composition_chain(device, kernels, &operands, layers as usize)
                    .and_then(|chain| chain.point(GraphTiming::Device))
                    .map_err(Unformed::Now)
            },
            share.forecast(layer * layers as f64, sets * (weight_bytes + history_bytes)),
        )?;
        Ok(ChainRead { timing, layers })
    }
}

/// What the composition chain read: its device time over its layers.
#[derive(Clone, Copy)]
struct ChainRead {
    timing: Timing,
    layers: u64,
}

impl ChainRead {
    /// The calls of one of its layers of `programs`' chain as a decode
    /// demand states them, at one row: the gate/up projection, the
    /// attention over its dense history, the down projection and the copy.
    /// Their rows' bytes, a few kilobytes against the weights' megabytes,
    /// are none.
    fn layer(programs: &ProgramSet) -> Result<[Work; CHAIN_LAYER_CALLS], ExecutorCostsError> {
        let format = programs.chain;
        let (features, _) = programs.expand(format);
        let output = format
            .canonical_byte_len(&[HIDDEN, features])
            .map_err(|error| failed("dense_output weights")(&error))?;
        Ok([
            Work {
                weights: vec![Bytes {
                    format,
                    bytes: expand_weight_bytes(format, features)?,
                }],
                row_flops: vec![Flops {
                    format,
                    flops: 2 * 2 * features * HIDDEN,
                }],
                ..Work::default()
            },
            Work {
                history: Some(HistoryRead {
                    layout: Some(KvCodec::Dense),
                    bytes_per_token: token_bytes(KvCodec::Dense)?,
                    flops_per_token: 2 * KV_HEADS * GROUP * (2 * HEAD_WIDTH),
                    window: None,
                }),
                ..Work::default()
            },
            Work {
                weights: vec![Bytes {
                    format,
                    bytes: output,
                }],
                row_flops: vec![Flops {
                    format,
                    flops: 2 * features * HIDDEN,
                }],
                ..Work::default()
            },
            Work::default(),
        ])
    }

    /// The composition ratio: the chain's time over its calls' as `costs`
    /// price them alone, the chain of `programs` ([`super::estimate::calls_seconds`]), so a step
    /// priced from the same costs is scaled to what composing its calls
    /// takes.
    fn composition(
        &self,
        programs: &ProgramSet,
        costs: &ExecutorCosts,
    ) -> Result<MeasuredCost, ExecutorCostsError> {
        let depth = u32::try_from(HISTORY_DEPTHS[0])
            .map_err(|_| ExecutorCostsError("the chain's depth exceeds u32".into()))?;
        let priced = self.layers as f64
            * super::estimate::calls_seconds(&Self::layer(programs)?, costs, depth)
                .map_err(|error| failed("pricing the composition chain")(&error))?;
        Ok(MeasuredCost {
            value: self.timing.median / priced,
            interval: self
                .timing
                .interval
                .map(|(low, high)| (low / priced, high / priced)),
            samples: self.timing.samples,
        })
    }
}

/// The weight rate of `format` along `route`: its `dense_expand` program at
/// one row, its calls rotating through the sets that span the cache where
/// `rotated`, else over one argument set.
fn weight_rate<T: Times>(
    backend: seismic::BackendName,
    programs: &ProgramSet,
    format: Element,
    route: &Route,
    cache: u64,
    rotated: bool,
    times: &mut T,
    share: Share,
) -> Result<WeightRate, Stop<T::Missing>> {
    let (features, formation) = programs.expand(format);
    let bytes = expand_weight_bytes(format, features)?;
    let sets = weight_sets(cache, bytes, rotated);
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
                    formed.settled()?;
                }
                formed.expand(format, Rows::Decode)
            },
            move |device, backings| {
                ExpandCase::new(device, backings, format, route, 1, features, sets)
            },
            ExpandCase::args,
        ),
        share,
    )?;
    Ok(WeightRate {
        bytes_per_second: MeasuredCost::rate(bytes as f64, &timing),
        features,
    })
}

/// The argument sets a weight rate's calls, each reading `bytes`, take:
/// those that span the cache where `rotated`, else one.
fn weight_sets(cache: u64, bytes: u64, rotated: bool) -> u64 {
    if rotated {
        rotation_sets(cache, bytes)
    } else {
        1
    }
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
    share: Share,
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
        share,
    )?;
    // A multiply and an add per weight, per row, of the gate and the up
    // projection.
    let flops = 2.0 * 2.0 * (rows * features * HIDDEN) as f64;
    Ok(MatrixCosts {
        flops: MeasuredCost::rate(flops, &timing),
        features,
    })
}

impl Walk<'_> {
    /// One point of `layout`'s ladder, read through `E`, its history along
    /// device storage's first route, into `ladder`, with history streaming
    /// from memory at `activation_rate`.
    ///
    /// The pilot times the first depth over one argument set: where its
    /// history's bytes, read from memory at the activation rate, are under
    /// [`UNROTATED`] of its time, where they come from cannot move it, and
    /// every rung is timed over one set, each sample one call, the first
    /// rung loading the pilot's record; else each rung rotates through the
    /// sets that span the cache, so its history comes from memory as it
    /// does at the depths estimates read. A rotation takes at most the
    /// walk's room: where the sets that span the cache exceed it, fewer
    /// sets read the rung, some of its bytes perhaps from a cache.
    ///
    /// A rung is read where its plane fits the largest allocation and one
    /// set of it the room, and, above the first, where its call, priced
    /// from the slowest rung below at its depth, fits a point's share of the
    /// job (a deep rung needs few samples, and reading deeper is what prices
    /// deep contexts: a Ryzen 5600X's attention, read to 16K tokens, priced
    /// 24K to 96K 19% slow, 2026-10-22); a rung above the first releases its
    /// rotation once it is timed ([`Times::time_apart`]). The ladder ends at
    /// the first rung it does not read, and its line runs through those it
    /// did ([`HistoryCosts::line`]).
    fn rung<E: Attention, T: Times>(
        &self,
        layout: KvCodec,
        rung: Rung,
        ladder: &mut Ladder,
        activation_rate: f64,
        times: &mut T,
        share: Share,
    ) -> Result<(), Stop<T::Missing>> {
        if ladder.ended {
            return Ok(());
        }
        let programs = &self.programs;
        let identity = E::program(programs).identity(self.backend)?;
        let per_token = token_bytes(layout)?;
        let route = self.routes[0];
        let point = |depth: u64, sets: u64| {
            PointSpec::new(
                vec![identity.clone()],
                &[("rows", 1)],
                &[("depth", depth), ("sets", sets)],
            )
        };
        let build = |depth: u64, sets: u64| {
            calls::<E, _>(
                |formed: &Programs| formed.kernel(E::formed(formed)),
                move |device, backings| {
                    AttendCase::new(device, backings, layout, route, depth, sets)
                },
                E::args,
            )
        };
        let (index, sets) = match rung {
            Rung::Pilot => (0, 1),
            Rung::Depth(index) => {
                let bytes = HISTORY_DEPTHS[index] * per_token;
                let rotated = ladder
                    .rotated
                    .expect("the plan reads a ladder's pilot before its rungs");
                let sets = if rotated {
                    rotation_sets(self.cache, bytes)
                        .min(self.room / bytes)
                        .max(1)
                } else {
                    1
                };
                (index, sets)
            }
        };
        let depth = HISTORY_DEPTHS[index];
        if largest_plane(layout, depth)? > self.largest || depth * per_token > self.room {
            ladder.ended = true;
            return Ok(());
        }
        let timing = match (rung, ladder.rungs.last()) {
            (Rung::Depth(_), Some(_)) => {
                let below = HISTORY_DEPTHS[index - 1];
                let slowest = ladder
                    .rungs
                    .iter()
                    .map(|cost| cost.value)
                    .fold(0., f64::max);
                let call = slowest * depth as f64 / below as f64;
                if call > self.point_seconds {
                    ladder.ended = true;
                    return Ok(());
                }
                times.time_apart(
                    point(depth, sets),
                    programs,
                    &mut build(depth, sets),
                    share.forecast(call * sets as f64, sets * depth * per_token),
                )?
            }
            _ => times.time(point(depth, sets), programs, &mut build(depth, sets), share)?,
        };
        match rung {
            Rung::Pilot => {
                ladder.rotated =
                    Some((depth * per_token) as f64 / activation_rate > UNROTATED * timing.median);
            }
            Rung::Depth(_) => ladder.rungs.push(MeasuredCost::of(&timing)),
        }
        Ok(())
    }
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
    /// with the first, [`ProgramSet::form`]), times it at the sustained
    /// clock for its share of the time left before `deadline` (the job's
    /// [`MEASUREMENT_SECONDS`] from its start) or until its median is
    /// resolved, and stores its record; `visit` is told what became of each
    /// point as soon as it is known. A point the deadline ends before any
    /// sample expires ([`Outcome::Expired`]), unrecorded, and the probe so
    /// ends the walk. `device` holds no path limit: the walk
    /// passes over a placement's paths by holding them. Fails when
    /// `device`'s executor is not of its own topology's revision; otherwise
    /// returns what ended the walk early, and what the job's points were
    /// made in and from ([`Measurement::held`]).
    pub fn measure(
        measurer: &mut Measurer,
        device: &Device,
        store: &dyn RecordStore<PointKey>,
        deadline: std::time::Instant,
        attempting: &mut dyn FnMut(&PointKey) -> Attempt,
        visit: &mut dyn FnMut(PointKey, Outcome<CallError, ExecutorCostsError>),
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
            visit,
            deadline,
        };
        // What the walk derives is the start's to derive from the records;
        // the job reports the records as they come, and why the walk ended
        // early, or left a figure derived, where no visited point says.
        let walked = walk(&topology, &executor, &mut measured);
        let held = Held {
            _backings: measured.backings,
            _programs: measured.programs.into_inner(),
        };
        let (ended, failures) = match walked {
            Ok(walked) => (None, walked.failures),
            Err(Stop::Point(_) | Stop::Expired) => (None, Vec::new()),
            Err(Stop::Failed(error)) => (Some(error), Vec::new()),
            Err(Stop::Missing(never)) => match never {},
        };
        Ok(Measurement {
            ended,
            failures,
            held,
        })
    }

    /// The costs `store`'s records give `executor`, retracing the walk the job
    /// measures and measuring nothing (forming nothing): complete once the
    /// probe is stored ([`Step::probe`]), pending while a later point no
    /// record holds is still to be measured (its figure priced meanwhile),
    /// and otherwise missing the probe (never attempted, or not to its end).
    /// Another point recorded unobtainable leaves its figure priced; the
    /// probe recorded so fails the costs, as does a record store that cannot
    /// be read.
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
        // A failure outside a point's timing, which leaves its figure
        // derived, is the job's to report: it recurs at each of its walks.
        match walk(topology, executor, &mut stored) {
            Ok(walked) => Ok(StoredCosts::Complete {
                costs: walked.costs,
                records: stored.read,
                pending: walked.pending,
            }),
            Err(Stop::Missing(key)) => Ok(StoredCosts::Missing(key)),
            Err(Stop::Point(error) | Stop::Failed(error)) => Err(error),
            Err(Stop::Expired) => unreachable!("timings from records alone have no deadline"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A depth is priced on the segment of the ladder it lies in, and one
    /// outside it on the nearest: a ladder whose time per token falls with
    /// depth prices each segment at its own rate, where one line through
    /// its first two rungs would price the deepest slow.
    #[test]
    fn a_depth_is_priced_on_its_ladder_segment() {
        let per_token = token_bytes(KvCodec::Dense).unwrap() as f64;
        let cost = |value| MeasuredCost {
            value,
            interval: None,
            samples: 8,
        };
        // 10 µs a call; history at 100 GB/s to 16K tokens, 200 GB/s beyond.
        let at = |depth: f64| {
            let shallow = depth.min(16384.) * per_token / 100e9;
            let deep = (depth - 16384.).max(0.) * per_token / 200e9;
            10e-6 + shallow + deep
        };
        let [first, second, third] = HISTORY_DEPTHS[..3] else {
            unreachable!("the ladder has three rungs and more")
        };
        let history = HistoryCosts::new(
            KvCodec::Dense,
            vec![cost(at(first as f64)), cost(at(second as f64)), cost(at(third as f64))],
        )
        .unwrap();
        let price = |depth: f64| {
            let line = history.line(depth, 1e9);
            line.fixed_seconds + depth * per_token * line.seconds_per_byte
        };
        for depth in [4096., 12000., 16384., 24576., 65536.] {
            assert!((price(depth) / at(depth) - 1.).abs() < 1e-9, "{depth}");
        }
        assert!((history.line(65536., 1e9).seconds_per_byte * 200e9 - 1.).abs() < 1e-9);
    }

    /// A rung that reads faster than one below it is priced at the slower
    /// one's time, as a call over more history takes no less: its segment is
    /// flat, each byte priced at none, and the segment past it rises from
    /// the slower time.
    #[test]
    fn a_rung_faster_than_one_below_it_prices_a_flat_segment() {
        let per_token = token_bytes(KvCodec::AffineK8V4).unwrap() as f64;
        let cost = |value| MeasuredCost {
            value,
            interval: None,
            samples: 8,
        };
        let history = HistoryCosts::new(
            KvCodec::AffineK8V4,
            vec![cost(2e-3), cost(1.5e-3), cost(4e-3)],
        )
        .unwrap();
        let flat = history.line(12000., 1e9);
        assert_eq!(flat.seconds_per_byte, 0.);
        assert_eq!(flat.fixed_seconds, 2e-3);
        let rising = history.line(24576., 1e9);
        let between = (HISTORY_DEPTHS[2] - HISTORY_DEPTHS[1]) as f64 * per_token;
        assert!((rising.seconds_per_byte * between / 2e-3 - 1.).abs() < 1e-12);
    }

    /// A ladder with one rung prices every depth on the line through it
    /// from no fixed cost, and one with none at the rate memory streams its
    /// bytes at.
    #[test]
    fn a_ladder_with_fewer_than_two_rungs_is_priced_through_the_origin() {
        let per_token = token_bytes(KvCodec::Dense).unwrap() as f64;
        let one = HistoryCosts::new(
            KvCodec::Dense,
            vec![MeasuredCost {
                value: 2e-3,
                interval: None,
                samples: 8,
            }],
        )
        .unwrap();
        let line = one.line(65536., 1e9);
        assert_eq!(line.fixed_seconds, 0.);
        assert!((line.seconds_per_byte * HISTORY_DEPTHS[0] as f64 * per_token / 2e-3 - 1.).abs() < 1e-12);
        let none = HistoryCosts::new(KvCodec::Dense, Vec::new()).unwrap();
        assert_eq!(
            none.line(65536., 1e9),
            HistoryLine {
                fixed_seconds: 0.,
                seconds_per_byte: 1e-9,
            }
        );
    }

    /// The plan reads everything after the probe, which it leaves out, in
    /// its order, each step one point: a job's share follows from the plan
    /// alone.
    #[test]
    fn the_plan_reads_every_step_after_the_probe_in_order() {
        let route = |storage| Route {
            route: AccessRoute::Device,
            storage,
            passed: Vec::new(),
            along: Vec::new(),
        };
        let (device, upload) = (route(StorageKind::Device), route(StorageKind::Upload));
        let (bf16, f32) = (Element::bf16(), Element::f32());
        let steps = plan(&[&device, &upload], &[bf16, f32]);
        let first = |layout| {
            [Rung::Pilot, Rung::Depth(0), Rung::Depth(1)].map(|rung| Step::Rung(layout, rung))
        };
        let deeper = |layout| (2..HISTORY_DEPTHS.len()).map(move |index| Step::Rung(layout, Rung::Depth(index)));
        let mut expected = vec![Step::Launch(GraphTiming::Device)];
        expected.extend(first(KvCodec::Dense));
        expected.extend([Step::Chain, Step::Matrix(bf16)]);
        expected.extend(first(KvCodec::AffineK8V4));
        expected.extend([
            Step::WeightRate(1, bf16, Pass::Rate),
            Step::Launch(GraphTiming::Waited),
            Step::Launch(GraphTiming::Queued),
            Step::WeightRate(0, f32, Pass::Pilot),
            Step::WeightRate(0, f32, Pass::Rate),
        ]);
        expected.extend(deeper(KvCodec::Dense).chain(deeper(KvCodec::AffineK8V4)));
        expected.push(Step::Matrix(f32));
        assert_eq!(steps, expected);
    }

    /// The composition ratio is the chain's time over its layers' calls as
    /// the costs price them alone, each a launch: here the fixture's 300
    /// GB/s weights, 20 TFLOP/s, 10 µs and 200 GB/s history and 5 µs
    /// launches, the chain read at 1.25 times their price.
    #[test]
    fn the_composition_ratio_is_the_chain_over_its_calls_priced() {
        let costs = crate::assessment::fixtures::costs();
        let backend = seismic::BackendName::Cpu;
        let chain = chain_element(backend).unwrap();
        let programs =
            ProgramSet::new(Sizes::of(backend), &measured_formats(backend), chain, 1 << 34)
                .unwrap();
        let (features, _) = programs.expand(chain);
        let depth = HISTORY_DEPTHS[0] as f64;
        let launch = 5e-6;
        let call = |bytes: f64, rate: f64, flops: f64| (bytes / rate).max(flops / 20e12) + launch;
        let output = chain.canonical_byte_len(&[HIDDEN, features]).unwrap() as f64;
        let history = depth * token_bytes(KvCodec::Dense).unwrap() as f64;
        let flops = (features * HIDDEN) as f64;
        let layer = call(expand_weight_bytes(chain, features).unwrap() as f64, 300e9, 4. * flops)
            + call(history, 200e9, depth * (2 * KV_HEADS * GROUP * 2 * HEAD_WIDTH) as f64)
            + 10e-6
            + call(output, 300e9, 2. * flops)
            + launch;
        let layers = 6;
        let read = ChainRead {
            timing: Timing {
                median: 1.25 * layers as f64 * layer,
                interval: None,
                samples: 8,
            },
            layers,
        };
        let ratio = read.composition(&programs, &costs).unwrap();
        assert!((ratio.value / 1.25 - 1.).abs() < 1e-12, "{}", ratio.value);
    }

    /// Records held in memory.
    #[derive(Default)]
    struct Held(std::cell::RefCell<Vec<(PointKey, Record)>>);

    impl RecordStore<PointKey> for Held {
        fn load(&self, key: &PointKey) -> Result<Option<Record>, seismic::measure::StoreError> {
            Ok(self
                .0
                .borrow()
                .iter()
                .find(|(held, _)| held == key)
                .map(|(_, record)| record.clone()))
        }

        fn store(
            &self,
            key: &PointKey,
            record: &Record,
        ) -> Result<(), seismic::measure::StoreError> {
            self.0.borrow_mut().push((key.clone(), record.clone()));
            Ok(())
        }
    }

    /// The stored records give the probe until it is stored; then complete
    /// costs, pending while every later point is not, each figure but the
    /// probe priced. The probe recorded unobtainable gives no costs.
    #[test]
    fn stored_costs_complete_with_the_probe() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let topology = catalog.topology();
        let executor = topology
            .executors()
            .iter()
            .find(|executor| executor.backend() == seismic::BackendName::Cpu)
            .unwrap()
            .clone();
        let measured = Record::Measured(Timing {
            median: 1e-3,
            interval: None,
            samples: 8,
        });
        let stored = Held::default();
        let StoredCosts::Missing(probe) =
            ExecutorCosts::stored(&topology, &executor, &stored).unwrap()
        else {
            panic!("nothing is stored")
        };
        stored.store(&probe, &measured).unwrap();
        let StoredCosts::Complete {
            costs,
            records,
            pending,
        } = ExecutorCosts::stored(&topology, &executor, &stored).unwrap()
        else {
            panic!("the probe is stored")
        };
        assert!(pending, "the later points are still to be measured");
        assert_eq!(records.len(), 1);
        assert!(costs.first.formats.is_empty());
        assert!(costs
            .routes
            .iter()
            .all(|route| route.activation.is_none() && route.formats.is_empty()));
        let derived = costs.derived(seismic::BackendName::Cpu);
        for figure in [
            Derived::ActivationMatrix,
            Derived::Composition,
            Derived::Launch,
            Derived::Step,
            Derived::History(KvCodec::Dense),
            Derived::History(KvCodec::AffineK8V4),
        ] {
            assert!(derived.contains(&figure), "{figure:?}");
        }

        let unobtainable = Held::default();
        let reason = Reason::new("no kernel");
        unobtainable
            .store(&probe, &Record::Unobtainable { reason })
            .unwrap();
        assert!(ExecutorCosts::stored(&topology, &executor, &unobtainable).is_err());
    }

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

    /// Every program of the walk gives each static its entry declares on
    /// every backend, so no figure goes unmeasured for want of one.
    #[test]
    fn every_walk_program_gives_its_statics_on_every_backend() {
        for backend in seismic::BackendName::ALL {
            let programs = ProgramSet::new(
                Sizes::of(backend),
                &measured_formats(backend),
                chain_element(backend).unwrap(),
                1 << 34,
            )
            .unwrap();
            let mut statics = vec![
                programs.output.statics(backend),
                programs.dense.statics(backend),
                programs.affine.statics(backend),
                programs.copy.statics(backend),
            ];
            statics.extend(programs.expands.iter().map(|(_, _, expand)| expand.statics(backend)));
            for given in statics {
                given.unwrap_or_else(|error| panic!("{backend:?}: {}", error.0));
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
                    for activations in [ScaledActivations::Rounded, ScaledActivations::Nvfp4] {
                        for width in [HIDDEN, HIDDEN + 32] {
                            if let Some(element) =
                                resident_element(encoding, activations, width, dense, layout)
                            {
                                assert!(formats.contains(&element), "{backend:?} {element:?}");
                            }
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
