//! The fixed device measurement behind a basis.
//!
//! Every entry of the backend's plan ([`super::plan`]) is formed with its
//! shipped default specialization; timed entries are timed over synthetic
//! device-resident tensors at the plan's generic sizes. Nothing inspects a
//! model artifact, tunes a parameter or loads a model. An entry the backend
//! cannot form is recorded unsupported. Measured times are taken as they
//! are.
//!
//! The measurement is built for a cold run of a few seconds:
//!
//! - Every planned native form is formed first, in parallel, then every
//!   timed entry is timed (see [`complete_basis`]).
//! - Synthetic weights are views into one zero-filled pool per element and
//!   row width, allocated once for the whole basis and reused by every
//!   class. A point's rotation of views spans the backend's rotation bytes.
//! - A point is timed as production runs its entries: one sealed native
//!   graph holding one launch per rotation view, run [`RUNS`] times back to
//!   back after one discarded run, with each run's device interval taken
//!   from a submission trace. The device is warmed once, not per point.
//!
//! Before each class allocates, the device's memory ceiling is refreshed, so
//! Seismic refuses a measurement allocation that would leave less headroom
//! than the planning reserve.

use super::basis::{
    median, BasisIdentity, ClassCost, ClassMeasurement, CostModel, HeadGeometry, MeasuredPoint,
    MeasurementBasis, MeasurementKey, OperationClass, PointShape,
};
use super::plan::{activation, measurement_plan, reference_weight, PlannedKey};

mod general_routed;
mod post_norm;
mod row_ops;
mod short_conv;
mod state_space;
use crate::platform::{refresh_device_ceiling, MemoryPolicyError, MemoryReserves};
use magnitude_kernels::{
    attention_decode, attention_decode_k8v4, attention_output, attention_project, dense_expand,
    dense_output, embedding_rows, gated_delta_output, gated_delta_project,
    gated_delta_step, readout_features_rows, readout_head_rows, routed_expand, routed_output,
    routed_route, sample_rows,
};
use magnitude_state::{BankComponent, ComponentDescriptor, KvCodec, LayerRef};
use seismic::{
    generated, BackendName, Device, DeviceCatalog, Element, Entry, LoadError, NativeGraph,
    NativeGraphPlan, NativeGraphSlot, NativeKernel, NativePort, NativeSpecialization, SlabLayout, SlabRegion, SlabTensor,
    SubmissionTrace, Tensor, TraceDetail, WorkflowTensor,
};
use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The output rows a weight-streaming class is timed at, at [`REDUCTION`]:
/// from a head-sized launch to a vocabulary-sized one. Its per-byte time
/// between them interpolates in log rows.
const LADDER_ROWS: [u64; 3] = [768, 6144, 49_152];
/// The reduction of the ladder.
const REDUCTION: u64 = 4096;
/// The ladder row count also timed at [`FLOOR_REDUCTION`]: the two sizes
/// give the class's launch floor.
const FLOOR_ROWS: u64 = 6144;
const FLOOR_REDUCTION: u64 = 1024;
/// The launch every weight representation is timed at through
/// `project_rows`: large enough to be bandwidth-bound, small enough that a
/// dense representation fits one rotation.
const FORMAT_ROWS: u64 = 16_384;

/// One weight-streaming launch of the plan: its nominal output rows and its
/// reduction. An entry realizes the rows in whole units of its own geometry
/// (heads, experts, segments) and records the rows it ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Launch {
    rows: u64,
    reduction: u64,
}

/// The launches a weight-streaming class is timed at: the ladder, then the
/// floor launch.
fn projection_launches() -> Vec<Launch> {
    LADDER_ROWS
        .iter()
        .map(|&rows| Launch {
            rows,
            reduction: REDUCTION,
        })
        .chain([Launch {
            rows: FLOOR_ROWS,
            reduction: FLOOR_REDUCTION,
        }])
        .collect()
}

/// The launch an exact representation binding is formed at.
const FORM_LAUNCH: Launch = Launch {
    rows: FLOOR_ROWS,
    reduction: REDUCTION,
};

/// The point of a weight-streaming launch that ran `rows` output rows.
fn launch_point(at: Launch, rows: u64, weight: Element, bytes: u64, samples: Vec<f64>) -> MeasuredPoint {
    MeasuredPoint {
        shape: PointShape::Launch {
            rows,
            reduction: at.reduction,
            weight,
        },
        bytes,
        samples,
    }
}

/// The point of a class whose cost follows bytes.
fn size_point(bytes: u64, samples: Vec<f64>) -> MeasuredPoint {
    MeasuredPoint {
        shape: PointShape::Size,
        bytes,
        samples,
    }
}

/// Distinct bytes one point's rotation of views spans, so that its launches
/// read from memory as a model's resident weights are read. Evidence
/// (2026-09-25, a streaming GEMV rotated over growing spans):
///
/// - Metal (M4 Max): flat within 1.4% from 64 MB to 8 GB.
/// - CUDA (GB10): 8–15% faster over 128 MB than over 512 MB and more, where
///   it matches the same kernel's time inside a decode step (nsys); flat
///   from 512 MB to 8 GB.
///
/// Vulkan and CPU have no evidence yet and take the larger span.
///
/// History and state rotate over the same span, and need it: over 64 MB
/// instead of 128 MB, M4 Pro dense attention at 4k history read 7–21% faster
/// and the largest state advances 7–8% faster (2026-09-27), so the GEMV's
/// flat 64 MB does not carry over.
fn rotation_bytes(backend: BackendName) -> u64 {
    match backend {
        BackendName::Metal => 128 << 20,
        _ => 512 << 20,
    }
}

/// Declared parameters whose every value is timed, each one alone from the
/// default configuration, with a point charged at its fastest value. Each
/// is a small fixed set, and each moves production speed by far more than
/// the defaults-only error budget:
///
/// - `INT8`: the arithmetic path of weight-streaming entries. CPU tuning
///   selects INT8 where the default is exact F32 (a 4B Q4_K_M plain decode
///   step: 39.5 ms measured against 137 ms predicted from defaults).
/// - `PARTS`: the history split of fused decode attention. With few KV
///   heads it is the attention's parallelism; GB10 tuning selects 24 where
///   the default is 12 (35B-A3B, two KV heads: +13% at 16k from defaults).
/// - `SLICES`: the query-group split of fused decode attention. Large query
///   groups (16 heads per KV head and more) are compute-bound on one slice;
///   tuning selects 2–4 on CUDA and Metal (M4 Pro K8/V4: 212 µs at the
///   default, 122 µs tuned).
///
/// Every variant is screened by one sample; only the fastest is timed.
const VARIED_PARAMETERS: [&str; 3] = ["INT8", "PARTS", "SLICES"];

/// The most launches one timed graph holds.
const MAX_LAUNCHES: u64 = 256;
/// Timed samples of every point, after one discarded sample.
const RUNS: usize = 3;
/// Device time the device is kept busy before the first timed run.
const WARM_SECONDS: f64 = 0.2;
/// Device time of one sample: runs of a point's graph queued as one
/// submission. On Metal, short submissions of one run each read 10–40%
/// slower and vary between runs of the measurement (2026-09-25).
const SAMPLE_SECONDS: f64 = 0.003;
/// The most runs one sample queues.
const MAX_PASSES: usize = 256;
/// Every synthetic extent is a multiple of this: it is a multiple of every
/// packed representation's group and of every native row and reduction
/// alignment.
const UNIT: u64 = 256;
/// Pool views start at multiples of this many rows (a packed layout's row
/// group).
const ROW_ALIGNMENT: u64 = 16;
/// Vocabulary widths sampling is timed at.
const SAMPLE_VOCABULARIES: [u64; 2] = [32_768, 1_048_576];

/// Decode attention's reference head geometry, timed at two context depths
/// (its floor and rate), and the depth every other geometry is timed at.
const HISTORY_REFERENCE: HeadGeometry = HeadGeometry {
    kv_heads: 2,
    group: 8,
    width: 256,
};
const HISTORY_DEPTHS: [u64; 2] = [4096, 32_768];
const HISTORY_DEPTH: u64 = 32_768;
/// The geometries decode attention is timed at beside the reference, each
/// differing from it in one axis: the key/value heads (the attention's
/// parallelism), the query heads per key/value head and the head width.
const HISTORY_KV_HEADS: [u64; 4] = [1, 4, 8, 16];
const HISTORY_GROUPS: [u64; 4] = [2, 4, 16, 32];
const HISTORY_WIDTHS: [u64; 3] = [64, 128, 512];
/// Rotated pairs of a timed head: the rotation is per query row, beside a
/// history the entry streams.
const HISTORY_ROTARY_PAIRS: u64 = 32;

/// Every decode attention point: the reference at both depths, then each
/// axis's other values at one depth.
fn history_points() -> Vec<(HeadGeometry, u64)> {
    let reference = HISTORY_REFERENCE;
    HISTORY_DEPTHS
        .iter()
        .map(|&depth| (reference, depth))
        .chain(HISTORY_KV_HEADS.iter().map(|&kv_heads| {
            (
                HeadGeometry {
                    kv_heads,
                    ..reference
                },
                HISTORY_DEPTH,
            )
        }))
        .chain(HISTORY_GROUPS.iter().map(|&group| {
            (
                HeadGeometry {
                    group,
                    ..reference
                },
                HISTORY_DEPTH,
            )
        }))
        .chain(HISTORY_WIDTHS.iter().map(|&width| {
            (
                HeadGeometry {
                    width,
                    ..reference
                },
                HISTORY_DEPTH,
            )
        }))
        .collect()
}

/// The two sizes of each class whose cost follows the bytes it touches, from
/// a small model's geometry to a large one's:
///
/// - gated delta steps of 16 key heads, by value heads (width 128);
const DELTA_STEP_HEADS: [u64; 2] = [16, 64];
/// - routing of 8 selected experts, by (hidden width, experts);
const ROUTING_SIZES: [(u64, u64); 2] = [(2048, 128), (8192, 512)];
/// - Mamba-2 steps and gates of 8 groups, heads 64 wide and 128 state
///   columns, by heads;
const STATE_SPACE_HEADS: [u64; 2] = [64, 256];
/// - short convolutions of 3 taps, by channels;
const SHORT_CONV_CHANNELS: [u64; 2] = [1024, 8192];
/// - per-layer inputs 256 wide, by layers;
const PER_LAYER_LAYERS: [u64; 2] = [8, 64];
/// - row conversions and copies, by elements.
const CONVERTED_ELEMENTS: [u64; 2] = [4096, 65_536];

/// The hidden width of every one-row launch.
const HIDDEN: u64 = 4096;
/// The head width of synthetic attention output projections.
const ATTENTION_WIDTH: u64 = 256;
/// The key and value rows of a synthetic attention projection: one kv head
/// of width 128.
const PROJECT_WIDTH: u64 = 128;
const RECURRENT_WIDTH: u64 = 128;
const ROUTED_SELECTED: u64 = 8;
/// Recurrent banks of the step measurement: the pristine bank it reads and
/// the successor it publishes.
const STEP_BANKS: u64 = 2;

/// The dependency reference: cycles of the four weight-streaming entries a
/// recurrent decoder block launches in order (recurrent projection, gated
/// output, paired expansion, down projection), q4k weights at the synthetic
/// geometry. One graph chains each call on the previous call's result, one
/// graph gives every call the same independent input.
const CHAIN_FEATURES: u64 = 8192;
const CHAIN_CYCLES: usize = 8;
const CHAIN_CALLS_PER_CYCLE: usize = 4;
const CHAIN_SAMPLES: usize = 7;

#[derive(Clone, Debug, PartialEq)]
pub enum MeasurementError {
    /// The device's memory ceiling could not be established.
    Ceiling(MemoryPolicyError),
    /// Allocation, formation infrastructure or graph sealing failed.
    Device {
        key: MeasurementKey,
        message: String,
    },
    /// A timed submission of the class failed on the device (for example an
    /// illegal memory access). The class is recorded unsupported; the
    /// device's context may be unusable, so measurement stops.
    Fault {
        key: MeasurementKey,
        message: String,
    },
    /// The class's measured points are not its plan's points.
    Fit {
        key: MeasurementKey,
        message: String,
    },
    /// The submission trace timing every run could not be started.
    Trace(String),
}

impl fmt::Display for MeasurementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ceiling(error) => write!(formatter, "measurement memory ceiling: {error}"),
            Self::Device { key, message } => write!(formatter, "measuring {key}: {message}"),
            Self::Fault { key, message } => {
                write!(formatter, "device fault while timing {key}: {message}")
            }
            Self::Fit { key, message } => {
                write!(formatter, "measured {key} does not fit its cost model: {message}")
            }
            Self::Trace(message) => write!(formatter, "measurement trace: {message}"),
        }
    }
}

impl std::error::Error for MeasurementError {}

/// A measurement that stopped early: the classes measured before the error,
/// valid and worth keeping (a fault's class is among them, recorded
/// unsupported), and the error.
#[derive(Debug)]
pub struct MeasurementFailure {
    pub basis: MeasurementBasis,
    pub error: MeasurementError,
}

impl fmt::Display for MeasurementFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(formatter)
    }
}

impl std::error::Error for MeasurementFailure {}

/// Where one class's measurement time went.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClassProfile {
    /// Implementation lookup, default configuration and any native formation
    /// the parallel formation pass did not do.
    pub formation: Duration,
    /// Synthetic tensor allocation and initialization.
    pub allocation: Duration,
    /// Graph sealing, the timed runs and waiting for them.
    pub timing: Duration,
    /// Breakdown of `timing` for diagnosing cold measurement cost.
    pub sealing: Duration,
    pub submission: Duration,
    pub encoding: Duration,
    pub dispatch: Duration,
    /// Sum of traced device intervals, which may overlap host work.
    pub device: Duration,
    pub waiting: Duration,
    pub tracing: Duration,
    pub total: Duration,
}

/// Measure the device's whole plan.
pub fn measure_basis(
    catalog: &DeviceCatalog,
    device: &Device,
    reserves: MemoryReserves,
    identity: BasisIdentity,
) -> Result<MeasurementBasis, MeasurementError> {
    complete_basis(
        catalog,
        device,
        reserves,
        MeasurementBasis {
            identity,
            classes: Vec::new(),
        },
        |_, _, _| {},
    )
    .map_err(|failure| failure.error)
}

/// Measure every entry of the device's plan that `basis` lacks and add it,
/// reporting each entry with its time profile as it completes. Entries
/// already in the basis (from a measurement that stopped early) are kept as
/// measured. The parallel formation pass is reported as the first measured
/// entry's formation. The result holds its entries in plan order.
///
/// Every form is formed before any class is timed: forming compiles kernels
/// on every host core, and host work during the timed samples biases device
/// times (M4 Pro: shader compilation during timing made one-row launches
/// 2-12x slower). Work by other processes is not observable and is timed as
/// it falls.
///
/// A measurement that stops early returns the entries measured before the
/// error with it, so the caller can keep them. A device fault records its
/// entry unsupported (a kernel that faults is not executable on the device)
/// and stops, since the device's context may be unusable.
pub fn complete_basis(
    catalog: &DeviceCatalog,
    device: &Device,
    reserves: MemoryReserves,
    basis: MeasurementBasis,
    mut observe: impl FnMut(&MeasurementKey, &ClassMeasurement, &ClassProfile),
) -> Result<MeasurementBasis, MeasurementFailure> {
    let MeasurementBasis {
        identity,
        mut classes,
    } = basis;
    let plan = measurement_plan(device.backend());
    let missing = plan
        .iter()
        .filter(|planned| !classes.iter().any(|(measured, _)| measured == planned.key()))
        .cloned()
        .collect::<Vec<_>>();
    let measured = (|| {
        if missing.is_empty() {
            return Ok(());
        }
        let session = Session::open(device)?;
        let formation = session.form_all(&missing);
        for (index, planned) in missing.into_iter().enumerate() {
            let (measurement, mut profile) = match session.measure(catalog, &reserves, &planned) {
                Ok(measured) => measured,
                Err(MeasurementError::Fault { key, message }) => {
                    classes.push((
                        key.clone(),
                        ClassMeasurement::Unsupported {
                            reason: format!("device fault: {message}"),
                        },
                    ));
                    return Err(MeasurementError::Fault { key, message });
                }
                Err(error) => return Err(error),
            };
            if index == 0 {
                profile.formation += formation;
                profile.total += formation;
            }
            let key = planned.key().clone();
            observe(&key, &measurement, &profile);
            classes.push((key, measurement));
        }
        Ok(())
    })();
    classes.sort_by_key(|(key, _)| plan.iter().position(|planned| planned.key() == key));
    let basis = MeasurementBasis { identity, classes };
    match measured {
        Ok(()) => Ok(basis),
        Err(error) => Err(MeasurementFailure { basis, error }),
    }
}

/// Measure one planned entry on its own.
pub fn measure_entry(
    catalog: &DeviceCatalog,
    device: &Device,
    reserves: &MemoryReserves,
    planned: &PlannedKey,
) -> Result<(ClassMeasurement, ClassProfile), MeasurementError> {
    Session::open(device)?.measure(catalog, reserves, planned)
}

/// Why a class produced no points.
enum Stop {
    /// The backend has no implementation, the default configuration is not
    /// admissible at the class's statics, or the native form fails to form.
    Unsupported(String),
    Failed(String),
    /// A timed submission failed on the device.
    Fault(String),
    /// The formation pass queued this point's forms and stopped.
    Queued,
    /// The compatibility pass formed this point's entry and stopped.
    Formed,
}

type Step<T> = Result<T, Stop>;

fn failed(error: impl fmt::Display) -> Stop {
    Stop::Failed(error.to_string())
}

fn fault(error: impl fmt::Display) -> Stop {
    Stop::Fault(error.to_string())
}

fn bytes(element: Element, extents: &[u64]) -> Step<u64> {
    element
        .canonical_byte_len(extents)
        .map_err(|error| failed(format!("{} {extents:?}: {error}", element.name())))
}

fn sum(parts: &[u64]) -> Step<u64> {
    parts
        .iter()
        .try_fold(0u64, |total, part| total.checked_add(*part))
        .ok_or_else(|| failed("synthetic byte count overflows"))
}

/// `value` rounded down to a multiple of `unit`, at least one unit.
fn multiple(value: u64, unit: u64) -> u64 {
    (value / unit).max(1) * unit
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The history planes of one attention layer, as the state layout encodes
/// them: element, per-head elements and bytes of one history row.
pub(crate) fn history_planes(
    affine: bool,
    activation: Element,
    kv_heads: u64,
    width: u64,
) -> Result<Vec<(Element, u64, u64)>, String> {
    let dtype = activation
        .dtype()
        .ok_or_else(|| format!("{} is not a dense activation", activation.name()))?;
    let codec = if affine {
        KvCodec::AffineK8V4
    } else {
        KvCodec::Dense
    };
    let width = usize::try_from(width).map_err(|_| "head width exceeds the host")?;
    let component = ComponentDescriptor::new(
        LayerRef::Target(0),
        codec.spec(dtype, width, width),
        usize::try_from(kv_heads).map_err(|_| "kv heads exceed the host")?,
    )
    .map_err(|error| error.to_string())?;
    component
        .planes()
        .iter()
        .map(|plane| {
            let per_head = *plane
                .row_extents
                .get(1)
                .ok_or("history plane row has no per-head extent")?;
            Ok((
                Element::dense(plane.dtype),
                per_head as u64,
                plane.row_bytes as u64,
            ))
        })
        .collect()
}

/// One zero-filled allocation of `rows` leading rows of `trailing` extents,
/// handed out as consecutive leading-axis views.
struct Pool {
    element: Element,
    trailing: Vec<u64>,
    tensor: Tensor,
    rows: u64,
    cursor: u64,
    /// Shaped for one class: released when the next class starts.
    transient: bool,
}

impl Pool {
    /// Views of a packed matrix pool start at the layout's row group; views
    /// of flat dense pools and of higher-rank packed pools need none.
    fn alignment(element: Element, trailing: &[u64]) -> u64 {
        if element.logical_group().is_some() && trailing.len() == 1 {
            ROW_ALIGNMENT
        } else {
            1
        }
    }
}

/// Formed native kernels by entry, bindings and specialization, each with
/// its variants; filled in parallel before timing.
type Formed = Mutex<HashMap<String, Box<dyn Any + Send>>>;
type FormJob<'a> = Box<dyn FnOnce() + Send + 'a>;

/// Points measured in this session, by key: the dependency reference
/// reuses the classes the plan already timed.
type Measured = RefCell<HashMap<MeasurementKey, Vec<MeasuredPoint>>>;

/// What one measurement run shares across its classes.
struct Session<'a> {
    device: &'a Device,
    rotation: u64,
    trace: SubmissionTrace,
    pools: RefCell<Vec<Pool>>,
    warmed: Cell<bool>,
    formed: Formed,
    measured: Measured,
    /// The reference chain's samples, shared by the dependency and the
    /// submission class.
    chained: RefCell<Option<Chained>>,
}

impl<'a> Session<'a> {
    fn open(device: &'a Device) -> Result<Self, MeasurementError> {
        Ok(Self {
            device,
            rotation: rotation_bytes(device.backend()),
            trace: device
                .trace_submissions(TraceDetail::Submissions)
                .map_err(|error| MeasurementError::Trace(error.to_string()))?,
            pools: RefCell::new(Vec::new()),
            warmed: Cell::new(false),
            formed: Mutex::new(HashMap::new()),
            measured: RefCell::new(HashMap::new()),
            chained: RefCell::new(None),
        })
    }

    /// Form every native kernel of `plan` in parallel: a formation pass over
    /// the plan queues each point's forms, then worker threads form them.
    fn form_all(&self, plan: &[PlannedKey]) -> Duration {
        let began = Instant::now();
        let jobs: Mutex<Vec<FormJob<'_>>> = Mutex::new(Vec::new());
        for planned in plan {
            let runner = Runner::new(self, Pass::Queue(&jobs), planned);
            // Every outcome of the pass is a queued form or a class that
            // needs none; the timing pass reports failures.
            let _ = runner.points(planned.key());
        }
        let jobs = jobs
            .into_inner()
            .expect("formation queue lock is never poisoned");
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(jobs.len().max(1));
        let queue = Mutex::new(jobs);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let job = queue
                        .lock()
                        .expect("formation queue lock is never poisoned")
                        .pop();
                    match job {
                        Some(job) => job(),
                        None => break,
                    }
                });
            }
        });
        began.elapsed()
    }

    fn measure(
        &self,
        catalog: &DeviceCatalog,
        reserves: &MemoryReserves,
        planned: &PlannedKey,
    ) -> Result<(ClassMeasurement, ClassProfile), MeasurementError> {
        let began = Instant::now();
        let key = planned.key();
        self.pools.borrow_mut().retain(|pool| !pool.transient);
        refresh_device_ceiling(catalog, self.device, reserves)
            .map_err(MeasurementError::Ceiling)?;
        let pass = match planned {
            PlannedKey::Timed(_) => Pass::Time,
            PlannedKey::Formed(_) => Pass::Form,
        };
        let runner = Runner::new(self, pass, planned);
        let measurement = match runner.points(key) {
            Ok(points) => ClassCost::from_points(key.class, &points)
                .map(|cost| ClassMeasurement::Measured { points, cost })
                .map_err(|message| MeasurementError::Fit {
                    key: key.clone(),
                    message,
                })?,
            Err(Stop::Formed) => ClassMeasurement::Formed,
            Err(Stop::Unsupported(reason)) => ClassMeasurement::Unsupported { reason },
            Err(Stop::Failed(message)) => {
                return Err(MeasurementError::Device {
                    key: key.clone(),
                    message,
                })
            }
            Err(Stop::Fault(message)) => {
                return Err(MeasurementError::Fault {
                    key: key.clone(),
                    message,
                })
            }
            Err(Stop::Queued) => {
                return Err(MeasurementError::Device {
                    key: key.clone(),
                    message: "the timing pass queued a formation".into(),
                })
            }
        };
        let profile = ClassProfile {
            total: began.elapsed(),
            ..runner.profile.get()
        };
        Ok((measurement, profile))
    }
}

/// A formation's outcome, shared by the pass that queued it and the pass
/// that times it.
enum Formation<E: Entry> {
    Formed(Vec<NativeKernel<E>>),
    Unsupported(String),
    Failed(String),
}

impl<E: Entry> Clone for Formation<E> {
    fn clone(&self) -> Self {
        match self {
            Self::Formed(kernels) => Self::Formed(kernels.clone()),
            Self::Unsupported(reason) => Self::Unsupported(reason.clone()),
            Self::Failed(message) => Self::Failed(message.clone()),
        }
    }
}

/// A graph under construction and the tensors its ports are bound to.
struct Timed {
    graph: NativeGraph,
    bindings: Vec<(NativePort, Tensor)>,
}

/// A sealed point graph with its bindings, sized to its samples.
struct Sealed {
    plan: NativeGraphPlan,
    slot: NativeGraphSlot,
    bindings: Vec<(NativePort, Tensor)>,
    /// Runs one sample queues.
    passes: usize,
    /// Launches one run holds.
    launches: u64,
    /// Device seconds of the one run that sized the samples.
    probe: f64,
}

/// Distinct slab-backed state for one launch in a measurement rotation.
struct SlabbedState {
    _slabs: SlabTensor,
    regions: Vec<Tensor>,
}

impl Timed {
    fn new(device: &Device) -> Self {
        Self {
            graph: device.native_graph(),
            bindings: Vec::new(),
        }
    }

    /// A port bound to `tensor` for every run.
    fn bound(&mut self, tensor: &Tensor) -> Step<NativePort> {
        let port = self
            .graph
            .port(tensor.element(), tensor.extents())
            .map_err(failed)?;
        self.bindings.push((port.clone(), tensor.clone()));
        Ok(port)
    }

    fn export(&mut self, value: &WorkflowTensor) -> Step<()> {
        self.graph.export(value).map_err(failed)
    }
}

/// What one run over a plan entry's points does.
#[derive(Clone, Copy)]
enum Pass<'q, 's> {
    /// Queue every form the points need and stop each point.
    Queue(&'q Mutex<Vec<FormJob<'s>>>),
    /// Form the first point's entry and stop: a compatibility binding.
    Form,
    /// Time every point.
    Time,
}

struct Runner<'q, 's, 'a> {
    session: &'s Session<'a>,
    pass: Pass<'q, 's>,
    /// Whether forms include the varied parameters' variants: only a timed
    /// entry picks its fastest variant.
    variants: bool,
    profile: Cell<ClassProfile>,
}

impl<'q, 's, 'a> Runner<'q, 's, 'a> {
    fn new(session: &'s Session<'a>, pass: Pass<'q, 's>, planned: &PlannedKey) -> Self {
        Self {
            session,
            pass,
            variants: matches!(planned, PlannedKey::Timed(_)),
            profile: Cell::new(ClassProfile::default()),
        }
    }

    fn device(&self) -> &'a Device {
        self.session.device
    }

    fn queuing(&self) -> bool {
        matches!(self.pass, Pass::Queue(_))
    }

    fn charge(&self, part: impl FnOnce(&mut ClassProfile) -> &mut Duration, began: Instant) {
        let mut profile = self.profile.get();
        *part(&mut profile) += began.elapsed();
        self.profile.set(profile);
    }

    /// `E`'s implementation on this backend at its default configuration
    /// for the statics among `dimensions`, and one variant per other value
    /// of each [`VARIED_PARAMETERS`] parameter the implementation declares
    /// (the precision gate admits every declared arithmetic option, so a
    /// point is timed at the fastest). `bindings` name the element
    /// assignment for the formation cache.
    fn form<E: Entry>(
        &self,
        bindings: &[Element],
        dimensions: &[(&str, u64)],
        prepare: impl Fn(&NativeSpecialization) -> Result<NativeKernel<E>, LoadError> + Send + 's,
    ) -> Step<Vec<NativeKernel<E>>>
    where
        NativeKernel<E>: Send,
    {
        let began = Instant::now();
        let backend = self.device().backend();
        let implementation = generated::native_implementation_for_backend::<E>(backend)
            .map_err(failed)?
            .ok_or_else(|| {
                Stop::Unsupported(format!(
                    "{} has no {} implementation",
                    E::NAME,
                    backend.as_str()
                ))
            })?;
        let mut statics = NativeSpecialization::new();
        for name in &implementation.statics {
            let value = dimensions
                .iter()
                .find_map(|(candidate, value)| (candidate == name).then_some(*value))
                .ok_or_else(|| {
                    failed(format!(
                        "{} declares `{name}` static, but the measurement supplies no value",
                        E::NAME
                    ))
                })?;
            statics = statics.with_static(name.clone(), value);
        }
        let defaults = implementation
            .default_specialization(&statics)
            .map_err(|error| {
                Stop::Unsupported(format!(
                    "{} default configuration at {dimensions:?}: {error}",
                    E::NAME
                ))
            })?;
        let mut specializations = vec![defaults.clone()];
        for parameter in &implementation.params {
            if self.variants
                && VARIED_PARAMETERS.contains(&parameter.name.as_str())
                && parameter.arithmetic
            {
                for value in &parameter.values[1..] {
                    let variant = defaults.clone().with_param(parameter.name.clone(), *value);
                    if implementation.validate(&variant).is_ok() {
                        specializations.push(variant);
                    }
                }
            }
        }
        let names = bindings
            .iter()
            .map(|element| element.name())
            .collect::<Vec<_>>()
            .join(",");
        let key = format!("{}|{names}|{specializations:?}", E::NAME);
        let formed = &self.session.formed;
        let form_now = move || -> Formation<E> {
            let mut kernels = Vec::with_capacity(specializations.len());
            for specialization in &specializations {
                match prepare(specialization) {
                    Ok(kernel) => kernels.push(kernel),
                    Err(LoadError::Bundle(error)) => return Formation::Failed(error.to_string()),
                    // A variant that does not form is not an option; the
                    // default not forming is the class not forming.
                    Err(error) if kernels.is_empty() => {
                        return Formation::Unsupported(format!(
                            "{} does not form: {error}",
                            E::NAME
                        ));
                    }
                    Err(_) => {}
                }
            }
            Formation::Formed(kernels)
        };
        if let Pass::Queue(queue) = self.pass {
            let mut cache = formed
                .lock()
                .expect("formation cache lock is never poisoned");
            if !cache.contains_key(&key) {
                cache.insert(key.clone(), Box::new(Option::<Formation<E>>::None));
                drop(cache);
                queue
                    .lock()
                    .expect("formation queue lock is never poisoned")
                    .push(Box::new(move || {
                        let formation = form_now();
                        formed
                            .lock()
                            .expect("formation cache lock is never poisoned")
                            .insert(key, Box::new(Some(formation)));
                    }));
            }
            return Err(Stop::Queued);
        }
        let cached = formed
            .lock()
            .expect("formation cache lock is never poisoned")
            .get(&key)
            .and_then(|entry| entry.downcast_ref::<Option<Formation<E>>>())
            .and_then(Clone::clone);
        let formation = cached.unwrap_or_else(form_now);
        self.charge(|profile| &mut profile.formation, began);
        match formation {
            Formation::Formed(kernels) => Ok(kernels),
            Formation::Unsupported(reason) => Err(Stop::Unsupported(reason)),
            Formation::Failed(message) => Err(Stop::Failed(message)),
        }
    }

    /// `count` distinct views of `rows` rows of `inner` elements.
    fn views(&self, element: Element, inner: u64, rows: u64, count: u64) -> Step<Vec<Tensor>> {
        self.shaped(element, &[rows, inner], count)
    }

    /// `count` distinct views of `extents` from the session's pools. A
    /// dense element's views are reshaped spans of one flat pool of that
    /// element; a packed element's views are leading-axis slices of a pool
    /// of the same trailing extents (packed rows cannot be reshaped).
    fn shaped(&self, element: Element, extents: &[u64], count: u64) -> Step<Vec<Tensor>> {
        if element.logical_group().is_none() {
            let length = extents.iter().product::<u64>();
            return self
                .pooled(element, &[], length, count, false)?
                .into_iter()
                .map(|view| {
                    view.reshape(extents)
                        .map_err(|error| failed(format!("pool view {extents:?}: {error}")))
                })
                .collect();
        }
        let [rows, trailing @ ..] = extents else {
            return Err(failed("a pooled view has at least one extent"));
        };
        // Packed views of higher rank are shaped for one class.
        self.pooled(element, trailing, *rows, count, trailing.len() > 1)
    }

    /// Fresh slab storage for each launch in a rotation. The kernel sees
    /// logical address-table views while the slabs retain the physical state.
    fn slabbed(
        &self,
        rows_per_slab: u64,
        logical_rows: u64,
        regions: Vec<SlabRegion>,
        count: u64,
    ) -> Step<Vec<SlabbedState>> {
        let began = Instant::now();
        let mut states = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let mut slabs =
                SlabTensor::new(self.device(), rows_per_slab, logical_rows, regions.clone())
                    .map_err(failed)?;
            for _ in 0..logical_rows.div_ceil(rows_per_slab) {
                slabs.add_slab().map_err(failed)?;
            }
            let views = (0..regions.len())
                .map(|region| slabs.logical_region(region).map_err(failed))
                .collect::<Step<Vec<_>>>()?;
            states.push(SlabbedState {
                _slabs: slabs,
                regions: views,
            });
        }
        self.charge(|profile| &mut profile.allocation, began);
        Ok(states)
    }

    /// Rotation count must reflect the committed slabs, including unused
    /// rows in the last slab, rather than only the synthetic logical span.
    fn slab_copies(
        &self,
        rows_per_slab: u64,
        logical_rows: u64,
        regions: &[SlabRegion],
    ) -> Step<u64> {
        let layout =
            SlabLayout::for_regions(rows_per_slab, logical_rows, regions).map_err(failed)?;
        let bytes = layout
            .slab_bytes
            .checked_mul(logical_rows.div_ceil(rows_per_slab))
            .and_then(|bytes| bytes.checked_add(layout.address_table_bytes))
            .ok_or_else(|| failed("measurement slab storage overflows"))?;
        Ok(self.copies(bytes))
    }

    /// Packed matrix views whose row width only this class uses: their pool
    /// is released when the next class starts.
    fn transient_views(
        &self,
        element: Element,
        inner: u64,
        rows: u64,
        count: u64,
    ) -> Step<Vec<Tensor>> {
        if element.logical_group().is_none() {
            return self.views(element, inner, rows, count);
        }
        self.pooled(element, &[inner], rows, count, true)
    }

    /// `count` consecutive leading-axis views of `rows` rows from the pool
    /// of `element` and `trailing` extents. The pool grows when a point asks
    /// for more than it holds; `begin` restarts a point's views at the
    /// pool's start.
    fn pooled(
        &self,
        element: Element,
        trailing: &[u64],
        rows: u64,
        count: u64,
        transient: bool,
    ) -> Step<Vec<Tensor>> {
        if self.queuing() {
            return Err(Stop::Queued);
        }
        let began = Instant::now();
        let alignment = Pool::alignment(element, trailing);
        let aligned = rows.div_ceil(alignment) * alignment;
        let needed = aligned
            .checked_mul(count)
            .ok_or_else(|| failed("synthetic pool rows overflow"))?;
        let mut pools = self.session.pools.borrow_mut();
        let index = match pools
            .iter()
            .position(|pool| pool.element == element && pool.trailing == trailing)
        {
            Some(index) => index,
            None => {
                pools.push(self.pool(element, trailing, needed, transient)?);
                pools.len() - 1
            }
        };
        let pool = &mut pools[index];
        if pool.cursor + needed > pool.rows {
            // Views already handed out keep the old allocation alive; the
            // point continues in a pool that holds all of its views.
            *pool = self.pool(
                element,
                trailing,
                (pool.cursor + needed).max(2 * pool.rows),
                pool.transient,
            )?;
        }
        let views = (0..count)
            .map(|_| {
                let start = pool.cursor;
                pool.cursor += aligned;
                pool.tensor
                    .slice_leading(start, start + rows)
                    .map_err(|error| failed(format!("{} pool view: {error}", element.name())))
            })
            .collect::<Step<Vec<_>>>();
        drop(pools);
        self.charge(|profile| &mut profile.allocation, began);
        views
    }

    /// A pool spanning the session's rotation bytes, at least `rows` rows.
    fn pool(&self, element: Element, trailing: &[u64], rows: u64, transient: bool) -> Step<Pool> {
        let alignment = Pool::alignment(element, trailing);
        let group = bytes(
            element,
            &std::iter::once(alignment)
                .chain(trailing.iter().copied())
                .collect::<Vec<_>>(),
        )?;
        let spanning = self.session.rotation.div_ceil(group) * alignment;
        let rows = rows.max(spanning).div_ceil(alignment) * alignment;
        let extents = std::iter::once(rows)
            .chain(trailing.iter().copied())
            .collect::<Vec<_>>();
        Ok(Pool {
            element,
            trailing: trailing.to_vec(),
            tensor: self.zeros(element, &extents)?,
            rows,
            cursor: 0,
            transient,
        })
    }

    /// Start a point once its entry is formed: its views begin at every
    /// pool's start. The compatibility pass stops here.
    fn begin(&self) -> Step<()> {
        if let Pass::Form = self.pass {
            return Err(Stop::Formed);
        }
        for pool in self.session.pools.borrow_mut().iter_mut() {
            pool.cursor = 0;
        }
        Ok(())
    }

    fn zeros(&self, element: Element, extents: &[u64]) -> Step<Tensor> {
        if self.queuing() {
            return Err(Stop::Queued);
        }
        let began = Instant::now();
        let tensor = Tensor::zeros(self.device(), element, extents)
            .map_err(|error| failed(format!("{} {extents:?}: {error}", element.name())));
        self.charge(|profile| &mut profile.allocation, began);
        tensor
    }

    fn i32s(&self, extents: &[u64], values: &[i32]) -> Step<Tensor> {
        Tensor::from_host(self.device(), Element::i32(), extents, &i32_bytes(values))
            .map_err(|error| failed(format!("i32 {extents:?}: {error}")))
    }

    /// Rotation views of a point streaming `point_bytes` per launch.
    fn copies(&self, point_bytes: u64) -> u64 {
        self.session
            .rotation
            .div_ceil(point_bytes.max(1))
            .clamp(1, MAX_LAUNCHES)
    }

    /// Seal `timed` (a graph of `launches` launches) and size its samples:
    /// one run sizes a sample to `passes` runs queued as one submission, at
    /// least [`SAMPLE_SECONDS`] of device work. Before the session's first
    /// point, samples follow until the device has been busy
    /// [`WARM_SECONDS`], bringing it to its sustained clock.
    fn seal(&self, timed: Timed, launches: u64) -> Step<Sealed> {
        let began = Instant::now();
        let plan = timed.graph.seal().map_err(failed)?;
        let slot = plan.new_slot().map_err(failed)?;
        self.charge(|profile| &mut profile.sealing, began);
        self.charge(|profile| &mut profile.timing, began);
        let mut sealed = Sealed {
            plan,
            slot,
            bindings: timed.bindings,
            passes: 1,
            launches,
            probe: 0.0,
        };
        sealed.probe = self.submit(&mut sealed, 1)?[0];
        sealed.passes =
            ((SAMPLE_SECONDS / sealed.probe.max(1e-7)).ceil() as usize).clamp(1, MAX_PASSES);
        if !self.session.warmed.get() {
            let mut busy = 0.0;
            while busy < WARM_SECONDS {
                busy += self.submit(&mut sealed, RUNS)?.iter().sum::<f64>();
            }
            self.session.warmed.set(true);
        }
        Ok(sealed)
    }

    /// `samples` samples of `sealed` back to back, each its device interval
    /// (the sum of its passes).
    fn submit(&self, sealed: &mut Sealed, samples: usize) -> Step<Vec<f64>> {
        let began = Instant::now();
        // Work outside these samples (pool fills) is not a sample.
        let trace_began = Instant::now();
        self.session.trace.collect().map_err(failed)?;
        self.charge(|profile| &mut profile.tracing, trace_began);
        let mut completions = Vec::with_capacity(samples);
        for _ in 0..samples {
            let submission_began = Instant::now();
            let mut sequence = self.device().native_sequence();
            for _ in 0..sealed.passes {
                let mut bindings = sealed.plan.bindings();
                for (port, tensor) in &sealed.bindings {
                    bindings.set(port, tensor).map_err(failed)?;
                }
                let outputs = sealed.plan.new_outputs().map_err(failed)?;
                sealed
                    .slot
                    .attach(bindings, outputs)
                    .map_err(failed)?
                    .queue(&mut sequence)
                    .map_err(failed)?;
            }
            self.charge(|profile| &mut profile.encoding, submission_began);
            let dispatch_began = Instant::now();
            completions.push(sequence.submit().map_err(fault)?);
            self.charge(|profile| &mut profile.dispatch, dispatch_began);
            self.charge(|profile| &mut profile.submission, submission_began);
        }
        let waiting_began = Instant::now();
        for completion in completions {
            completion.wait().map_err(fault)?;
        }
        self.charge(|profile| &mut profile.waiting, waiting_began);
        let trace_began = Instant::now();
        let traced = self.session.trace.collect().map_err(failed)?;
        self.charge(|profile| &mut profile.tracing, trace_began);
        if traced.len() != samples {
            return Err(failed(format!(
                "{samples} timed samples recorded {} submissions",
                traced.len()
            )));
        }
        let intervals = traced
            .iter()
            .map(|submission| submission.device.1 - submission.device.0)
            .collect::<Vec<_>>();
        let mut profile = self.profile.get();
        profile.device += Duration::from_secs_f64(intervals.iter().sum());
        self.profile.set(profile);
        self.charge(|profile| &mut profile.timing, began);
        Ok(intervals)
    }

    /// Per-launch seconds of `samples` samples of `sealed`.
    fn per_launch(&self, sealed: &mut Sealed, samples: usize) -> Step<Vec<f64>> {
        let runs = (sealed.passes as u64 * sealed.launches) as f64;
        Ok(self
            .submit(sealed, samples)?
            .into_iter()
            .map(|sample| sample / runs)
            .collect())
    }

    /// The samples of the fastest of `graphs` (each a graph and its launch
    /// count) and its position. Several variants are screened by one sample
    /// each (the run that sized it, when one run fills a sample); the fastest is then timed as a single variant is: a leading
    /// sample brings it to its sustained behavior and [`RUNS`] samples
    /// follow back to back.
    fn fastest_graph(
        &self,
        graphs: impl IntoIterator<Item = Step<(Timed, u64)>>,
    ) -> Step<(usize, Vec<f64>)> {
        let mut sealed = graphs
            .into_iter()
            .map(|graph| graph.and_then(|(timed, launches)| self.seal(timed, launches)))
            .collect::<Step<Vec<_>>>()?;
        let chosen = match sealed.len() {
            0 => return Err(failed("no formed variant was timed")),
            1 => 0,
            _ => {
                let mut best: Option<(usize, f64)> = None;
                for (index, variant) in sealed.iter_mut().enumerate() {
                    // A run as long as a sample already is one.
                    let sample = if variant.passes == 1 {
                        variant.probe / variant.launches as f64
                    } else {
                        self.per_launch(variant, 1)?[0]
                    };
                    if best.is_none_or(|(_, fastest)| sample < fastest) {
                        best = Some((index, sample));
                    }
                }
                best.expect("several variants were screened").0
            }
        };
        let samples = self.per_launch(&mut sealed[chosen], 1 + RUNS)?[1..].to_vec();
        Ok((chosen, samples))
    }

    /// The samples of the fastest variant among `kernels`, each timed on the
    /// graph `graph` builds for it.
    fn fastest<E: Entry>(
        &self,
        kernels: &[NativeKernel<E>],
        graph: impl Fn(&NativeKernel<E>) -> Step<(Timed, u64)>,
    ) -> Step<Vec<f64>> {
        Ok(self.fastest_graph(kernels.iter().map(graph))?.1)
    }

    /// Every point of `targets`; the formation pass visits all of them.
    fn each<T: Copy>(
        &self,
        targets: &[T],
        point: impl Fn(T) -> Step<MeasuredPoint>,
    ) -> Step<Vec<MeasuredPoint>> {
        let mut points = Vec::with_capacity(targets.len());
        let mut queued = false;
        for target in targets {
            match point(*target) {
                Ok(measured) => points.push(measured),
                Err(Stop::Queued) => queued = true,
                Err(stop) => return Err(stop),
            }
        }
        if queued {
            Err(Stop::Queued)
        } else {
            Ok(points)
        }
    }

    /// The points of `key`, timed once per session: a class the plan already
    /// timed is reused by the dependency reference.
    fn points(&self, key: &MeasurementKey) -> Step<Vec<MeasuredPoint>> {
        let timing = matches!(self.pass, Pass::Time);
        if timing {
            if let Some(points) = self.session.measured.borrow().get(key) {
                return Ok(points.clone());
            }
        }
        let points = self.class_points(key)?;
        if timing {
            self.session
                .measured
                .borrow_mut()
                .insert(key.clone(), points.clone());
        }
        Ok(points)
    }

    fn class_points(&self, key: &MeasurementKey) -> Step<Vec<MeasuredPoint>> {
        use OperationClass as C;
        let reference = reference_weight(self.device().backend());
        let unexpected = || failed(format!("{key} has an unexpected binding list"));
        // A weight-streaming entry's bindings: its exact ones, or for its cost
        // key the reference representation with the activation as the norm.
        let weighted = || -> Step<(Element, Element, Element)> {
            match key.bindings.as_slice() {
                &[activation] => Ok((activation, reference, activation)),
                &[weight, activation] => Ok((activation, weight, activation)),
                &[norm, weight, activation] => Ok((norm, weight, activation)),
                _ => Err(unexpected()),
            }
        };
        // A timed class runs its ladder; a formed binding its one launch.
        let launches = if self.variants {
            projection_launches()
        } else {
            vec![FORM_LAUNCH]
        };
        let project = |point: &dyn Fn(Element, Element, Element, Launch) -> Step<MeasuredPoint>| {
            let (norm, weight, activation) = weighted()?;
            self.each(&launches, |at| point(norm, weight, activation, at))
        };
        let a = activation();
        match (key.class, key.bindings.as_slice()) {
            (C::EmbeddingRows, &[activation]) => {
                self.each(&[()], |()| self.embedding_rows(reference, activation))
            }
            (C::EmbeddingRows, &[table, activation]) => {
                self.each(&[()], |()| self.embedding_rows(table, activation))
            }
            (C::AttentionDecode | C::AttentionDecodeK8V4, &[activation]) => {
                let affine = key.class == C::AttentionDecodeK8V4;
                self.each(&history_points(), |(heads, depth)| {
                    self.attention_decode(affine, activation, heads, depth)
                })
            }
            (C::DeltaStep, &[activation]) => {
                self.each(&DELTA_STEP_HEADS, |heads| self.delta_step(activation, heads))
            }
            (C::RoutedSelect | C::RoutedRoute, bindings) => {
                let (norm, router, activation) = match bindings {
                    &[activation] => (activation, activation, activation),
                    &[norm, router, activation] => (norm, router, activation),
                    _ => return Err(unexpected()),
                };
                let sizes = if self.variants {
                    &ROUTING_SIZES[..]
                } else {
                    &ROUTING_SIZES[..1]
                };
                self.each(sizes, |(hidden, experts)| {
                    if key.class == C::RoutedSelect {
                        self.routed_select(norm, router, activation, hidden, experts)
                    } else {
                        self.routed_route(norm, router, activation, hidden, experts)
                    }
                })
            }
            (C::StateSpaceStep, &[activation]) => self.each(&STATE_SPACE_HEADS, |heads| {
                self.state_space_step(activation, heads)
            }),
            (C::StateSpaceGate, &[activation]) => self.each(&STATE_SPACE_HEADS, |heads| {
                self.state_space_gate(activation, heads)
            }),
            (C::ShortConvRows, &[activation]) => self.each(&SHORT_CONV_CHANNELS, |channels| {
                self.short_conv_rows(activation, channels)
            }),
            (C::AttentionProject, _) => project(&|n, w, a, at| self.attention_project(n, w, a, at)),
            (C::AttentionOutput, _) => project(&|_, w, a, at| self.attention_output(w, a, at)),
            (C::DeltaProject, _) => project(&|n, w, a, at| self.delta_project(n, w, a, at)),
            (C::DeltaOutput, _) => project(&|n, w, a, at| self.delta_output(n, w, a, at)),
            (C::ShortConvProject, _) => {
                project(&|n, w, a, at| self.short_conv_project(n, w, a, at))
            }
            (C::DenseExpand, _) => project(&|n, w, a, at| self.dense_expand(n, w, a, at)),
            (C::DenseUp, _) => project(&|n, w, a, at| self.dense_up(n, w, a, at)),
            (C::DenseOutput, _) => project(&|_, w, a, at| self.dense_output(w, a, at)),
            (C::RoutedGateUp, _) => project(&|_, w, a, at| self.routed_gate_up(w, a, at)),
            (C::RoutedUp, _) => project(&|_, w, a, at| self.routed_up(w, a, at)),
            (C::RoutedDown, _) => project(&|_, w, a, at| self.routed_down(w, a, at)),
            (C::RoutedExpand, _) => project(&|_, w, a, at| self.routed_expand(w, a, at)),
            (C::RoutedOutput, _) => project(&|_, w, a, at| self.routed_output(w, a, at)),
            (C::ProjectRows, _) => project(&|_, w, a, at| self.project_rows(w, a, at)),
            (C::PerLayerGate, _) => project(&|_, w, a, at| self.per_layer_gate(w, a, at)),
            (C::ReadoutHead, _) => project(&|n, w, a, at| self.readout_head(n, w, a, at)),
            (C::WeightFormat, &[weight, activation]) => self.each(
                &[Launch {
                    rows: FORMAT_ROWS,
                    reduction: REDUCTION,
                }],
                |at| self.project_rows(weight, activation, at),
            ),
            (C::PostNormResidual, &[norm]) => {
                self.each(&[()], |()| self.post_norm_residual(norm))
            }
            (C::MoeTail, &[norm]) => self.each(&[()], |()| self.moe_tail(norm)),
            (C::PerLayerInputs, bindings) => {
                let (table, norm) = match bindings {
                    &[norm] => (reference, norm),
                    &[table, norm] => (table, norm),
                    _ => return Err(unexpected()),
                };
                self.each(&PER_LAYER_LAYERS, |layers| {
                    self.per_layer_inputs(table, norm, layers)
                })
            }
            (C::ImportRows, bindings) => {
                let (source, destination) = match bindings {
                    &[] => (Element::f32(), a),
                    &[source, destination] => (source, destination),
                    _ => return Err(unexpected()),
                };
                self.each(&CONVERTED_ELEMENTS, |elements| {
                    self.convert_rows(false, source, destination, elements)
                })
            }
            (C::RepackRows, bindings) => {
                let (source, destination) = match bindings {
                    &[] => (
                        crate::source_element(magnitude_artifacts::gguf::Encoding::Q4K)
                            .ok_or_else(|| failed("q4_k has a source element"))?,
                        reference,
                    ),
                    &[source, destination] => (source, destination),
                    _ => return Err(unexpected()),
                };
                self.each(&CONVERTED_ELEMENTS, |elements| {
                    self.convert_rows(true, source, destination, elements)
                })
            }
            (C::CopyRows, &[]) => {
                self.each(&CONVERTED_ELEMENTS, |elements| self.copy_rows(elements))
            }
            (C::TableUpload, &[]) => self.table_uploads(),
            (C::ReadoutFeatures, &[norm, activation]) => {
                self.each(&[()], |()| self.readout_features(norm, activation))
            }
            (C::SampleRows, &[]) => self.each(&SAMPLE_VOCABULARIES, |vocabulary| {
                self.sample_rows(vocabulary)
            }),
            (C::LaunchDependency, &[]) => Ok(vec![self.chain_samples()?.dependency]),
            (C::StepSubmission, &[]) => Ok(vec![self.chain_samples()?.submission]),
            _ => Err(unexpected()),
        }
    }

    /// One table row gathered and decoded: a launch-dominated class.
    fn embedding_rows(&self, table: Element, activation: Element) -> Step<MeasuredPoint> {
        let (vocabulary, hidden) = (UNIT, HIDDEN);
        let device = self.device();
        let kernels = self.form::<embedding_rows::Entry>(
            &[table, activation],
            &[("M", 1), ("V", vocabulary), ("D", hidden)],
            move |specialization| {
                embedding_rows::native_for_device_with(
                    device,
                    embedding_rows::Elements {
                        EW: table,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let table_bytes = bytes(table, &[vocabulary, hidden])?;
        let launches = self.copies(table_bytes);
        let tables = self.views(table, hidden, vocabulary, launches)?;
        let tokens = self.zeros(Element::i32(), &[1, 2])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let tokens = timed.bound(&tokens)?;
            for table in &tables {
                let table = timed.bound(table)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        embedding_rows::WorkflowArgs {
                            table: table.tensor().into(),
                            tokens: tokens.tensor().into(),
                            scale: 1.0,
                            normalize: 0,
                            epsilon: 0.0,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.r1)?;
            }
            Ok((timed, launches))
        })?;
        Ok(size_point(table_bytes / vocabulary, samples))
    }

    /// Query (with interleaved gate) / key / value projection over one kv
    /// head of width 128: the query rows fill the launch's rows beside the
    /// key and value rows, in whole heads with their gates.
    fn attention_project(
        &self,
        norm: Element,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let (key_rows, hidden) = (PROJECT_WIDTH, at.reduction);
        let query = multiple(at.rows.saturating_sub(2 * key_rows), 2 * PROJECT_WIDTH);
        let point_bytes = sum(&[
            bytes(weight, &[query, hidden])?,
            2 * bytes(weight, &[key_rows, hidden])?,
        ])?;
        let device = self.device();
        let kernels = self.form::<attention_project::Entry>(
            &[norm, weight, activation],
            &[
                ("M", 1),
                ("D", hidden),
                ("Q", query),
                ("GR", 0),
                ("K", key_rows),
                ("V", key_rows),
            ],
            move |specialization| {
                attention_project::native_for_device_with(
                    device,
                    attention_project::Elements {
                        NW: norm,
                        QW: weight,
                        GW: weight,
                        KW: weight,
                        VW: weight,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let queries = self.views(weight, hidden, query, launches)?;
        let keys = self.views(weight, hidden, key_rows, launches)?;
        let values = self.views(weight, hidden, key_rows, launches)?;
        let input = self.zeros(Element::f32(), &[1, hidden])?;
        let input_norm = self.zeros(norm, &[hidden])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let input = timed.bound(&input)?;
            let input_norm = timed.bound(&input_norm)?;
            for ((query, key), value) in queries.iter().zip(&keys).zip(&values) {
                let query = timed.bound(query)?;
                let key = timed.bound(key)?;
                let value = timed.bound(value)?;
                // No separate gate: zero rows of the query weight.
                let gate = query.tensor().slice_leading(0, 0);
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        attention_project::WorkflowArgs {
                            hidden: input.tensor().into(),
                            input_norm: input_norm.tensor().into(),
                            query_weight: query.tensor().into(),
                            gate_weight: (&gate).into(),
                            key_weight: key.tensor().into(),
                            value_weight: value.tensor().into(),
                            epsilon: 1e-6,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.r0)?;
                timed.export(&result.r2)?;
                timed.export(&result.r3)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, query + 2 * key_rows, weight, point_bytes, samples))
    }

    /// Fused attention of one decode row over `depth` history rows at
    /// `heads`. The streamed bytes are the history it reads. The measured
    /// form has an interleaved gate and head norms, whose per-row work is
    /// small beside the history the entry streams.
    fn attention_decode(
        &self,
        affine: bool,
        activation: Element,
        heads: HeadGeometry,
        depth: u64,
    ) -> Step<MeasuredPoint> {
        let HeadGeometry {
            kv_heads,
            group,
            width,
        } = heads;
        let pairs = HISTORY_ROTARY_PAIRS.min(width / 2);
        let rows = depth + 1;
        let dimensions = [
            ("M", 1),
            ("T", rows),
            ("KV", kv_heads),
            ("G", group),
            ("P", pairs),
            ("S", width - 2 * pairs),
            ("I", width),
            ("U", 0),
            ("F", 1),
            ("N", 1),
            ("NV", 0),
            ("R", 1),
        ];
        let device = self.device();
        let (dense, k8v4) = if affine {
            let kernels = self.form::<attention_decode_k8v4::Entry>(
                &[activation],
                &dimensions,
                move |specialization| {
                    attention_decode_k8v4::native_for_device_with(
                        device,
                        attention_decode_k8v4::Elements { A: activation },
                        specialization,
                    )
                },
            )?;
            (Vec::new(), kernels)
        } else {
            let kernels = self.form::<attention_decode::Entry>(
                &[activation],
                &dimensions,
                move |specialization| {
                    attention_decode::native_for_device_with(
                        device,
                        attention_decode::Elements { A: activation },
                        specialization,
                    )
                },
            )?;
            (kernels, Vec::new())
        };
        self.begin()?;
        let planes = history_planes(affine, activation, kv_heads, width).map_err(failed)?;
        let row_bytes = sum(&planes
            .iter()
            .map(|(_, _, bytes)| *bytes)
            .collect::<Vec<_>>())?;
        let history_bytes = depth * row_bytes;
        let slab_rows = magnitude_state::history_rows_per_slab(row_bytes).map_err(failed)? as u64;
        let regions = planes
            .iter()
            .map(|(element, per_head, _)| SlabRegion {
                element: *element,
                row_shape: vec![kv_heads, *per_head],
            })
            .collect::<Vec<_>>();
        let launches = self.slab_copies(slab_rows, rows, &regions)?;
        let histories = self.slabbed(slab_rows, rows, regions, launches)?;
        let depth = i32::try_from(depth).map_err(failed)?;
        let query = self.zeros(activation, &[1, kv_heads * group, 2 * width])?;
        let gate = self.zeros(activation, &[1, kv_heads * group, 0])?;
        let fresh_key = self.zeros(activation, &[1, 1, kv_heads * width])?;
        let fresh_value = self.zeros(activation, &[1, 1, kv_heads * width])?;
        let query_norm = self.zeros(Element::f32(), &[1, width])?;
        let key_norm = self.zeros(Element::f32(), &[1, width])?;
        let value_norm = self.zeros(Element::f32(), &[0, width])?;
        let rotary_components = self.zeros(Element::i32(), &[pairs])?;
        let rotary_frequencies = self.zeros(Element::f32(), &[pairs])?;
        let rotary_amplitudes = self.zeros(Element::f32(), &[pairs])?;
        let coordinates = self.zeros(Element::i32(), &[1, 4])?;
        let visible = self.i32s(&[1, 1, 2], &[0, depth])?;
        let fresh = self.i32s(&[1, 2], &[0, 1])?;
        let destinations = self.i32s(&[1], &[depth])?;
        let scale = 1.0 / (width as f32).sqrt();
        // The inputs every launch shares, bound once per graph.
        let shared = |timed: &mut Timed| -> Step<[NativePort; 14]> {
            Ok([
                timed.bound(&query)?,
                timed.bound(&gate)?,
                timed.bound(&fresh_key)?,
                timed.bound(&fresh_value)?,
                timed.bound(&query_norm)?,
                timed.bound(&key_norm)?,
                timed.bound(&value_norm)?,
                timed.bound(&rotary_components)?,
                timed.bound(&rotary_frequencies)?,
                timed.bound(&rotary_amplitudes)?,
                timed.bound(&coordinates)?,
                timed.bound(&visible)?,
                timed.bound(&fresh)?,
                timed.bound(&destinations)?,
            ])
        };
        let samples = if affine {
            self.fastest(&k8v4, |kernel| {
                let mut timed = Timed::new(device);
                let [q, g, k, v, qn, kn, vn, rc, rf, ra, co, vi, fr, de] = shared(&mut timed)?;
                for index in 0..launches as usize {
                    let mut planes = histories[index]
                        .regions
                        .iter()
                        .map(|view| timed.bound(view))
                        .collect::<Step<Vec<_>>>()?;
                    let [key_codes, key_coefficients, value_codes, value_coefficients] =
                        planes.as_mut_slice()
                    else {
                        return Err(failed("affine history is not four planes"));
                    };
                    let result = timed
                        .graph
                        .enqueue(
                            kernel,
                            attention_decode_k8v4::WorkflowArgs {
                                query: q.tensor().into(),
                                gate: g.tensor().into(),
                                key: k.tensor().into(),
                                value: v.tensor().into(),
                                query_norm: qn.tensor().into(),
                                key_norm: kn.tensor().into(),
                                value_norm: vn.tensor().into(),
                                rotary_components: rc.tensor().into(),
                                rotary_frequencies: rf.tensor().into(),
                                rotary_amplitudes: ra.tensor().into(),
                                coordinates: co.tensor().into(),
                                visible: vi.tensor().into(),
                                fresh: fr.tensor().into(),
                                destinations: de.tensor().into(),
                                history_key_codes: key_codes.tensor_mut().into(),
                                history_key_coefficients: key_coefficients.tensor_mut().into(),
                                history_value_codes: value_codes.tensor_mut().into(),
                                history_value_coefficients: value_coefficients.tensor_mut().into(),
                                epsilon: 1e-6,
                                scale,
                                gate_function: 0,
                                slab_rows: u32::try_from(slab_rows)
                                    .map_err(|_| failed("history slab rows exceed u32"))?,
                            },
                        )
                        .map_err(failed)?;
                    timed.export(&result.value)?;
                }
                Ok((timed, launches))
            })?
        } else {
            self.fastest(&dense, |kernel| {
                let mut timed = Timed::new(device);
                let [q, g, k, v, qn, kn, vn, rc, rf, ra, co, vi, fr, de] = shared(&mut timed)?;
                for index in 0..launches as usize {
                    let mut planes = histories[index]
                        .regions
                        .iter()
                        .map(|view| timed.bound(view))
                        .collect::<Step<Vec<_>>>()?;
                    let [history_key, history_value] = planes.as_mut_slice() else {
                        return Err(failed("dense history is not two planes"));
                    };
                    let result = timed
                        .graph
                        .enqueue(
                            kernel,
                            attention_decode::WorkflowArgs {
                                query: q.tensor().into(),
                                gate: g.tensor().into(),
                                key: k.tensor().into(),
                                value: v.tensor().into(),
                                query_norm: qn.tensor().into(),
                                key_norm: kn.tensor().into(),
                                value_norm: vn.tensor().into(),
                                rotary_components: rc.tensor().into(),
                                rotary_frequencies: rf.tensor().into(),
                                rotary_amplitudes: ra.tensor().into(),
                                coordinates: co.tensor().into(),
                                visible: vi.tensor().into(),
                                fresh: fr.tensor().into(),
                                destinations: de.tensor().into(),
                                history_key: history_key.tensor_mut().into(),
                                history_value: history_value.tensor_mut().into(),
                                epsilon: 1e-6,
                                scale,
                                gate_function: 0,
                                slab_rows: u32::try_from(slab_rows)
                                    .map_err(|_| failed("history slab rows exceed u32"))?,
                            },
                        )
                        .map_err(failed)?;
                    timed.export(&result.value)?;
                }
                Ok((timed, launches))
            })?
        };
        Ok(MeasuredPoint {
            shape: PointShape::Heads(heads),
            bytes: history_bytes,
            samples,
        })
    }

    /// Output projection of heads of width 256 (the launch's reduction in
    /// whole heads) into the launch's rows.
    fn attention_output(
        &self,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let heads = (at.reduction / ATTENTION_WIDTH).max(1);
        let reduction = heads * ATTENTION_WIDTH;
        let hidden = at.rows;
        let point_bytes = bytes(weight, &[hidden, reduction])?;
        let device = self.device();
        let kernels = self.form::<attention_output::Entry>(
            &[weight, activation],
            &[
                ("M", 1),
                ("D", hidden),
                ("Q", heads),
                ("W", ATTENTION_WIDTH),
            ],
            move |specialization| {
                attention_output::native_for_device_with(
                    device,
                    attention_output::Elements {
                        A: activation,
                        OW: weight,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let weights = self.views(weight, reduction, hidden, launches)?;
        let input = self.zeros(Element::f32(), &[1, hidden])?;
        let gated = self.zeros(activation, &[1, heads, ATTENTION_WIDTH])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let input = timed.bound(&input)?;
            let gated = timed.bound(&gated)?;
            for weight in &weights {
                let weight = timed.bound(weight)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        attention_output::WorkflowArgs {
                            hidden: input.tensor().into(),
                            gated: gated.tensor().into(),
                            output_weight: weight.tensor().into(),
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, hidden, weight, point_bytes, samples))
    }

    /// Recurrent projection at width 128, twice as many value heads as key
    /// heads, in whole key heads filling the launch's rows. The small
    /// decay-rate matrices are shared by every launch.
    fn delta_project(
        &self,
        norm: Element,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let (width, hidden) = (RECURRENT_WIDTH, at.reduction);
        // A key head adds 2 key, 2 value and 2 gate rows of the width, and
        // two decay rows.
        let key_heads = (at.rows / (6 * width + 4)).max(1);
        let value_heads = 2 * key_heads;
        let (channels, inner) = ((2 * key_heads + value_heads) * width, value_heads * width);
        let rows = channels + inner + 2 * value_heads;
        let point_bytes = sum(&[
            bytes(weight, &[channels, hidden])?,
            bytes(weight, &[inner, hidden])?,
            2 * bytes(weight, &[value_heads, hidden])?,
        ])?;
        let device = self.device();
        let kernels = self.form::<gated_delta_project::Entry>(
            &[norm, weight, activation],
            &[
                ("M", 1),
                ("H", hidden),
                ("NK", key_heads),
                ("NV", value_heads),
                ("W", width),
            ],
            move |specialization| {
                gated_delta_project::native_for_device_with(
                    device,
                    gated_delta_project::Elements {
                        NW: norm,
                        QW: weight,
                        GW: weight,
                        AW: weight,
                        BW: weight,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let projections = self.views(weight, hidden, channels, launches)?;
        let gates = self.views(weight, hidden, inner, launches)?;
        let alpha = self.zeros(weight, &[value_heads, hidden])?;
        let beta = self.zeros(weight, &[value_heads, hidden])?;
        let input = self.zeros(Element::f32(), &[1, hidden])?;
        let input_norm = self.zeros(norm, &[hidden])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let input = timed.bound(&input)?;
            let input_norm = timed.bound(&input_norm)?;
            let alpha = timed.bound(&alpha)?;
            let beta = timed.bound(&beta)?;
            for (projection, gate) in projections.iter().zip(&gates) {
                let projection = timed.bound(projection)?;
                let gate = timed.bound(gate)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        gated_delta_project::WorkflowArgs {
                            hidden: input.tensor().into(),
                            input_norm: input_norm.tensor().into(),
                            qkv_weight: projection.tensor().into(),
                            gate_weight: gate.tensor().into(),
                            alpha_weight: alpha.tensor().into(),
                            beta_weight: beta.tensor().into(),
                            epsilon: 1e-6,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, rows, weight, point_bytes, samples))
    }

    /// One row's recurrent state advance of 16 key heads and `value_heads`
    /// value heads of width 128 with 4 taps, over the state layout's bank
    /// components (a one-row tape, as plain decoding plans it). Its bytes are
    /// the bank's.
    fn delta_step(&self, activation: Element, value_heads: u64) -> Step<MeasuredPoint> {
        let (key_heads, width, convolution_width) = (16, RECURRENT_WIDTH, 4);
        let host = |value: u64| {
            usize::try_from(value).map_err(|_| failed("recurrent geometry exceeds host domain"))
        };
        // The gated delta bank as the state layout forms it, with the one-row
        // tape plain decoding plans. The window keeps raw projection rows in
        // activation precision.
        let [window, delta, tape] = [
            BankComponent::ConvWindow {
                width: host(convolution_width)?,
                channels: host((2 * key_heads + value_heads) * width)?,
                dtype: activation
                    .dtype()
                    .ok_or_else(|| failed("a decoder activation is dense"))?,
            },
            BankComponent::DeltaState {
                heads: host(value_heads)?,
                width: host(width)?,
            },
            BankComponent::DeltaTape {
                value_heads: host(value_heads)?,
                key_heads: host(key_heads)?,
                width: host(width)?,
            },
        ]
        .map(|component| component.spec(0).map_err(failed));
        let (window, delta, tape) = (&window?, &delta?, &tape?);
        let tape_rows = *tape
            .shape
            .first()
            .ok_or_else(|| failed("recurrent tape has no row axis"))?
            as u64;
        let bank_bytes = [window, delta, tape]
            .iter()
            .try_fold(0u64, |total, component| {
                total
                    .checked_add(component.bytes().map_err(failed)? as u64)
                    .ok_or_else(|| failed("recurrent bank bytes overflow"))
            })?;
        let channels = (2 * key_heads + value_heads) * width;
        let device = self.device();
        let kernels = self.form::<gated_delta_step::Entry>(
            &[activation],
            &[
                ("M", 1),
                ("B", 1),
                ("S", STEP_BANKS),
                ("NK", key_heads),
                ("NV", value_heads),
                ("W", width),
                ("C", convolution_width),
                ("T", tape_rows),
            ],
            move |specialization| {
                gated_delta_step::native_for_device_with(
                    device,
                    gated_delta_step::Elements { A: activation },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let slab_banks = magnitude_state::banks_per_slab(bank_bytes).map_err(failed)? as u64;
        let regions = [window, delta, tape]
            .iter()
            .map(|component| SlabRegion {
                element: Element::dense(component.dtype),
                row_shape: component
                    .shape
                    .iter()
                    .map(|&extent| extent as u64)
                    .collect(),
            })
            .collect::<Vec<_>>();
        let launches = self.slab_copies(slab_banks, STEP_BANKS, &regions)?;
        let states = self.slabbed(slab_banks, STEP_BANKS, regions, launches)?;
        let projection = self.zeros(
            activation,
            &[1, channels + value_heads * width + 2 * value_heads],
        )?;
        let convolution = self.zeros(Element::f32(), &[channels, convolution_width])?;
        let rate = self.zeros(Element::f32(), &[value_heads])?;
        let time_bias = self.zeros(Element::f32(), &[value_heads])?;
        // One slot of one row reads the pristine bank 0 and publishes bank 1
        // after its row; the terminal segment row closes the table.
        let segments = self.i32s(&[2, 2], &[0, 1, 1, 1])?;
        let stop = self.i32s(&[1], &[1])?;
        let previous_bank = self.i32s(&[1], &[0])?;
        let previous_tape = self.i32s(&[1], &[0])?;
        let following_bank = self.i32s(&[1], &[1])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let projection = timed.bound(&projection)?;
            let convolution = timed.bound(&convolution)?;
            let rate = timed.bound(&rate)?;
            let time_bias = timed.bound(&time_bias)?;
            let segments = timed.bound(&segments)?;
            let stop = timed.bound(&stop)?;
            let previous_bank = timed.bound(&previous_bank)?;
            let previous_tape = timed.bound(&previous_tape)?;
            let following_bank = timed.bound(&following_bank)?;
            for state in &states {
                let [window, delta, tape] = state.regions.as_slice() else {
                    return Err(failed("recurrent state is not three regions"));
                };
                let mut window = timed.bound(window)?;
                let mut delta = timed.bound(delta)?;
                let mut tape = timed.bound(tape)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        gated_delta_step::WorkflowArgs {
                            projection: projection.tensor().into(),
                            convolution: convolution.tensor().into(),
                            rate: rate.tensor().into(),
                            time_bias: time_bias.tensor().into(),
                            segments: segments.tensor().into(),
                            stop: stop.tensor().into(),
                            previous_bank: previous_bank.tensor().into(),
                            previous_tape: previous_tape.tensor().into(),
                            following_bank: following_bank.tensor().into(),
                            window: window.tensor_mut().into(),
                            delta: delta.tensor_mut().into(),
                            tape: tape.tensor_mut().into(),
                            norm_epsilon: 1e-6 * width as f32,
                            grouped: false,
                            slab_banks: u32::try_from(slab_banks)
                                .map_err(|_| failed("bank slab count exceeds u32"))?,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(size_point(bank_bytes, samples))
    }

    /// Gated recurrent output of value heads of width 128 (the launch's
    /// reduction in whole heads, half as many key heads) into the launch's
    /// rows.
    fn delta_output(
        &self,
        norm: Element,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let value_heads = multiple(at.reduction / RECURRENT_WIDTH, 2);
        let key_heads = value_heads / 2;
        let inner = value_heads * RECURRENT_WIDTH;
        let projection_width =
            (2 * key_heads + value_heads) * RECURRENT_WIDTH + inner + 2 * value_heads;
        let hidden = at.rows;
        let point_bytes = bytes(weight, &[hidden, inner])?;
        let device = self.device();
        let kernels = self.form::<gated_delta_output::Entry>(
            &[norm, weight, activation],
            &[
                ("M", 1),
                ("H", hidden),
                ("NK", key_heads),
                ("NV", value_heads),
                ("W", RECURRENT_WIDTH),
            ],
            move |specialization| {
                gated_delta_output::native_for_device_with(
                    device,
                    gated_delta_output::Elements {
                        A: activation,
                        RN: norm,
                        OW: weight,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let weights = self.views(weight, inner, hidden, launches)?;
        let input = self.zeros(Element::f32(), &[1, hidden])?;
        let mixed = self.zeros(activation, &[1, value_heads, RECURRENT_WIDTH])?;
        let projection = self.zeros(activation, &[1, projection_width])?;
        let recurrent_norm = self.zeros(norm, &[RECURRENT_WIDTH])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let input = timed.bound(&input)?;
            let mixed = timed.bound(&mixed)?;
            let projection = timed.bound(&projection)?;
            let recurrent_norm = timed.bound(&recurrent_norm)?;
            for weight in &weights {
                let weight = timed.bound(weight)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        gated_delta_output::WorkflowArgs {
                            hidden: input.tensor().into(),
                            mixed: mixed.tensor().into(),
                            projection: projection.tensor().into(),
                            recurrent_norm: recurrent_norm.tensor().into(),
                            output_weight: weight.tensor().into(),
                            epsilon: 1e-6,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, hidden, weight, point_bytes, samples))
    }

    /// Paired gate/up projection: half the launch's rows each.
    fn dense_expand(
        &self,
        norm: Element,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let (features, hidden) = (multiple(at.rows / 2, ROW_ALIGNMENT), at.reduction);
        let point_bytes = bytes(weight, &[features, hidden])? * 2;
        let device = self.device();
        let kernels = self.form::<dense_expand::Entry>(
            &[norm, weight, activation],
            &[("M", 1), ("O", 1), ("H", hidden), ("F", features), ("GS", 0), ("US", 0)],
            move |specialization| {
                dense_expand::native_for_device_with(
                    device,
                    dense_expand::Elements {
                        NW: norm,
                        GW: weight,
                        UW: weight,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let gates = self.views(weight, hidden, features, launches)?;
        let ups = self.views(weight, hidden, features, launches)?;
        let residual = self.zeros(Element::f32(), &[1, hidden])?;
        let norm = self.zeros(norm, &[hidden])?;
        let out_rows = self.zeros(Element::i32(), &[1])?;
        let scale = self.zeros(Element::f32(), &[0])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let residual = timed.bound(&residual)?;
            let norm = timed.bound(&norm)?;
            let out_rows = timed.bound(&out_rows)?;
            let scale = timed.bound(&scale)?;
            for (gate, up) in gates.iter().zip(&ups) {
                let gate = timed.bound(gate)?;
                let up = timed.bound(up)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        dense_expand::WorkflowArgs {
                            residual: residual.tensor().into(),
                            norm: norm.tensor().into(),
                            gate_weight: gate.tensor().into(),
                            up_weight: up.tensor().into(),
                            out_rows: out_rows.tensor().into(),
                            eps: 1e-5,
                            activation: 0,
                            gate_scale: scale.tensor().into(),
                            up_scale: scale.tensor().into(),
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, 2 * features, weight, point_bytes, samples))
    }

    /// Down projection plus residual into the launch's rows.
    fn dense_output(
        &self,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let (hidden, features) = (at.rows, at.reduction);
        let point_bytes = bytes(weight, &[hidden, features])?;
        let device = self.device();
        let kernels = self.form::<dense_output::Entry>(
            &[weight, activation],
            &[("M", 1), ("O", 1), ("H", hidden), ("F", features), ("DS", 0)],
            move |specialization| {
                dense_output::native_for_device_with(
                    device,
                    dense_output::Elements {
                        DW: weight,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let weights = self.views(weight, features, hidden, launches)?;
        let residual = self.zeros(Element::f32(), &[1, hidden])?;
        let product = self.zeros(activation, &[1, features])?;
        let out_rows = self.zeros(Element::i32(), &[1])?;
        let scale = self.zeros(Element::f32(), &[0])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let residual = timed.bound(&residual)?;
            let product = timed.bound(&product)?;
            let out_rows = timed.bound(&out_rows)?;
            let scale = timed.bound(&scale)?;
            for weight in &weights {
                let weight = timed.bound(weight)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        dense_output::WorkflowArgs {
                            residual: residual.tensor().into(),
                            product: product.tensor().into(),
                            down_weight: weight.tensor().into(),
                            out_rows: out_rows.tensor().into(),
                            down_scale: scale.tensor().into(),
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, hidden, weight, point_bytes, samples))
    }

    /// Router logits and top-k selection of 8 of `experts` over a
    /// `hidden`-wide row. Its bytes are the router's.
    fn routed_route(
        &self,
        norm: Element,
        router: Element,
        activation: Element,
        hidden: u64,
        experts: u64,
    ) -> Step<MeasuredPoint> {
        let dimensions = [("M", 1), ("H", hidden), ("E", experts), ("K", ROUTED_SELECTED)];
        let device = self.device();
        let kernels = self.form::<routed_route::Entry>(
            &[norm, router, activation],
            &dimensions,
            move |specialization| {
                routed_route::native_for_device_with(
                    device,
                    routed_route::Elements {
                        NW: norm,
                        RW: router,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let router_bytes = bytes(router, &[experts, hidden])?;
        let launches = self.copies(router_bytes);
        let routers = self.views(router, hidden, experts, launches)?;
        let residual = self.zeros(Element::f32(), &[1, hidden])?;
        let norm = self.zeros(norm, &[hidden])?;
        let shared_router = self.zeros(Element::f32(), &[hidden])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let residual = timed.bound(&residual)?;
            let norm = timed.bound(&norm)?;
            let shared_router = timed.bound(&shared_router)?;
            for router in &routers {
                let router = timed.bound(router)?;
                let mut routes = timed
                    .graph
                    .local_for(kernel, "routes", &dimensions)
                    .map_err(failed)?;
                let mut scores = timed
                    .graph
                    .local_for(kernel, "scores", &dimensions)
                    .map_err(failed)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        routed_route::WorkflowArgs {
                            residual: residual.tensor().into(),
                            norm: norm.tensor().into(),
                            router: router.tensor().into(),
                            shared_router: shared_router.tensor().into(),
                            routes: routes.tensor_mut().into(),
                            scores: scores.tensor_mut().into(),
                            eps: 1e-6,
                            normalize: 1,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.r0)?;
                timed.export(scores.tensor())?;
            }
            Ok((timed, launches))
        })?;
        Ok(size_point(router_bytes, samples))
    }

    /// Decode expansion of 8 selected experts and a shared expert of equal
    /// features, gate and up each: the launch's rows in 18 equal parts. Only
    /// the selected experts are allocated: routes name each once, as a
    /// decode row's choices do.
    fn routed_expand(
        &self,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let experts = ROUTED_SELECTED;
        let features = multiple(at.rows / (2 * (experts + 1)), ROW_ALIGNMENT);
        let hidden = at.reduction;
        let point_bytes = sum(&[
            bytes(weight, &[experts, features, hidden])?,
            bytes(weight, &[features, hidden])?,
        ])? * 2;
        let device = self.device();
        let kernels = self.form::<routed_expand::Entry>(
            &[weight, activation],
            &[
                ("M", 1),
                ("H", hidden),
                ("E", experts),
                ("K", ROUTED_SELECTED),
                ("F", features),
                ("S", features),
            ],
            move |specialization| {
                routed_expand::native_for_device_with(
                    device,
                    routed_expand::Elements {
                        A: activation,
                        EGW: weight,
                        EUW: weight,
                        SGW: weight,
                        SUW: weight,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let expert_gates = self.shaped(weight, &[experts, features, hidden], launches)?;
        let expert_ups = self.shaped(weight, &[experts, features, hidden], launches)?;
        let shared_gates = self.transient_views(weight, hidden, features, launches)?;
        let shared_ups = self.transient_views(weight, hidden, features, launches)?;
        let normalized = self.zeros(activation, &[1, hidden])?;
        let routes = self.i32s(
            &[1, ROUTED_SELECTED],
            &(0..ROUTED_SELECTED as i32).collect::<Vec<_>>(),
        )?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let normalized = timed.bound(&normalized)?;
            let routes = timed.bound(&routes)?;
            for ((expert_gate, expert_up), (shared_gate, shared_up)) in expert_gates
                .iter()
                .zip(&expert_ups)
                .zip(shared_gates.iter().zip(&shared_ups))
            {
                let expert_gate = timed.bound(expert_gate)?;
                let expert_up = timed.bound(expert_up)?;
                let shared_gate = timed.bound(shared_gate)?;
                let shared_up = timed.bound(shared_up)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        routed_expand::WorkflowArgs {
                            normalized: normalized.tensor().into(),
                            routes: routes.tensor().into(),
                            expert_gate: expert_gate.tensor().into(),
                            expert_up: expert_up.tensor().into(),
                            shared_gate: shared_gate.tensor().into(),
                            shared_up: shared_up.tensor().into(),
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.r0)?;
                timed.export(&result.r1)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, 2 * (experts + 1) * features, weight, point_bytes, samples))
    }

    /// Decode down projection of 8 selected experts and a shared expert over
    /// the launch's reduction: its rows in 9 equal parts.
    fn routed_output(
        &self,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let experts = ROUTED_SELECTED;
        let hidden = multiple(at.rows / (experts + 1), ROW_ALIGNMENT);
        let features = at.reduction;
        let point_bytes = sum(&[
            bytes(weight, &[experts, hidden, features])?,
            bytes(weight, &[hidden, features])?,
        ])?;
        let device = self.device();
        let kernels = self.form::<routed_output::Entry>(
            &[weight, activation],
            &[
                ("M", 1),
                ("H", hidden),
                ("E", experts),
                ("K", ROUTED_SELECTED),
                ("F", features),
                ("S", features),
            ],
            move |specialization| {
                routed_output::native_for_device_with(
                    device,
                    routed_output::Elements {
                        A: activation,
                        EDW: weight,
                        SDW: weight,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let expert_downs = self.shaped(weight, &[experts, hidden, features], launches)?;
        let shared_downs = self.views(weight, features, hidden, launches)?;
        let residual = self.zeros(Element::f32(), &[1, hidden])?;
        let expert_product = self.zeros(activation, &[1, ROUTED_SELECTED, features])?;
        let shared_product = self.zeros(activation, &[1, features])?;
        let routes = self.i32s(
            &[1, ROUTED_SELECTED],
            &(0..ROUTED_SELECTED as i32).collect::<Vec<_>>(),
        )?;
        let scores = self.zeros(Element::f32(), &[1, ROUTED_SELECTED])?;
        let coefficient = self.zeros(Element::f32(), &[1])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let residual = timed.bound(&residual)?;
            let expert_product = timed.bound(&expert_product)?;
            let shared_product = timed.bound(&shared_product)?;
            let routes = timed.bound(&routes)?;
            let scores = timed.bound(&scores)?;
            let coefficient = timed.bound(&coefficient)?;
            for (expert_down, shared_down) in expert_downs.iter().zip(&shared_downs) {
                let expert_down = timed.bound(expert_down)?;
                let shared_down = timed.bound(shared_down)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        routed_output::WorkflowArgs {
                            residual: residual.tensor().into(),
                            expert_product: expert_product.tensor().into(),
                            shared_product: shared_product.tensor().into(),
                            routes: routes.tensor().into(),
                            scores: scores.tensor().into(),
                            coefficient: coefficient.tensor().into(),
                            expert_down: expert_down.tensor().into(),
                            shared_down: shared_down.tensor().into(),
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, (experts + 1) * hidden, weight, point_bytes, samples))
    }

    /// The final norm of one 4096-wide output row: a launch-dominated class.
    fn readout_features(&self, norm: Element, activation: Element) -> Step<MeasuredPoint> {
        let device = self.device();
        let kernels = self.form::<readout_features_rows::Entry>(
            &[norm, activation],
            &[("M", 1), ("O", 1), ("D", HIDDEN)],
            move |specialization| {
                readout_features_rows::native_for_device_with(
                    device,
                    readout_features_rows::Elements {
                        NW: norm,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = MAX_LAUNCHES;
        let hidden = self.zeros(Element::f32(), &[1, HIDDEN])?;
        let norms = self.views(norm, HIDDEN, 1, launches)?;
        let out_rows = self.zeros(Element::i32(), &[1])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let hidden = timed.bound(&hidden)?;
            let out_rows = timed.bound(&out_rows)?;
            for norm in &norms {
                let norm = timed.bound(&norm.reshape(&[HIDDEN]).map_err(failed)?)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        readout_features_rows::WorkflowArgs {
                            hidden: hidden.tensor().into(),
                            norm: norm.tensor().into(),
                            out_rows: out_rows.tensor().into(),
                            epsilon: 1e-6,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(size_point(bytes(norm, &[HIDDEN])?, samples))
    }

    /// Vocabulary projection of one row: the vocabulary is the launch's
    /// rows.
    fn readout_head(
        &self,
        norm: Element,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let (vocabulary, hidden) = (at.rows, at.reduction);
        let point_bytes = bytes(weight, &[vocabulary, hidden])?;
        let device = self.device();
        let kernels = self.form::<readout_head_rows::Entry>(
            &[norm, weight, activation],
            &[("M", 1), ("O", 1), ("V", vocabulary), ("D", hidden)],
            move |specialization| {
                readout_head_rows::native_for_device_with(
                    device,
                    readout_head_rows::Elements {
                        NW: norm,
                        OW: weight,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let weights = self.views(weight, hidden, vocabulary, launches)?;
        let input = self.zeros(Element::f32(), &[1, hidden])?;
        let norm = self.zeros(norm, &[hidden])?;
        let out_rows = self.zeros(Element::i32(), &[1])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let hidden = timed.bound(&input)?;
            let norm = timed.bound(&norm)?;
            let out_rows = timed.bound(&out_rows)?;
            for weight in &weights {
                let weight = timed.bound(weight)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        readout_head_rows::WorkflowArgs {
                            hidden: hidden.tensor().into(),
                            norm: norm.tensor().into(),
                            weight: weight.tensor().into(),
                            out_rows: out_rows.tensor().into(),
                            epsilon: 1e-6,
                            softcap: 0.0,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, vocabulary, weight, point_bytes, samples))
    }

    /// Unconstrained selection over one F32 logits row of `vocabulary`.
    fn sample_rows(&self, vocabulary: u64) -> Step<MeasuredPoint> {
        let dimensions = [("M", 1), ("V", vocabulary)];
        let device = self.device();
        let kernels = self.form::<sample_rows::Entry>(&[], &dimensions, move |specialization| {
            sample_rows::native_for_device(device, specialization)
        })?;
        self.begin()?;
        let point_bytes = bytes(Element::f32(), &[1, vocabulary])?;
        let launches = self.copies(point_bytes);
        let logits = self.views(Element::f32(), vocabulary, 1, launches)?;
        let mask = self.zeros(Element::u32(), &[1, vocabulary.div_ceil(32)])?;
        let constrained = self.zeros(Element::i32(), &[1])?;
        let draws = self.zeros(Element::u32(), &[1, 6])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let mask = timed.bound(&mask)?;
            let constrained = timed.bound(&constrained)?;
            let draws = timed.bound(&draws)?;
            for logits in &logits {
                let logits = timed.bound(logits)?;
                let mut result = timed
                    .graph
                    .local_for(kernel, "result", &dimensions)
                    .map_err(failed)?;
                timed
                    .graph
                    .enqueue(
                        kernel,
                        sample_rows::WorkflowArgs {
                            logits: logits.tensor().into(),
                            mask: mask.tensor().into(),
                            constrained: constrained.tensor().into(),
                            draws: draws.tensor().into(),
                            result: result.tensor_mut().into(),
                        },
                    )
                    .map_err(failed)?;
                timed.export(result.tensor())?;
            }
            Ok((timed, launches))
        })?;
        Ok(size_point(point_bytes, samples))
    }

    /// The reference chain's samples, measured once per session.
    fn chain_samples(&self) -> Step<Chained> {
        if let Some(chained) = self.session.chained.borrow().as_ref() {
            return Ok(chained.clone());
        }
        let chained = self.chained()?;
        *self.session.chained.borrow_mut() = Some(chained.clone());
        Ok(chained)
    }

    /// The dependency and step-submission costs of the reference cycle.
    ///
    /// One graph of [`CHAIN_CYCLES`] cycles of the four reference entries, each
    /// call consuming the previous call's result, as a decoder block's calls
    /// do. A sample's dependency cost is its device time per call beyond
    /// what the basis's own classes predict for those calls standing alone,
    /// so it is exactly the time those classes miss inside a step. Its
    /// submission cost is the host time of submitting the chained graph and
    /// waiting for it, beyond its device time.
    fn chained(&self) -> Step<Chained> {
        let activation = Element::bf16();
        let weight = Element::stored(
            "q4k",
            crate::resident_layout(crate::ExecutionPath::Native, self.device().backend()),
        )
        .ok_or_else(|| failed("q4k has no resident form on this backend"))?;
        let (hidden, features) = (HIDDEN, CHAIN_FEATURES);
        let (key_heads, value_heads, width) = (16, 32, RECURRENT_WIDTH);
        let channels = (2 * key_heads + value_heads) * width;
        let inner = value_heads * width;
        let recurrent = [
            ("M", 1),
            ("H", hidden),
            ("NK", key_heads),
            ("NV", value_heads),
            ("W", width),
        ];
        let device = self.device();
        // Every form is attempted before any stop, so the formation pass
        // queues all four.
        let project = self.form::<gated_delta_project::Entry>(
            &[activation, weight, activation],
            &recurrent,
            move |specialization| {
                gated_delta_project::native_for_device_with(
                    device,
                    gated_delta_project::Elements {
                        NW: activation,
                        QW: weight,
                        GW: weight,
                        AW: weight,
                        BW: weight,
                        A: activation,
                    },
                    specialization,
                )
            },
        );
        let output = self.form::<gated_delta_output::Entry>(
            &[activation, weight, activation],
            &recurrent,
            move |specialization| {
                gated_delta_output::native_for_device_with(
                    device,
                    gated_delta_output::Elements {
                        A: activation,
                        RN: activation,
                        OW: weight,
                    },
                    specialization,
                )
            },
        );
        let expand = self.form::<dense_expand::Entry>(
            &[activation, weight, activation],
            &[("M", 1), ("O", 1), ("H", hidden), ("F", features), ("GS", 0), ("US", 0)],
            move |specialization| {
                dense_expand::native_for_device_with(
                    device,
                    dense_expand::Elements {
                        NW: activation,
                        GW: weight,
                        UW: weight,
                        A: activation,
                    },
                    specialization,
                )
            },
        );
        let down = self.form::<dense_output::Entry>(
            &[weight, activation],
            &[("M", 1), ("O", 1), ("H", hidden), ("F", features), ("DS", 0)],
            move |specialization| {
                dense_output::native_for_device_with(
                    device,
                    dense_output::Elements {
                        DW: weight,
                        A: activation,
                    },
                    specialization,
                )
            },
        );
        let (project, output, expand, down) = (project?, output?, expand?, down?);
        // Each entry's default, then its INT8 variant where it declares one:
        // the chain runs the variants the classes are timed at, never a
        // slower default the classes would not charge.
        let variants = [project.len(), output.len(), expand.len(), down.len()]
            .into_iter()
            .max()
            .unwrap_or(1);
        fn variant<K>(kernels: &[K], choice: usize) -> &K {
            &kernels[choice.min(kernels.len() - 1)]
        }
        self.begin()?;
        let cycles = CHAIN_CYCLES as u64;
        let qkv = self.views(weight, hidden, channels, cycles)?;
        let gates = self.views(weight, hidden, inner, cycles)?;
        let alpha = self.zeros(weight, &[value_heads, hidden])?;
        let beta = self.zeros(weight, &[value_heads, hidden])?;
        let outputs = self.views(weight, inner, hidden, cycles)?;
        let expand_gates = self.views(weight, hidden, features, cycles)?;
        let expand_ups = self.views(weight, hidden, features, cycles)?;
        let downs = self.views(weight, features, hidden, cycles)?;
        let residual = self.zeros(Element::f32(), &[1, hidden])?;
        let norm = self.zeros(activation, &[hidden])?;
        let recurrent_norm = self.zeros(activation, &[width])?;
        let mixed = self.zeros(activation, &[1, value_heads, width])?;
        let chain = |choice: usize| -> Step<Timed> {
            let (project, output, expand, down) = (
                variant(&project, choice),
                variant(&output, choice),
                variant(&expand, choice),
                variant(&down, choice),
            );
            let mut timed = Timed::new(device);
            let input = timed.bound(&residual)?;
            let norm = timed.bound(&norm)?;
            let recurrent_norm = timed.bound(&recurrent_norm)?;
            let mixed = timed.bound(&mixed)?;
            let alpha = timed.bound(&alpha)?;
            let beta = timed.bound(&beta)?;
            let rows = timed.bound(&self.zeros(Element::i32(), &[1])?)?;
            let scale = timed.bound(&self.zeros(Element::f32(), &[0])?)?;
            let mut current: Option<WorkflowTensor> = None;
            for cycle in 0..CHAIN_CYCLES {
                let qkv = timed.bound(&qkv[cycle])?;
                let gate = timed.bound(&gates[cycle])?;
                let output_weight = timed.bound(&outputs[cycle])?;
                let expand_gate = timed.bound(&expand_gates[cycle])?;
                let expand_up = timed.bound(&expand_ups[cycle])?;
                let down_weight = timed.bound(&downs[cycle])?;
                let hidden = match &current {
                    Some(previous) => previous.clone(),
                    None => input.tensor().clone(),
                };
                let projected = timed
                    .graph
                    .enqueue(
                        project,
                        gated_delta_project::WorkflowArgs {
                            hidden: (&hidden).into(),
                            input_norm: norm.tensor().into(),
                            qkv_weight: qkv.tensor().into(),
                            gate_weight: gate.tensor().into(),
                            alpha_weight: alpha.tensor().into(),
                            beta_weight: beta.tensor().into(),
                            epsilon: 1e-6,
                        },
                    )
                    .map_err(failed)?
                    .value;
                let mixed_residual = timed
                    .graph
                    .enqueue(
                        output,
                        gated_delta_output::WorkflowArgs {
                            hidden: (&hidden).into(),
                            mixed: mixed.tensor().into(),
                            projection: (&projected).into(),
                            recurrent_norm: recurrent_norm.tensor().into(),
                            output_weight: output_weight.tensor().into(),
                            epsilon: 1e-6,
                        },
                    )
                    .map_err(failed)?
                    .value;
                let residual_in = mixed_residual.clone();
                let expanded = timed
                    .graph
                    .enqueue(
                        expand,
                        dense_expand::WorkflowArgs {
                            residual: (&residual_in).into(),
                            norm: norm.tensor().into(),
                            gate_weight: expand_gate.tensor().into(),
                            up_weight: expand_up.tensor().into(),
                            out_rows: rows.tensor().into(),
                            eps: 1e-5,
                            activation: 0,
                            gate_scale: scale.tensor().into(),
                            up_scale: scale.tensor().into(),
                        },
                    )
                    .map_err(failed)?
                    .value;
                let result = timed
                    .graph
                    .enqueue(
                        down,
                        dense_output::WorkflowArgs {
                            residual: (&residual_in).into(),
                            product: (&expanded).into(),
                            down_weight: down_weight.tensor().into(),
                            out_rows: rows.tensor().into(),
                            down_scale: scale.tensor().into(),
                        },
                    )
                    .map_err(failed)?
                    .value;
                current = Some(result);
            }
            if let Some(last) = &current {
                timed.export(last)?;
            }
            Ok(timed)
        };
        let calls = (CHAIN_CYCLES * CHAIN_CALLS_PER_CYCLE) as u64;
        // What the basis's own classes predict for one cycle's four calls at
        // the bytes they stream: the dependency cost is the chained time
        // those standalone costs do not account for.
        // The chain streams the reference representation, so no weight
        // format enters.
        let class_seconds = |class: OperationClass, rows: u64, streamed: u64| -> Step<f64> {
            let key = MeasurementKey::new(class, &[activation]);
            let points = self.points(&key)?;
            match ClassCost::from_points(class, &points).map_err(failed)?.model {
                CostModel::Projection(projection) => Ok(projection.launch_seconds
                    + projection.seconds_per_byte(rows) * streamed as f64),
                _ => Err(failed(format!("{} is not a projection", class.name()))),
            }
        };
        let project_bytes = sum(&[
            bytes(weight, &[channels, hidden])?,
            bytes(weight, &[inner, hidden])?,
            bytes(weight, &[value_heads, hidden])?,
            bytes(weight, &[value_heads, hidden])?,
        ])?;
        let expand_bytes = sum(&[
            bytes(weight, &[features, hidden])?,
            bytes(weight, &[features, hidden])?,
        ])?;
        let standalone_per_call = (class_seconds(
            OperationClass::DeltaProject,
            channels + inner + 2 * value_heads,
            project_bytes,
        )? + class_seconds(
            OperationClass::DeltaOutput,
            hidden,
            bytes(weight, &[hidden, inner])?,
        )? + class_seconds(OperationClass::DenseExpand, 2 * features, expand_bytes)?
            + class_seconds(
                OperationClass::DenseOutput,
                hidden,
                bytes(weight, &[hidden, features])?,
            )?)
            / CHAIN_CALLS_PER_CYCLE as f64;
        let (choice, chained) =
            self.fastest_graph((0..variants).map(|choice| Ok((chain(choice)?, calls))))?;
        let chained_call = median(&chained).ok_or_else(|| failed("no chained median"))?;
        let submission =
            self.submissions(chain(choice)?, chained_call * calls as f64, CHAIN_SAMPLES)?;
        Ok(Chained {
            // A dependent call cannot cost less than its standalone launch: a
            // difference at or below zero measures none.
            dependency: size_point(
                0,
                chained
                    .iter()
                    .map(|call| (call - standalone_per_call).max(0.0))
                    .collect(),
            ),
            submission: size_point(0, submission),
        })
    }

    /// Host seconds of submitting and waiting for the same sealed chain,
    /// beyond its measured device work. Reusing the plan keeps graph formation
    /// outside the samples, as it is for a served step.
    fn submissions(&self, timed: Timed, device_seconds: f64, count: usize) -> Step<Vec<f64>> {
        let plan = timed.graph.seal().map_err(failed)?;
        let mut slot = plan.new_slot().map_err(failed)?;
        (0..count)
            .map(|_| {
                let mut bindings = plan.bindings();
                for (port, tensor) in &timed.bindings {
                    bindings.set(port, tensor).map_err(failed)?;
                }
                let outputs = plan.new_outputs().map_err(failed)?;
                self.session.trace.collect().map_err(failed)?;
                let began = Instant::now();
                let (_, completion) = slot
                    .attach(bindings, outputs)
                    .map_err(failed)?
                    .submit()
                    .map_err(fault)?;
                completion.wait().map_err(fault)?;
                let wall = began.elapsed().as_secs_f64();
                self.session.trace.collect().map_err(failed)?;
                Ok((wall - device_seconds).max(0.0))
            })
            .collect()
    }
}

/// Samples of the per-call dependency cost and the per-step submission cost.
#[derive(Clone)]
struct Chained {
    dependency: MeasuredPoint,
    submission: MeasuredPoint,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_points_differ_from_the_reference_in_one_axis() {
        let points = history_points();
        assert_eq!(&points[..2], &[(HISTORY_REFERENCE, 4096), (HISTORY_REFERENCE, 32_768)]);
        for (heads, depth) in &points[2..] {
            assert_eq!(*depth, HISTORY_DEPTH);
            let differing = [
                heads.kv_heads != HISTORY_REFERENCE.kv_heads,
                heads.group != HISTORY_REFERENCE.group,
                heads.width != HISTORY_REFERENCE.width,
            ];
            assert_eq!(differing.iter().filter(|differs| **differs).count(), 1);
        }
    }

    #[test]
    fn the_floor_launch_shares_a_ladder_row_count() {
        let launches = projection_launches();
        assert!(LADDER_ROWS.contains(&FLOOR_ROWS));
        assert!(FLOOR_REDUCTION < REDUCTION);
        assert_eq!(launches.len(), LADDER_ROWS.len() + 1);
        assert_eq!(FORM_LAUNCH.reduction, REDUCTION);
    }

    #[test]
    fn measured_history_planes_are_the_state_layout_rows() {
        let configuration = super::super::plan::tests::QWEN35_CONFIGURATIONS[3];
        let (definition, _) = super::super::plan::tests::declared_model(&configuration);
        for (codec, affine) in [(KvCodec::Dense, false), (KvCodec::AffineK8V4, true)] {
            let layout =
                magnitude_state::ModelStateLayout::derive(&definition.decoder, &[], codec, 0)
                    .unwrap();
            let layout_row = layout.target_history[0].components()[0]
                .planes()
                .iter()
                .map(|plane| plane.row_bytes as u64)
                .sum::<u64>();
            let planes = history_planes(
                affine,
                Element::bf16(),
                configuration.kv_heads,
                configuration.head_width,
            )
            .unwrap();
            assert_eq!(
                planes.iter().map(|(_, _, bytes)| bytes).sum::<u64>(),
                layout_row
            );
            for (element, per_head, row_bytes) in planes {
                assert_eq!(
                    element
                        .canonical_byte_len(&[1, configuration.kv_heads, per_head])
                        .unwrap(),
                    row_bytes
                );
            }
        }
    }
}
