//! The fixed device measurement behind a basis.
//!
//! Every key of the backend's declared plan is formed with its shipped
//! default specialization and timed over synthetic device-resident tensors.
//! Nothing inspects a model artifact, tunes a parameter or loads a model. A
//! class the backend cannot form is recorded unsupported; any other failure,
//! including a nonphysical fit, is a measurement error.
//!
//! The measurement is built for a cold run of a few seconds:
//!
//! - Every planned native form is formed first, in parallel.
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
    median, BasisIdentity, ClassCost, ClassMeasurement, MeasuredPoint, MeasurementBasis,
    MeasurementKey, OperationClass,
};
use super::plan::measurement_plan;
use crate::platform::{refresh_device_ceiling, MemoryPolicyError, MemoryReserves};
use magnitude_model_contracts::{
    ActivationDType, BlockGeometry, DecoderGeometry, FeedForwardGeometry, MixerGeometry,
    RecurrentGeometry, RecurrentHeadMapping,
};
use magnitude_model_kernels::{
    attention_output, dense_expand, dense_output, embedding_rows, gated_attention_decode,
    gated_attention_decode_k8v4, gated_attention_project, gated_delta_output, gated_delta_project,
    gated_delta_step, readout_features_rows, readout_head_rows, routed_expand, routed_output,
    routed_route, sample_rows,
};
use magnitude_model_state::{ComponentDescriptor, KvCodec, LayerRef, ModelStateLayout};
use seismic::{
    generated, BackendName, Device, DeviceCatalog, Element, Entry, LoadError, NativeGraph,
    NativeKernel, NativePort, NativeSpecialization, SubmissionTrace, Tensor, TraceDetail,
    WorkflowTensor,
};
use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Streamed bytes of the two sizes a linear weight-streaming class is timed
/// at: a launch-dominated size and a bandwidth-dominated size.
const SMALL_BYTES: u64 = 2 * 1024 * 1024;
const LARGE_BYTES: u64 = 64 * 1024 * 1024;

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
const VARIED_PARAMETERS: [&str; 2] = ["INT8", "PARTS"];

/// The most launches one timed graph holds.
const MAX_LAUNCHES: u64 = 256;
/// Timed samples of every point, after one discarded sample.
const RUNS: usize = 5;
/// Device time the device is kept busy before the first timed run.
const WARM_SECONDS: f64 = 0.2;
/// Device time of one sample: runs of a point's graph queued as one
/// submission. On Metal, short submissions of one run each read 10–40%
/// slower and vary between runs of the measurement (2026-09-25).
const SAMPLE_SECONDS: f64 = 0.002;
/// The most runs one sample queues.
const MAX_PASSES: usize = 256;
/// Every synthetic extent a size search chooses is a multiple of this: it is
/// a multiple of every packed representation's group and of every native
/// row and reduction alignment.
const UNIT: u64 = 256;
/// Pool views start at multiples of this many rows (a packed layout's row
/// group).
const ROW_ALIGNMENT: u64 = 16;
/// Context depths the attention history classes are timed at.
const HISTORY_DEPTHS: [u64; 2] = [4096, 32_768];
/// Vocabulary widths sampling is timed at.
const SAMPLE_VOCABULARIES: [u64; 2] = [32_768, 1_048_576];

/// The row width of every synthetic weight-streaming matrix: each class
/// varies the other dimension, so all its views share one pool per
/// representation.
const HIDDEN: u64 = 4096;
const ATTENTION_QUERY_HEADS: u64 = 16;
const ATTENTION_WIDTH: u64 = 256;
/// The projection is sized by its query heads per kv head, over one kv head
/// of width 128.
const PROJECT_WIDTH: u64 = 128;
const RECURRENT_KEY_HEADS: u64 = 16;
const RECURRENT_VALUE_HEADS: u64 = 32;
const RECURRENT_WIDTH: u64 = 128;
const ROUTED_SELECTED: u64 = 8;
const ROUTED_FEATURES: u64 = 512;
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
    /// Allocation, formation infrastructure or a timed launch failed.
    Device {
        key: MeasurementKey,
        message: String,
    },
    /// The class's samples do not establish a physical cost.
    Cost {
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
            Self::Device { key, message } => {
                write!(
                    formatter,
                    "measuring {} {key:?}: {message}",
                    key.class.name()
                )
            }
            Self::Cost { key, message } => write!(
                formatter,
                "measured {} {key:?} has no physical cost: {message}",
                key.class.name()
            ),
            Self::Trace(message) => write!(formatter, "measurement trace: {message}"),
        }
    }
}

impl std::error::Error for MeasurementError {}

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
    pub total: Duration,
}

/// Measure every key of `device`'s declared plan.
pub fn measure_basis(
    catalog: &DeviceCatalog,
    device: &Device,
    reserves: MemoryReserves,
    identity: BasisIdentity,
) -> Result<MeasurementBasis, MeasurementError> {
    measure_basis_observed(catalog, device, reserves, identity, |_, _, _| {})
}

/// [`measure_basis`], reporting each class with its time profile as it
/// completes. The parallel formation pass is reported as the first class's
/// formation.
pub fn measure_basis_observed(
    catalog: &DeviceCatalog,
    device: &Device,
    reserves: MemoryReserves,
    identity: BasisIdentity,
    mut observe: impl FnMut(&MeasurementKey, &ClassMeasurement, &ClassProfile),
) -> Result<MeasurementBasis, MeasurementError> {
    let plan = measurement_plan(device.backend());
    let session = Session::open(device)?;
    let formation = session.form_all(&plan);
    let mut classes = Vec::with_capacity(plan.len());
    for (index, key) in plan.into_iter().enumerate() {
        let (measurement, mut profile) = session.measure(catalog, &reserves, &key)?;
        if index == 0 {
            profile.formation += formation;
            profile.total += formation;
        }
        observe(&key, &measurement, &profile);
        classes.push((key, measurement));
    }
    Ok(MeasurementBasis { identity, classes })
}

/// Measure one class on its own.
pub fn measure_class(
    catalog: &DeviceCatalog,
    device: &Device,
    reserves: &MemoryReserves,
    key: &MeasurementKey,
) -> Result<(ClassMeasurement, ClassProfile), MeasurementError> {
    Session::open(device)?.measure(catalog, reserves, key)
}

/// Why a class produced no points.
enum Stop {
    /// The backend has no implementation, the default configuration is not
    /// admissible at the class's statics, or the native form fails to form.
    Unsupported(String),
    Failed(String),
    /// The formation pass queued this point's forms and stopped.
    Queued,
}

type Step<T> = Result<T, Stop>;

fn failed(error: impl fmt::Display) -> Stop {
    Stop::Failed(error.to_string())
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

/// The smallest multiple of `unit` whose streamed bytes reach `target`.
fn size_for(target: u64, unit: u64, streamed: impl Fn(u64) -> Step<u64>) -> Step<u64> {
    let mut upper = 1u64;
    while streamed(upper * unit)? < target {
        upper = upper
            .checked_mul(2)
            .ok_or_else(|| failed("synthetic size search overflows"))?;
    }
    let mut lower = 0u64;
    while upper - lower > 1 {
        let middle = lower + (upper - lower) / 2;
        if streamed(middle * unit)? < target {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    Ok(upper * unit)
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn geometry(key: &MeasurementKey, name: &str) -> Step<u64> {
    key.geometry_value(name)
        .ok_or_else(|| failed(format!("{} key has no {name}", key.class.name())))
}

fn activation_dtype(activation: Element) -> Step<ActivationDType> {
    if activation == Element::bf16() {
        Ok(ActivationDType::BF16)
    } else if activation == Element::f16() {
        Ok(ActivationDType::F16)
    } else {
        Err(failed(format!(
            "{} is not a decoder activation",
            activation.name()
        )))
    }
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

/// What one measurement run shares across its classes.
struct Session<'a> {
    device: &'a Device,
    rotation: u64,
    trace: SubmissionTrace,
    pools: RefCell<Vec<Pool>>,
    warmed: Cell<bool>,
    formed: Formed,
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
            chained: RefCell::new(None),
        })
    }

    /// Form every native kernel of `plan` in parallel: a formation pass over
    /// the plan queues each point's forms, then worker threads form them.
    fn form_all(&self, plan: &[MeasurementKey]) -> Duration {
        let began = Instant::now();
        let jobs: Mutex<Vec<FormJob<'_>>> = Mutex::new(Vec::new());
        {
            let runner = Runner {
                session: self,
                queue: Some(&jobs),
                profile: Cell::new(ClassProfile::default()),
            };
            for key in plan {
                // Every outcome of the pass is a queued form or a class that
                // needs none; the timing pass reports failures.
                let _ = runner.points(key);
            }
        }
        let jobs = jobs.into_inner().expect("formation queue lock is never poisoned");
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
        key: &MeasurementKey,
    ) -> Result<(ClassMeasurement, ClassProfile), MeasurementError> {
        let began = Instant::now();
        self.pools.borrow_mut().retain(|pool| !pool.transient);
        refresh_device_ceiling(catalog, self.device, reserves)
            .map_err(MeasurementError::Ceiling)?;
        let runner = Runner {
            session: self,
            queue: None,
            profile: Cell::new(ClassProfile::default()),
        };
        let points = runner.points(key);
        let profile = ClassProfile {
            total: began.elapsed(),
            ..runner.profile.get()
        };
        let measurement = match points {
            Ok(points) => ClassCost::from_points(key.class, &points)
                .map(|cost| ClassMeasurement::Measured { points, cost })
                .map_err(|message| MeasurementError::Cost {
                    key: key.clone(),
                    message,
                }),
            Err(Stop::Unsupported(reason)) => Ok(ClassMeasurement::Unsupported { reason }),
            Err(Stop::Failed(message)) => Err(MeasurementError::Device {
                key: key.clone(),
                message,
            }),
            Err(Stop::Queued) => Err(MeasurementError::Device {
                key: key.clone(),
                message: "the timing pass queued a formation".into(),
            }),
        }?;
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

struct Runner<'q, 's, 'a> {
    session: &'s Session<'a>,
    /// Present during the formation pass: forms are queued here and the
    /// point stops.
    queue: Option<&'q Mutex<Vec<FormJob<'s>>>>,
    profile: Cell<ClassProfile>,
}

impl<'q, 's, 'a> Runner<'q, 's, 'a> {
    fn device(&self) -> &'a Device {
        self.session.device
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
        prepare: impl Fn(&NativeSpecialization) -> Result<NativeKernel<E>, LoadError>
            + Send
            + 's,
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
            if VARIED_PARAMETERS.contains(&parameter.name.as_str()) && parameter.arithmetic {
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
                        return Formation::Unsupported(format!("{} does not form: {error}", E::NAME))
                    }
                    Err(_) => {}
                }
            }
            Formation::Formed(kernels)
        };
        if let Some(queue) = self.queue {
            let mut cache = formed.lock().expect("formation cache lock is never poisoned");
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
        if self.queue.is_some() {
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

    /// Start a point: its views begin at every pool's start.
    fn begin(&self) {
        for pool in self.session.pools.borrow_mut().iter_mut() {
            pool.cursor = 0;
        }
    }

    fn zeros(&self, element: Element, extents: &[u64]) -> Step<Tensor> {
        if self.queue.is_some() {
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

    /// Seal `timed` and time it. One run sizes a sample: `passes` runs
    /// queued as one submission, at least [`SAMPLE_SECONDS`] of device work.
    /// A leading sample (and, before the session's first point, samples
    /// until the device has been busy [`WARM_SECONDS`]) brings the device to
    /// its sustained clock; then [`RUNS`] samples follow back to back. Each
    /// sample's device interval, per launch.
    fn run(&self, timed: Timed, launches: u64) -> Step<Vec<f64>> {
        let began = Instant::now();
        let plan = timed.graph.seal().map_err(failed)?;
        let mut slot = plan.new_slot().map_err(failed)?;
        let mut submit = |samples: usize, passes: usize| -> Step<Vec<f64>> {
            // Work outside these samples (pool fills) is not a sample.
            self.session.trace.collect().map_err(failed)?;
            let mut completions = Vec::with_capacity(samples);
            for _ in 0..samples {
                let mut sequence = self.device().native_sequence();
                for _ in 0..passes {
                    let mut bindings = plan.bindings();
                    for (port, tensor) in &timed.bindings {
                        bindings.set(port, tensor).map_err(failed)?;
                    }
                    let outputs = plan.new_outputs().map_err(failed)?;
                    slot.attach(bindings, outputs)
                        .map_err(failed)?
                        .queue(&mut sequence)
                        .map_err(failed)?;
                }
                completions.push(sequence.submit().map_err(failed)?);
            }
            for completion in completions {
                completion.wait().map_err(failed)?;
            }
            let traced = self.session.trace.collect().map_err(failed)?;
            if traced.len() != samples {
                return Err(failed(format!(
                    "{samples} timed samples recorded {} submissions",
                    traced.len()
                )));
            }
            Ok(traced
                .iter()
                .map(|submission| (submission.device.1 - submission.device.0) / passes as f64)
                .collect())
        };
        let run = submit(1, 1)?[0];
        let passes = ((SAMPLE_SECONDS / run.max(1e-7)).ceil() as usize).clamp(1, MAX_PASSES);
        if !self.session.warmed.get() {
            let mut busy = 0.0;
            while busy < WARM_SECONDS {
                busy += submit(RUNS, passes)?.iter().sum::<f64>() * passes as f64;
            }
            self.session.warmed.set(true);
        }
        let seconds = submit(1 + RUNS, passes)?;
        self.charge(|profile| &mut profile.timing, began);
        Ok(seconds[1..]
            .iter()
            .map(|run| run / launches as f64)
            .collect())
    }

    /// The samples of the variant with the smallest median.
    fn fastest<E: Entry>(
        &self,
        kernels: &[NativeKernel<E>],
        time: impl Fn(&NativeKernel<E>) -> Step<Vec<f64>>,
    ) -> Step<Vec<f64>> {
        let mut best: Option<(f64, Vec<f64>)> = None;
        for kernel in kernels {
            let samples = time(kernel)?;
            let seconds =
                median(&samples).ok_or_else(|| failed("timed runs have no median"))?;
            if best.as_ref().is_none_or(|(fastest, _)| seconds < *fastest) {
                best = Some((seconds, samples));
            }
        }
        best.map(|(_, samples)| samples)
            .ok_or_else(|| failed("no formed variant was timed"))
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

    fn points(&self, key: &MeasurementKey) -> Step<Vec<MeasuredPoint>> {
        use OperationClass as C;
        let sized = [SMALL_BYTES, LARGE_BYTES];
        match (key.class, key.bindings.as_slice()) {
            (C::EmbeddingRows, &[table, activation]) => {
                self.each(&[()], |()| self.embedding_rows(table, activation))
            }
            (C::AttentionProject, &[norm, weight, activation]) => self.each(&sized, |target| {
                self.attention_project(norm, weight, activation, target)
            }),
            (C::AttentionDecode | C::AttentionDecodeK8V4, &[activation]) => {
                self.each(&HISTORY_DEPTHS, |depth| {
                    self.attention_decode(key, activation, depth)
                })
            }
            (C::AttentionOutput, &[weight, activation]) => self.each(&sized, |target| {
                self.attention_output(weight, activation, target)
            }),
            (C::DeltaProject, &[norm, weight, activation]) => self.each(&sized, |target| {
                self.delta_project(norm, weight, activation, target)
            }),
            (C::DeltaStep, &[activation]) => {
                self.each(&[()], |()| self.delta_step(key, activation))
            }
            (C::DeltaOutput, &[norm, weight, activation]) => self.each(&sized, |target| {
                self.delta_output(norm, weight, activation, target)
            }),
            (C::DenseExpand, &[norm, weight, activation]) => self.each(&sized, |target| {
                self.dense_expand(norm, weight, activation, target)
            }),
            (C::DenseOutput, &[weight, activation]) => self.each(&sized, |target| {
                self.dense_output(weight, activation, target)
            }),
            (C::RoutedRoute, &[norm, router, activation]) => {
                self.each(&[()], |()| self.routed_route(key, norm, router, activation))
            }
            (C::RoutedExpand, &[weight, activation]) => self.each(&sized, |target| {
                self.routed_expand(weight, activation, target)
            }),
            (C::RoutedOutput, &[weight, activation]) => self.each(&sized, |target| {
                self.routed_output(weight, activation, target)
            }),
            (C::ReadoutFeatures, &[norm, activation]) => {
                self.each(&[()], |()| self.readout_features(norm, activation))
            }
            (C::ReadoutHead, &[norm, weight, activation]) => self.each(&sized, |target| {
                self.readout_head(norm, weight, activation, target)
            }),
            (C::SampleRows, &[]) => {
                self.each(&SAMPLE_VOCABULARIES, |vocabulary| self.sample_rows(vocabulary))
            }
            (C::LaunchDependency, &[]) => Ok(vec![self.chain_samples()?.dependency]),
            (C::StepSubmission, &[]) => Ok(vec![self.chain_samples()?.submission]),
            _ => Err(failed(format!("{key:?} has an unexpected binding list"))),
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
        self.begin();
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
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.r1)?;
            }
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: table_bytes / vocabulary,
            samples,
        })
    }

    /// Q/K/V projection over one kv head of width 128 from a 4096-wide
    /// residual, the query heads per kv head sized to the target.
    fn attention_project(
        &self,
        norm: Element,
        weight: Element,
        activation: Element,
        target: u64,
    ) -> Step<MeasuredPoint> {
        let key_rows = PROJECT_WIDTH;
        let query_rows = |group: u64| group * 2 * PROJECT_WIDTH;
        let streamed = |group| {
            sum(&[
                bytes(weight, &[query_rows(group), HIDDEN])?,
                2 * bytes(weight, &[key_rows, HIDDEN])?,
            ])
        };
        let group = size_for(target, 1, streamed)?;
        let device = self.device();
        let kernels = self.form::<gated_attention_project::Entry>(
            &[norm, weight, activation],
            &[
                ("M", 1),
                ("D", HIDDEN),
                ("KV", 1),
                ("G", group),
                ("W", PROJECT_WIDTH),
            ],
            move |specialization| {
                gated_attention_project::native_for_device_with(
                    device,
                    gated_attention_project::Elements {
                        NW: norm,
                        QW: weight,
                        KW: weight,
                        VW: weight,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin();
        let point_bytes = streamed(group)?;
        let launches = self.copies(point_bytes);
        let queries = self.views(weight, HIDDEN, query_rows(group), launches)?;
        let keys = self.views(weight, HIDDEN, key_rows, launches)?;
        let values = self.views(weight, HIDDEN, key_rows, launches)?;
        let input = self.zeros(Element::f32(), &[1, HIDDEN])?;
        let input_norm = self.zeros(norm, &[HIDDEN])?;
        let query_norm = self.zeros(Element::f32(), &[PROJECT_WIDTH])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let input = timed.bound(&input)?;
            let input_norm = timed.bound(&input_norm)?;
            let query_norm = timed.bound(&query_norm)?;
            for ((query, key), value) in queries.iter().zip(&keys).zip(&values) {
                let query = timed.bound(query)?;
                let key = timed.bound(key)?;
                let value = timed.bound(value)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        gated_attention_project::WorkflowArgs {
                            hidden: input.tensor().into(),
                            input_norm: input_norm.tensor().into(),
                            query_norm: query_norm.tensor().into(),
                            query_gate_weight: query.tensor().into(),
                            key_weight: key.tensor().into(),
                            value_weight: value.tensor().into(),
                            epsilon: 1e-6,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.r0)?;
                timed.export(&result.r1)?;
                timed.export(&result.r2)?;
            }
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: point_bytes,
            samples,
        })
    }

    /// Fused attention of one decode row over `depth` history rows at the
    /// key's head geometry. The streamed bytes are the history it reads.
    fn attention_decode(
        &self,
        key: &MeasurementKey,
        activation: Element,
        depth: u64,
    ) -> Step<MeasuredPoint> {
        let affine = key.class == OperationClass::AttentionDecodeK8V4;
        let kv_heads = geometry(key, "kv_heads")?;
        let group = geometry(key, "group")?;
        let pairs = geometry(key, "rotary_pairs")?;
        let width = geometry(key, "width")?;
        let rows = depth + 1;
        let dimensions = [
            ("M", 1),
            ("T", rows),
            ("KV", kv_heads),
            ("G", group),
            ("P", pairs),
            ("S", width - 2 * pairs),
            ("R", 1),
        ];
        let device = self.device();
        let (dense, k8v4) = if affine {
            let kernels = self.form::<gated_attention_decode_k8v4::Entry>(
                &[activation],
                &dimensions,
                move |specialization| {
                    gated_attention_decode_k8v4::native_for_device_with(
                        device,
                        gated_attention_decode_k8v4::Elements { A: activation },
                        specialization,
                    )
                },
            )?;
            (Vec::new(), kernels)
        } else {
            let kernels = self.form::<gated_attention_decode::Entry>(
                &[activation],
                &dimensions,
                move |specialization| {
                    gated_attention_decode::native_for_device_with(
                        device,
                        gated_attention_decode::Elements { A: activation },
                        specialization,
                    )
                },
            )?;
            (kernels, Vec::new())
        };
        self.begin();
        let planes = history_planes(affine, activation, kv_heads, width).map_err(failed)?;
        let row_bytes = sum(&planes.iter().map(|(_, _, bytes)| *bytes).collect::<Vec<_>>())?;
        let history_bytes = depth * row_bytes;
        let launches = self.copies(rows * row_bytes);
        let histories = planes
            .iter()
            .map(|(element, per_head, _)| {
                self.shaped(*element, &[rows, kv_heads, *per_head], launches)
            })
            .collect::<Step<Vec<_>>>()?;
        let depth = i32::try_from(depth).map_err(failed)?;
        let query_gate = self.zeros(activation, &[1, kv_heads * group * 2 * width])?;
        let fresh_key = self.zeros(activation, &[1, kv_heads * width])?;
        let fresh_value = self.zeros(activation, &[1, kv_heads * width])?;
        let query_norm = self.zeros(Element::f32(), &[width])?;
        let key_norm = self.zeros(Element::f32(), &[width])?;
        let rotary_components = self.zeros(Element::i32(), &[pairs])?;
        let rotary_frequencies = self.zeros(Element::f32(), &[pairs])?;
        let coordinates = self.zeros(Element::i32(), &[1, 4])?;
        let visible = self.i32s(&[1, 1, 2], &[0, depth])?;
        let fresh = self.i32s(&[1, 2], &[0, 1])?;
        let destinations = self.i32s(&[1], &[depth])?;
        let scale = 1.0 / (width as f32).sqrt();
        // The inputs every launch shares, bound once per graph.
        let shared = |timed: &mut Timed| -> Step<[NativePort; 11]> {
            Ok([
                timed.bound(&query_gate)?,
                timed.bound(&fresh_key)?,
                timed.bound(&fresh_value)?,
                timed.bound(&query_norm)?,
                timed.bound(&key_norm)?,
                timed.bound(&rotary_components)?,
                timed.bound(&rotary_frequencies)?,
                timed.bound(&coordinates)?,
                timed.bound(&visible)?,
                timed.bound(&fresh)?,
                timed.bound(&destinations)?,
            ])
        };
        let samples = if affine {
            self.fastest(&k8v4, |kernel| {
                let mut timed = Timed::new(device);
                let [qg, k, v, qn, kn, rc, rf, co, vi, fr, de] = shared(&mut timed)?;
                for index in 0..launches as usize {
                    let mut planes = histories
                        .iter()
                        .map(|views| timed.bound(&views[index]))
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
                            gated_attention_decode_k8v4::WorkflowArgs {
                                query_gate: qg.tensor().into(),
                                key: k.tensor().into(),
                                value: v.tensor().into(),
                                query_norm: qn.tensor().into(),
                                key_norm: kn.tensor().into(),
                                rotary_components: rc.tensor().into(),
                                rotary_frequencies: rf.tensor().into(),
                                coordinates: co.tensor().into(),
                                visible: vi.tensor().into(),
                                fresh: fr.tensor().into(),
                                destinations: de.tensor().into(),
                                history_key_codes: key_codes.tensor_mut().into(),
                                history_key_coefficients: key_coefficients.tensor_mut().into(),
                                history_value_codes: value_codes.tensor_mut().into(),
                                history_value_coefficients: value_coefficients
                                    .tensor_mut()
                                    .into(),
                                epsilon: 1e-6,
                                scale,
                            },
                        )
                        .map_err(failed)?;
                    timed.export(&result.value)?;
                }
                self.run(timed, launches)
            })?
        } else {
            self.fastest(&dense, |kernel| {
                let mut timed = Timed::new(device);
                let [qg, k, v, qn, kn, rc, rf, co, vi, fr, de] = shared(&mut timed)?;
                for index in 0..launches as usize {
                    let mut planes = histories
                        .iter()
                        .map(|views| timed.bound(&views[index]))
                        .collect::<Step<Vec<_>>>()?;
                    let [history_key, history_value] = planes.as_mut_slice() else {
                        return Err(failed("dense history is not two planes"));
                    };
                    let result = timed
                        .graph
                        .enqueue(
                            kernel,
                            gated_attention_decode::WorkflowArgs {
                                query_gate: qg.tensor().into(),
                                key: k.tensor().into(),
                                value: v.tensor().into(),
                                query_norm: qn.tensor().into(),
                                key_norm: kn.tensor().into(),
                                rotary_components: rc.tensor().into(),
                                rotary_frequencies: rf.tensor().into(),
                                coordinates: co.tensor().into(),
                                visible: vi.tensor().into(),
                                fresh: fr.tensor().into(),
                                destinations: de.tensor().into(),
                                history_key: history_key.tensor_mut().into(),
                                history_value: history_value.tensor_mut().into(),
                                epsilon: 1e-6,
                                scale,
                            },
                        )
                        .map_err(failed)?;
                    timed.export(&result.value)?;
                }
                self.run(timed, launches)
            })?
        };
        Ok(MeasuredPoint {
            bytes: history_bytes,
            samples,
        })
    }

    /// Output projection of 16 heads of width 256, the output width sized.
    fn attention_output(
        &self,
        weight: Element,
        activation: Element,
        target: u64,
    ) -> Step<MeasuredPoint> {
        let reduction = ATTENTION_QUERY_HEADS * ATTENTION_WIDTH;
        let streamed = |hidden| bytes(weight, &[hidden, reduction]);
        let hidden = size_for(target, UNIT, streamed)?;
        let device = self.device();
        let kernels = self.form::<attention_output::Entry>(
            &[weight, activation],
            &[
                ("M", 1),
                ("D", hidden),
                ("Q", ATTENTION_QUERY_HEADS),
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
        self.begin();
        let point_bytes = streamed(hidden)?;
        let launches = self.copies(point_bytes);
        let weights = self.views(weight, reduction, hidden, launches)?;
        let input = self.zeros(Element::f32(), &[1, hidden])?;
        let gated = self.zeros(activation, &[1, ATTENTION_QUERY_HEADS, ATTENTION_WIDTH])?;
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
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: point_bytes,
            samples,
        })
    }

    /// Recurrent projection from a 4096-wide residual at width 128, twice
    /// as many value heads as key heads, the key heads sized. The small
    /// decay-rate matrices are shared by every launch.
    fn delta_project(
        &self,
        norm: Element,
        weight: Element,
        activation: Element,
        target: u64,
    ) -> Step<MeasuredPoint> {
        let width = RECURRENT_WIDTH;
        let rows = |key_heads: u64| {
            let value_heads = 2 * key_heads;
            ((2 * key_heads + value_heads) * width, value_heads * width, value_heads)
        };
        let streamed = |key_heads| {
            let (channels, inner, value_heads) = rows(key_heads);
            sum(&[
                bytes(weight, &[channels, HIDDEN])?,
                bytes(weight, &[inner, HIDDEN])?,
                2 * bytes(weight, &[value_heads, HIDDEN])?,
            ])
        };
        let key_heads = size_for(target, 1, streamed)?;
        let (channels, inner, value_heads) = rows(key_heads);
        let device = self.device();
        let kernels = self.form::<gated_delta_project::Entry>(
            &[norm, weight, activation],
            &[
                ("M", 1),
                ("H", HIDDEN),
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
        self.begin();
        let point_bytes = streamed(key_heads)?;
        let launches = self.copies(point_bytes);
        let projections = self.views(weight, HIDDEN, channels, launches)?;
        let gates = self.views(weight, HIDDEN, inner, launches)?;
        let alpha = self.zeros(weight, &[value_heads, HIDDEN])?;
        let beta = self.zeros(weight, &[value_heads, HIDDEN])?;
        let input = self.zeros(Element::f32(), &[1, HIDDEN])?;
        let input_norm = self.zeros(norm, &[HIDDEN])?;
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
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: point_bytes,
            samples,
        })
    }

    /// One row's recurrent state advance at the key's head geometry, over
    /// the state layout's bank components (a one-row tape, as plain decoding
    /// plans it).
    fn delta_step(&self, key: &MeasurementKey, activation: Element) -> Step<MeasuredPoint> {
        let key_heads = geometry(key, "key_heads")?;
        let value_heads = geometry(key, "value_heads")?;
        let width = geometry(key, "width")?;
        let convolution_width = geometry(key, "convolution_width")?;
        let layout = ModelStateLayout::derive(
            &DecoderGeometry {
                activation_dtype: activation_dtype(activation)?,
                hidden: 1,
                vocabulary: 1,
                context_limit: 1,
                epsilon: 1e-6,
                blocks: vec![BlockGeometry {
                    mixer: MixerGeometry::Recurrent(RecurrentGeometry {
                        convolution_width,
                        key_heads,
                        value_heads,
                        width,
                        head_mapping: RecurrentHeadMapping::Tiled,
                    }),
                    feedforward: FeedForwardGeometry::Dense { intermediate: 1 },
                }],
            },
            0,
            KvCodec::Dense,
            0,
        )
        .map_err(failed)?;
        let [window, delta, tape] = layout.target_recurrent.as_slice() else {
            return Err(failed("a recurrent layer has three state components"));
        };
        let tape_rows = *tape
            .shape
            .first()
            .ok_or_else(|| failed("recurrent tape has no row axis"))? as u64;
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
        self.begin();
        let launches = self.copies(STEP_BANKS * bank_bytes);
        let arena = |component: &magnitude_model_state::ComponentSpec| -> Step<Vec<Tensor>> {
            let extents = std::iter::once(STEP_BANKS)
                .chain(component.shape.iter().map(|extent| *extent as u64))
                .collect::<Vec<_>>();
            self.shaped(Element::dense(component.dtype), &extents, launches)
        };
        let windows = arena(window)?;
        let deltas = arena(delta)?;
        let tapes = arena(tape)?;
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
            for ((window, delta), tape) in windows.iter().zip(&deltas).zip(&tapes) {
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
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: bank_bytes,
            samples,
        })
    }

    /// Gated recurrent output of 32 value heads of width 128, the output
    /// width sized.
    fn delta_output(
        &self,
        norm: Element,
        weight: Element,
        activation: Element,
        target: u64,
    ) -> Step<MeasuredPoint> {
        let inner = RECURRENT_VALUE_HEADS * RECURRENT_WIDTH;
        let projection_width = (2 * RECURRENT_KEY_HEADS + RECURRENT_VALUE_HEADS)
            * RECURRENT_WIDTH
            + inner
            + 2 * RECURRENT_VALUE_HEADS;
        let streamed = |hidden| bytes(weight, &[hidden, inner]);
        let hidden = size_for(target, UNIT, streamed)?;
        let device = self.device();
        let kernels = self.form::<gated_delta_output::Entry>(
            &[norm, weight, activation],
            &[
                ("M", 1),
                ("H", hidden),
                ("NK", RECURRENT_KEY_HEADS),
                ("NV", RECURRENT_VALUE_HEADS),
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
        self.begin();
        let point_bytes = streamed(hidden)?;
        let launches = self.copies(point_bytes);
        let weights = self.views(weight, inner, hidden, launches)?;
        let input = self.zeros(Element::f32(), &[1, hidden])?;
        let mixed = self.zeros(activation, &[1, RECURRENT_VALUE_HEADS, RECURRENT_WIDTH])?;
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
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: point_bytes,
            samples,
        })
    }

    /// Paired gate/up projection from a 4096-wide residual, the feature
    /// width sized; both matrices are streamed.
    fn dense_expand(
        &self,
        norm: Element,
        weight: Element,
        activation: Element,
        target: u64,
    ) -> Step<MeasuredPoint> {
        let streamed = |features| Ok(bytes(weight, &[features, HIDDEN])? * 2);
        let features = size_for(target, UNIT, streamed)?;
        let device = self.device();
        let kernels = self.form::<dense_expand::Entry>(
            &[norm, weight, activation],
            &[("M", 1), ("O", 1), ("H", HIDDEN), ("F", features)],
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
        self.begin();
        let point_bytes = streamed(features)?;
        let launches = self.copies(point_bytes);
        let gates = self.views(weight, HIDDEN, features, launches)?;
        let ups = self.views(weight, HIDDEN, features, launches)?;
        let residual = self.zeros(Element::f32(), &[1, HIDDEN])?;
        let norm = self.zeros(norm, &[HIDDEN])?;
        let out_rows = self.zeros(Element::i32(), &[1])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let residual = timed.bound(&residual)?;
            let norm = timed.bound(&norm)?;
            let out_rows = timed.bound(&out_rows)?;
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
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: point_bytes,
            samples,
        })
    }

    /// Down projection from a 4096-wide product, the residual width sized.
    fn dense_output(
        &self,
        weight: Element,
        activation: Element,
        target: u64,
    ) -> Step<MeasuredPoint> {
        let features = HIDDEN;
        let streamed = |hidden| bytes(weight, &[hidden, features]);
        let hidden = size_for(target, UNIT, streamed)?;
        let device = self.device();
        let kernels = self.form::<dense_output::Entry>(
            &[weight, activation],
            &[("M", 1), ("O", 1), ("H", hidden), ("F", features)],
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
        self.begin();
        let point_bytes = streamed(hidden)?;
        let launches = self.copies(point_bytes);
        let weights = self.views(weight, features, hidden, launches)?;
        let residual = self.zeros(Element::f32(), &[1, hidden])?;
        let product = self.zeros(activation, &[1, features])?;
        let out_rows = self.zeros(Element::i32(), &[1])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let residual = timed.bound(&residual)?;
            let product = timed.bound(&product)?;
            let out_rows = timed.bound(&out_rows)?;
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
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: point_bytes,
            samples,
        })
    }

    /// Router logits and top-k selection at the key's routing geometry.
    fn routed_route(
        &self,
        key: &MeasurementKey,
        norm: Element,
        router: Element,
        activation: Element,
    ) -> Step<MeasuredPoint> {
        let hidden = geometry(key, "hidden")?;
        let experts = geometry(key, "experts")?;
        let selected = geometry(key, "selected")?;
        let dimensions = [("M", 1), ("H", hidden), ("E", experts), ("K", selected)];
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
        self.begin();
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
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: router_bytes,
            samples,
        })
    }

    /// Decode expansion of 8 selected experts and a shared expert, 512
    /// features each (the catalog's routed widths), the hidden width sized.
    /// Only the selected experts are allocated: routes name each once, as a
    /// decode row's choices do.
    fn routed_expand(
        &self,
        weight: Element,
        activation: Element,
        target: u64,
    ) -> Step<MeasuredPoint> {
        let (experts, features) = (ROUTED_SELECTED, ROUTED_FEATURES);
        let streamed = |hidden| {
            Ok(sum(&[
                bytes(weight, &[experts, features, hidden])?,
                bytes(weight, &[features, hidden])?,
            ])? * 2)
        };
        let hidden = size_for(target, UNIT, streamed)?;
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
        self.begin();
        let point_bytes = streamed(hidden)?;
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
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: point_bytes,
            samples,
        })
    }

    /// Decode down projection of 8 selected experts and a shared expert,
    /// 512 features each, the hidden width sized.
    fn routed_output(
        &self,
        weight: Element,
        activation: Element,
        target: u64,
    ) -> Step<MeasuredPoint> {
        let (experts, features) = (ROUTED_SELECTED, ROUTED_FEATURES);
        let streamed = |hidden| {
            sum(&[
                bytes(weight, &[experts, hidden, features])?,
                bytes(weight, &[hidden, features])?,
            ])
        };
        let hidden = size_for(target, UNIT, streamed)?;
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
        self.begin();
        let point_bytes = streamed(hidden)?;
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
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: point_bytes,
            samples,
        })
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
        self.begin();
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
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: bytes(norm, &[HIDDEN])?,
            samples,
        })
    }

    /// Vocabulary projection of one 4096-wide row, the vocabulary sized.
    fn readout_head(
        &self,
        norm: Element,
        weight: Element,
        activation: Element,
        target: u64,
    ) -> Step<MeasuredPoint> {
        let streamed = |vocabulary| bytes(weight, &[vocabulary, HIDDEN]);
        let vocabulary = size_for(target, UNIT, streamed)?;
        let device = self.device();
        let kernels = self.form::<readout_head_rows::Entry>(
            &[norm, weight, activation],
            &[("M", 1), ("O", 1), ("V", vocabulary), ("D", HIDDEN)],
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
        self.begin();
        let point_bytes = streamed(vocabulary)?;
        let launches = self.copies(point_bytes);
        let weights = self.views(weight, HIDDEN, vocabulary, launches)?;
        let hidden = self.zeros(Element::f32(), &[1, HIDDEN])?;
        let norm = self.zeros(norm, &[HIDDEN])?;
        let out_rows = self.zeros(Element::i32(), &[1])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let hidden = timed.bound(&hidden)?;
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
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: point_bytes,
            samples,
        })
    }

    /// Unconstrained selection over one F32 logits row of `vocabulary`.
    fn sample_rows(&self, vocabulary: u64) -> Step<MeasuredPoint> {
        let dimensions = [("M", 1), ("V", vocabulary)];
        let device = self.device();
        let kernels = self.form::<sample_rows::Entry>(&[], &dimensions, move |specialization| {
            sample_rows::native_for_device(device, specialization)
        })?;
        self.begin();
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
            self.run(timed, launches)
        })?;
        Ok(MeasuredPoint {
            bytes: point_bytes,
            samples,
        })
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
        let (key_heads, value_heads, width) =
            (RECURRENT_KEY_HEADS, RECURRENT_VALUE_HEADS, RECURRENT_WIDTH);
        let channels = (2 * key_heads + value_heads) * width;
        let inner = value_heads * width;
        let recurrent = [
            ("M", 1),
            ("H", hidden),
            ("NK", key_heads),
            ("NV", value_heads),
            ("W", width),
        ];
        let dense = [("M", 1), ("O", 1), ("H", hidden), ("F", features)];
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
            &dense,
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
            &dense,
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
        self.begin();
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
        let class_seconds = |key: MeasurementKey, streamed: u64| -> Step<f64> {
            let points = self.points(&key)?;
            let cost = ClassCost::from_points(key.class, &points).map_err(failed)?;
            Ok(cost.seconds(1, streamed, streamed).median)
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
            MeasurementKey::delta_project(activation, weight, activation),
            project_bytes,
        )? + class_seconds(
            MeasurementKey::delta_output(activation, weight, activation),
            bytes(weight, &[hidden, inner])?,
        )? + class_seconds(
            MeasurementKey::dense_expand(activation, weight, activation),
            expand_bytes,
        )? + class_seconds(
            MeasurementKey::dense_output(weight, activation),
            bytes(weight, &[hidden, features])?,
        )?) / CHAIN_CALLS_PER_CYCLE as f64;
        let mut fastest: Option<(usize, f64, Vec<f64>)> = None;
        for choice in 0..variants {
            let samples = self.run(chain(choice)?, calls)?;
            let call = median(&samples).ok_or_else(|| failed("no chained median"))?;
            if fastest.as_ref().is_none_or(|(_, best, _)| call < *best) {
                fastest = Some((choice, call, samples));
            }
        }
        let (choice, chained_call, chained) =
            fastest.ok_or_else(|| failed("no chained variant was timed"))?;
        let submission = (0..CHAIN_SAMPLES)
            .map(|_| self.submission(chain(choice)?, chained_call * calls as f64))
            .collect::<Step<Vec<_>>>()?;
        Ok(Chained {
            dependency: MeasuredPoint {
                bytes: 0,
                // A dependent call cannot cost less than its standalone
                // launch: a difference at or below zero measures none.
                samples: chained
                    .iter()
                    .map(|call| (call - standalone_per_call).max(0.0))
                    .collect(),
            },
            submission: MeasuredPoint {
                bytes: 0,
                samples: submission,
            },
        })
    }

    /// Host seconds of submitting one run of `timed` and waiting for it,
    /// beyond `device_seconds` of its device work. The difference cannot be
    /// negative; a device whose submission is free measures zero.
    fn submission(&self, timed: Timed, device_seconds: f64) -> Step<f64> {
        let plan = timed.graph.seal().map_err(failed)?;
        let mut slot = plan.new_slot().map_err(failed)?;
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
            .map_err(failed)?;
        completion.wait().map_err(failed)?;
        let wall = began.elapsed().as_secs_f64();
        self.session.trace.collect().map_err(failed)?;
        Ok((wall - device_seconds).max(0.0))
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
    use seismic::Layout;

    #[test]
    fn size_search_finds_the_smallest_unit_multiple_reaching_the_target() {
        let weight = Element::stored("q4k", Layout::Rows16).unwrap();
        for target in [SMALL_BYTES, LARGE_BYTES] {
            let streamed = |rows| bytes(weight, &[rows, HIDDEN]);
            let Ok(rows) = size_for(target, UNIT, streamed) else {
                panic!("size search failed");
            };
            assert_eq!(rows % UNIT, 0);
            assert!(streamed(rows).ok().unwrap() >= target);
            assert!(rows == UNIT || streamed(rows - UNIT).ok().unwrap() < target);
        }
    }

    #[test]
    fn measured_history_planes_are_the_state_layout_rows() {
        let configuration = super::super::plan::QWEN35_CONFIGURATIONS[3];
        let (definition, _) = super::super::plan::tests::declared_model(&configuration);
        for (codec, affine) in [(KvCodec::Dense, false), (KvCodec::AffineK8V4, true)] {
            let layout = ModelStateLayout::derive(&definition.geometry, 0, codec, 0).unwrap();
            let layout_row = layout.target_history[0]
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
