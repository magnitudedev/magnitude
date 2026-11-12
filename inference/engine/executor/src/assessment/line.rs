//! The machine's line: what the hardware itself reaches on an executor,
//! which floors are judged against (a kernel's floor is the larger of its
//! bytes over the stream rate and its multiply-adds over its operand's
//! peak): the rate device memory streams at, and the arithmetic peak of
//! each operand type the backend's forms multiply in. The third figure of
//! the line, an empty dependent launch, is the executor's costs' launch
//! ([`super::ExecutorCosts::launch`]), which the launch graph measures.
//!
//! The line is a development measurement and runs only on request, and is
//! built only with the `machine-line` feature (so no served engine carries
//! it or its probe kernels): nothing in loading, planning or serving reads
//! it, so no standing request of the runtime's assessment asks for it, and
//! a tool that needs it has the measurement job take its requests alone
//! ([`super::requests::line_outstanding`]).
//!
//! Each figure is a dedicated kernel's (`line/line.seismic`) doing only what it
//! measures, never one of our kernels, timed by the measurement job at
//! every value of its one declared parameter, UNROLL (the independent loads
//! or multiply chains a lane keeps in flight): the figure is the fastest,
//! and it has reached the line where one more step did not raise it
//! ([`Rate`]). A peak is timed at S, 2S and 4S multiply-adds a lane and is
//! the marginal rate between S and 4S, so a launch's and a call's fixed
//! cost cancel; it stands only where the work it credits is shown to be
//! done, the second increment (2S to 4S) twice the first (S to 2S) within
//! the timings' intervals, which rule out equal increments (a ratio of 1),
//! the line being the fastest UNROLL whose work does ([`Peak::Unscaled`]
//! where none does): a rate that is flat in UNROLL says nothing of that,
//! since work a compiler drops is flat too. No rate is reported, fitted or
//! named by device: the records are the measurement job's, of the machine
//! and build, stored with the executor's other records and read back from
//! them ([`MachineLine::stored`]). Nothing prices a figure of the line, so
//! a point the job's time ends is left to the next job, never priced; a
//! CPU's stream is its own sweeps of host memory, recorded under keys of the
//! line's, apart from the host memory's line the runtime prices a host tier
//! by.

use super::executor_costs::{
    executor_identity, failed, largest_cache, named, rotation_sets, ExecutorCostsError, Formation, MeasuredCost, Walks,
};
use super::streaming_costs::{MemoryCosts, Sweeps};
use magnitude_kernels::{line_f32, line_int8, line_matrix, line_stream};
use seismic::measure::{Reason, Record, RecordStore, StoreError, Timing, Unformed};
use seismic::{BackendName, Device, DeviceTopology, Element, Entry, Executor, NativeKernel, Point, PointKey, Tensor};
use std::cell::OnceCell;
use std::fmt;

/// The parameter every probe is timed at each value of.
const UNROLL: &str = "UNROLL";

/// The f32 values of one row of the stream: one 32-lane group's 16-byte
/// loads at the largest UNROLL, so every value divides it.
const ROW_VALUES: u64 = 32 * 4 * 16;

/// The bytes one call of the stream reads: 0.3 ms at an RTX 3090's
/// 0.9 TB/s, 1 ms at an M4 Pro's or a GB10's quarter of that, so a launch's
/// microseconds are under 1% of a call, and a point's share of the job
/// holds tens of samples (each at least tuning's least device time).
const STREAM_CALL_BYTES: u64 = 256 << 20;

/// What a peak probe's lanes are seeded with, and write back: an f32 whose
/// half and bf16 are exact, twice which truncates to the int8 operand 1, so
/// every product stays far from overflow over a probe's steps.
const SEED: f32 = 0.5;

/// An operand type the forms multiply in, whose peak the line holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    /// Fused f32 multiply-adds.
    F32,
    /// 16-bit matrix multiply-adds into f32: Metal's half simdgroup
    /// matrices, CUDA's bf16 MMA, Vulkan's f16 cooperative matrices.
    Matrix,
    /// int8 multiply-adds into i32: CUDA's s8 MMA, the CPU tier's
    /// quantized dots.
    Int8,
}

impl Operand {
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Matrix => "matrix",
            Self::Int8 => "int8",
        }
    }
}

/// One probe's figure: the fastest of its timings, and the UNROLL it ran at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    /// Bytes (the stream) or operations (a peak: a multiply-add is two) a
    /// second.
    pub per_second: MeasuredCost,
    pub unroll: u64,
}

/// A rate and whether one more step did not raise it: a larger step than the
/// fastest was timed and ran no faster, or the fastest is the largest and
/// the step to it from the one before rose less than their timings resolve
/// (their intervals overlap). An `Open` rate is a floor: the hardware's line
/// may lie above it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rate<T> {
    Reached(T),
    Open(T),
}

impl<T> Rate<T> {
    pub fn figure(&self) -> &T {
        match self {
            Self::Reached(figure) | Self::Open(figure) => figure,
        }
    }

    pub fn is_reached(&self) -> bool {
        matches!(self, Self::Reached(_))
    }

    fn map<U>(self, f: impl FnOnce(T) -> U) -> Rate<U> {
        match self {
            Self::Reached(figure) => Rate::Reached(f(figure)),
            Self::Open(figure) => Rate::Open(f(figure)),
        }
    }
}

/// An operand's peak, or why the executor's device has none.
#[derive(Clone, Debug, PartialEq)]
pub enum Peak {
    /// The fastest UNROLL whose work scales.
    Rate(Rate<Line>),
    /// No UNROLL's work is shown to scale with S: of the fastest, its
    /// increments' ratio (2S to 4S over S to 2S, 2 where the work is done)
    /// is not shown to be 2, `ratio` its interval (none where the timings
    /// resolve no increment), and `seconds` its timings at S, 2S and 4S.
    Unscaled { unroll: u64, ratio: Option<(f64, f64)>, seconds: [MeasuredCost; 3] },
    /// The probe measured nothing of the device: no configuration's time
    /// grew with its work.
    Defect(Reason),
    /// The device lacks the operand's instructions (a configuration its
    /// declaration or the device refuses).
    Unobtainable(Reason),
}

/// The host memory's stream as the CPU line holds it: the figure at the
/// readers of the CPU peaks' pool, or at more (every hardware thread where
/// the pool has fewer, else twice the pool) where that is faster.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sweep {
    pub per_second: MeasuredCost,
    pub readers: u64,
}

/// The rate an executor's memory is read at: a device's by a probe kernel,
/// the host's by sweeps of host memory read by threads, which is the CPU
/// executor's stream (records of the line's own, apart from the host
/// memory's line the runtime prices a host tier by); or why it could not be
/// had, which leaves the peaks standing.
#[derive(Clone, Debug, PartialEq)]
pub enum Stream {
    Kernel(Rate<Line>),
    Host(Rate<Sweep>),
    /// The sweeps rose with no bytes: a defect of the probe.
    Defect(Reason),
    Unobtainable(Reason),
}

/// The executor's line.
#[derive(Clone, Debug, PartialEq)]
pub struct MachineLine {
    /// The rate its memory is read at.
    pub stream: Stream,
    /// The peak of each operand type the backend's forms multiply in.
    pub peaks: Vec<(Operand, Peak)>,
}

/// Why the line could not be had.
#[derive(Clone, Debug, PartialEq)]
pub enum MachineLineError {
    Executor(ExecutorCostsError),
    Store(StoreError),
    /// A point no record holds: the job has not measured it yet.
    Unrecorded { probe: &'static str, unroll: u64 },
    /// A sweep of the CPU's stream no record holds: the job has not measured
    /// it yet.
    UnrecordedSweep { readers: u64, bytes: u64 },
}

impl fmt::Display for MachineLineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Executor(error) => write!(formatter, "{error}"),
            Self::Store(error) => write!(formatter, "a line's record: {error}"),
            Self::Unrecorded { probe, unroll } => write!(formatter, "no record of `{probe}` at UNROLL {unroll}"),
            Self::UnrecordedSweep { readers, bytes } => {
                write!(formatter, "no record of the host memory sweep of {bytes} bytes by {readers} readers")
            }
        }
    }
}

impl std::error::Error for MachineLineError {}

impl From<ExecutorCostsError> for MachineLineError {
    fn from(error: ExecutorCostsError) -> Self {
        Self::Executor(error)
    }
}

impl From<StoreError> for MachineLineError {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}

/// A probe kernel: a call's arguments, one of its operands' inputs and an
/// output.
trait Probe: Entry + Send + Sync {
    fn args<'a>(operands: &'a Operands, input: &'a Tensor, out: &'a mut Tensor) -> Self::Args<'a>;

    /// Why `device` lacks the probe's operand, where a property of the
    /// device says so, which the probe's declaration cannot: a kernel's
    /// identity, which keys its records, names no device.
    fn lacking(_device: &Device) -> Option<String> {
        None
    }
}

impl Probe for line_stream::Entry {
    fn args<'a>(_operands: &'a Operands, input: &'a Tensor, out: &'a mut Tensor) -> Self::Args<'a> {
        line_stream::Args { input, out }
    }
}

impl Probe for line_f32::Entry {
    fn args<'a>(operands: &'a Operands, x: &'a Tensor, out: &'a mut Tensor) -> Self::Args<'a> {
        line_f32::Args { x, steps: operands.steps(), out }
    }
}

impl Probe for line_matrix::Entry {
    fn args<'a>(operands: &'a Operands, x: &'a Tensor, out: &'a mut Tensor) -> Self::Args<'a> {
        line_matrix::Args { x, steps: operands.steps(), out }
    }

    fn lacking(device: &Device) -> Option<String> {
        match device.properties() {
            // The forms run tensor operations here, which the probe does not time.
            seismic::device::DeviceProperties::Metal(properties) if properties.tensor_ops.holds() => {
                Some("the forms run tensor operations on this device, which the probe does not time".to_owned())
            }
            _ => (device.forms_matrix_path() == Some(false)).then(|| {
                "the device does not form the cooperative-matrix path (unsupported, or off for its driver: \
                 `facts::matrix_path`)"
                    .to_owned()
            }),
        }
    }
}

impl Probe for line_int8::Entry {
    fn args<'a>(operands: &'a Operands, x: &'a Tensor, out: &'a mut Tensor) -> Self::Args<'a> {
        line_int8::Args { x, steps: operands.steps(), out }
    }

    /// The CPU's quantized dots are the tier's instructions; a tier with none
    /// would time a scalar loop of ours, not the hardware.
    fn lacking(device: &Device) -> Option<String> {
        match device.properties() {
            seismic::device::DeviceProperties::Cpu(_)
                if seismic::cpu::Tier::detected().is_none_or(|tier| tier.quantized_dots() == 0) =>
            {
                Some("the CPU's tier has no quantized dot instructions".to_owned())
            }
            _ => None,
        }
    }
}

/// The shape of the probes on an executor.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// The stream's rows of [`ROW_VALUES`] a call ([`STREAM_CALL_BYTES`],
    /// within the largest allocation), and the inputs its calls rotate
    /// through: as many as span beyond the largest cache that can serve
    /// device storage ([`rotation_sets`]), so each call reads memory.
    rows: u64,
    sets: u64,
    /// A peak's lanes, and the multiply-adds each makes of each operand
    /// ([`Operand`]'s order).
    lanes: u64,
    steps: [u64; 3],
}

impl Shape {
    /// A peak is timed at 1, 2 and 4 times a lane's steps, and the job gives
    /// a point a fixed share of its time: a point whose call is 8 ms or
    /// more holds six samples or fewer, which bound no interval, so no
    /// increment is resolved. On a GPU, 2^20 lanes are several waves of the
    /// most any device holds resident (an RTX 3090's 82 multiprocessors hold
    /// 126K threads), and a lane's 2^12 f32 or matrix multiply-adds, or 2^14
    /// int8, make a call of 0.1 to 1.1 ms at S on every measured unit (an M4
    /// Pro's 8 TFLOPS the longest), so 4S holds eight samples and a launch's
    /// microseconds are cancelled by the marginal rate. On a CPU a work item
    /// is a lane and its multiply-adds are its tier's vectors: 2^10 items
    /// are tens a core on every core, and an item's 2^18 f32 or 2^21 int8
    /// multiply-adds (tens of microseconds, far past claiming it) make a
    /// call of 0.6 to 1 ms at S on an M4 Pro's 0.9 TFLOPS and 4 TOPS. A
    /// call's repetitions in a sample follow from `MIN_SAMPLE_SECONDS`, so
    /// these sizes hold for the sample time the job takes.
    fn of(topology: &DeviceTopology, executor: &Executor) -> Self {
        let row = ROW_VALUES * 4;
        let rows = STREAM_CALL_BYTES.min(executor.limits().max_allocation.bytes()) / row;
        let (lanes, steps) = match executor.backend() {
            BackendName::Cpu => (1 << 10, [1 << 18, 1 << 18, 1 << 21]),
            BackendName::Metal | BackendName::Cuda | BackendName::Vulkan => (1 << 20, [1 << 12, 1 << 12, 1 << 14]),
        };
        Self { rows, sets: rotation_sets(largest_cache(topology, executor), rows * row), lanes, steps }
    }

    /// What a stream's call reads, its key's point.
    fn stream(self) -> (f64, Vec<(&'static str, u64)>) {
        ((self.rows * ROW_VALUES * 4) as f64, vec![("rows", self.rows), ("columns", ROW_VALUES), ("sets", self.sets)])
    }

    /// The multiply-adds a lane of `operand`'s peak makes.
    fn steps(self, operand: Operand) -> u64 {
        self.steps[operand as usize]
    }

    /// `operand`'s peak's operations a call at `multiple` times its steps,
    /// a multiply-add two, and its key's point.
    fn peak(self, operand: Operand, multiple: u64) -> (f64, Vec<(&'static str, u64)>) {
        let steps = self.steps(operand) * multiple;
        (2. * (self.lanes * steps) as f64, vec![("lanes", self.lanes), ("steps", steps)])
    }
}

/// One probe on an executor: its formation, its declared UNROLL values in
/// ascending order, and what a call does (its units and its key's point).
struct Plan<E: Entry> {
    formation: Formation<E>,
    unrolls: Vec<u64>,
    units: f64,
    point: Vec<(&'static str, u64)>,
}

impl<E: Probe> Plan<E> {
    /// The probe of `E` formed at `statics` on `backend`, none where the
    /// backend has no implementation of it (its forms multiply no such
    /// operand).
    fn of(
        backend: BackendName,
        entry: seismic::BoundEntry<E>,
        statics: &[(&'static str, u64)],
        (units, point): (f64, Vec<(&'static str, u64)>),
    ) -> Result<Option<Self>, ExecutorCostsError> {
        let Some(implementation) = seismic::generated::native_implementation_for_backend::<E>(backend)
            .map_err(|error| failed(E::NAME)(&error))?
        else {
            return Ok(None);
        };
        let parameter = implementation
            .params
            .iter()
            .find(|parameter| parameter.name == UNROLL)
            .ok_or_else(|| ExecutorCostsError(format!("`{}` declares no {UNROLL} on {}", E::NAME, backend.as_str())))?;
        let mut unrolls = parameter.values.clone();
        unrolls.sort_unstable();
        Ok(Some(Self { formation: Formation::new(entry, statics), unrolls, units, point }))
    }

    /// The key of its point at `unroll`.
    fn key(
        &self,
        identity: &seismic::Identity,
        backend: BackendName,
        unroll: u64,
    ) -> Result<PointKey, ExecutorCostsError> {
        Ok(PointKey {
            executor: identity.clone(),
            kernels: vec![self.formation.identity(backend)?],
            configuration: named(&[("unroll", unroll)]),
            point: named(&self.point),
        })
    }

    /// The record at `unroll`: its timing, or why it is unobtainable.
    fn load(
        &self,
        identity: &seismic::Identity,
        backend: BackendName,
        store: &dyn RecordStore<PointKey>,
        unroll: u64,
    ) -> Result<Result<Timing, Reason>, MachineLineError> {
        match store.load(&self.key(identity, backend, unroll)?)? {
            Some(Record::Measured(timing)) => Ok(Ok(timing)),
            Some(Record::Unobtainable { reason }) => Ok(Err(reason)),
            None => Err(MachineLineError::Unrecorded { probe: E::NAME, unroll }),
        }
    }

    /// Its figure from `store`'s records: the fastest configuration
    /// measured, or why none was (each recorded unobtainable).
    fn stored(
        &self,
        identity: &seismic::Identity,
        backend: BackendName,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Result<Rate<Line>, Reason>, MachineLineError> {
        let mut rates = Vec::new();
        let mut unobtainable = None;
        for &unroll in &self.unrolls {
            match self.load(identity, backend, store, unroll)? {
                Ok(timing) => rates.push((unroll, MeasuredCost::rate(self.units, &timing))),
                Err(reason) => {
                    unobtainable.get_or_insert(reason);
                }
            }
        }
        match reached(&rates) {
            Some(line) => Ok(Ok(line)),
            None => Ok(Err(unobtainable.expect("a probe declares an UNROLL value"))),
        }
    }

    /// Time each configuration in the measurement job (`walks`), each
    /// formed, all at once on the host's threads, when the first is made,
    /// its calls rotating through `operands`' inputs (made once), each into
    /// an output of its own; none on a device that lacks the operand
    /// ([`Probe::lacking`]).
    fn measure(
        &self,
        walks: &mut Walks<'_, '_>,
        operands: &OnceCell<Result<Operands, ExecutorCostsError>>,
        make: impl Fn(&Device) -> Result<Operands, ExecutorCostsError>,
    ) -> Result<(), ExecutorCostsError> {
        let device = walks.device();
        let identity = walks.identity();
        let kernels = OnceCell::<Vec<Result<NativeKernel<E>, ExecutorCostsError>>>::new();
        for (index, &unroll) in self.unrolls.iter().enumerate() {
            let (kernels, make) = (&kernels, &make);
            walks.time(self.key(&identity, device.backend(), unroll)?, false, move || {
                if let Some(lacking) = E::lacking(device) {
                    return Err(Unformed::Never(ExecutorCostsError(lacking)));
                }
                let kernel = kernels.get_or_init(|| {
                    seismic::Workers::of_the_host().each(&self.unrolls, |&unroll| {
                        self.formation.form_at(device, (UNROLL, unroll))
                    })
                })[index]
                    .as_ref()
                    .map_err(|error| Unformed::Never(error.clone()))?;
                let operands = operands
                    .get_or_init(|| make(device))
                    .as_ref()
                    .map_err(|error| Unformed::Now(error.clone()))?;
                let mut outs = operands
                    .inputs
                    .iter()
                    .map(|_| Tensor::zeros(device, Element::f32(), &[operands.outputs]))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| Unformed::Now(failed("a probe's output")(&error)))?;
                kernel
                    .point(
                        operands
                            .inputs
                            .iter()
                            .zip(&mut outs)
                            .map(|(input, out)| E::args(operands, input, out))
                            .collect(),
                    )
                    .map(Point::Kernel)
                    .map_err(|error| Unformed::Now(failed(E::NAME)(&error)))
            });
        }
        Ok(())
    }
}

/// The inputs a probe's calls rotate through (a peak's one), a peak's steps
/// (`[0, S]`, no values) and the length of a call's output.
struct Operands {
    inputs: Vec<Tensor>,
    steps: Option<Tensor>,
    outputs: u64,
}

impl Operands {
    /// A peak's steps.
    fn steps(&self) -> &Tensor {
        self.steps.as_ref().expect("a peak's operands hold its steps")
    }
}

/// The multiples of an operand's steps its peak is timed at: its marginal
/// rate is between the first and the last, the middle one showing the work
/// scales.
const MULTIPLES: [u64; 3] = [1, 2, 4];

/// One operand's peak probe, formed at each of [`MULTIPLES`] of its steps.
struct Ladder<E: Entry> {
    plans: [Plan<E>; 3],
}

impl<E: Probe> Ladder<E> {
    fn of(
        backend: BackendName,
        entry: &seismic::BoundEntry<E>,
        shape: Shape,
        operand: Operand,
    ) -> Result<Option<Self>, ExecutorCostsError> {
        let mut plans = Vec::with_capacity(MULTIPLES.len());
        for multiple in MULTIPLES {
            let statics = [("S", shape.steps(operand) * multiple)];
            // A backend with no implementation has none at any multiple.
            let Some(plan) = Plan::of(backend, entry.clone(), &statics, shape.peak(operand, multiple))? else {
                return Ok(None);
            };
            plans.push(plan);
        }
        let plans = plans.try_into().unwrap_or_else(|plans: Vec<_>| {
            unreachable!(
                "a plan of each of the {} multiples, not {}",
                MULTIPLES.len(),
                plans.len()
            )
        });
        Ok(Some(Self { plans }))
    }

    fn points(&self) -> usize {
        self.plans.iter().map(|plan| plan.unrolls.len()).sum()
    }

    /// Its peak from `store`'s records: the fastest UNROLL's marginal rate
    /// where its work scales, otherwise the figure unscaled; unobtainable
    /// where every configuration was.
    fn stored(
        &self,
        identity: &seismic::Identity,
        backend: BackendName,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Peak, MachineLineError> {
        let mut points = Vec::new();
        let mut unobtainable = None;
        for &unroll in &self.plans[0].unrolls {
            let mut timings = Vec::with_capacity(MULTIPLES.len());
            for plan in &self.plans {
                match plan.load(identity, backend, store, unroll)? {
                    Ok(timing) => timings.push(timing),
                    Err(reason) => {
                        unobtainable.get_or_insert(reason);
                    }
                }
            }
            if let Ok(timings) = <[Timing; 3]>::try_from(timings) {
                points.push((unroll, Marginal::of(self.plans[0].units, &timings)));
            }
        }
        if points.is_empty() {
            return Ok(Peak::Unobtainable(unobtainable.expect("a probe declares an UNROLL value")));
        }
        // The line is the fastest UNROLL whose work scales: an unscaled
        // figure, noise or dropped work, may be the highest and is no line.
        let scaled = points
            .iter()
            .filter(|(_, marginal)| marginal.scaled)
            .filter_map(|(unroll, marginal)| marginal.rate.map(|rate| (*unroll, rate)))
            .collect::<Vec<_>>();
        if let Some(rate) = reached(&scaled) {
            return Ok(Peak::Rate(rate));
        }
        let Some((unroll, marginal)) = points
            .iter()
            .filter_map(|(unroll, marginal)| marginal.rate.map(|rate| (unroll, marginal, rate)))
            .max_by(|(_, _, a), (_, _, b)| a.value.total_cmp(&b.value))
            .map(|(unroll, marginal, _)| (unroll, marginal))
        else {
            return Ok(Peak::Defect(Reason::new("no configuration's time grew with its work".to_owned())));
        };
        Ok(Peak::Unscaled { unroll: *unroll, ratio: marginal.ratio, seconds: marginal.seconds })
    }

    fn measure(&self, walks: &mut Walks<'_, '_>, shape: Shape, operand: Operand) -> Result<(), ExecutorCostsError> {
        for (plan, multiple) in self.plans.iter().zip(MULTIPLES) {
            let steps = shape.steps(operand) * multiple;
            plan.measure(walks, &OnceCell::new(), |device: &Device| {
                let bytes = (0..shape.lanes).flat_map(|_| SEED.to_le_bytes()).collect::<Vec<_>>();
                let input = Tensor::from_host(device, Element::f32(), &[shape.lanes], &bytes)
                    .map_err(|error| failed("a peak's seeds")(&error))?;
                let steps = Tensor::zeros(device, Element::i32(), &[0, steps])
                    .map_err(|error| failed("a peak's steps")(&error))?;
                Ok(Operands {
                    inputs: vec![input],
                    steps: Some(steps),
                    outputs: shape.lanes,
                })
            })?;
        }
        Ok(())
    }
}

/// A peak at one UNROLL from its timings at [`MULTIPLES`]: the rate between
/// the first and last (the operations of three first-multiples over the time
/// they add), whether the work scaled, and the ratio of the increments.
struct Marginal {
    rate: Option<MeasuredCost>,
    scaled: bool,
    ratio: Option<(f64, f64)>,
    seconds: [MeasuredCost; 3],
}

impl Marginal {
    fn of(units: f64, [one, two, four]: &[Timing; 3]) -> Self {
        let added = four.median - one.median;
        let intervals = one.interval.zip(two.interval).zip(four.interval).map(|((one, two), four)| (one, two, four));
        let rate = (added > 0.).then(|| MeasuredCost {
            value: 3. * units / added,
            interval: intervals.and_then(|((one_fastest, one_slowest), _, (four_fastest, four_slowest))| {
                let (least, most) = (four_fastest - one_slowest, four_slowest - one_fastest);
                (least > 0.).then(|| (3. * units / most, 3. * units / least))
            }),
            samples: one.samples.min(four.samples),
        });
        // The first increment (S to 2S) and the second (2S to 4S), each
        // bounded by its two timings' intervals.
        let ratio = intervals.and_then(
            |((one_fastest, one_slowest), (two_fastest, two_slowest), (four_fastest, four_slowest))| {
                let (first_least, first_most) = (two_fastest - one_slowest, two_slowest - one_fastest);
                let (second_least, second_most) = (four_fastest - two_slowest, four_slowest - two_fastest);
                (first_least > 0.).then(|| (second_least / first_most, second_most / first_least))
            },
        );
        let scaled = ratio.is_some_and(|(least, most)| 1. < least && least <= 2. && 2. <= most);
        let seconds = [one, two, four].map(|timing| MeasuredCost {
            value: timing.median,
            interval: timing.interval,
            samples: timing.samples,
        });
        Self {
            rate,
            scaled,
            ratio,
            seconds,
        }
    }
}

/// The fastest of `rates` (by UNROLL, ascending), reached where a larger
/// UNROLL was timed and ran no faster, or the fastest is the largest and the
/// step to it from the one before rose less than their timings resolve;
/// none where there are no rates.
fn reached(rates: &[(u64, MeasuredCost)]) -> Option<Rate<Line>> {
    let fastest = rates
        .iter()
        .enumerate()
        .max_by(|(_, (_, a)), (_, (_, b))| a.value.total_cmp(&b.value))
        .map(|(index, _)| index)?;
    let (unroll, per_second) = rates[fastest];
    let reached = fastest + 1 < rates.len()
        || fastest
            .checked_sub(1)
            .is_some_and(|before| !raised(rates[before].1, per_second));
    let line = Line { per_second, unroll };
    Some(if reached { Rate::Reached(line) } else { Rate::Open(line) })
}

/// The probes on an executor.
struct Probes {
    shape: Shape,
    /// The stream's probe, on a device with memory of its own.
    stream: Option<Plan<line_stream::Entry>>,
    /// The readers of the CPU's stream, its host memory's sweeps
    /// ([`host_readers`]); none on a device.
    host: Option<Result<[u64; 2], ExecutorCostsError>>,
    peaks: (Option<Ladder<line_f32::Entry>>, Option<Ladder<line_matrix::Entry>>, Option<Ladder<line_int8::Entry>>),
}

impl Probes {
    fn of(topology: &DeviceTopology, executor: &Executor) -> Result<Self, ExecutorCostsError> {
        let backend = executor.backend();
        let shape = Shape::of(topology, executor);
        let stream = Plan::of(backend, line_stream::native_entry(), &[], shape.stream())?;
        assert_eq!(stream.is_none(), backend == BackendName::Cpu, "only the CPU's stream is the host's sweeps");
        Ok(Self {
            shape,
            stream,
            host: (backend == BackendName::Cpu).then(host_readers),
            peaks: (
                Ladder::of(backend, &line_f32::native_entry(), shape, Operand::F32)?,
                Ladder::of(backend, &line_matrix::native_entry(), shape, Operand::Matrix)?,
                Ladder::of(backend, &line_int8::native_entry(), shape, Operand::Int8)?,
            ),
        })
    }

    /// The points of every probe.
    fn points(&self) -> usize {
        let (f32, matrix, int8) = &self.peaks;
        self.stream.as_ref().map_or(0, |stream| stream.unrolls.len())
            + self.host.as_ref().map_or(0, |readers| if readers.is_ok() { HOST_POINTS } else { 0 })
            + f32.as_ref().map_or(0, Ladder::points)
            + matrix.as_ref().map_or(0, Ladder::points)
            + int8.as_ref().map_or(0, Ladder::points)
    }
}

impl MachineLine {
    /// The points the measurement job takes of it on `executor`.
    pub(super) fn points(topology: &DeviceTopology, executor: &Executor) -> Result<usize, ExecutorCostsError> {
        Ok(Probes::of(topology, executor)?.points())
    }

    /// The line `store`'s records give `executor`, measuring and forming
    /// nothing; why there is none.
    pub fn stored(
        topology: &DeviceTopology,
        executor: &Executor,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Self, MachineLineError> {
        let identity = executor_identity(topology, executor)?;
        let backend = executor.backend();
        let Probes { stream, host, peaks: (f32, matrix, int8), .. } = Probes::of(topology, executor)?;
        let mut peaks = Vec::new();
        if let Some(ladder) = f32 {
            peaks.push((Operand::F32, ladder.stored(&identity, backend, store)?));
        }
        if let Some(ladder) = matrix {
            peaks.push((Operand::Matrix, ladder.stored(&identity, backend, store)?));
        }
        if let Some(ladder) = int8 {
            peaks.push((Operand::Int8, ladder.stored(&identity, backend, store)?));
        }
        // The stream last; an unobtainable one leaves the peaks.
        let stream = match (stream, host) {
            (Some(stream), _) => match stream.stored(&identity, backend, store)? {
                Ok(rate) => Stream::Kernel(rate),
                Err(reason) => Stream::Unobtainable(reason),
            },
            (None, Some(readers)) => host_stream(readers, topology, &identity, store)?,
            (None, None) => unreachable!("a device without a stream probe is the CPU"),
        };
        Ok(Self { stream, peaks })
    }

    /// Measure the line's points in the measurement job (`walks`): the
    /// stream's (a device's, or the CPU's sweeps), then each peak's, every
    /// configuration of each. Its figures need their intervals, so each
    /// point samples to bound its median whatever its part of the time
    /// ([`Walks::bounded`]).
    pub(super) fn measure(walks: &mut Walks<'_, '_>) -> Result<(), ExecutorCostsError> {
        walks.bounded();
        let device = walks.device();
        let Probes { shape, stream, host, peaks: (f32, matrix, int8) } = Probes::of(device.topology(), device.info())?;
        if let Some(Ok(readers)) = host {
            measure_host(readers, walks)?;
        }
        if let Some(stream) = stream {
            stream.measure(walks, &OnceCell::new(), |device| {
                let inputs = (0..shape.sets)
                    .map(|_| Tensor::zeros(device, Element::f32(), &[shape.rows, ROW_VALUES]))
                    .collect::<Result<_, _>>()
                    .map_err(|error| failed("the stream's rows")(&error))?;
                Ok(Operands { inputs, steps: None, outputs: shape.rows })
            })?;
        }
        if let Some(ladder) = f32 {
            ladder.measure(walks, shape, Operand::F32)?;
        }
        if let Some(ladder) = matrix {
            ladder.measure(walks, shape, Operand::Matrix)?;
        }
        if let Some(ladder) = int8 {
            ladder.measure(walks, shape, Operand::Int8)?;
        }
        Ok(())
    }
}

/// The readers the CPU's stream is read by: the pool the CPU peaks run on
/// (the performance cores the process may run on, within its quota), and one
/// step more: every hardware thread where the pool has fewer, else twice the
/// pool. Why there are none where the host's processors cannot be read or
/// the process may run on no performance core.
pub(super) fn host_readers() -> Result<[u64; 2], ExecutorCostsError> {
    let threads = std::thread::available_parallelism()
        .map_err(|error| ExecutorCostsError(format!("the host's processors: {error}")))?
        .get();
    let processors = seismic::Processors::discover()
        .map_err(|error| ExecutorCostsError(format!("the host's processors: {error}")))?;
    readers_among(threads, processors.performance_cores())
}

/// [`host_readers`] on a host of `threads` hardware threads, `performance`
/// of them performance cores.
fn readers_among(threads: usize, performance: usize) -> Result<[u64; 2], ExecutorCostsError> {
    let pool = performance.min(threads);
    if pool == 0 {
        return Err(ExecutorCostsError("the process may run on no performance core".to_owned()));
    }
    Ok([pool as u64, if threads > pool { threads } else { 2 * pool } as u64])
}

/// The sweeps of the CPU's stream: [`HOST_POINTS`] points, a sweep of none
/// and of the whole span at each of the two reader counts.
const HOST_POINTS: usize = 4;

/// The key of the sweep of `bytes` of a span `span` by `readers` threads on
/// the CPU executor `identity` names: the line's own, apart from the host
/// memory's line (`MemoryCosts`) the runtime prices a host tier by.
fn sweep_key(identity: &seismic::Identity, readers: u64, span: u64, bytes: u64) -> PointKey {
    PointKey {
        executor: identity.clone(),
        kernels: Vec::new(),
        configuration: named(&[("line_host_stream", 1), ("readers", readers), ("span", span)]),
        point: named(&[("bytes", bytes)]),
    }
}

/// Measure the CPU's stream in the measurement job (`walks`): for each of
/// `readers`, sweeps by that many threads of none and of the whole span
/// (the span [`MemoryCosts::span`] gives, beyond every cache that can serve
/// host memory), every page of it written once before it is timed, in the
/// readers' shares ([`Sweeps::written`]).
fn measure_host(readers: [u64; 2], walks: &mut Walks<'_, '_>) -> Result<(), ExecutorCostsError> {
    let span = MemoryCosts::span(walks.device().topology());
    let identity = walks.identity();
    for readers in readers {
        let threads = usize::try_from(readers)
            .map_err(|_| ExecutorCostsError(format!("{readers} readers exceed the host")))?;
        let words = OnceCell::new();
        for bytes in [0, span] {
            let words = &words;
            walks.time_host(sweep_key(&identity, readers, span, bytes), move || {
                let words = words.get_or_init(|| Sweeps::written((span / 8) as usize, threads));
                Ok(Sweeps { words: &words[..(bytes / 8) as usize], readers: threads })
            });
        }
    }
    Ok(())
}

/// The CPU's stream from the records of its sweeps at `readers`
/// ([`host_readers`]): the fastest, reached where more readers did not raise
/// it; why none was had where every sweep was unobtainable or the readers
/// were not had (the stream is unobtainable and the peaks stand).
fn host_stream(
    readers: Result<[u64; 2], ExecutorCostsError>,
    topology: &DeviceTopology,
    identity: &seismic::Identity,
    store: &dyn RecordStore<PointKey>,
) -> Result<Stream, MachineLineError> {
    let readers = match readers {
        Ok(readers) => readers,
        Err(error) => return Ok(Stream::Unobtainable(Reason::new(error.to_string()))),
    };
    let span = MemoryCosts::span(topology);
    let mut rates = Vec::new();
    let mut unmeasured = None;
    for readers in readers {
        match sweep_rate(identity, span, readers, store)? {
            Ok(per_second) => rates.push((readers, per_second)),
            Err(fault) => {
                unmeasured.get_or_insert(fault);
            }
        }
    }
    Ok(match reached(&rates) {
        Some(rate) => Stream::Host(rate.map(|line| Sweep { per_second: line.per_second, readers: line.unroll })),
        None => unmeasured.expect("a stream is read by two reader counts"),
    })
}

/// The bytes a second `readers` threads read, the slope through the sweeps
/// of none and of `span` bytes (starting and joining the readers, their
/// fixed cost, apart), its interval from the sweeps'; the stream it is a
/// fault of where a sweep is unobtainable, or rose with no bytes.
fn sweep_rate(
    identity: &seismic::Identity,
    span: u64,
    readers: u64,
    store: &dyn RecordStore<PointKey>,
) -> Result<Result<MeasuredCost, Stream>, MachineLineError> {
    let mut sweeps = [None, None];
    for (sweep, bytes) in sweeps.iter_mut().zip([0, span]) {
        match store.load(&sweep_key(identity, readers, span, bytes))? {
            Some(Record::Measured(timing)) => *sweep = Some(timing),
            Some(Record::Unobtainable { reason }) => return Ok(Err(Stream::Unobtainable(reason))),
            None => return Err(MachineLineError::UnrecordedSweep { readers, bytes }),
        }
    }
    let [none, all] = sweeps.map(|sweep| sweep.expect("each sweep was read"));
    if all.median <= none.median {
        let reason = format!("{readers} readers read {span} bytes in {} s, none in {} s", all.median, none.median);
        return Ok(Err(Stream::Defect(Reason::new(reason))));
    }
    let interval =
        none.interval.zip(all.interval).and_then(|((none_fastest, none_slowest), (all_fastest, all_slowest))| {
            let (least, most) = (all_fastest - none_slowest, all_slowest - none_fastest);
            (least > 0.).then(|| (span as f64 / most, span as f64 / least))
        });
    Ok(Ok(MeasuredCost {
        value: span as f64 / (all.median - none.median),
        interval,
        samples: none.samples.min(all.samples),
    }))
}

/// Whether `after`, the next step's rate, rose above `before` by more than
/// their timings resolve: its interval lies wholly above `before`'s. Where
/// either has none (one sample) no plateau is shown, so it did.
fn raised(before: MeasuredCost, after: MeasuredCost) -> bool {
    match (before.interval, after.interval) {
        (Some((_, highest)), Some((lowest, _))) => lowest > highest,
        _ => true,
    }
}

/// A timing of `median` seconds, its interval `spread` either side where
/// given, for tests.
#[cfg(test)]
fn timing(median: f64, spread: Option<f64>) -> seismic::measure::Timing {
    seismic::measure::Timing {
        median,
        interval: spread.map(|spread| (median - spread, median + spread)),
        samples: if spread.is_some() { 8 } else { 1 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::measure::HeldStore;

    /// A peak is the marginal rate between S and 4S: a fixed cost of a call
    /// cancels. It scales where the 2S-to-4S increment is twice the S-to-2S
    /// one within the timings' intervals, and does not where the work is
    /// dropped (no added time) or sub-linear, or where one sample resolves
    /// no increment.
    #[test]
    fn a_peak_is_the_marginal_rate_and_stands_where_its_work_scales() {
        // 10 operations a first multiple; a call costs 1 s beside its work.
        let linear = Marginal::of(
            10.,
            &[
                super::timing(2., Some(0.1)),
                super::timing(3., Some(0.1)),
                super::timing(5., Some(0.1)),
            ],
        );
        assert!(linear.scaled);
        assert_eq!(linear.rate.unwrap().value, 3. * 10. / 3.);
        let (least, most) = linear.ratio.unwrap();
        assert!(least <= 2. && 2. <= most);
        // Work a compiler dropped: the time does not grow.
        let dropped = Marginal::of(
            10.,
            &[
                super::timing(2., Some(0.1)),
                super::timing(2., Some(0.1)),
                super::timing(2., Some(0.1)),
            ],
        );
        assert!(!dropped.scaled && dropped.rate.is_none());
        // Sub-linear: the second increment is the first.
        let sublinear = Marginal::of(
            10.,
            &[
                super::timing(2., Some(0.1)),
                super::timing(3., Some(0.1)),
                super::timing(4., Some(0.1)),
            ],
        );
        assert!(!sublinear.scaled && sublinear.rate.is_some());
        // One sample each shows no increment.
        let unresolved = Marginal::of(
            10.,
            &[
                super::timing(2., None),
                super::timing(3., None),
                super::timing(5., None),
            ],
        );
        assert!(!unresolved.scaled && unresolved.ratio.is_none());
    }

    /// The line is the fastest configuration, reached where a larger UNROLL
    /// was timed and ran no faster, or where the step to the largest rose
    /// less than its timings resolve.
    #[test]
    fn the_line_is_the_fastest_configuration_reached_past_it() {
        let rate = |value: f64, spread: Option<f64>| MeasuredCost::rate(1., &super::timing(10. - value, spread));
        let rates = |spread, values: &[f64]| {
            values
                .iter()
                .enumerate()
                .map(|(index, &value)| (1 << index, rate(value, spread)))
                .collect::<Vec<_>>()
        };
        assert_eq!(reached(&[]), None);
        // The fastest has a larger one after it that ran slower.
        let line = reached(&rates(Some(0.1), &[1., 3., 2.])).unwrap();
        assert!(matches!(line, Rate::Reached(Line { unroll: 2, .. })), "{line:?}");
        // The fastest is the largest and its step overlaps the one before.
        assert!(reached(&rates(Some(0.6), &[1., 2., 3.])).unwrap().is_reached());
        // It rose past the one before, beyond the intervals, or has no interval to resolve.
        assert!(!reached(&rates(Some(0.4), &[1., 2., 3.])).unwrap().is_reached());
        assert!(!reached(&rates(None, &[1., 2., 3.])).unwrap().is_reached());
    }

    /// The stored line reads each operand's three timings per UNROLL: the
    /// fastest scaled UNROLL's marginal rate, the figure unscaled where
    /// its work does not scale, a peak every configuration of which is
    /// unobtainable as unobtainable, and a stream that is unobtainable
    /// leaving the peaks.
    #[test]
    fn the_stored_line_reads_each_peak_through_its_scaling() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let topology = catalog.topology();
        let topology = &*topology;
        let executor = topology
            .executors()
            .iter()
            .find(|executor| executor.backend() == BackendName::Cpu)
            .expect("the host's CPU is an executor");
        let identity = executor_identity(topology, executor).unwrap();
        let probes = Probes::of(topology, executor).unwrap();
        let store = HeldStore::default();
        assert!(matches!(
            MachineLine::stored(topology, executor, &store),
            Err(MachineLineError::Unrecorded { probe: "line_f32", .. })
        ));
        assert!(probes.stream.is_none(), "the CPU's stream is the host's sweeps");
        let backend = executor.backend();
        let time = |key: PointKey, timing| store.store(&key, &Record::Measured(timing)).unwrap();
        // The stream is the host's sweeps: at the pool's readers a span
        // read in 2 s beyond the sweep of none, at more readers in 3 s.
        let [pool, wider] = host_readers().unwrap();
        let span = MemoryCosts::span(topology);
        let sweeps = [(pool, 0), (pool, span), (wider, 0), (wider, span)]
            .map(|(readers, bytes)| sweep_key(&identity, readers, span, bytes));
        for (key, seconds) in sweeps.iter().zip([1., 3., 1., 4.]) {
            time(key.clone(), super::timing(seconds, None));
        }
        let (f32, matrix, int8) = &probes.peaks;
        assert!(matrix.is_none(), "the CPU's forms multiply no 16-bit operand");
        let (f32, int8) = (f32.as_ref().unwrap(), int8.as_ref().unwrap());
        // Every UNROLL a second faster than the one before at its multiples
        // (a call's fixed 1 s beside a work of `work` seconds a multiple),
        // or with the work not growing past 2S.
        let f32_peak = |scales: bool| {
            for (index, &unroll) in f32.plans[0].unrolls.iter().enumerate() {
                let work = 8. - index as f64 * 0.5;
                for (plan, multiple) in f32.plans.iter().zip(MULTIPLES) {
                    let grown = if scales { multiple as f64 } else { multiple.min(2) as f64 };
                    time(plan.key(&identity, backend, unroll).unwrap(), super::timing(1. + work * grown, Some(0.05)));
                }
            }
        };
        let reason = Reason::new("no int8 dots here".to_owned());
        for plan in &int8.plans {
            for &unroll in &plan.unrolls {
                store
                    .store(
                        &plan.key(&identity, backend, unroll).unwrap(),
                        &Record::Unobtainable { reason: reason.clone() },
                    )
                    .unwrap();
            }
        }
        f32_peak(true);
        let line = MachineLine::stored(topology, executor, &store).unwrap();
        let Stream::Host(Rate::Reached(stream)) = line.stream else {
            panic!("the CPU's stream is the host's, reached: {:?}", line.stream)
        };
        assert!((stream.per_second.value - span as f64 / 2.).abs() < 1e-6 * span as f64);
        assert_eq!(stream.readers, pool);
        let largest = *f32.plans[0].unrolls.last().unwrap();
        // Every UNROLL is faster than the one before, beyond the intervals: the largest is the fastest and
        // more UNROLL may raise it, so the peak is open.
        let Peak::Rate(Rate::Open(peak)) = &line.peaks[0].1 else {
            panic!(
                "a scaling peak still rising at the largest UNROLL is open: {:?}",
                line.peaks[0]
            )
        };
        // The marginal rate: 3 multiples of operations over 3 multiples' work.
        let work = 8. - (f32.plans[0].unrolls.len() - 1) as f64 * 0.5;
        assert_eq!((peak.unroll, line.peaks[0].0), (largest, Operand::F32));
        assert!((peak.per_second.value - f32.plans[0].units / work).abs() < 1e-6 * f32.plans[0].units);
        assert_eq!(line.peaks[1], (Operand::Int8, Peak::Unobtainable(reason.clone())));
        // The work stopped growing past 2S: not a peak.
        f32_peak(false);
        let line = MachineLine::stored(topology, executor, &store).unwrap();
        let Peak::Unscaled { unroll, ratio, .. } = &line.peaks[0].1 else { panic!("unscaled: {:?}", line.peaks[0]) };
        assert!(*unroll == largest && ratio.is_some_and(|(_, most)| most < 2.));
        for key in &sweeps {
            store.store(key, &Record::Unobtainable { reason: reason.clone() }).unwrap();
        }
        // An unobtainable stream leaves the peaks standing.
        let line = MachineLine::stored(topology, executor, &store).unwrap();
        assert!(matches!(line.stream, Stream::Unobtainable(_)));
        assert!(matches!(line.peaks[0].1, Peak::Unscaled { .. }));
        // A sweep the job has not measured is no record of the line's.
        let unrecorded = sweep_rate(&identity, span, pool, &HeldStore::default());
        assert!(matches!(unrecorded, Err(MachineLineError::UnrecordedSweep { bytes: 0, .. })), "{unrecorded:?}");
    }

    /// A host whose readers cannot be had has no stream, and its peaks
    /// stand: the pool is the performance cores within the process's
    /// threads, none of them is no pool, and a failed discovery is a
    /// stream unobtainable for its cause.
    #[test]
    fn a_host_without_readers_has_no_stream() {
        assert_eq!(readers_among(8, 4).unwrap(), [4, 8]);
        assert_eq!(readers_among(4, 4).unwrap(), [4, 8]);
        assert_eq!(readers_among(2, 4).unwrap(), [2, 4]);
        assert!(readers_among(8, 0).is_err());
        let topology = seismic::Devices::discover(&[]).unwrap().topology();
        let cause = ExecutorCostsError("the host's processors: no".to_owned());
        let host = topology.executors().iter().find(|executor| executor.backend() == BackendName::Cpu).unwrap();
        let identity = executor_identity(&topology, host).unwrap();
        let stream = host_stream(Err(cause.clone()), &topology, &identity, &HeldStore::default()).unwrap();
        assert_eq!(stream, Stream::Unobtainable(Reason::new(cause.to_string())));
    }
}
