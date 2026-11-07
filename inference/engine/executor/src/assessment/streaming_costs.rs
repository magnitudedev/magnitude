//! What moving a load's experts costs on this machine, beside what its
//! kernels cost ([`super::executor_costs`]) and where a miss is computed
//! ([`super::cost`]): the reads, landings, moves and transfers the cost
//! function, the prompt-group model and the shrink take from measurement.
//!
//! - **The served file's read line** ([`RequestReads::line`]): a read of
//!   `b` bytes takes `λ + b / D`, from the load's own reads of its served
//!   file: `D` the volume's bytes over its busy time, `λ` what its demand
//!   reads took beyond that. A run is judged by it, and its miss record
//!   carries it to the estimates made from the record.
//! - **The landing line** ([`FileLine`]): what a request's file misses add
//!   past their read as they land into the host tier.
//! - **The shrink's move line** ([`MoveCosts`]), per bank group: moving `n`
//!   records of the group's own record form takes `μ₀ + n · μ₁`, its
//!   `routed_move` launches (one per routed tensor) timed moving none and
//!   the most one shrink moves, at most a move point's records (a move's
//!   time grows by one copy pass per `LIST_ROWS` records, so the point holds
//!   only what it moves). A shrink is paced by each moving group's
//!   line at the records it moves ([`MoveCosts::of_bank`], [`MoveLines`]).
//! - **The convert's ceiling** ([`ConvertCosts`]), per bank group: a stored
//!   record's conversion into its resident form by a scratch round's
//!   `routed_convert` launches (one per routed tensor), from a conveyor and
//!   into a scratch both in the device's own memory, wherever a load lays
//!   them: the slope through rounds of one record and of
//!   [`CONVERTED_RECORDS`].
//! - **The host memory's read line** ([`MemoryCosts`]), where a bank has a
//!   host tier (a record's read from it,
//!   [`super::cost::HostLines::memory`]): reading `b` bytes of host
//!   memory with every CPU at once takes `b · ρ` past the readers' start,
//!   the slope through sweeps of none and of a span beyond the host's
//!   caches.
//!
//! The move, convert and host memory lines are points a load requests of
//! the measurement job (`PointRequest`), keyed by the executor and the form
//! or readers they depend on ([`PointKey`]); the job measures them
//! (`MoveForm::measure`, `ConvertForm::measure`, `MemoryCosts::measure`) and
//! a load reads them from the records ([`MoveCosts::stored`]), measuring
//! nothing: until they are stored, a load's decision runs its policy that
//! needs no cost. No figure is fitted: each is a line through measured
//! points (`seismic::measure::Sums::fit`), of no negative term by
//! construction ([`Terms`]), and a recorded line that breaks them is
//! refused as it is read.

use super::executor_costs::{executor_identity, failed, named, ExecutorCostsError, Formation, Walks};
use crate::planning::{BankBinding, ExpertBankPlan, RecordTensor, RecordTensors};
use magnitude_kernels::{constants, routed_convert, routed_copy, routed_move};
use seismic::measure::{
    Gauge, InvalidLine, Pending, Reason, Record, RecordStore, StoreError, Sums, Terms, Timing, Unformed, Work,
};
use seismic::{
    Device, DeviceTopology, Element, Executor, GraphTiming, NativeKernel, Point,
    PointKey, Tensor,
};
use std::cell::OnceCell;
use std::fmt;
use std::path::Path;
use std::time::Instant;

/// Why the requests counted make no read line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReadLineError {
    /// No request read, or the volume delivered nothing yet.
    Unread { requests: u64 },
    /// Times that are not finite or are negative (a broken clock or sum).
    Terms(InvalidLine),
}

impl fmt::Display for ReadLineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unread { requests } => write!(formatter, "{requests} demand requests make no read line yet"),
            Self::Terms(invalid) => write!(formatter, "the requests' times: {invalid}"),
        }
    }
}

impl std::error::Error for ReadLineError {}

/// A read line: a read of `b` bytes takes `λ + b / D`, `λ` (its latency)
/// not negative and `1/D` (its seconds a byte) not negative, by
/// construction ([`Self::new`]): a per-byte cost of zero is a path whose
/// reads grow no slower with their bytes, as reads served from memory do.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "ReadTerms", into = "ReadTerms")]
pub struct ReadLine(Terms<2>);

/// A read line as it is recorded.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
struct ReadTerms {
    latency_seconds: f64,
    seconds_per_byte: f64,
}

impl TryFrom<ReadTerms> for ReadLine {
    type Error = InvalidLine;
    fn try_from(terms: ReadTerms) -> Result<Self, InvalidLine> {
        Self::new(terms.latency_seconds, terms.seconds_per_byte)
    }
}

impl From<ReadLine> for ReadTerms {
    fn from(line: ReadLine) -> Self {
        Self { latency_seconds: line.latency_seconds(), seconds_per_byte: line.seconds_per_byte() }
    }
}

impl ReadLine {
    /// The line of latency `latency_seconds` and `seconds_per_byte`, or why
    /// they make none.
    pub const fn new(latency_seconds: f64, seconds_per_byte: f64) -> Result<Self, InvalidLine> {
        match Terms::new([latency_seconds, seconds_per_byte]) {
            Ok(terms) => Ok(Self(terms)),
            Err(invalid) => Err(invalid),
        }
    }

    /// `λ`.
    pub const fn latency_seconds(&self) -> f64 {
        self.0.fixed()
    }

    /// `1/D`.
    pub const fn seconds_per_byte(&self) -> f64 {
        self.0.slope()
    }

    /// The transfer term of a read of `bytes` bytes, `b/D`, beside its
    /// latency `λ`; fractional bytes are an expectation's.
    pub fn transfer_seconds(&self, bytes: f64) -> f64 {
        bytes * self.seconds_per_byte()
    }

    /// What a read of `bytes` stored bytes costs, `λ + b/D`: the one price
    /// of reading experts from the file, which a decode layer's and a prompt
    /// group's file misses and a stored table's misses take.
    pub fn seconds(&self, bytes: u64) -> f64 {
        self.latency_seconds() + self.transfer_seconds(bytes as f64)
    }
}

/// The served file's reads as a load's real work made them: its requests'
/// demand reads, each its stored bytes and the seconds from a worker taking
/// its first piece to its last's end, and what the volume read and how long
/// it held a piece in flight (`seismic_io::accounting`). Sums of a
/// window are the difference of the sums at its ends ([`Self::since`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RequestReads {
    pub requests: u64,
    pub bytes: f64,
    pub seconds: f64,
    /// Every read's bytes (demand, hint, fill), and the volume's busy
    /// seconds over them.
    pub delivered: f64,
    pub busy_seconds: f64,
}

impl RequestReads {
    /// Count one request of `bytes` read in `seconds`.
    pub fn add(&mut self, bytes: u64, seconds: f64) {
        self.requests += 1;
        self.bytes += bytes as f64;
        self.seconds += seconds;
    }

    /// Count `bytes` the volume read over `seconds` busy.
    pub fn deliver(&mut self, bytes: u64, seconds: f64) {
        self.delivered += bytes as f64;
        self.busy_seconds += seconds;
    }

    /// The reads of `self` and then `later`, of one served path.
    pub fn merge(&self, later: &Self) -> Self {
        Self {
            requests: self.requests + later.requests,
            bytes: self.bytes + later.bytes,
            seconds: self.seconds + later.seconds,
            delivered: self.delivered + later.delivered,
            busy_seconds: self.busy_seconds + later.busy_seconds,
        }
    }

    /// The reads counted after `earlier`.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            requests: self.requests - earlier.requests,
            bytes: self.bytes - earlier.bytes,
            seconds: self.seconds - earlier.seconds,
            delivered: self.delivered - earlier.delivered,
            busy_seconds: self.busy_seconds - earlier.busy_seconds,
        }
    }

    /// The read line of the file actually served: its bytes at the drive's
    /// own rate, the volume's busy time over what it read (no job's time
    /// gives it: a job's holds the bytes ahead of it), and of its latency
    /// what the requests' demand reads took beyond that, at least none.
    /// There is none until a request read and the volume delivered.
    pub fn line(&self) -> Result<ReadLine, ReadLineError> {
        if self.requests == 0 || self.delivered == 0. {
            return Err(ReadLineError::Unread { requests: self.requests });
        }
        let byte_seconds = self.busy_seconds / self.delivered;
        let latency = ((self.seconds - self.bytes * byte_seconds) / self.requests as f64).max(0.);
        ReadLine::new(latency, byte_seconds).map_err(ReadLineError::Terms)
    }

    /// The read line once the reads measure one ([`Self::line`]): none
    /// while unread, and an error only for times that make no line.
    pub fn measured(&self) -> Result<Option<ReadLine>, ReadLineError> {
        match self.line() {
            Ok(line) => Ok(Some(line)),
            Err(ReadLineError::Unread { .. }) => Ok(None),
            Err(error) => Err(error),
        }
    }
}

/// A per-record line: `n` records take `fixed_seconds + n · record_seconds`,
/// neither term negative, by construction ([`Self::new`]).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "RecordTerms", into = "RecordTerms")]
pub struct RecordLine(Terms<2>);

/// A per-record line as it is recorded.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
struct RecordTerms {
    fixed_seconds: f64,
    record_seconds: f64,
}

impl TryFrom<RecordTerms> for RecordLine {
    type Error = InvalidLine;
    fn try_from(terms: RecordTerms) -> Result<Self, InvalidLine> {
        Self::new(terms.fixed_seconds, terms.record_seconds)
    }
}

impl From<RecordLine> for RecordTerms {
    fn from(line: RecordLine) -> Self {
        Self { fixed_seconds: line.0.fixed(), record_seconds: line.0.slope() }
    }
}

impl RecordLine {
    /// The line of `fixed_seconds` and `record_seconds`, or why they make
    /// none.
    pub const fn new(fixed_seconds: f64, record_seconds: f64) -> Result<Self, InvalidLine> {
        match Terms::new([fixed_seconds, record_seconds]) {
            Ok(terms) => Ok(Self(terms)),
            Err(invalid) => Err(invalid),
        }
    }

    /// The seconds of `records` records.
    pub fn seconds(&self, records: u64) -> f64 {
        self.0.at([1., records as f64])
    }
}

/// What a layer's file misses cost beside their read (`λ + b/D`, the read
/// line's) as they land into the host tier: a
/// request's reads are one job, which ends once its last piece is read;
/// each landing is then submitted to the host compute executor, whose
/// workers convert its pieces (`RowConversion`) beside the other landings',
/// then map and answer it. `n` landings at once end `fixed + n · record`
/// after the read: observed from a load's requests' own landings
/// (`Telemetry::landings_ended`) on the one generic line. Not in it: the
/// park's notice of the answer, an in-kernel poll of a host word where the
/// park spins (one link read), unmeasured where a park is a submission
/// boundary.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FileLine {
    pub landing: RecordLine,
}

impl FileLine {
    /// The line of `terms`, a fixed part and a record's.
    pub fn of(terms: Terms<2>) -> Self {
        Self { landing: RecordLine(terms) }
    }

    /// What `misses` file misses of a layer add past their reads' end.
    pub fn seconds(&self, misses: u64) -> f64 {
        self.landing.seconds(misses)
    }

    /// Whether `misses` land sooner by `device`, the device's convert in
    /// their barrier, than on the host.
    pub fn device_lands(&self, device: &RecordLine, misses: u64) -> bool {
        device.seconds(misses) < self.seconds(misses)
    }
}

/// A host-timed submission, complete when submitted.
struct Done(f64);

impl Pending<String> for Done {
    fn seconds(&self) -> Result<f64, String> {
        Ok(self.0)
    }
    fn wait(&self) -> Result<(), String> {
        Ok(())
    }
}

/// A bank group's records as its shrink moves them: each routed tensor's
/// record (one `routed_move` launch apiece, as the bank moves them), and
/// the most records one shrink moves (`L`, `ExpertBankPlan::move_records`).
/// What a move line's points are requested by (`PointRequest::Moves`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MoveForm {
    pub record: RecordTensors<RecordTensor>,
    pub moves: u64,
}

/// A bank's shrink move lines ([`MoveCosts::of_bank`]): the line of each
/// group whose shrink can move records (`ExpertBankPlan::move_records`, its
/// `routed_move` list), at its own record form, with the group. Any other
/// group holds the same records at every slots from the floor to the
/// widest, so no shrink retires or moves a record of it.
#[derive(Clone, Debug, PartialEq)]
pub struct MoveLines(Vec<(usize, RecordLine)>);

impl MoveLines {
    /// The device seconds a shrink takes that moves `moved(g)` records of
    /// each group `g` that can move: each moving group's launches, a group
    /// moving none launching nothing.
    pub fn seconds(&self, moved: impl Fn(usize) -> u64) -> f64 {
        self.0
            .iter()
            .map(|&(group, line)| match moved(group) {
                0 => 0.,
                records => line.seconds(records),
            })
            .sum()
    }
}

/// Why a move form's line could not be had; each point is named by the
/// records it moves.
#[derive(Clone, Debug, PartialEq)]
pub enum MoveError {
    Executor(ExecutorCostsError),
    Store(StoreError),
    Unrecorded { moved: u64 },
    Unobtainable { moved: u64, reason: Reason },
    /// Moving more records took no longer: `(records, seconds)` each.
    Flat { points: [(u64, f64); 2] },
}

impl fmt::Display for MoveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Executor(error) => write!(formatter, "{error}"),
            Self::Store(error) => write!(formatter, "a move record: {error}"),
            Self::Unrecorded { moved } => write!(formatter, "no record of a move of {moved} records"),
            Self::Unobtainable { moved, reason } => {
                write!(formatter, "a move of {moved} records is unobtainable: {reason}")
            }
            Self::Flat { points: [first, second] } => write!(
                formatter,
                "moving {} records took {:.3e} s, no longer than {} took {:.3e} s",
                second.0, second.1, first.0, first.1
            ),
        }
    }
}

impl std::error::Error for MoveError {}

impl From<ExecutorCostsError> for MoveError {
    fn from(error: ExecutorCostsError) -> Self {
        Self::Executor(error)
    }
}

impl From<StoreError> for MoveError {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}

/// A move form's line on an executor, through a move of none and of the
/// most records a point moves ([`MoveForm::moved`]), the launches' fixed
/// cost in its fixed term.
#[derive(Clone, Debug, PartialEq)]
pub struct MoveCosts {
    pub line: RecordLine,
}

impl MoveForm {
    /// The most records a point moves: the form's moves, at most a move
    /// point's (`MOVED_RECORDS`). A move's time is a launch's plus one copy
    /// pass per `LIST_ROWS` records, so the line through it holds at every
    /// move up to the form's, and the point holds only the records it moves
    /// (twice them, the upper moved onto the lower), not a whole shrink's.
    fn moved(&self) -> u64 {
        self.moves.min(super::executor_costs::MOVED_RECORDS)
    }

    /// Each routed tensor's `routed_move` over a point's records, its list
    /// as the bank forms it at the form's moves (`L`).
    fn moves(&self) -> RecordTensors<Formation<routed_move::Entry>> {
        let (moves, records) = (self.moves, 2 * self.moved());
        self.record.map(|_, tensor| {
            let [n, k] = tensor.matrix;
            Formation::new(
                routed_move::native_entry_with(routed_move::Elements { U: tensor.resident }),
                &[("D", records), ("L", moves), ("N", n), ("K", k)],
            )
        })
    }

    /// The records a point moves: none, and [`Self::moved`].
    fn points(&self) -> [u64; 2] {
        [0, self.moved()]
    }

    fn key(
        &self,
        executor: &seismic::Identity,
        backend: seismic::BackendName,
        moved: u64,
    ) -> Result<PointKey, ExecutorCostsError> {
        let mut configuration = named(&[("L", self.moves)]);
        for (index, (_, tensor)) in self.record.iter().enumerate() {
            configuration.extend(named(&[
                (&format!("{index}.n"), tensor.matrix[0]),
                (&format!("{index}.k"), tensor.matrix[1]),
            ]));
        }
        Ok(PointKey {
            executor: executor.clone(),
            kernels: self
                .moves()
                .iter()
                .map(|(_, formation)| formation.identity(backend))
                .collect::<Result<_, _>>()?,
            configuration,
            point: named(&[("moved", moved)]),
        })
    }
}

/// One move's graph: each routed tensor's `routed_move` of the first
/// `moved` list entries, record `M + i` onto record `i`, over `2M` zero
/// records (`M` the form's [`MoveForm::moved`]).
fn move_point(
    device: &Device,
    kernels: &RecordTensors<NativeKernel<routed_move::Entry>>,
    form: &MoveForm,
    moved: u64,
) -> Result<Point, ExecutorCostsError> {
    let error = failed("move point");
    let most = form.moved();
    let mut graph = device.native_graph();
    let words = magnitude_kernels::list_extent(form.moves);
    let mut ports = Vec::new();
    for ((_, kernel), (_, tensor)) in kernels.iter().zip(form.record.iter()) {
        let [n, k] = tensor.matrix;
        let mut records = graph.port(tensor.resident, &[2 * most, n, k]).map_err(|cause| error(&cause))?;
        let list = graph.port(Element::i32(), &[words]).map_err(|cause| error(&cause))?;
        graph
            .enqueue(
                kernel,
                routed_move::WorkflowArgs {
                    records: records.tensor_mut().into(),
                    list: list.tensor().into(),
                },
            )
            .map_err(|cause| error(&cause))?;
        ports.push((records, list, *tensor));
    }
    let plan = graph.seal().map_err(|cause| error(&cause))?;
    let mut list = vec![0i32; words as usize];
    list[constants::LIST_COUNT as usize] = moved as i32;
    for entry in 0..most {
        let at = magnitude_kernels::list_extent(entry) as usize;
        list[at + constants::LIST_FROM as usize] = (most + entry) as i32;
        list[at + constants::LIST_TO as usize] = entry as i32;
    }
    let list = list.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    let mut bound = Vec::new();
    for (records, port, tensor) in &ports {
        let [n, k] = tensor.matrix;
        bound.push((records, Tensor::zeros(device, tensor.resident, &[2 * most, n, k]).map_err(|cause| error(&cause))?));
        bound.push((
            port,
            Tensor::from_host(device, Element::i32(), &[words], &list).map_err(|cause| error(&cause))?,
        ));
    }
    let mut bindings = plan.bindings();
    for (port, tensor) in &bound {
        bindings.set(port, tensor).map_err(|cause| error(&cause))?;
    }
    plan.point(bindings, GraphTiming::Device)
        .map(Point::Graph)
        .map_err(|cause| error(&cause))
}

impl MoveCosts {
    /// `form`'s move line on `executor` that `store`'s records give,
    /// measuring and forming nothing; why there is none.
    pub fn stored(
        topology: &DeviceTopology,
        executor: &Executor,
        form: MoveForm,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Self, MoveError> {
        let identity = executor_identity(topology, executor)?;
        let [none, most] = form.points().map(|moved| {
            let key = form.key(&identity, executor.backend(), moved)?;
            let timing = match store.load(&key)? {
                Some(Record::Measured(timing)) => timing,
                Some(Record::Unobtainable { reason }) => return Err(MoveError::Unobtainable { moved, reason }),
                None => return Err(MoveError::Unrecorded { moved }),
            };
            Ok((moved, timing.median))
        });
        let medians = [none?, most?];
        let line = rising(medians).ok_or(MoveError::Flat { points: medians })?;
        Ok(Self { line: RecordLine(line) })
    }

    /// The move lines of a shrink of `plan`'s bank on `executor`, which pace
    /// a shrink to a decode step: each group's whose shrink can move records,
    /// at its own records and the most records one shrink moves, as
    /// `store`'s records give them ([`Self::stored`]); why there are none.
    pub fn of_bank(
        topology: &DeviceTopology,
        executor: &Executor,
        plan: &ExpertBankPlan,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<MoveLines, MoveError> {
        Self::forms(plan)
            .map(|(group, form)| Ok((group, Self::stored(topology, executor, form, store)?.line)))
            .collect::<Result<_, _>>()
            .map(MoveLines)
    }

    /// Each group of `plan` whose shrink can move records, with its form:
    /// what a load requests the move lines of (`PointRequest::Moves`).
    pub fn forms(plan: &ExpertBankPlan) -> impl Iterator<Item = (usize, MoveForm)> + '_ {
        plan.groups
            .iter()
            .enumerate()
            .map(|(group, members)| (group, MoveForm { record: members.record, moves: plan.move_records(group) }))
            .filter(|(_, form)| form.moves > 0)
    }
}

impl MoveForm {
    /// Measure the form's points in the measurement job (`walks`): moves of
    /// none and of [`Self::moved`] records, the kernels formed with the
    /// first attempted.
    pub(super) fn measure(&self, walks: &mut Walks<'_, '_>) -> Result<(), ExecutorCostsError> {
        let device = walks.device();
        type Formed = Result<RecordTensors<NativeKernel<routed_move::Entry>>, ExecutorCostsError>;
        let formed = OnceCell::<Formed>::new();
        for moved in self.points() {
            let key = self.key(&walks.identity(), device.backend(), moved)?;
            let formed = &formed;
            walks.time(key, false, move || {
                let kernels = formed
                    .get_or_init(|| self.moves().try_map(|_, formation| formation.form(device, None)))
                    .as_ref()
                    .map_err(|error| Unformed::Never(error.clone()))?;
                move_point(device, kernels, self, moved).map_err(Unformed::Now)
            });
        }
        Ok(())
    }
}

/// The records a convert ceiling's larger round converts
/// ([`ConvertCosts`]): enough of a record's row tiles that the device is
/// full however few rows a record holds.
pub const CONVERTED_RECORDS: u64 = 64;

/// Why the convert's ceiling ([`ConvertCosts`]) could not be had; each
/// point is named by the records it converts.
#[derive(Clone, Debug, PartialEq)]
pub enum CeilingError {
    Executor(ExecutorCostsError),
    Store(StoreError),
    Unobtainable { count: u64, reason: Reason },
    /// A point no record holds: the job has not measured it yet.
    Unrecorded { count: u64 },
    /// More took no longer: `(count, seconds)` each.
    Flat { points: [(u64, f64); 2] },
}

impl fmt::Display for CeilingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Executor(error) => write!(formatter, "{error}"),
            Self::Store(error) => write!(formatter, "a ceiling's record: {error}"),
            Self::Unobtainable { count, reason } => write!(formatter, "a point of {count} is unobtainable: {reason}"),
            Self::Unrecorded { count } => write!(formatter, "no record of a point of {count}"),
            Self::Flat { points: [first, second] } => write!(
                formatter,
                "{} took {:.3e} s, no longer than {} took {:.3e} s",
                second.0, second.1, first.0, first.1
            ),
        }
    }
}

impl std::error::Error for CeilingError {}

impl From<ExecutorCostsError> for CeilingError {
    fn from(error: ExecutorCostsError) -> Self {
        Self::Executor(error)
    }
}

impl From<StoreError> for CeilingError {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}

/// The line through two measured points `(count, seconds)` of a figure
/// that rises with its count ([`Sums::fit`]); none where more took no
/// longer.
fn rising(points: [(u64, f64); 2]) -> Option<Terms<2>> {
    let mut sums = Sums::default();
    for (count, seconds) in points {
        sums.add([1., count as f64], seconds, 1.);
    }
    sums.fit().and_then(Result::ok).filter(|terms| terms.slope() > 0.)
}

/// A bank group's record as a scratch round converts it: each routed
/// tensor's stored form into its resident one. What a convert ceiling's
/// points are requested by (`PointRequest::Convert`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ConvertForm {
    pub record: RecordTensors<RecordTensor>,
}

impl ConvertForm {
    /// Each routed tensor's `routed_convert` over [`CONVERTED_RECORDS`]
    /// conveyor records into as many scratch records, its list as long.
    fn converts(&self) -> RecordTensors<Formation<routed_convert::Entry>> {
        let records = CONVERTED_RECORDS;
        self.record.map(|_, tensor| {
            let [n, k] = tensor.matrix;
            Formation::new(
                routed_convert::native_entry_with(routed_convert::Elements { ST: tensor.source, U: tensor.resident }),
                &[("V", records), ("L", records), ("S", records), ("N", n), ("K", k)],
            )
        })
    }

    fn key(&self, executor: &seismic::Identity, backend: seismic::BackendName, converted: u64) -> Result<PointKey, ExecutorCostsError> {
        // Measured with any load beside it holding its own traffic
        // (`SCHEMA`).
        let mut configuration = named(&[("convert", CONVERTED_RECORDS), SCHEMA]);
        for (index, (_, tensor)) in self.record.iter().enumerate() {
            configuration.extend(named(&[(&format!("{index}.n"), tensor.matrix[0]), (&format!("{index}.k"), tensor.matrix[1])]));
        }
        Ok(PointKey {
            executor: executor.clone(),
            kernels: self.converts().iter().map(|(_, formation)| formation.identity(backend)).collect::<Result<_, _>>()?,
            configuration,
            point: named(&[("converted", converted)]),
        })
    }
}

/// One round's graph: each routed tensor's `routed_convert` of the first
/// `converted` list entries, conveyor record `i` into scratch record `i`,
/// both zero and in the device's own memory, on the grid
/// `routed_scratch_grids` writes a round of `converted` records.
fn convert_point(
    device: &Device,
    kernels: &RecordTensors<NativeKernel<routed_convert::Entry>>,
    form: &ConvertForm,
    converted: u64,
) -> Result<Point, ExecutorCostsError> {
    let error = failed("convert point");
    let records = CONVERTED_RECORDS;
    let words = magnitude_kernels::list_extent(records);
    // The step and its list: the list written for the step's sequence.
    let sequence = 1;
    let mut step = vec![0i32; constants::STEP_WORDS as usize];
    step[constants::STEP_LOW as usize] = sequence;
    let mut list = vec![0i32; words as usize];
    list[constants::LIST_SEQUENCE as usize] = sequence;
    list[constants::LIST_COUNT as usize] = converted as i32;
    for entry in 0..records {
        let at = magnitude_kernels::list_extent(entry) as usize;
        list[at + constants::LIST_FROM as usize] = entry as i32;
        list[at + constants::LIST_TO as usize] = entry as i32;
    }
    let bytes = |words: &[i32]| words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    let mut graph = device.native_graph();
    let port = |graph: &mut seismic::NativeGraph, element, extents: &[u64]| graph.port(element, extents).map_err(|cause| error(&cause));
    let step_port = port(&mut graph, Element::i32(), &[constants::STEP_WORDS])?;
    let list_port = port(&mut graph, Element::i32(), &[words])?;
    let step_tensor = Tensor::from_host(device, Element::i32(), &[constants::STEP_WORDS], &bytes(&step)).map_err(|cause| error(&cause))?;
    let list_tensor = Tensor::from_host(device, Element::i32(), &[words], &bytes(&list)).map_err(|cause| error(&cause))?;
    let mut bound = Vec::new();
    for ((_, kernel), (_, tensor)) in kernels.iter().zip(form.record.iter()) {
        let [n, k] = tensor.matrix;
        let conveyor = port(&mut graph, tensor.source, &[records, n, k])?;
        let mut destination = port(&mut graph, tensor.resident, &[records, n, k])?;
        let grid = port(&mut graph, Element::i32(), &[3])?;
        graph
            .enqueue(
                kernel,
                routed_convert::WorkflowArgs {
                    conveyor: conveyor.tensor().into(),
                    list: list_port.tensor().into(),
                    step: step_port.tensor().into(),
                    destination: destination.tensor_mut().into(),
                    grid: grid.tensor().into(),
                },
            )
            .map_err(|cause| error(&cause))?;
        let groups = crate::operators::routed::streamed::convert_groups(n).map_err(|cause| error(&cause))?;
        bound.push((conveyor, Tensor::zeros(device, tensor.source, &[records, n, k]).map_err(|cause| error(&cause))?));
        bound.push((destination, Tensor::zeros(device, tensor.resident, &[records, n, k]).map_err(|cause| error(&cause))?));
        bound.push((grid, Tensor::from_host(device, Element::i32(), &[3], &bytes(&[groups, converted as i32, 1])).map_err(|cause| error(&cause))?));
    }
    let plan = graph.seal().map_err(|cause| error(&cause))?;
    let mut bindings = plan.bindings();
    for (port, tensor) in [(&step_port, &step_tensor), (&list_port, &list_tensor)].into_iter().chain(bound.iter().map(|(port, tensor)| (port, tensor))) {
        bindings.set(port, tensor).map_err(|cause| error(&cause))?;
    }
    plan.point(bindings, GraphTiming::Device).map(Point::Graph).map_err(|cause| error(&cause))
}

/// The convert's ceiling on an executor, per bank group: a record of the
/// group's form converted from the device's own memory into it, the
/// slope's seconds through rounds of one record and of
/// [`CONVERTED_RECORDS`], each timed by the measurement job on the device
/// at its sustained clock and kept in the record store, so a form is
/// measured once per executor. It holds wherever
/// a load lays its conveyor and scratch: what the convert costs at the
/// device's own memory.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvertCosts {
    pub record_seconds: f64,
    /// The convert of a count of records, launches and all.
    pub line: RecordLine,
}

impl ConvertCosts {
    /// `form`'s convert ceiling on `executor` that `store`'s records give,
    /// measuring and forming nothing; why there is none.
    pub fn stored(
        topology: &DeviceTopology,
        executor: &Executor,
        form: ConvertForm,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Self, CeilingError> {
        let identity = executor_identity(topology, executor)?;
        let [one, most] = [1, CONVERTED_RECORDS].map(|converted| {
            let key = form.key(&identity, executor.backend(), converted)?;
            let timing = match store.load(&key)? {
                Some(Record::Measured(timing)) => timing,
                Some(Record::Unobtainable { reason }) => return Err(CeilingError::Unobtainable { count: converted, reason }),
                None => return Err(CeilingError::Unrecorded { count: converted }),
            };
            Ok((converted, timing.median))
        });
        let medians = [one?, most?];
        let line = rising(medians).ok_or(CeilingError::Flat { points: medians })?;
        Ok(Self { record_seconds: line.slope(), line: RecordLine(line) })
    }

    /// Each bank group's convert line of `plan`'s bank ([`Self::stored`]).
    pub fn lines(
        topology: &DeviceTopology,
        executor: &Executor,
        plan: &ExpertBankPlan,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Vec<RecordLine>, CeilingError> {
        Self::forms(plan).map(|form| Ok(Self::stored(topology, executor, form, store)?.line)).collect()
    }

    /// Each streamed layer of `plan`'s bank's convert ceiling, a record's
    /// seconds, by its group's form ([`Self::stored`]).
    pub fn of_bank(
        topology: &DeviceTopology,
        executor: &Executor,
        plan: &ExpertBankPlan,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Vec<f64>, CeilingError> {
        let groups = Self::forms(plan)
            .map(|form| Ok(Self::stored(topology, executor, form, store)?.record_seconds))
            .collect::<Result<Vec<_>, CeilingError>>()?;
        Ok(plan.layers.iter().map(|layer| groups[layer.group]).collect())
    }

    /// Each group's form of `plan`'s bank, in group order: what the convert
    /// ceilings are requested by (`PointRequest::Convert`).
    pub fn forms(plan: &ExpertBankPlan) -> impl Iterator<Item = ConvertForm> + '_ {
        plan.groups.iter().map(|members| ConvertForm { record: members.record })
    }
}

impl ConvertForm {
    /// Measure the form's points in the measurement job (`walks`): rounds of
    /// one record and of [`CONVERTED_RECORDS`], the kernels formed with the
    /// first attempted.
    pub(super) fn measure(&self, walks: &mut Walks<'_, '_>) -> Result<(), ExecutorCostsError> {
        let device = walks.device();
        type Formed = Result<RecordTensors<NativeKernel<routed_convert::Entry>>, ExecutorCostsError>;
        let formed = OnceCell::<Formed>::new();
        for converted in [1, CONVERTED_RECORDS] {
            let key = self.key(&walks.identity(), device.backend(), converted)?;
            let formed = &formed;
            walks.time(key, false, move || {
                let kernels = formed
                    .get_or_init(|| self.converts().try_map(|_, formation| formation.form(device, None)))
                    .as_ref()
                    .map_err(|error| Unformed::Never(error.clone()))?;
                convert_point(device, kernels, self, converted).map_err(Unformed::Now)
            });
        }
        Ok(())
    }
}

/// The blocks a copy over the link takes (`routed_copy` reading host
/// storage in place into device memory: a prompt fill's tier copy): the
/// fewest whose reads reach the link's rate, `⌈R_link / r_block⌉`, from a
/// copy of [`TIER_COPY_PROBE`] on one block (`r_block`) and on a list's
/// whole grid (`magnitude_kernels::LIST_BLOCKS`; `R_link`: the link, every
/// block's reads in flight). More blocks move no faster and
/// take the SMs compute beside them needs: a 59 MB copy on 512 blocks hid
/// 2% of itself under the 35B's expert pass on an RTX 3090, on 2 blocks 78%,
/// each at the link's 11.8–11.9 GB/s, one block's 6.40 GB/s. Both points
/// are kept in the record store, so the count is measured once per executor.
///
/// A copy on the device's transfer queue also takes from the device's work
/// beside it while it runs: on an RTX 3090 under Vulkan a submission beside
/// a 13 GB/s host-to-device copy starts about 90 µs later and runs 2.9x
/// longer. So the device's work, each run its own submission, is timed
/// alone and beside such a copy ([`Self::stall_share`]), both kept in the
/// record store.
#[derive(Clone, Debug, PartialEq)]
pub struct TierCopyCosts {
    pub blocks: u32,
    /// The probe's copy on a list's whole grid.
    pub whole_seconds: f64,
    /// The share of the device's time a copy on the transfer queue takes
    /// from work queued beside it, for as long as the copy runs: `1 −
    /// alone / beside` of the probe's work submitted back to back.
    pub stall_share: f64,
}

/// Whether the stall probe's queued work runs alone or beside a copy on
/// the transfer queue.
#[derive(Clone, Copy)]
enum Beside {
    Alone,
    Copy,
}

impl Beside {
    fn point(self) -> u64 {
        match self {
            Self::Alone => 0,
            Self::Copy => 1,
        }
    }

    /// The [`TierCopyPoint`] of the stall probe's work run `self`.
    fn named(self) -> TierCopyPoint {
        match self {
            Self::Alone => TierCopyPoint::StallAlone,
            Self::Copy => TierCopyPoint::StallBesideCopy,
        }
    }
}

/// The schema of the points a tool may measure in a load's own process,
/// the tier copy's, its stall's and the convert's, a term of their keys:
/// 2, each taken with the load's traffic held and drained
/// (`ExecutorDomain::hold_expert_traffic`); a record of an earlier schema
/// may have been taken beside the load's copies and reads, so none is read
/// and each is measured again.
const SCHEMA: (&str, u64) = ("schema", 2);

/// The probe a tier copy's blocks are measured over: records of a MiB,
/// enough of them that a launch's fixed cost (about 10 µs) is under 1% of
/// the link's time to copy them (2.8 ms at 12 GB/s).
pub const TIER_COPY_PROBE: (u64, u64) = (32, 1 << 20);

/// A point of [`TierCopyCosts`]: the probe's copy on `blocks`, or the
/// stall probe's work, alone or beside a copy on the transfer queue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TierCopyPoint {
    Copy { blocks: u32 },
    StallAlone,
    StallBesideCopy,
}

impl fmt::Display for TierCopyPoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Copy { blocks } => write!(formatter, "a tier copy on {blocks} blocks"),
            Self::StallAlone => formatter.write_str("the stall probe's work alone"),
            Self::StallBesideCopy => formatter.write_str("the stall probe's work beside a transfer-queue copy"),
        }
    }
}

/// Why a tier copy's costs could not be had, each point named
/// ([`TierCopyPoint`]).
#[derive(Clone, Debug, PartialEq)]
pub enum TierCopyError {
    Executor(ExecutorCostsError),
    Store(StoreError),
    Unrecorded { point: TierCopyPoint },
    Unobtainable { point: TierCopyPoint, reason: Reason },
    /// The stall probe's work ran faster beside a copy than alone, beyond
    /// both timings' intervals: a bad measurement, never a stall of zero.
    FasterBeside { alone: f64, beside: f64 },
}

impl fmt::Display for TierCopyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Executor(error) => write!(formatter, "{error}"),
            Self::Store(error) => write!(formatter, "a tier copy record: {error}"),
            Self::Unrecorded { point } => write!(formatter, "no record of {point}"),
            Self::Unobtainable { point, reason } => write!(formatter, "{point} is unobtainable: {reason}"),
            Self::FasterBeside { alone, beside } => write!(
                formatter,
                "the stall probe ran faster beside a copy ({beside:e} s) than alone ({alone:e} s), beyond both intervals: a bad measurement"
            ),
        }
    }
}

impl std::error::Error for TierCopyError {}

impl From<ExecutorCostsError> for TierCopyError {
    fn from(error: ExecutorCostsError) -> Self {
        Self::Executor(error)
    }
}

impl From<StoreError> for TierCopyError {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}

impl TierCopyCosts {
    fn formation() -> Formation<routed_copy::Entry> {
        let (records, bytes) = TIER_COPY_PROBE;
        Formation::new(
            routed_copy::native_entry_with(routed_copy::Elements { U: Element::bool() }),
            &[("S", records), ("D", records), ("L", records), ("N", 1), ("K", bytes)],
        )
    }

    fn key(identity: &seismic::Identity, backend: seismic::BackendName, blocks: u32) -> Result<PointKey, ExecutorCostsError> {
        let (records, bytes) = TIER_COPY_PROBE;
        Ok(PointKey {
            executor: identity.clone(),
            kernels: vec![Self::formation().identity(backend)?],
            configuration: named(&[("tier_copy", 1), ("records", records), ("bytes", bytes), SCHEMA]),
            point: named(&[("blocks", u64::from(blocks))]),
        })
    }

    /// The stall probe's key: the probe's records copied within device
    /// memory on a list's whole grid, each run its own submission, `beside`.
    fn stall_key(identity: &seismic::Identity, backend: seismic::BackendName, beside: Beside) -> Result<PointKey, ExecutorCostsError> {
        let (records, bytes) = TIER_COPY_PROBE;
        Ok(PointKey {
            executor: identity.clone(),
            kernels: vec![Self::formation().identity(backend)?],
            configuration: named(&[("copy_stall", 1), ("records", records), ("bytes", bytes), SCHEMA]),
            point: named(&[("beside_copy", beside.point())]),
        })
    }

    /// The costs on `executor` that `store`'s records give, measuring and
    /// forming nothing; why there are none: a load copies on a list's whole
    /// grid until the job has measured them.
    pub fn stored(topology: &DeviceTopology, executor: &Executor, store: &dyn RecordStore<PointKey>) -> Result<Self, TierCopyError> {
        let (backend, identity) = (executor.backend(), executor_identity(topology, executor)?);
        let grid = magnitude_kernels::LIST_BLOCKS;
        let timing = |key: PointKey, point: TierCopyPoint| match store.load(&key)? {
            Some(Record::Measured(timing)) => Ok(timing),
            Some(Record::Unobtainable { reason }) => Err(TierCopyError::Unobtainable { point, reason }),
            None => Err(TierCopyError::Unrecorded { point }),
        };
        let median = |key: PointKey, point: TierCopyPoint| timing(key, point).map(|timing| timing.median);
        let [one, whole] = [1, grid].map(|blocks| median(Self::key(&identity, backend, blocks)?, TierCopyPoint::Copy { blocks }));
        let [one, whole] = [one?, whole?];
        // A block's rate against the link's, `R_link / r_block`, is the one
        // block's time over the whole grid's.
        let blocks = (one / whole).ceil().clamp(1., f64::from(grid)) as u32;
        let [alone, beside] = [Beside::Alone, Beside::Copy].map(|beside| timing(Self::stall_key(&identity, backend, beside)?, beside.named()));
        let (alone, beside) = (alone?, beside?);
        // The work beside a copy takes at least its time alone. Faster beside
        // it within the timings' intervals, a copy stalls nothing the probe
        // resolves; faster beyond them, the measurement is bad.
        let stall_share = match beside.median >= alone.median {
            true => 1. - alone.median / beside.median,
            false => match (alone.interval, beside.interval) {
                (Some((alone_low, _)), Some((_, beside_high))) if beside_high < alone_low => {
                    return Err(TierCopyError::FasterBeside { alone: alone.median, beside: beside.median });
                }
                _ => 0.,
            },
        };
        Ok(Self { blocks, whole_seconds: whole, stall_share })
    }

    /// The link's ceiling, a byte's seconds: the whole grid's copy from
    /// host memory the device reads in place into device memory, every block
    /// keeping its reads in flight, what the link carries at its peak.
    pub fn link_byte_seconds(&self) -> f64 {
        let (records, bytes) = TIER_COPY_PROBE;
        self.whole_seconds / (records * bytes) as f64
    }

    /// Measure the copy in the measurement job (`walks`), on a device with
    /// host storage it reads in place (a linked host tier's): on one block
    /// and on a list's whole grid, the kernel formed with the first
    /// attempted; then the stall probe's work alone and beside a copy the
    /// transfer queue runs throughout ([`Self::stall_share`]). Why the
    /// device has no transfer queue fails it.
    pub(super) fn measure(walks: &mut Walks<'_, '_>) -> Result<(), ExecutorCostsError> {
        let device = walks.device();
        let formed = OnceCell::<Result<NativeKernel<routed_copy::Entry>, ExecutorCostsError>>::new();
        for blocks in [1, magnitude_kernels::LIST_BLOCKS] {
            let key = Self::key(&walks.identity(), device.backend(), blocks)?;
            let formed = &formed;
            walks.time(key, true, move || {
                let kernel = formed
                    .get_or_init(|| Self::formation().form(device, None))
                    .as_ref()
                    .map_err(|error| Unformed::Never(error.clone()))?;
                tier_copy_point(device, kernel, blocks, Source::Host).map_err(Unformed::Now)
            });
        }
        // The device's work queued alone, then beside a copy the transfer
        // queue runs throughout its timing.
        let error = failed("the stall probe's copy");
        let transfer = seismic::Transfer::new(device).map_err(|cause| error(&cause))?;
        let (records, bytes) = TIER_COPY_PROBE;
        let shape = [records, 1, bytes];
        let copied = Tensor::shared(device, Element::bool(), &shape).map_err(|cause| error(&cause))?;
        let into = Tensor::zeros(device, Element::bool(), &shape).map_err(|cause| error(&cause))?;
        for beside in [Beside::Alone, Beside::Copy] {
            let key = Self::stall_key(&walks.identity(), device.backend(), beside)?;
            let formed = &formed;
            let point = move || {
                let kernel = formed
                    .get_or_init(|| Self::formation().form(device, None))
                    .as_ref()
                    .map_err(|error| Unformed::Never(error.clone()))?;
                tier_copy_point(device, kernel, magnitude_kernels::LIST_BLOCKS, Source::Device).map_err(Unformed::Now)
            };
            // Timed with no compile beside it; beside a copy, its record kept
            // only where the copy ran throughout ([`Copying`]).
            let store = walks.store();
            match beside {
                Beside::Alone => walks.time_in(key, true, store, point),
                Beside::Copy => {
                    let copying = Copying::start(&transfer, &copied, &into, store);
                    walks.time_in(key, true, &copying, point);
                    copying.stop().map_err(|cause| error(&cause))?;
                }
            }
        }
        Ok(())
    }

    /// The device time a copy taking `time` on the transfer queue takes
    /// from the work beside it ([`Self::stall_share`]), in `time`'s unit.
    pub fn stall(&self, time: f64) -> f64 {
        time * self.stall_share
    }
}

/// A copy the transfer queue runs, batch after batch, from host memory into
/// device memory until stopped: what a promotion is to the device's work
/// beside it. As the store of a point timed beside it, it stops before the
/// point's record is kept, and keeps none where it failed: the point was
/// not timed beside a copy throughout.
struct Copying<'s> {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: std::sync::Mutex<Option<std::thread::JoinHandle<Result<(), String>>>>,
    store: &'s dyn RecordStore<PointKey>,
}

impl<'s> Copying<'s> {
    fn start(transfer: &seismic::Transfer, copied: &Tensor, into: &Tensor, store: &'s dyn RecordStore<PointKey>) -> Self {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (transfer, copied, into, stopping) = (transfer.clone(), copied.clone(), into.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !stopping.load(std::sync::atomic::Ordering::Relaxed) {
                let run = seismic::TransferRun { source: &copied, source_offset: 0, destination: &into, destination_offset: 0, bytes: copied.byte_len() };
                let value = transfer.copy(&[seismic::TransferItem::Run(run)]).map_err(|error| error.to_string())?;
                while !transfer.wait(value, std::time::Duration::from_secs(1)).map_err(|error| error.to_string())? {}
            }
            Ok(())
        });
        Self { stop, thread: std::sync::Mutex::new(Some(thread)), store }
    }

    /// Stop after the batch running: why it failed, if it did and no stop
    /// before told it.
    fn stop(&self) -> Result<(), String> {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        match self.thread.lock().expect("the copier").take() {
            Some(thread) => thread.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
            None => Ok(()),
        }
    }
}

impl RecordStore<PointKey> for Copying<'_> {
    fn load(&self, key: &PointKey) -> Result<Option<Record>, StoreError> {
        self.store.load(key)
    }

    fn store(&self, key: &PointKey, record: &Record) -> Result<(), StoreError> {
        self.stop().map_err(|cause| StoreError(format!("the copy beside the point failed: {cause}")))?;
        self.store.store(key, record)
    }
}

/// Where a copy point's records lie: in host memory the device reads in
/// place over the link (a tier copy), or in device memory (the stall
/// probe's work, which loads no link).
#[derive(Clone, Copy)]
enum Source {
    Host,
    Device,
}

/// A copy of [`TIER_COPY_PROBE`]'s records from host-shared memory, read in
/// place over the link as a tier's records are, into device memory on
/// `blocks` blocks, its list staged in device memory as the copy reads it.
fn tier_copy_point(device: &Device, kernel: &NativeKernel<routed_copy::Entry>, blocks: u32, source: Source) -> Result<Point, ExecutorCostsError> {
    let error = failed("tier copy point");
    let (records, bytes) = TIER_COPY_PROBE;
    let byte = Element::bool();
    let shape = [records, 1, bytes];
    let sequence = 1i32;
    let words = magnitude_kernels::list_extent(records);
    let mut list = vec![0i32; words as usize];
    list[constants::LIST_SEQUENCE as usize] = sequence;
    list[constants::LIST_COUNT as usize] = records as i32;
    for entry in 0..records {
        let at = magnitude_kernels::list_extent(entry) as usize;
        list[at + constants::LIST_FROM as usize] = entry as i32;
        list[at + constants::LIST_TO as usize] = entry as i32;
    }
    let i32_bytes = |values: &[i32]| values.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    let tensors = [
        match source {
            // Host memory the device reads in place over the link, as a tier's.
            Source::Host => Tensor::shared(device, byte, &shape),
            Source::Device => Tensor::zeros(device, byte, &shape),
        }
        .map_err(|cause| error(&cause))?,
        Tensor::from_host(device, Element::i32(), &[words], &i32_bytes(&list)).map_err(|cause| error(&cause))?,
        Tensor::from_host(device, Element::i32(), &[constants::STEP_WORDS], &i32_bytes(&[sequence, 0])).map_err(|cause| error(&cause))?,
        Tensor::zeros(device, byte, &shape).map_err(|cause| error(&cause))?,
        Tensor::from_host(device, Element::i32(), &[3], &i32_bytes(&[blocks as i32, 1, 1])).map_err(|cause| error(&cause))?,
    ];
    let mut graph = device.native_graph();
    let mut ports = Vec::with_capacity(tensors.len());
    for tensor in &tensors {
        ports.push(graph.port(tensor.element(), &tensor.extents()).map_err(|cause| error(&cause))?);
    }
    let [source_port, list_port, step_port, destination_port, grid_port] = &mut ports[..] else {
        unreachable!("five ports");
    };
    graph
        .enqueue(
            kernel,
            routed_copy::WorkflowArgs {
                source: source_port.tensor().into(),
                list: list_port.tensor().into(),
                step: step_port.tensor().into(),
                destination: destination_port.tensor_mut().into(),
                grid: grid_port.tensor().into(),
            },
        )
        .map_err(|cause| error(&cause))?;
    let plan = graph.seal().map_err(|cause| error(&cause))?;
    let mut bindings = plan.bindings();
    for (port, tensor) in ports.iter().zip(&tensors) {
        bindings.set(port, tensor).map_err(|cause| error(&cause))?;
    }
    // The stall probe's runs are submitted back to back, each its own
    // submission, as a step's are: a copy beside them delays each.
    let timing = match source {
        Source::Host => GraphTiming::Device,
        Source::Device => GraphTiming::Queued,
    };
    plan.point(bindings, timing).map(Point::Graph).map_err(|cause| error(&cause))
}

/// The value `store` records measured at `key`, none while unrecorded. The
/// jobs that write these records ([`GraphMemory`], [`VolumeCosts`],
/// [`CpuCosts`]) store only measurements: one recorded unobtainable is a
/// store that is not theirs, an error.
fn measured(store: &dyn RecordStore<PointKey>, key: &PointKey, what: &str) -> Result<Option<f64>, ExecutorCostsError> {
    match store.load(key).map_err(|cause| failed(what)(&cause))? {
        Some(Record::Measured(timing)) => Ok(Some(timing.median)),
        Some(Record::Unobtainable { reason }) => Err(failed(what)(&format!("recorded unobtainable ({reason}), which no job writes"))),
        None => Ok(None),
    }
}

/// What an executor's recorded graphs hold (`seismic::GraphCost`), a
/// requested point (`PointRequest::GraphCost`): the job's own device
/// measures it on opening, and its terms are kept as records, each a number
/// of bytes, so a load opens nothing to plan.
pub struct GraphMemory;

impl GraphMemory {
    /// The terms `backend`'s cost holds.
    fn terms(backend: seismic::BackendName) -> &'static [&'static str] {
        match backend {
            seismic::BackendName::Cuda => &["graph", "launch", "granule"],
            seismic::BackendName::Vulkan => &["launch", "chunk"],
            seismic::BackendName::Cpu | seismic::BackendName::Metal => &[],
        }
    }

    fn key(identity: &seismic::Identity, term: &str) -> PointKey {
        PointKey {
            executor: identity.clone(),
            kernels: Vec::new(),
            configuration: named(&[("graph_memory", 1)]),
            point: named(&[(term, 1)]),
        }
    }

    /// The cost on `executor` that `store` records, none where a term of it
    /// is unrecorded.
    pub fn stored(
        topology: &DeviceTopology,
        executor: &Executor,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Option<seismic::GraphCost>, ExecutorCostsError> {
        let identity = executor_identity(topology, executor)?;
        let backend = executor.backend();
        let values = Self::terms(backend)
            .iter()
            .map(|term| measured(store, &Self::key(&identity, term), "graph memory record"))
            .collect::<Result<Option<Vec<f64>>, _>>()?;
        let Some(values) = values else {
            return Ok(None);
        };
        let values = values.into_iter().map(|value| value as u64).collect::<Vec<_>>();
        Ok(Some(match (backend, values.as_slice()) {
            (seismic::BackendName::Cuda, &[graph, launch, granule]) => seismic::GraphCost::Pooled { graph, launch, granule },
            (seismic::BackendName::Vulkan, &[launch, chunk]) => seismic::GraphCost::Chunked { launch, chunk },
            (seismic::BackendName::Cpu | seismic::BackendName::Metal, &[]) => seismic::GraphCost::None,
            (backend, _) => unreachable!("{backend:?}'s graph memory holds its terms"),
        }))
    }

    /// Record what `device`, opened with its backend's measurement, holds.
    pub fn record(device: &Device, store: &dyn RecordStore<PointKey>) -> Result<(), ExecutorCostsError> {
        let identity = executor_identity(device.topology(), device.info())?;
        let values = match device.graph_cost() {
            seismic::GraphCost::Pooled { graph, launch, granule } => vec![graph, launch, granule],
            seismic::GraphCost::Chunked { launch, chunk } => vec![launch, chunk],
            seismic::GraphCost::None => Vec::new(),
        };
        let error = failed("graph memory record");
        for (term, value) in Self::terms(device.backend()).iter().zip(values) {
            let timing = Timing { median: value as f64, interval: None, samples: 1 };
            store.store(&Self::key(&identity, term), &Record::Measured(timing)).map_err(|cause| error(&cause))?;
        }
        Ok(())
    }
}

/// A volume's stored curve, or the rule before it's choice: the current
/// ([`Curve::SCHEMA`](seismic_io::Curve::SCHEMA)), or the one before it,
/// which a load reads as it was while the job probes the current one, so
/// no load waits on the probe or reads meanwhile at the pre-probe bounds (a
/// 32 MiB depth for hints, where the earlier choice held GB10's to 1 MiB).
#[derive(Clone, Debug, PartialEq)]
pub enum StoredVolume {
    Current(seismic_io::Curve),
    Earlier(seismic_io::EarlierChoice),
}

/// How a volume is read (`seismic_io::Curve`), a requested point
/// (`PointRequest::Volume`) keyed by the volume and its urgent request:
/// the job probes it, and a load adopts what the records hold, so no load
/// reads or evicts a file to time it. Each point of the curve is three
/// terms (its rate, spread and cores, keyed by its read size and window),
/// each lone time two (its seconds and their error, keyed by its read
/// size), each point of its range line three (keyed by its depth in
/// ranges).
pub struct VolumeCosts;

impl VolumeCosts {
    /// The urgent request a [`PointRequest::Volume`](super::PointRequest) names.
    pub fn urgent(urgent: &Option<(Vec<u64>, u64)>) -> Option<seismic_io::Urgent> {
        urgent.as_ref().map(|(parts, records)| seismic_io::Urgent { parts: parts.clone(), records: *records })
    }

    /// A term's key under `schema`, at `at` (its read size and window, as
    /// the term has them): the rule's ([`Curve::SCHEMA`](seismic_io::Curve::SCHEMA)
    /// or [`EarlierChoice::SCHEMA`](seismic_io::EarlierChoice::SCHEMA)).
    fn key(identity: &seismic::Identity, file: &Path, urgent: Option<&seismic_io::Urgent>, schema: u64, term: &str, at: &[(&str, u64)]) -> Result<PointKey, ExecutorCostsError> {
        let volume = seismic::volume_identity(file).map_err(|cause| failed("volume identity")(&cause))?;
        let mut configuration = named(&[("volume_device", volume.device)]);
        configuration.insert(format!("volume_{}", volume.uuid.as_deref().unwrap_or("unnamed")), 1);
        configuration.insert("schema".into(), schema);
        if let Some(urgent) = urgent {
            configuration.insert("records".into(), urgent.records);
            configuration.extend(urgent.parts.iter().enumerate().map(|(part, &bytes)| (format!("part_{part}"), bytes)));
        }
        let mut point = named(&[(term, 1)]);
        point.extend(at.iter().map(|&(name, value)| (name.to_owned(), value)));
        Ok(PointKey { executor: identity.clone(), kernels: Vec::new(), configuration, point })
    }

    /// The curve `store` records of `file`'s volume, whole; else the rule
    /// before it's choice ([`StoredVolume`]); none until the job probed it.
    pub fn stored(
        topology: &DeviceTopology,
        executor: &Executor,
        file: &Path,
        urgent: Option<&seismic_io::Urgent>,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Option<StoredVolume>, ExecutorCostsError> {
        let identity = executor_identity(topology, executor)?;
        let term = |schema: u64, term: &str, at: &[(&str, u64)]| measured(store, &Self::key(&identity, file, urgent, schema, term, at)?, "volume record");
        let (grid, lone, depths) = seismic_io::Curve::grid(urgent);
        let schema = seismic_io::Curve::SCHEMA;
        let mut points = Vec::with_capacity(grid.len());
        for (read, window) in grid {
            let at = [("read", read), ("window", window)];
            let (Some(rate), Some(spread), Some(cores)) = (term(schema, "rate", &at)?, term(schema, "spread", &at)?, term(schema, "cores", &at)?) else {
                points.clear();
                break;
            };
            points.push(seismic_io::Point { read, window, rate, spread, cores });
        }
        let mut lones = Vec::with_capacity(lone.len());
        for read in lone {
            let at = [("read", read)];
            let (Some(seconds), Some(error)) = (term(schema, "lone", &at)?, term(schema, "lone_error", &at)?) else {
                points.clear();
                break;
            };
            lones.push(seismic_io::Lone { read, seconds, error });
        }
        // A range line ends at its first flat depth ("A volume's curve"):
        // its points are the depths stored before the first missing one.
        let mut ranges = Vec::with_capacity(depths.len());
        for depth in depths {
            let at = [("depth", depth)];
            let (Some(rate), Some(spread), Some(cores)) = (term(schema, "range_rate", &at)?, term(schema, "range_spread", &at)?, term(schema, "range_cores", &at)?) else {
                break;
            };
            ranges.push(seismic_io::Curve::range_point(urgent.expect("a range line is a request's"), depth, rate, spread, cores));
        }
        if urgent.is_some() && ranges.is_empty() {
            points.clear();
        }
        if !points.is_empty() {
            return Ok(Some(StoredVolume::Current(seismic_io::Curve { points, lone: lones, ranges })));
        }
        let schema = seismic_io::EarlierChoice::SCHEMA;
        let (window, demand, read, demand_read, cores) = (term(schema, "window", &[])?, term(schema, "demand", &[])?, term(schema, "read", &[])?, term(schema, "demand_read", &[])?, term(schema, "cores", &[])?);
        let (Some(window), Some(demand), Some(read), Some(demand_read), Some(cores)) = (window, demand, read, demand_read, cores) else {
            return Ok(None);
        };
        Ok(Some(StoredVolume::Earlier(seismic_io::EarlierChoice { window: window as u64, demand: demand as u64, read: read as u64, demand_read: demand_read as u64, cores })))
    }

    /// Probe `file`'s volume in the job, its curve stored.
    pub(super) fn measure(
        device: &Device,
        file: &Path,
        urgent: Option<&seismic_io::Urgent>,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<(), ExecutorCostsError> {
        let error = failed("volume probe");
        let identity = executor_identity(device.topology(), device.info())?;
        let source = magnitude_artifacts::FileSource::open(file).map_err(|cause| error(&cause))?;
        let curve = seismic_io::probe_volume(&source, urgent).map_err(|cause| error(&cause))?;
        let schema = seismic_io::Curve::SCHEMA;
        let points = curve.points.iter().flat_map(|point| {
            let at = vec![("read", point.read), ("window", point.window)];
            [("rate", point.rate), ("spread", point.spread), ("cores", point.cores)].map(|(term, value)| (term, at.clone(), value))
        });
        let lone = curve.lone.iter().flat_map(|lone| [("lone", lone.seconds), ("lone_error", lone.error)].map(|(term, value)| (term, vec![("read", lone.read)], value)));
        let ranges = curve.ranges.iter().flat_map(|point| {
            let at = vec![("depth", point.window / point.read)];
            [("range_rate", point.rate), ("range_spread", point.spread), ("range_cores", point.cores)].map(|(term, value)| (term, at.clone(), value))
        });
        for (term, at, value) in points.chain(lone).chain(ranges) {
            let timing = Timing { median: value, interval: None, samples: 1 };
            store.store(&Self::key(&identity, file, urgent, schema, term, &at)?, &Record::Measured(timing)).map_err(|cause| error(&cause))?;
        }
        Ok(())
    }
}

/// A layer's CPU site as its pool's lines depend on it (`CpuCosts`): its
/// record form and extents, nothing of a model's weights, so a site of any
/// model of this form reads the same lines. Every extent of a site derives
/// from it (`experts::SiteShape::of`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CpuForm {
    pub(crate) record: RecordTensors<RecordTensor>,
    pub(crate) choices: u64,
    pub(crate) experts: u64,
    /// The site's expansion function (`functions::activate`).
    pub(crate) function: i32,
    /// The site table's slots, and the choices of the decode form's widest
    /// launch it holds.
    pub(crate) flight: u64,
    pub(crate) widest: u64,
}

/// A CPU site's pool on this executor, a requested point: the workers it
/// sizes to and its lines there. Pool sizing and the lines are timed by the
/// job on records of the form, so no load times the CPU; a load opens its
/// pool at the stored workers, or the machine's widest, and its site stays
/// off (`SiteLines::Pending`) until the lines are stored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpuCosts {
    pub workers: usize,
    pub lines: super::cost::CpuLines,
}

impl CpuForm {
    /// The form of a site over `binding`, expanding by `function`.
    pub(crate) fn of(binding: &BankBinding, function: i32) -> Self {
        Self {
            record: binding.record,
            choices: binding.selected,
            experts: binding.experts,
            function,
            flight: binding.words.site().slots(),
            widest: binding.words.choices,
        }
    }

    fn key(&self, identity: &seismic::Identity, term: &str) -> PointKey {
        let mut configuration = named(&[
            ("cpu_site", 1),
            ("rows", magnitude_kernels::constants::DECODE_ROWS),
            ("choices", self.choices),
            ("experts", self.experts),
            ("hidden", self.record.down.matrix[0]),
            ("features", self.record.up.matrix[0]),
            ("function", u64::from(self.function as u32)),
            ("flight", self.flight),
            ("widest", self.widest),
        ]);
        for (name, tensor) in self.record.iter() {
            configuration.insert(format!("{name:?}_{}_n", tensor.resident.name()), tensor.matrix[0]);
            configuration.insert(format!("{name:?}_{}_k", tensor.resident.name()), tensor.matrix[1]);
        }
        PointKey { executor: identity.clone(), kernels: Vec::new(), configuration, point: named(&[(term, 1)]) }
    }
}

impl CpuCosts {
    /// What `store` records of `form`'s site on `executor`; none until the
    /// job measured it.
    pub(super) fn stored(
        topology: &DeviceTopology,
        executor: &Executor,
        form: &CpuForm,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Option<Self>, ExecutorCostsError> {
        use super::cost::{CpuLines, Nanoseconds, RowCost, RowLine};
        let identity = executor_identity(topology, executor)?;
        let error = failed("CPU site record");
        let term = |term: &str| measured(store, &form.key(&identity, term), "CPU site record");
        let (workers, one, handshake) = (term("workers")?, term("one")?, term("handshake")?);
        let (exact, at_most) = (term("row")?, term("row_at_most")?);
        let (Some(workers), Some(one), Some(handshake)) = (workers, one, handshake) else {
            return Ok(None);
        };
        let ns = |value: f64| Nanoseconds::of(value).map_err(|cause| error(&cause));
        let row = match (exact, at_most) {
            (Some(row), _) => RowCost::Measured(ns(row)?),
            (None, Some(row)) => RowCost::AtMost(ns(row)?),
            (None, None) => return Ok(None),
        };
        let lines = CpuLines { expert: RowLine { one: ns(one)?, row }, handshake: ns(handshake)? };
        Ok(Some(Self { workers: workers as usize, lines }))
    }

    /// Size `form`'s pool and measure its lines in the job, stored.
    pub(super) fn measure(device: &Device, form: &CpuForm, store: &dyn RecordStore<PointKey>) -> Result<(), ExecutorCostsError> {
        use super::cost::RowCost;
        let error = failed("CPU site");
        let identity = executor_identity(device.topology(), device.info())?;
        let (workers, lines) = crate::experts::measure_cpu_site(device, form).map_err(|cause| error(&cause))?;
        let seconds = |nanoseconds: f64| nanoseconds * 1e-9;
        let (row, row_term) = match lines.expert.row {
            RowCost::Measured(row) => (row, "row"),
            RowCost::AtMost(row) => (row, "row_at_most"),
        };
        let terms = [
            ("workers", workers as f64),
            ("one", seconds(lines.expert.one.get())),
            ("handshake", seconds(lines.handshake.get())),
            (row_term, seconds(row.get())),
        ];
        for (term, value) in terms {
            let timing = Timing { median: value, interval: None, samples: 1 };
            store.store(&form.key(&identity, term), &Record::Measured(timing)).map_err(|cause| error(&cause))?;
        }
        Ok(())
    }
}

/// Why the host memory's read line could not be had.
#[derive(Clone, Debug, PartialEq)]
pub enum MemoryError {
    Executor(ExecutorCostsError),
    Store(StoreError),
    Unobtainable { bytes: u64, reason: Reason },
    /// A sweep no record holds: the job has not measured it yet.
    Unrecorded { bytes: u64 },
    /// Sweeping the span took no longer than sweeping none: `(bytes,
    /// seconds)` each.
    Flat { points: [(u64, f64); 2] },
}

impl fmt::Display for MemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Executor(error) => write!(formatter, "{error}"),
            Self::Store(error) => write!(formatter, "a host memory record: {error}"),
            Self::Unobtainable { bytes, reason } => {
                write!(formatter, "a sweep of {bytes} bytes of host memory is unobtainable: {reason}")
            }
            Self::Unrecorded { bytes } => write!(formatter, "no record of a sweep of {bytes} bytes of host memory"),
            Self::Flat { points } => write!(formatter, "host memory sweeps of {points:?} (bytes, seconds) rise with no bytes"),
        }
    }
}

impl std::error::Error for MemoryError {}

impl From<ExecutorCostsError> for MemoryError {
    fn from(error: ExecutorCostsError) -> Self {
        Self::Executor(error)
    }
}

impl From<StoreError> for MemoryError {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}

/// Sweeps of host memory, one a pass: `readers` threads started together,
/// each reading its contiguous share of `words` once, then joined.
struct Sweeps<'a> {
    words: &'a [u64],
    readers: usize,
}

impl Sweeps<'_> {
    fn sweep(&self) {
        let length = self.words.len();
        std::thread::scope(|scope| {
            for reader in 0..self.readers {
                let share = &self.words[reader * length / self.readers..(reader + 1) * length / self.readers];
                scope.spawn(move || std::hint::black_box(share.iter().fold(0u64, |sum, &word| sum.wrapping_add(word))));
            }
        });
    }
}

impl Work for Sweeps<'_> {
    type Error = String;
    type Pending = Done;
    /// Host memory holds no clock a gauge raises.
    type Clock = ();

    fn submit(&self, passes: usize) -> Result<Done, String> {
        let started = Instant::now();
        for _ in 0..passes {
            self.sweep();
        }
        Ok(Done(started.elapsed().as_secs_f64()))
    }

    fn calls_per_pass(&self) -> usize {
        1
    }

    /// The span lies beyond every cache: every pass reads it from memory.
    fn submit_cold(&self, passes: usize) -> Result<Done, String> {
        self.submit(passes)
    }

    fn cold_calls_per_pass(&self) -> usize {
        1
    }

    fn clock(&self) -> Option<()> {
        None
    }

    fn gauge(&self) -> Result<Box<dyn Gauge<String>>, String> {
        Err("host memory sweeps hold no clock to gauge".into())
    }
}

/// The host memory's read line: host RAM read by every CPU the process may
/// run on at once, each thread a contiguous share of a span
/// [`CACHE_MULTIPLE`](super::executor_costs::CACHE_MULTIPLE) times the
/// largest cache that can serve host memory
/// ([`super::executor_costs::host_cache`]), so under the sweeps' cyclic
/// access its bytes come from memory, as a rotation's do. Sweeps of none and
/// of the whole span are timed by the measurement job and kept in the
/// record store, so the line is measured once per executor and reader
/// count; the slope through them is a byte's seconds,
/// starting and joining the readers its fixed cost, apart.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryCosts {
    pub byte_seconds: f64,
}

impl MemoryCosts {
    /// The span a sweep reads on `topology`'s host: [`CACHE_MULTIPLE`] times
    /// the largest cache that can serve host memory.
    fn span(topology: &DeviceTopology) -> u64 {
        super::executor_costs::CACHE_MULTIPLE * super::executor_costs::host_cache(topology)
    }

    /// The key of a sweep of `bytes` by `readers` threads on `topology`'s
    /// host, beside `executor` (whose identity names the hardware and
    /// driver), of a span `span`.
    fn key(identity: &seismic::Identity, readers: u64, span: u64, bytes: u64) -> PointKey {
        PointKey {
            executor: identity.clone(),
            kernels: Vec::new(),
            configuration: named(&[("host_memory", 1), ("readers", readers), ("span", span)]),
            point: named(&[("bytes", bytes)]),
        }
    }

    /// The host memory's read line beside `executor` with `readers` reading
    /// at once (a load's CPUs, `std::thread::available_parallelism` in its
    /// process) that `store`'s records give, measuring nothing; why there is
    /// none.
    pub fn stored(
        topology: &DeviceTopology,
        executor: &Executor,
        readers: u64,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Self, MemoryError> {
        let identity = executor_identity(topology, executor)?;
        let span = Self::span(topology);
        let [none, all] = [0, span].map(|bytes| {
            let key = Self::key(&identity, readers, span, bytes);
            let timing = match store.load(&key)? {
                Some(Record::Measured(timing)) => timing,
                Some(Record::Unobtainable { reason }) => return Err(MemoryError::Unobtainable { bytes, reason }),
                None => return Err(MemoryError::Unrecorded { bytes }),
            };
            Ok((bytes, timing.median))
        });
        let medians = [none?, all?];
        let byte_seconds = rising(medians).ok_or(MemoryError::Flat { points: medians })?.slope();
        Ok(Self { byte_seconds })
    }

    /// Measure the line's sweeps by `readers` threads in the measurement job
    /// (`walks`), every page of the span written once before it is timed
    /// (an untouched page reads as the one zero page).
    pub(super) fn measure(readers: u64, walks: &mut Walks<'_, '_>) {
        let span = Self::span(walks.device().topology());
        let identity = walks.identity();
        let words = OnceCell::new();
        for bytes in [0, span] {
            let words = &words;
            walks.time_host(Self::key(&identity, readers, span, bytes), move || {
                let words = words.get_or_init(|| (0..span / 8).collect::<Vec<u64>>());
                let readers = usize::try_from(readers).map_err(|_| Unformed::Never(ExecutorCostsError(format!("{readers} readers exceed the host"))))?;
                Ok(Sweeps { words: &words[..(bytes / 8) as usize], readers })
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::measure::{Attempt, Measure, Outcome, Stop};
    use seismic::Measurer;

    use crate::source_element;
    use magnitude_artifacts::gguf::Encoding;

    /// What a job's device holds per graph is what a load then reads, with
    /// nothing opened; nothing recorded reads as none.
    #[test]
    fn a_graph_cost_recorded_is_read_as_it_was_measured() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let topology = catalog.topology();
        let available = topology.executors().iter().filter(|info| matches!(info.availability(), seismic::Availability::Available));
        for info in available {
            let device = catalog.open(info.id()).unwrap_or_else(|error| panic!("an available executor opens: {error}"));
            let store = seismic::measure::HeldStore::default();
            let unrecorded = GraphMemory::stored(device.topology(), device.info(), &store).unwrap();
            GraphMemory::record(&device, &store).unwrap();
            let recorded = GraphMemory::stored(device.topology(), device.info(), &store).unwrap();
            assert_eq!(recorded, Some(device.graph_cost()));
            assert!(matches!(unrecorded, None | Some(seismic::GraphCost::None)), "{unrecorded:?}");
        }
    }

    /// A tier copy's points give the link's peak, what a discrete prefill's
    /// bar prices its copies by: a copy from pinned host memory on every
    /// block, near the link's own rate, far above a route's kernel rate. A
    /// CUDA or Vulkan host's measurement, run on one by name with `--ignored`.
    #[test]
    #[ignore]
    fn a_tier_copy_reads_the_links_ceiling() {
        let devices = seismic::Devices::discover(&[]).expect("devices are discovered");
        let backends = [seismic::BackendName::Cuda, seismic::BackendName::Vulkan]
            .into_iter()
            .filter(|&backend| crate::test_support::available(&devices, backend))
            .collect::<Vec<_>>();
        assert!(!backends.is_empty(), "the host has a CUDA or Vulkan executor");
        for backend in backends {
            let device = devices
                .open_backend(backend)
                .unwrap_or_else(|error| panic!("an available {backend:?} executor opens: {error}"));
            let store = seismic::measure::HeldStore::default();
            let mut measurer = Measurer::new(crate::SEARCH_SETTINGS.error_rate, crate::MIN_SAMPLE_SECONDS);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            let requests = [super::super::requests::PointRequest::TierCopy];
            super::super::requests::measure(&mut measurer, &device, &store, &requests, deadline, &mut |_| Attempt::Proceed, &mut |_, _| ()).unwrap();
            let costs = TierCopyCosts::stored(device.topology(), device.info(), &store).unwrap();
            let rate = 1. / costs.link_byte_seconds();
            eprintln!("{backend:?}: link ceiling {:.2} GB/s on {} blocks, a copy stalls {:.0}% of the work beside it", rate / 1e9, costs.blocks, costs.stall_share * 100.);
            assert!(rate > 5e9, "{backend:?}: {rate} bytes a second");
        }
    }

    /// A site's pool and lines measured at a form of no model are read as
    /// measured; none stands before they are.
    #[test]
    fn a_cpu_sites_points_are_read_as_measured() {
        let device = seismic::Devices::discover(&[]).unwrap().open_backend(seismic::BackendName::Cpu).unwrap();
        let tensor = |matrix| RecordTensor { source: Element::f32(), resident: Element::f32(), matrix };
        let form = CpuForm {
            record: RecordTensors { gate: None, up: tensor([32, 64]), down: tensor([64, 32]) },
            choices: 4,
            experts: 16,
            function: 0,
            flight: 4,
            widest: 4 * magnitude_kernels::constants::DECODE_ROWS,
        };
        let store = seismic::measure::HeldStore::default();
        assert_eq!(CpuCosts::stored(device.topology(), device.info(), &form, &store).unwrap(), None);
        CpuCosts::measure(&device, &form, &store).unwrap();
        let stored = CpuCosts::stored(device.topology(), device.info(), &form, &store).unwrap().expect("stored");
        assert!(stored.workers > 0 && stored.lines.expert.one.get() > 0., "{stored:?}");
    }

    /// The read line's bytes run at what the volume delivered over its busy
    /// time, its latency what requests took beyond that, none below zero;
    /// no line stands before a request read and the volume delivered; a
    /// window's sums give its own.
    #[test]
    fn the_read_line_is_the_drives_rate_and_the_requests_excess() {
        let mut reads = RequestReads::default();
        assert_eq!(reads.line(), Err(ReadLineError::Unread { requests: 0 }));
        reads.add(1_000, 1e-3);
        assert_eq!(reads.line(), Err(ReadLineError::Unread { requests: 1 }));
        reads.deliver(1_000_000, 0.5);
        let line = reads.line().unwrap();
        assert!((line.seconds_per_byte() - 5e-7).abs() < 1e-18);
        assert!((line.latency_seconds() - 5e-4).abs() < 1e-12);
        let first = reads;
        reads.add(1_000, 1e-6);
        assert_eq!(reads.since(&first).line(), Err(ReadLineError::Unread { requests: 1 }));
        assert_eq!(reads.since(&first).merge(&first).requests, 2);
        let mut quick = RequestReads::default();
        quick.add(1_000, 1e-6);
        quick.deliver(1_000, 1.);
        assert_eq!(quick.line().unwrap().latency_seconds(), 0.);
    }


    /// `routed_copy`'s device time at a prompt group's seed shape (Q4_K
    /// records of 589,824 bytes, copied as bytes): the seed copy of 256
    /// records into 512, each side the device's own memory or reserved
    /// device storage (a lent scratch's, the bank's records'), at a list for another step
    /// (copying nothing) and at 1, 16 and every entry, the list host-shared
    /// as the bank holds it, the copy on the grid `routed_list_grid` writes
    /// of its list, reading the list it stages, as a step launches it. It
    /// prints them. A CUDA host's profile, run on one by name with
    /// `--ignored`.
    #[test]
    #[ignore]
    fn the_seed_copy_at_a_prompt_groups_shape() {
        use magnitude_kernels::{routed_copy, routed_list_grid};
        let device = seismic::Devices::discover(&[])
            .expect("devices are discovered")
            .open_backend(seismic::BackendName::Cuda)
            .expect("a CUDA device opens");
        let byte = Element::bool();
        let per = 589_824u64;
        let formation = Formation::new(routed_copy::native_entry_with(routed_copy::Elements { U: byte }), &[]);
        let copy = formation.form(&device, None).unwrap();
        let grid_formation = Formation::new(routed_list_grid::native_entry(), &[]);
        let list_grid = grid_formation.form(&device, None).unwrap();
        let identity = executor_identity(device.topology(), device.info()).unwrap();
        let error_rate = crate::SEARCH_SETTINGS.error_rate;
        let how = Measure { stop: Stop::Resolved { error_rate }, min_sample_seconds: crate::MIN_SAMPLE_SECONDS };
        let step_words = (0..constants::STEP_WORDS).map(|word| i32::from(word == 0)).flat_map(i32::to_le_bytes).collect::<Vec<_>>();
        let step = Tensor::from_host(&device, Element::i32(), &[constants::STEP_WORDS], &step_words).unwrap();
        let (sources, destinations) = (256u64, 512u64);
        // Each side either the device's own memory or reserved as the bank
        // is, as a lent scratch and the bank's records are.
        let tensor = |reserved: bool, records: u64| match reserved {
            true => {
                let (tensor, resizer) =
                    crate::planning::BankAllocation::reserve(&device, crate::planning::BankTier::Bank.contract_on(&device), byte, &[records, 1, per], records, None).unwrap();
                (tensor, Some(resizer))
            }
            false => (Tensor::zeros(&device, byte, &[records, 1, per]).unwrap(), None),
        };
        for (source_reserved, destination_reserved) in [(false, true), (false, false), (true, false), (true, true)] {
            let (source, _source_resizer) = tensor(source_reserved, sources);
            let (destination, _resizer) = tensor(destination_reserved, destinations);
            let placed = |reserved: bool| if reserved { "reserved" } else { "device" };
            for (copied, sequence) in [(0u64, 0), (1, 1), (16, 1), (sources, 1)] {
                let words = magnitude_kernels::list_extent(sources);
                let mut list = vec![0i32; words as usize];
                list[constants::LIST_SEQUENCE as usize] = sequence;
                list[constants::LIST_COUNT as usize] = copied.max(1) as i32;
                for entry in 0..copied.max(1) {
                    let at = magnitude_kernels::list_extent(entry) as usize;
                    list[at + constants::LIST_FROM as usize] = entry as i32;
                    list[at + constants::LIST_TO as usize] = entry as i32;
                }
                let list_tensor = Tensor::shared(&device, Element::i32(), &[words]).unwrap();
                let bytes = list.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
                // SAFETY: no device work uses the list yet.
                unsafe { list_tensor.write_shared(0, &bytes).unwrap() };
                let point = || {
                    let mut graph = device.native_graph();
                    let port = |graph: &mut seismic::NativeGraph, element, extents: &[u64]| graph.port(element, extents);
                    let source_port = port(&mut graph, byte, &[sources, 1, per]).map_err(|cause| Unformed::Now(cause.to_string()))?;
                    let list_port = port(&mut graph, Element::i32(), &[words]).map_err(|cause| Unformed::Now(cause.to_string()))?;
                    let step_port = port(&mut graph, Element::i32(), &[constants::STEP_WORDS]).map_err(|cause| Unformed::Now(cause.to_string()))?;
                    let mut destination_port =
                        port(&mut graph, byte, &[destinations, 1, per]).map_err(|cause| Unformed::Now(cause.to_string()))?;
                    let mut grid_port = port(&mut graph, Element::i32(), &[3]).map_err(|cause| Unformed::Now(cause.to_string()))?;
                    let mut staged_port = port(&mut graph, Element::i32(), &[words]).map_err(|cause| Unformed::Now(cause.to_string()))?;
                    graph
                        .enqueue(
                            &list_grid,
                            routed_list_grid::WorkflowArgs {
                                list: list_port.tensor().into(),
                                step: step_port.tensor().into(),
                                grid: grid_port.tensor_mut().into(),
                                staged: staged_port.tensor_mut().into(),
                            },
                        )
                        .map_err(|cause| Unformed::Now(cause.to_string()))?;
                    graph
                        .enqueue(
                            &copy,
                            routed_copy::WorkflowArgs {
                                source: source_port.tensor().into(),
                                list: staged_port.tensor().into(),
                                step: step_port.tensor().into(),
                                destination: destination_port.tensor_mut().into(),
                                grid: grid_port.tensor().into(),
                            },
                        )
                        .map_err(|cause| Unformed::Now(cause.to_string()))?;
                    let plan = graph.seal().map_err(|cause| Unformed::Now(cause.to_string()))?;
                    let mut bindings = plan.bindings();
                    // A seed copy's blocks, the grid's first word, which
                    // its host writes (`ExpertBank::open`).
                    let blocks = (magnitude_kernels::LIST_BLOCKS as i32).to_le_bytes().into_iter().chain([0; 8]).collect::<Vec<_>>();
                    let grid = Tensor::from_host(&device, Element::i32(), &[3], &blocks).map_err(|cause| Unformed::Now(cause.to_string()))?;
                    let staged = Tensor::zeros(&device, Element::i32(), &[words]).map_err(|cause| Unformed::Now(cause.to_string()))?;
                    for (port, tensor) in [
                        (&source_port, &source),
                        (&list_port, &list_tensor),
                        (&step_port, &step),
                        (&destination_port, &destination),
                        (&grid_port, &grid),
                        (&staged_port, &staged),
                    ] {
                        bindings.set(port, tensor).map_err(|cause| Unformed::Now(cause.to_string()))?;
                    }
                    plan.point(bindings, GraphTiming::Device).map(Point::Graph).map_err(|cause| Unformed::Now(cause.to_string()))
                };
                let key = PointKey {
                    executor: identity.clone(),
                    kernels: vec![
                        grid_formation.identity(device.backend()).unwrap(),
                        formation.identity(device.backend()).unwrap(),
                    ],
                    configuration: named(&[("S", sources), ("D", destinations)]),
                    point: named(&[("copied", copied)]),
                };
                let store = seismic::measure::HeldStore::default();
                let mut measurer = Measurer::new(error_rate, crate::MIN_SAMPLE_SECONDS);
                let median = match measurer
                    .measure_all(vec![(key, point)], &how, seismic::measure::Deadline::new(None), &store, &mut |_| Attempt::Proceed)
                    .pop()
                {
                    Some((_, Outcome::Measured(timing))) => timing.median,
                    outcome => panic!("the copy was not measured: {outcome:?}"),
                };
                eprintln!(
                    "seed copy, {} {sources} -> {} {destinations}, {copied} records: {:.1} µs, {:.1} µs a record, {:.2} GB/s",
                    placed(source_reserved),
                    placed(destination_reserved),
                    median * 1e6,
                    if copied > 0 { median * 1e6 / copied as f64 } else { 0. },
                    copied as f64 * per as f64 / median / 1e9,
                );
            }
        }
    }

    /// A shrink's moves take each moving group's line at the records it
    /// moves, its launches' fixed cost once; a group moving none launches
    /// nothing.
    #[test]
    fn a_shrinks_moves_take_each_moving_groups_line() {
        let line = |fixed_seconds, record_seconds| RecordLine::new(fixed_seconds, record_seconds).unwrap();
        // Groups 0 and 2 can move; group 1 cannot.
        let lines = MoveLines(vec![(0, line(1e-4, 1e-6)), (2, line(2e-4, 3e-6))]);
        let moving = |moved: [u64; 3]| move |group: usize| moved[group];
        assert_eq!(lines.seconds(moving([0, 0, 0])), 0.);
        assert_eq!(lines.seconds(moving([10, 0, 0])), 1e-4 + 10. * 1e-6);
        assert_eq!(lines.seconds(moving([10, 0, 2])), (1e-4 + 10. * 1e-6) + (2e-4 + 2. * 3e-6));
    }

    /// A shrink's move line is measured at each form's own records and
    /// moves, on every native backend with an available executor, as the
    /// measurement job measures a load's request of it with nothing stored: it rises with the records
    /// moved, its fixed cost the launches', each form its own line, restored
    /// once measured. Every such backend opens and is measured, and the
    /// backends measured are printed.
    #[test]
    fn the_move_line_is_measured_at_the_banks_own_form() {
        let devices = seismic::Devices::discover(&[]).unwrap();
        let available = [seismic::BackendName::Metal, seismic::BackendName::Cuda, seismic::BackendName::Vulkan]
            .into_iter()
            .filter(|&backend| crate::test_support::available(&devices, backend))
            .collect::<Vec<_>>();
        let mut measured = Vec::new();
        for &backend in &available {
            let device = devices
                .open_backend(backend)
                .unwrap_or_else(|error| panic!("an available {backend:?} executor opens: {error}"));
            move_lines_are_measured_at_the_banks_own_form(&device);
            measured.push(backend);
        }
        eprintln!("move lines measured on {measured:?}");
        assert!(!measured.is_empty(), "the host has a native backend to measure on");
    }

    fn move_lines_are_measured_at_the_banks_own_form(device: &Device) {
        let records = std::env::temp_dir().join(format!("move-lines-{}-{:?}", std::process::id(), device.backend()));
        let store = seismic::measure::JsonFileStore::new(&records);
        let layout = crate::resident_layout(crate::ExecutionPath::Native, device.backend());
        let form = |[features, hidden]: [u64; 2], moves| {
            let tensor = |matrix: [u64; 2]| RecordTensor {
                source: source_element(Encoding::Q4K, matrix[1]).unwrap(),
                resident: crate::resident_element(
                    Encoding::Q4K,
                    magnitude_family_contracts::ScaledActivations::Rounded,
                    matrix[1],
                    seismic::DType::BF16,
                    layout,
                )
                .unwrap(),
                matrix,
            };
            MoveForm {
                record: RecordTensors {
                    gate: Some(tensor([features, hidden])),
                    up: tensor([features, hidden]),
                    down: tensor([hidden, features]),
                },
                moves,
            }
        };
        // Each form requested of the job, which measures its points, then
        // read from the records.
        let measured = |form: MoveForm| {
            let mut measurer = Measurer::new(crate::SEARCH_SETTINGS.error_rate, crate::MIN_SAMPLE_SECONDS);
            let deadline = Instant::now() + std::time::Duration::from_secs(120);
            let requests = [super::super::PointRequest::Moves(form)];
            let ended = super::super::requests::measure(&mut measurer, device, &store, &requests, deadline, &mut |_| Attempt::Proceed, &mut |_, _| {}).unwrap();
            assert!(ended.ended.is_empty() && ended.failures.is_empty(), "{ended:?}");
            MoveCosts::stored(device.topology(), device.info(), form, &store).unwrap()
        };
        // Q4_K rows of whole 256-value blocks.
        let (large, small) = (form([512, 2048], 32), form([256, 256], 32));
        let costs = [large, small].map(measured);
        for costs in &costs {
            assert!(costs.line.0.slope() > 0. && costs.line.0.fixed() > 0., "{:?}", costs.line);
        }
        assert_ne!(costs[0].line, costs[1].line, "each form has its own line");
        for (form, costs) in [large, small].into_iter().zip(&costs) {
            assert_eq!(MoveCosts::stored(device.topology(), device.info(), form, &store).unwrap().line, costs.line);
        }
        // A whole bank's shrink (here a million records' list, gigabytes
        // apiece were they all held) is measured at a move point's records:
        // its point holds only what it moves.
        let whole = form([512, 2048], 1 << 20);
        assert_eq!(whole.points(), [0, 32]);
        let whole = measured(whole);
        assert!(whole.line.0.slope() > 0., "{:?}", whole.line);
        std::fs::remove_dir_all(records).unwrap();
    }

    }
