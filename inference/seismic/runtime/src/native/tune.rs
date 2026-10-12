//! Native tuning serves no candidate its unit's numerical requirement has
//! not admitted against the unit's native default at every timed point. A
//! challenger takes its two tests, its matched rounds and its numerical
//! witnesses, in the order the unit's observations give ([`Sequence`]):
//! witnessed first, it is admitted in its first observed execution at each
//! timed point before it is sampled; timed first, it is sampled
//! unwitnessed and witnessed only when its rounds leave it ahead, a
//! challenger that could be served. The witnesses outside the timed points
//! are covered once, for a leader about to be served
//! ([`Stage::Promote`]). Host comparison remains outside the device
//! interval.
//! A search and a survey use the same evidence. Timing normalization and
//! reuse do not confer numerical eligibility.
//!
//! A unit serves a legal choice from its mandatory work on ([`UnitSearch`]):
//! its canonical defaults, replaced only by a challenger whose lead over it
//! was resolved in matched rounds ([`super::sampling`]) and that was
//! covered. Its optional work comes in
//! pieces, each priced before it starts from this load's observations of
//! its operations' classes ([`Quote`], [`Footprint`]). The consumer owns
//! admission and the original cutoff. Runtime checks that cutoff between
//! submissions and witnesses, retaining completed evidence when work is
//! interrupted. Expiry is neither a numerical nor an execution rejection.

use super::plan::{self, PointShape};
use super::deadline::{Deadline, WorkError};
use super::sampling::{Rates, Sequence, Shuffle, Tally, Verdict};
use super::search::{Cost, ParameterValues, PointKey, SearchSettings};
use super::timing::{self, OutputPool, PointPrices, PointTiming};
pub use super::timing::{
    Footprint, Linear, Observed, Quote, Raise, RaiseOutcome, SustainedClock,
};
pub use seismic_native_target::process::{return_freed_pages, ProcessMemory};
pub use seismic_native_target::workers::{Narrowed, Workers};
pub use super::admission::{
    Constraint, GOVERNING_STEP, GoverningStep, NumericalObservation, NumericalRequirement,
    Reselection, Violation, admit, governing_ratio, reselect,
};
pub use super::comparison::{
    COMPARISON_VERSION, Difference, StepPower, SubjectComparison, SubjectMeasurement,
};
use super::validation::{PreparedPoint, Validator};
use super::{
    FormationClass, HeldPrograms, MeasureOptions, NativePrepared, Rendered, median, median_of,
};
use crate::api::device::DeviceInner;
use crate::api::kernel::{EncodedArgs, PrepareError};
use crate::api::{CallError, TensorError};
use seismic_compiler::prepared::{InvocationContract, validate_invocation};
use seismic_lang::checked::{CheckedModule, NativeImplementation, NativeSpecialization};
use seismic_lang::entry::{ElementBindings, LogicalEntry, ParameterKind, TensorAccess};
use seismic_lang::expr::SymbolValue;
use seismic_lang::ids::EntryId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::cpu::CpuNativeKernels;

/// Restores a point's `&mut` tensors to their initial contents.
pub type TuningInitializer<'a> = Box<dyn FnMut() -> Result<(), TensorError> + 'a>;

/// One workload the tuned implementation serves, described before its
/// inputs exist: they are built only when the tuner first needs the point.
#[derive(Clone, Debug)]
pub struct PointSpec {
    pub label: String,
    /// Invocation demand for this workload; objective and allocation use
    /// demand times actual elapsed service, not independently normalized classes.
    pub weight: f64,
    /// Served-region identity for discovery and keyed reuse. Region labels
    /// do not renormalize the global expected elapsed-service objective.
    pub class: Option<String>,
    /// The point's cost relative to the unit's other points. Points come in
    /// ascending cost; the tuner predicts a point's time from the previous
    /// one's by the ratio of their costs.
    pub cost: f64,
    /// The census measures the defaults here: the point's time stands for
    /// its workload when the consumer divides its tuning time among units.
    /// This discovery membership is independent of the full numerical contract.
    /// A zero-demand bootstrap point may be selected to price later discovery.
    pub census: bool,
}

fn discovery_selected(specs: &[PointSpec], index: usize) -> bool {
    if specs.iter().any(|spec| spec.census) {
        specs[index].census
    } else {
        specs[index].weight > 0.0
    }
}

/// Which numerical witnesses represent the performance workload. Distinct
/// adversarial witnesses do not imply additional served invocations; genuine
/// layer/weight rotations can explicitly retain every witness in the workload.
#[derive(Clone, Debug)]
pub enum WorkloadRotation {
    All,
    Selected(Vec<usize>),
}

impl WorkloadRotation {
    fn indices(&self, witnesses: usize) -> Result<Vec<usize>, TuneError> {
        let indices = match self {
            Self::All => (0..witnesses).collect(),
            Self::Selected(indices) => indices.clone(),
        };
        let unique = indices.iter().copied().collect::<BTreeSet<_>>();
        if indices.is_empty()
            || unique.len() != indices.len()
            || indices.iter().any(|index| *index >= witnesses)
        {
            return Err(TuneError::Declaration(
                "performance workload must select nonempty, distinct numerical witness indices"
                    .into(),
            ));
        }
        Ok(indices)
    }
}

/// A point's inputs.
pub struct PointInputs<'a> {
    /// Complete argument sets required by numerical validation.
    pub rotation: Vec<EncodedArgs>,
    /// The declared performance workload, independent of adversarial witnesses.
    pub workload: WorkloadRotation,
    /// Required when the entry has `&mut` parameters: called before every
    /// reference and validated invocation, it restores all writable tensors
    /// in every rotation. Timed passes after validation run without it.
    pub initialize: Option<TuningInitializer<'a>>,
    /// The leading-axis rows each named `&mut` parameter's entry writes. A
    /// `&mut` parameter absent here is observed whole, as written
    /// everywhere.
    pub written: BTreeMap<String, WrittenRows>,
}

/// The leading-axis rows of a `&mut` parameter an entry writes at a point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrittenRows {
    pub rows: Range<u64>,
    /// Observe the parameter's other rows too. The entry may not write
    /// them: a candidate must leave them the default's bit for bit. They
    /// are not read otherwise.
    pub observe_rest: bool,
}

/// The points of one tuning unit, whose inputs are built on request and
/// held by the tuner only while its work uses them.
pub trait PointSource<'a> {
    /// The tuner gave up the inputs it was built for `point`: drop what the
    /// source kept of them. A later [`PointSource::build`] builds them again.
    fn release(&mut self, _point: usize) {}
    /// Every point, in ascending cost.
    fn points(&self) -> Vec<PointSpec>;
    /// The dimensions of calls the unit's configuration serves at shapes its
    /// points' inputs do not take (a point's rows over more history segments
    /// than its inputs hold), each naming every dimension its launches read.
    /// A configuration is formed for all of them, so one whose launches at
    /// any exceed the formed functions' or the device's limits is excluded
    /// as it is formed, before it is measured or chosen; the defaults must
    /// launch at every one.
    fn served(&self) -> Vec<BTreeMap<String, u64>> {
        Vec::new()
    }
    /// Build point `point`'s inputs, or say why they cannot be built.
    fn build(&mut self, point: usize) -> Result<PointInputs<'a>, String>;
}

/// A point with its inputs built.
pub(crate) struct TuningPoint<'a> {
    pub label: String,
    /// Invocation demand for this workload; objective and allocation use
    /// demand times actual elapsed service, not independently normalized classes.
    pub weight: f64,
    /// Served-region identity for discovery and keyed reuse. Region labels
    /// do not renormalize the global expected elapsed-service objective.
    pub class: Option<String>,
    pub rotation: Vec<EncodedArgs>,
    pub(super) workload: Vec<usize>,
    pub initialize: Option<TuningInitializer<'a>>,
    pub written: BTreeMap<String, WrittenRows>,
}

impl<'a> TuningPoint<'a> {
    fn new(spec: &PointSpec, inputs: PointInputs<'a>) -> Result<Self, TuneError> {
        if inputs.rotation.is_empty() {
            return Err(TuneError::NoPoints);
        }
        let workload = inputs.workload.indices(inputs.rotation.len())?;
        Ok(Self {
            label: spec.label.clone(),
            weight: spec.weight,
            class: spec.class.clone(),
            rotation: inputs.rotation,
            workload,
            initialize: inputs.initialize,
            written: inputs.written,
        })
    }
}

/// A configuration as recorded: its static values and parameter values.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Configuration {
    pub statics: BTreeMap<String, u64>,
    pub params: BTreeMap<String, u64>,
    /// Launch-local choices, indexed by launch ordinal.
    #[serde(default)]
    pub launches: Vec<BTreeMap<String, u64>>,
}

impl Configuration {
    /// `specialization` as a record states it.
    pub fn of(specialization: &NativeSpecialization) -> Self {
        Self {
            statics: specialization.statics().clone(),
            params: specialization.params().clone(),
            launches: {
                let mut launches = Vec::new();
                for ((ordinal, name), value) in specialization.launch_params() {
                    launches.resize_with(ordinal + 1, BTreeMap::new);
                    launches[*ordinal].insert(name.clone(), *value);
                }
                launches
            },
        }
    }

    pub fn specialization(&self) -> NativeSpecialization {
        let mut specialization = NativeSpecialization::new();
        for (name, value) in &self.statics {
            specialization = specialization.with_static(name.clone(), *value);
        }
        for (name, value) in &self.params {
            specialization = specialization.with_param(name.clone(), *value);
        }
        for (ordinal, launch) in self.launches.iter().enumerate() {
            for (name, value) in launch {
                specialization = specialization.with_launch_param(ordinal, name.clone(), *value);
            }
        }
        specialization
    }
}

/// Why a configuration was not chosen.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Exclusion {
    /// Formation failed: an authoring defect (compile error) or toolchain
    /// failure.
    Formation(String),
    /// The device rejected a call, for example a launch beyond its limits.
    Execution(String),
    /// Results or writable state disagree with the selected reference.
    Validation { point: String, detail: String },
    /// Measuring the configuration at a point failed.
    Measurement { point: String, detail: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointMeasurement {
    pub point: String,
    /// What the measurement depends on; configurations with the same key at
    /// a point share it.
    pub key: PointKey,
    /// The time of one call there. For the defaults, the median of their
    /// samples. For a challenger of a search, its time in the unit's
    /// canonical units: the median over its rounds of its sample's ratio
    /// to the served choice's sample of the same round, times the served
    /// choice's recorded time, and so through every choice served back to
    /// the defaults. Every recorded time of one unit is thereby comparable
    /// with every other, whichever choice each was measured beside and
    /// whatever clock the device ran at then; each choice between a
    /// configuration and the defaults adds the spread of one median of
    /// ratios to that comparison. The deviation is in the same units.
    pub median_seconds: f64,
    pub deviation_seconds: f64,
    /// The device seconds per call of every sample taken, as the device
    /// read them in that session: not in canonical units.
    pub samples: Vec<f64>,
    pub repetitions: usize,
    pub rotation_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Outcome {
    Measured {
        /// Candidate measurements at every point measured.
        points: Vec<PointMeasurement>,
        /// A nominal leader's measurement in the last session that gave
        /// its comparison with the served choice more rounds, which is
        /// then also `points` (empty for every other configuration, a
        /// challenger whose first session resolved its lead among them).
        confirmed: Vec<PointMeasurement>,
    },
    Excluded(Exclusion),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfigurationRecord {
    pub configuration: Configuration,
    /// The artifact formed for it, stated once; none when none was formed.
    pub artifact: Option<String>,
    /// Its measurements against the default: the observation at every
    /// point it was held to, admitted or not, as indices into
    /// [`TuningResult::observations`]. None for the default, which is the
    /// reference. Whether a requirement admits it is a function of these
    /// ([`TuningResult::admitted`]), recorded nowhere.
    pub observations: Vec<usize>,
    pub outcome: Outcome,
    /// How the search formed it before it was first measured; none for the
    /// defaults, which the mandatory work forms, and for a survey's
    /// configurations.
    #[serde(default)]
    pub formation: Option<FormationRecord>,
}

/// How a search formed a configuration before it first measured it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FormationRecord {
    /// What the formation calls made for its piece took: the compile of
    /// what it runs that no held program had. A call is shared by every
    /// program it forms, so these are the seconds of all `programs`, not
    /// this configuration's part of them. Nothing when everything it runs
    /// was held.
    pub seconds: f64,
    /// The programs those calls formed (launch variants, or whole
    /// programs), its own among them and those of the configurations
    /// measured with it and next.
    pub programs: usize,
    /// No compiler ran for it: every program it runs was held by the
    /// search, live on the device or kept by the artifact store.
    pub reused: bool,
}

/// A tuning parameter as the tuner saw it: its declared values, the first
/// being the default.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclaredParameter {
    pub name: String,
    #[serde(default)]
    pub launch: Option<usize>,
    pub arithmetic: bool,
    pub form: bool,
    pub values: Vec<u64>,
}

/// How the recorded configurations were reached.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TuningMethod {
    /// The defaults, measured at the points discovered so far.
    Census,
    /// Every admissible configuration, `samples` per point.
    Survey { samples: usize },
    /// A search of each independent group of the planned domain
    /// ([`GroupSearch`]): the groups of launches that run together, or the
    /// one group that is a program formed whole. `candidates` are the
    /// configurations the search asked the declared domain for, which is
    /// never produced whole: every base, nearby change, form
    /// representative, leader's neighbour and generated combination.
    /// `complete` when all of them were measured and the order asks for
    /// none more: every form's leader has every neighbour and every form
    /// one parameter from it measured.
    Search {
        groups: usize,
        candidates: usize,
        complete: bool,
    },
}

/// Where tuning time went.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TuningTime {
    /// Building the points' inputs.
    pub building_seconds: f64,
    /// Case fingerprinting and reference preparation/execution.
    pub reference_seconds: f64,
    pub forming_seconds: f64,
    pub measuring_seconds: f64,
    pub validating_seconds: f64,
    /// Untimed full-scope validation execution, excluding separately recorded references.
    #[serde(default)]
    pub scope_validation_seconds: f64,
}

/// A tuning point as recorded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointRecord {
    pub label: String,
    pub weight: f64,
    #[serde(default)]
    pub class: Option<String>,
}

/// Terminal serving authority, distinct from numerical witness evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TuningDisposition {
    QualifiedCandidate,
    /// The formed declaration default remains usable when the first allowance
    /// cannot qualify a candidate. This never supplies fidelity or service evidence.
    CanonicalFallback {
        artifact: String,
    },
}

/// The complete, serializable outcome of tuning one implementation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuningResult {
    pub disposition: TuningDisposition,
    pub tuning_identity: String,
    pub entry: String,
    pub backend: String,
    /// Points in the order given, with their weights and classes.
    pub points: Vec<PointRecord>,
    /// What `overall` was chosen under.
    pub requirement: NumericalRequirement,
    /// What every observation is a measurement of besides its point and
    /// key: the comparison's version, the default's artifact, the entry and
    /// the device. No requirement's α is part of it.
    pub measurement_identity: String,
    /// Every observation made of the unit's candidates, each made once, in
    /// the order made. A candidate's record names its own.
    pub observations: Vec<NumericalObservation>,
    /// Every choice the unit's search served, in turn: its canonical
    /// default, then each promotion. A promotion was timed, confirmed
    /// faster than the choice before it and covered at every point; no
    /// other candidate has that evidence.
    pub served: Vec<Configuration>,
    /// Set when the record is served under a requirement other than the
    /// one it was searched under ([`reselect`]).
    pub reselection: Option<Reselection>,
    pub implementation_identity: String,
    /// The parameters tuned, in declaration order.
    pub parameters: Vec<DeclaredParameter>,
    /// Every configuration reached, in the order reached.
    pub configurations: Vec<ConfigurationRecord>,
    /// The chosen configuration, for the consumer to prepare.
    pub overall: Configuration,
    pub method: TuningMethod,
    pub time: TuningTime,
    /// The coverage checks of this unit's leaders at the untimed served
    /// points; `None` when no leader reached one.
    #[serde(default)]
    pub coverage: Option<CoverageCheck>,
}

/// How the coverage checks of a unit's leaders stand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageOutcome {
    /// The served choice passed its check.
    Passed,
    /// Every leader checked failed: the choice before them is served.
    Failed,
}

/// The coverage checks of a unit's leaders: how they stand, and how many
/// leaders failed one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoverageCheck {
    pub outcome: CoverageOutcome,
    pub failed: usize,
}

impl TuningResult {
    /// Whether the record's requirement admits `record`'s measurements at
    /// every point it was held to, or the first constraint it violates.
    pub fn admitted(&self, record: &ConfigurationRecord) -> Result<(), Violation> {
        record
            .observations
            .iter()
            .try_for_each(|observation| admit(&self.observations[*observation], &self.requirement))
    }

    /// Configurations that could not be formed or that the requirement
    /// does not admit. Exceeding the caller's requirement does not itself
    /// prove a kernel defect.
    pub fn rejections(&self) -> impl Iterator<Item = &ConfigurationRecord> {
        self.configurations.iter().filter(|record| {
            matches!(record.outcome, Outcome::Excluded(Exclusion::Formation(_)))
                || self.admitted(record).is_err()
        })
    }
}

#[derive(Debug)]
pub enum TuneError {
    /// The entry has no native implementation for the device's backend, or
    /// the static values are incomplete.
    Declaration(String),
    NoPoints,
    Reference(String),
    NoValidatedCandidate(Vec<Exclusion>),
    /// A survey's domain override names an undeclared parameter or moves its
    /// default.
    Domain(String),
    /// The all-defaults configuration could not be formed, run or measured;
    /// it is the search's start and the validation reference.
    DefaultUnusable(Exclusion),
    /// The entry writes `parameter` in place, and `point` binds the same
    /// tensor for every configuration without an initializer to restore it.
    SharedMutableState {
        point: String,
        parameter: String,
    },
    /// A point's initializer failed.
    Initialization {
        point: String,
        detail: String,
    },
    /// Building a point's inputs failed.
    Inputs {
        point: String,
        detail: String,
    },
    /// The chosen artifact could not be retained for serving without forming it again.
    Publication(String),
}

impl std::fmt::Display for TuneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Declaration(message)
            | Self::Domain(message)
            | Self::Reference(message)
            | Self::Publication(message) => {
                write!(f, "{message}")
            }
            Self::NoValidatedCandidate(failures) => write!(
                f,
                "no candidate passed numerical validation and measurement: {failures:?}"
            ),
            Self::NoPoints => f.write_str("tuning needs at least one point"),
            Self::DefaultUnusable(exclusion) => {
                write!(
                    f,
                    "the all-defaults configuration is unusable: {exclusion:?}"
                )
            }
            Self::SharedMutableState { point, parameter } => write!(
                f,
                "point `{point}` binds `&mut {parameter}` across configurations without an initializer"
            ),
            Self::Initialization { point, detail } => {
                write!(f, "initializing point `{point}` failed: {detail}")
            }
            Self::Inputs { point, detail } => {
                write!(f, "building the inputs of point `{point}` failed: {detail}")
            }
        }
    }
}

impl std::error::Error for TuneError {}

/// A time-bounded search (production).
#[derive(Clone, Debug)]
pub struct SearchPlan {
    /// The unit's budget: work is started only while its whole price fits
    /// what remains of it ([`Quote::fits`]). Work begun runs to its end.
    pub allowance: Duration,
    pub settings: SearchSettings,
    /// Minimum device time of one sample; sets repetitions per sample.
    pub min_sample_seconds: f64,
    /// Consumer hints for the search: values of entry parameters. An
    /// admissible hint with another value of a form parameter is that
    /// form's representative, in place of the nearest configuration of the
    /// form; the other hints are measured with the representatives.
    /// Inadmissible hints are skipped.
    pub start: Vec<ParameterValues>,
}

/// Every admissible configuration, measured and validated (development).
/// What is surveyed is what the caller names: the declared domain with
/// `domains` in place of the value lists they name. The admissible
/// configurations are visited in the declaration's order, a batch at a
/// time, and never held: what a survey holds is the record it returns, one
/// entry for each configuration the caller asked for.
#[derive(Clone, Debug)]
pub struct SurveyPlan {
    /// Samples per point of every configuration.
    pub samples: usize,
    pub min_sample_seconds: f64,
    /// Replacement value lists for some parameters, widening the declared
    /// space. Each keeps its declared default first.
    pub domains: BTreeMap<String, Vec<u64>>,
}

#[derive(Clone, Debug)]
pub enum Strategy {
    Search(SearchPlan),
    Survey(SurveyPlan),
}

/// How a [`UnitSearch`] measures and searches.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitPlan {
    pub settings: SearchSettings,
    /// Minimum device time of one sample; sets repetitions per sample.
    pub min_sample_seconds: f64,
    /// Consumer hints, as [`SearchPlan::start`].
    pub start: Vec<ParameterValues>,
}

/// A unit's next piece of work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Build one point's inputs and measure the defaults there.
    Discovery,
    /// Form the first representative of each structurally different form
    /// and measure and admit it at the timed points: a measurement like
    /// any other, of what no refinement inside a form can stand for. A
    /// consumer starts it ahead of every refinement.
    Form,
    /// Form a challenger and measure and admit it at the timed points, in
    /// rounds matched with the served choice ([`super::sampling`]).
    Measure,
    /// Convert a resolved lead into the served choice: coverage of the
    /// untimed witnesses, promotion. Its matched rounds beside the served
    /// choice have resolved it; it samples nothing more.
    Promote,
    /// Give a lead that is only nominal more matched rounds beside the
    /// served choice: the fewest in which it can resolve. Resolved as a
    /// lead, it is converted at once; resolved as a trail, it is dropped;
    /// unresolved, it stays a nominal lead. It is no budget already spent
    /// with nothing to show; it competes by value with the challengers
    /// still to measure.
    Confirm,
}

/// A whole piece of work that yields value, priced in full before it
/// starts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Work {
    pub stage: Stage,
    pub quote: Quote,
    /// The part of `quote` that is the work's compile: it runs off the
    /// device, so a caller may run it ahead of the rest while the device
    /// does other work ([`UnitSearch::forming`]).
    pub forming: Quote,
    /// The share of the unit's default service the work is expected to
    /// recover. A lead ([`Stage::Promote`]): what it measured. A challenger
    /// ([`Stage::Measure`]): what the unit still costs, times the chance
    /// that one more measured configuration is the fastest of them all,
    /// one in their number. Discovery recovers nothing itself.
    pub gain: f64,
    /// The bytes the work adds, at its most, to what is held when it
    /// starts: the inputs it has built that are not resident, and what its
    /// own operations hold while they run.
    pub footprint: Footprint,
}

/// What the load holds, as work is priced and admitted in bytes: the
/// process's `footprint` and, where a driver holds the device's
/// allocations outside the process (CUDA, Vulkan), the device's charges
/// `charged` beside it. The footprint holds what no ledger counts: the
/// heap, live and freed and not yet returned, with everything a toolchain
/// or a framework keeps (rendered sources and compiles, prepared kernels,
/// observations and records). A CPU device's allocations are the process's
/// own memory and a Metal device's are wired into its footprint, so there
/// the footprint is everything. Where the system reports no footprint,
/// what is known is the device's charges and `retained`, the bytes of the
/// unit's reference observations.
///
/// The footprint is the process's, not one unit's: what another unit's
/// compile holds while this unit's piece runs is counted with the piece.
/// Every byte is counted with exactly one piece that way (the pieces run
/// one after another on the device's thread), so the classes' prices sum to
/// what the load's work adds, whichever piece each byte is read with.
fn held_bytes(device: &DeviceInner, charged: u64, retained: u64, footprint: Option<u64>) -> u64 {
    match footprint {
        Some(footprint) if apart(device) => charged + footprint,
        Some(footprint) => footprint,
        None => charged + retained,
    }
}

/// Build point `index`'s inputs on `device`. A build the device's enforced
/// memory limit refused is not the unit's failure: its piece was admitted
/// on a footprint below what the build demands, which an unbuilt point's
/// price only forecasts. The point's price (`resident`) is raised to what
/// the build demanded of the ledger, what it had charged when refused with
/// what it asked for, and `None` ends the piece. Offered again at that
/// price, it is admitted as any priced piece: once enough is free, so the
/// same allocation is not refused again, or never, its unit then stopped
/// for want of room.
fn build_point<'a>(
    device: &DeviceInner,
    source: &mut dyn PointSource<'a>,
    spec: &PointSpec,
    index: usize,
    resident: &mut Observed,
) -> Result<Option<PointInputs<'a>>, TuneError> {
    let before = device.memory_usage();
    match source.build(index) {
        Ok(inputs) => Ok(Some(inputs)),
        Err(detail) => match device.memory_usage().refused_demand {
            Some(demand) if Some(demand) > before.refused_demand => {
                resident.observe(demand.saturating_sub(before.charged) as f64);
                Ok(None)
            }
            _ => Err(TuneError::Inputs {
                point: spec.label.clone(),
                detail,
            }),
        },
    }
}

/// What a class of operation is taken to add to the bytes held when a
/// piece is admitted: the most one of its operations has added. A piece is
/// started only if its bytes fit what is free, and half of a class's
/// operations add more than its mean: admitted on the mean, they would
/// pass what admitted them. None before the class was observed.
fn most_bytes(class: &Observed) -> Option<f64> {
    class.price().map(|_| class.longest())
}

/// Whether a driver holds `device`'s allocations outside the process's
/// footprint.
fn apart(device: &DeviceInner) -> bool {
    match &device.kind {
        super::OpenedKind::Cuda(_) => true,
        #[cfg(not(target_os = "macos"))]
        super::OpenedKind::Vulkan(_) => true,
        _ => false,
    }
}

/// The most the process's footprint is known to have been over a piece of
/// work on `device`: the most of its `readings`, and, where the device's
/// allocations are in the footprint, no less than the footprint when the
/// piece `began` with what the ledgers rose by within the piece (`rise`:
/// the device's charges and the reference observations, each at its
/// high-water mark over what it was when the piece began). Every byte the
/// ledgers count is footprint there, so a piece that placed and released
/// device memory between two readings held at least that.
fn footprint_peak(
    device: &DeviceInner,
    began: Option<u64>,
    readings: Option<u64>,
    rise: u64,
) -> Option<u64> {
    if apart(device) {
        return readings;
    }
    readings
        .zip(began)
        .map(|(readings, began)| readings.max(began.saturating_add(rise)))
}

/// What was held over one piece of work ([`held_bytes`]) when the piece
/// began, at its most while it ran, and when it ended.
///
/// The most is exact for what Seismic's ledgers count, the device's
/// charges and the reference observations: their high-water marks are
/// started again with each piece. Where the device is apart the device's
/// mark stands beside the footprint; where its allocations are in the
/// footprint, the most is no less than the footprint when the piece began
/// with what the ledgers rose by within it ([`footprint_peak`]).
///
/// The footprint itself is read when the piece began, when its inputs were
/// resident and when it ended. What is not seen is what no ledger counts
/// and the process held and gave back between those readings: host heap of
/// the piece's own operations and of compiles running beside it, and what
/// a framework holds for a submission. The system keeps no high-water mark
/// of the footprint that can be started again; its mark for the life of
/// the process says so only for a piece that sets it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PieceMemory {
    pub began_bytes: u64,
    pub peak_bytes: u64,
    pub ended_bytes: u64,
    /// The most Seismic's own ledgers counted over the piece: the device's
    /// high-water mark and the reference observations'. What a consumer
    /// holds to a ceiling on the device's charges; the fields above are
    /// everything held, which on a device whose allocations are in the
    /// process is the process's footprint, from before the load's tuning
    /// began.
    pub ledger_peak_bytes: u64,
}

impl PieceMemory {
    /// What the piece added at its most to what was held when it began.
    pub fn added(&self) -> u64 {
        self.peak_bytes.saturating_sub(self.began_bytes)
    }
}

/// Wall seconds by operation class: compiles, building points' inputs,
/// placing kernels and restoring state, observed invocations with their
/// readback and comparison, warming the device, timed submissions, and
/// whole checks of untimed points.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Spent {
    pub compiling: f64,
    pub building: f64,
    pub placing: f64,
    pub witnessing: f64,
    pub warming: f64,
    pub sampling: f64,
    pub checking: f64,
    /// Planning the next piece: the search's requests of its declared
    /// domain.
    pub planning: f64,
}

impl Spent {
    fn since(self, earlier: Spent) -> Spent {
        Spent {
            compiling: self.compiling - earlier.compiling,
            building: self.building - earlier.building,
            placing: self.placing - earlier.placing,
            witnessing: self.witnessing - earlier.witnessing,
            warming: self.warming - earlier.warming,
            sampling: self.sampling - earlier.sampling,
            checking: self.checking - earlier.checking,
            planning: self.planning - earlier.planning,
        }
    }
}

/// A piece of work completed or interrupted at the consumer's cutoff.
#[derive(Clone, Debug, PartialEq)]
pub struct Performed {
    pub stage: Stage,
    pub quote: Quote,
    pub actual: Duration,
    /// Optional work stopped at the consumer's cutoff; completed evidence
    /// remains, and no incomplete witness becomes a rejection.
    pub interrupted: bool,
    /// Where `actual` went, by operation class; the rest is host work
    /// between operations.
    pub spent: Spent,
    /// What each raise of the device's clock did within the piece, and
    /// each establishment of its gauge, in order ([`timing::Clock`]).
    pub raises: Vec<Raise>,
    /// The served choice changed. The unit's record changed either way: a
    /// measurement, a verdict or the choice.
    pub promoted: bool,
    /// The footprint the piece was quoted, and what was held over it.
    pub footprint: Footprint,
    pub memory: PieceMemory,
}

/// Where a unit's search stands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Standing {
    /// The cost of the served choice relative to the defaults: 1 until a
    /// challenger is promoted.
    pub cost: f64,
    /// Configurations measured so far.
    pub measured: usize,
    /// Configurations the search has asked its domain for, the defaults
    /// among them: all of the domain that exists, whichever way the
    /// implementation's programs are formed.
    pub generated: usize,
    /// Requests the search's sweeps have made of its declared domain: one
    /// for each assignment a sweep asked for, admitted or not. A sweep
    /// asks for an assignment once, so this follows `generated`, not the
    /// pieces planned times what was measured before each.
    pub requests: usize,
    /// Configurations the declaration's value lists name: the product of
    /// its parameters' value counts, which `where` can only reduce. None
    /// before the search is planned. The admissible ones are not counted
    /// before the search has asked for them all: once `exhausted`,
    /// `generated` is their number.
    pub declared: Option<u128>,
    /// The search has asked for everything its order reaches and measured
    /// it ([`GroupSearch::exhausted`]): the one way a search ends before
    /// its load's budget does.
    pub exhausted: bool,
    /// Programs the search formed so far, with `forming_seconds` what
    /// forming them took: launch variants where launches are formed
    /// separately, configurations where the program is formed whole. The
    /// defaults' programs, which the census formed as preparation, are not
    /// among them.
    pub programs: usize,
    /// What forming programs has taken so far.
    pub forming_seconds: f64,
}

pub struct TuneRequest<'r, 'a> {
    pub device: &'r Arc<DeviceInner>,
    pub module: &'r CheckedModule,
    pub entry: EntryId,
    pub bindings: ElementBindings,
    /// Values of every static dimension.
    pub statics: NativeSpecialization,
    pub cpu: Option<&'static CpuNativeKernels>,
    pub points: &'r mut dyn PointSource<'a>,
    /// What every candidate is held to against the native default.
    pub validation: NumericalRequirement,
    pub strategy: Strategy,
}

/// Forms configurations of one entry's implementation.
struct Formation<'s, 'm> {
    device: &'s Arc<DeviceInner>,
    module: &'m CheckedModule,
    entry: EntryId,
    logical: &'s Arc<LogicalEntry>,
    bindings: &'s ElementBindings,
    cpu: Option<&'static CpuNativeKernels>,
    implementation: &'s NativeImplementation,
    /// The unit's defaults: what a launch's source is rendered at.
    default: &'s NativeSpecialization,
    /// The served shapes beyond the points' ([`PointSource::served`]).
    served: &'s [BTreeMap<String, u64>],
    /// What formations of this unit took: the one price of every formation
    /// of it, wherever it is quoted.
    price: &'s Mutex<FormationPrice>,
}

/// What one formation call of a unit's challengers forms: the launch
/// sources its challengers need, where launches are formed separately, or
/// one challenger's program, where the program is formed whole.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Formable {
    Launches(Vec<plan::LaunchSource>),
    Program(NativeSpecialization),
}

impl Formable {
    /// How many programs the call forms for its unit to hold: launch
    /// variants, or the one whole program.
    fn programs(&self) -> usize {
        match self {
            Self::Launches(sources) => sources.iter().map(|source| source.variants.len()).sum(),
            Self::Program(_) => 1,
        }
    }
}

/// What a formation call formed.
enum Product {
    Launches(Result<HeldPrograms, String>),
    Program(Result<Arc<NativePrepared>, Exclusion>),
}

/// How many formation calls run at once: each on a thread of its own, as
/// many as the host runs in parallel.
fn formation_threads(calls: usize) -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(calls)
        .max(1)
}

/// `form` each of `work` on the threads lent to `device`'s former, at most
/// [`formation_threads`] at once, and in turn on this thread without them;
/// what each formed, in order. No thread is started here: whoever owns the
/// load's formation owns its threads ([`Workers`]).
fn concurrently<W: Sync, R: Send>(
    device: &DeviceInner,
    work: &[W],
    form: impl Fn(&W) -> R + Sync,
) -> Vec<R> {
    match device.programs.workers() {
        Some(workers) => workers.each(work, form),
        None => work.iter().map(form).collect(),
    }
}

/// What a formation call does, known before it runs: its class, and for a
/// call that compiles the most entries one of its sources' compiles forms
/// (the sources compile at once).
type Formative = (FormationClass, usize);

/// What formations of a unit's challengers took, by what a formation does
/// ([`FormationClass`]): one that finds everything live on the device or
/// kept by the artifact store runs no compiler and takes about nothing,
/// and one that compiles took 0.9 to 3.5 s on an M4 Max. Which it is is
/// read from the device and the store before the formation starts, so
/// each is priced by its own class. One mean over both priced a compile
/// that took 2.21 s at 0.71 s.
///
/// A kept formation is priced by the mean of the unit's kept formations. A
/// call that compiles is priced by what it forms ([`Linear`]): the cost of
/// the call and the cost of each entry, from the most specific observations
/// this load has, in this order.
///
/// 1. The unit's own compile calls, once they were of two sizes: its
///    source's call cost and entry cost.
/// 2. The unit's own compile calls of one size, with the toolchain's
///    compiles of this load ([`Linear`] of the device's former) saying how
///    a call's time follows its size: the unit's mean call, scaled as the
///    toolchain's price moves from that size to the one asked. A unit's
///    source is its own (one took 3.1 to 4.3 s where another took 2.4 to
///    2.8 s under the same compiler), so its level is its own; how much of
///    a call is its entries is the compiler's.
/// 3. No compile of the unit: the toolchain's compiles of this load. Every
///    unit's compiles are calls of the same compiler, so one unit's are
///    evidence of what another's takes; a unit that starts late is priced
///    by them and not by the longest operation of any kind the load has
///    seen ([`Quote::fits`]).
///
/// Only a load whose toolchain has compiled nothing leaves a compile
/// unpriced. A piece of a program formed whole makes one call for each of
/// its challengers, [`formation_threads`] at once, and is priced by the
/// calls one thread makes in turn ([`Formation::quote`]). The canonical
/// program's formation is of neither class: it prices nothing; it is only
/// an operation the load completed.
#[derive(Default)]
struct FormationPrice {
    canonical: Duration,
    kept: Observed,
    compiled: Linear,
    /// What each program a compile of the unit formed added to the bytes
    /// held ([`held_bytes`]), read on the compile's own thread as it began
    /// and ended: the class the unit's held programs are priced by. The
    /// reading is the process's, so what the device's thread took or gave
    /// back meanwhile is in it, either way; the class's mean is over every
    /// compile of the unit, and a mean below nothing prices nothing.
    program_bytes: Observed,
    /// The formation calls made since a piece last took them, each with
    /// what it formed: what the configurations the next piece measures
    /// record of their formation ([`Programs::records`]).
    calls: Vec<(Formable, FormationRecord)>,
    /// The programs each formation asked about since the unit last formed
    /// anything forms, rendered: its class is read from them whenever it
    /// is asked ([`Rendered`]).
    rendered: BTreeMap<Formable, Rendered>,
    /// The programs publishing the serving artifact of each choice asked
    /// about since the unit last formed anything forms, rendered
    /// ([`UnitSearch::publication`]).
    serving: BTreeMap<NativeSpecialization, Rendered>,
}

/// What publishing a choice's serving artifact does, known before it runs
/// ([`UnitSearch::artifact_publication`]): what a load's publications are
/// observed against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Publication {
    /// The programs of the choice the toolchain compiles; none when the
    /// store keeps each as a load requests it.
    pub sources: usize,
    /// What forming the unit's canonical program took in this process:
    /// the same source text, compiled once already, with each launch's
    /// single variant as a publication compiles it.
    pub canonical: Duration,
}

impl FormationPrice {
    /// The price of one challenger formation call that does `formative`;
    /// none while nothing it can be priced by was observed. `toolchain`:
    /// what the device's toolchain's compiles took in this load.
    fn of(&self, (class, entries): Formative, toolchain: &Linear) -> Option<f64> {
        match class {
            FormationClass::Kept => self.kept.price(),
            FormationClass::Compiled => match (self.compiled.each(), self.compiled.mean()) {
                (Some(_), _) => self.compiled.price(entries),
                (None, Some((seconds, size))) => {
                    let scale = toolchain
                        .price(entries)
                        .zip(toolchain.price(size.round() as usize))
                        .filter(|(_, observed)| *observed > 0.)
                        .map_or(1., |(asked, observed)| asked / observed);
                    Some(seconds * scale)
                }
                (None, None) => toolchain.price(entries),
            },
        }
    }

    /// A challenger formation that did `formative`, as classed before it
    /// ran, took `seconds`. What it formed is not asked about again.
    fn observe(&mut self, (class, entries): Formative, seconds: Duration) {
        match class {
            FormationClass::Kept => self.kept.observe(seconds.as_secs_f64()),
            FormationClass::Compiled => self.compiled.observe(entries, seconds.as_secs_f64()),
        }
        self.rendered.clear();
        self.serving.clear();
    }

    /// The longest formation observed.
    fn longest(&self) -> Duration {
        self.canonical.max(Duration::from_secs_f64(
            self.kept.longest().max(self.compiled.longest()),
        ))
    }

    /// Everything formation has taken.
    fn total(&self) -> f64 {
        self.canonical.as_secs_f64() + self.kept.total() + self.compiled.total()
    }
}

/// The compile of a unit's next piece of work ([`UnitSearch::forming`]):
/// the launch sources its challengers run that no held program has, or the
/// whole programs of its challengers. It reads nothing of the unit's
/// search and submits nothing to the device, so it runs on any thread
/// while the device measures other work.
pub struct Forming<'m> {
    device: Arc<DeviceInner>,
    module: &'m CheckedModule,
    entry: EntryId,
    logical: Arc<LogicalEntry>,
    bindings: ElementBindings,
    default: NativeSpecialization,
    cpu: Option<&'static CpuNativeKernels>,
    implementation: NativeImplementation,
    served: Vec<BTreeMap<String, u64>>,
    /// Each formation call, with the programs it forms rendered: what it
    /// does is read from them as it starts ([`Forming::run`]).
    work: Vec<(Formable, Rendered)>,
}

impl Forming<'_> {
    /// The threads the compile's calls run on at once: what its consumer
    /// counts against the threads the host has, so that every compile in
    /// flight runs on threads of its own, as it was priced
    /// ([`Formation::quote`]).
    pub fn threads(&self) -> usize {
        formation_threads(self.work.len())
    }

    /// Make every formation call, to the end. What each call does is read
    /// as it starts, not as it was priced: what other units formed and the
    /// store kept since may have made a compile priced as one a load, and
    /// observed in the class it was priced in, a load of 0.04 s priced the
    /// unit's next compile of 0.5 s at that (a GB10 load, 2026-10-12).
    pub fn run(self) -> Formed {
        let began = Instant::now();
        let held = |device: &DeviceInner| {
            held_bytes(
                device,
                device.memory_usage().charged,
                0,
                ProcessMemory::footprint_now(),
            )
        };
        let before = held(&self.device);
        let products = concurrently(&self.device, &self.work, |(formable, rendered)| {
            let class = rendered.class(&self.device);
            let began = Instant::now();
            let product = match formable {
                Formable::Launches(sources) => Product::Launches(
                    NativePrepared::hold_launch_variants(
                        &self.device,
                        self.module,
                        self.entry,
                        &self.bindings,
                        &self.default,
                        &self.implementation,
                        sources,
                    )
                    .map_err(prepare_message),
                ),
                Formable::Program(specialization) => Product::Program(
                    NativePrepared::prepare_implementation(
                        &self.device,
                        self.module,
                        self.entry,
                        &self.logical,
                        self.bindings.clone(),
                        specialization.clone(),
                        self.cpu,
                        self.implementation.clone(),
                    )
                    .map_err(|error| Exclusion::Formation(prepare_message(error)))
                    .and_then(|kernel| serving(kernel, &self.served)),
                ),
            };
            (class, product, began.elapsed())
        });
        Formed {
            added_bytes: held(&self.device) as f64 - before as f64,
            calls: self
                .work
                .into_iter()
                .zip(products)
                .map(|((formable, _), (class, product, elapsed))| FormedCall {
                    formable,
                    class,
                    product,
                    elapsed,
                })
                .collect(),
            elapsed: began.elapsed(),
        }
    }
}

/// What compiling one of a unit's programs takes when nothing the toolchain
/// formed or kept serves it, the system's caches included
/// ([`UnitSearch::cold_compile`]): it reads nothing of the unit, so it
/// runs on any thread. A compile cannot be interrupted, and where a system
/// cache serves most of a load's compiles the load observes only its hits:
/// an M6 load (2026-10-12) whose compiles had all taken under 0.17 s
/// started one at 59.75 s, priced at 0.044 s, that missed the cache and
/// took 0.39 s past its budget. This is what such a miss takes.
pub struct ColdCompile {
    device: Arc<DeviceInner>,
    source: seismic_native_target::ProgramSource,
}

impl ColdCompile {
    /// The length of the program's text: what of a load's units is the
    /// largest compile.
    pub fn size(&self) -> usize {
        self.source.text.len()
    }

    /// Compile the program cold, and drop it. Returns what it took; none
    /// if the compile failed, which says nothing of what one takes. The
    /// program is the canonical default's, which the load has formed.
    pub fn run(self) -> Option<Duration> {
        self.device.programs.cold(&self.source).ok()
    }
}

/// A compile that ran ([`Forming::run`]), for its unit to hold
/// ([`UnitSearch::formed`]).
pub struct Formed {
    calls: Vec<FormedCall>,
    /// What the calls took together.
    elapsed: Duration,
    /// What was held when the calls ended beyond what was when they began.
    added_bytes: f64,
}

/// One formation call that ran: what it was to form, its class as it
/// started, what it formed and what it took.
struct FormedCall {
    formable: Formable,
    class: Formative,
    product: Product,
    elapsed: Duration,
}

/// Publication of the served choice's serving artifact: what a later
/// process requests when it loads the unit's published choice.
impl UnitSearch<'_, '_> {
    /// The unit's served choice as it stands: the canonical default until
    /// a challenger is promoted.
    fn served_choice(&self) -> &NativeSpecialization {
        match &self.progress {
            Progress::Searching(searching) => &searching.ledger.incumbent().specialization,
            Progress::Discovering(_) => &self.default,
        }
    }

    /// The choice the unit's next piece serves if it promotes: the leader
    /// of a conversion. None when the next piece is no conversion.
    fn converting(&self) -> Option<&NativeSpecialization> {
        let Progress::Searching(searching) = &self.progress else {
            return None;
        };
        let ledger = &searching.ledger;
        let leader = match self.next()?.stage {
            Stage::Promote => ledger.leader()?,
            Stage::Confirm => ledger.nominal()?.0,
            Stage::Discovery | Stage::Form | Stage::Measure => return None,
        };
        Some(&ledger.contenders[leader].specialization)
    }

    /// What publishing the serving artifact of `choice` does, read from
    /// the store before it runs ([`Rendered::publication`]).
    fn publication(&self, choice: &NativeSpecialization) -> (Formative, usize) {
        let mut price = self.formation_price.lock().expect("formation price lock");
        price
            .serving
            .entry(choice.clone())
            .or_insert_with(|| {
                NativePrepared::publication_rendered(
                    &self.device,
                    self.module,
                    self.entry,
                    &self.logical,
                    &self.bindings,
                    choice,
                    &self.implementation,
                )
            })
            .publication(&self.device)
    }

    /// The price of [`UnitSearch::publish_artifact`]: of the served choice
    /// as it stands, or, `worked`, as it
    /// stands once the unit's next piece has run, which changes it only if
    /// that piece is a conversion and promotes its leader. A choice whose
    /// programs the store keeps as a load requests them (the canonical
    /// default's, and any the search formed alone) owes nothing: its
    /// publication runs no compiler. One that the store lacks owes one call
    /// of the toolchain.
    ///
    /// This is the price before the load has published an artifact that
    /// compiled: a load's publications are a class of their own, priced by
    /// its consumer from the ones it has run
    /// ([`UnitSearch::artifact_publication`]). Until one has, nothing the
    /// unit observed is a publication, and two things stand in for one:
    /// a challenger formation call of that size ([`FormationPrice::of`]),
    /// and the unit's canonical program's formation, the same programs at
    /// the default choice. Neither is it: one load's publications took
    /// eight times the first (its challenger compiles had found the
    /// system's shader cache warm, 64 to 73 ms against 3 to 8 ms) and
    /// another's a third of the second. The price is the larger of the
    /// two: holding back too little runs past the budget, and holding back
    /// too much delays a piece until the first publication is observed.
    ///
    /// Every promoted unit was owed a compile whatever the store kept: a
    /// load of twenty units held back 5.6 s of its last 5.6 s for
    /// publications that then took 0.29 s, thirteen of sixteen compiling
    /// nothing.
    pub fn artifact_owed(&self, worked: bool) -> Quote {
        let choice = worked
            .then(|| self.converting())
            .flatten()
            .unwrap_or_else(|| self.served_choice());
        let (formative, _) = self.publication(choice);
        let mut owed = Quote::default();
        if formative.0 == FormationClass::Compiled {
            let toolchain = self.device.programs.compiled();
            let price = self.formation_price.lock().expect("formation price lock");
            let canonical = price.canonical.as_secs_f64();
            let formed = price.of(formative, &toolchain);
            owed.add(Some(formed.map_or(canonical, |formed| formed.max(canonical))), 1);
        }
        owed
    }

    /// What [`UnitSearch::publish_artifact`] does, known before it runs:
    /// of the served choice as it stands, or,
    /// `worked`, as it stands once the unit's next piece has run.
    pub fn artifact_publication(&self, worked: bool) -> Publication {
        let choice = worked
            .then(|| self.converting())
            .flatten()
            .unwrap_or_else(|| self.served_choice());
        let (_, sources) = self.publication(choice);
        let canonical = self
            .formation_price
            .lock()
            .expect("formation price lock")
            .canonical;
        Publication { sources, canonical }
    }

    /// What building the inputs the unit holds took, each point's at its
    /// class's price: what giving them up is held to before the load has
    /// observed inputs given up. It frees what building them allocated and
    /// writes nothing.
    pub fn inputs_built(&self) -> Quote {
        let resident: Box<dyn Iterator<Item = usize> + '_> = match &self.progress {
            Progress::Discovering(slots) => Box::new(
                slots
                    .iter()
                    .enumerate()
                    .filter(|(_, timed)| timed.as_ref().is_some_and(|timed| timed.point.resident()))
                    .map(|(index, _)| index),
            ),
            Progress::Searching(searching) => Box::new(
                searching
                    .timed
                    .iter()
                    .copied()
                    .zip(searching.points.iter())
                    .filter(|(_, point)| point.resident())
                    .map(|(index, _)| index),
            ),
        };
        let mut built = Quote::default();
        for index in resident {
            built.add(self.builds[index].price(), 1);
        }
        built
    }

    /// A cold compile of the largest program of the unit's canonical
    /// default ([`ColdCompile`]); none where that renders nothing, as on
    /// the CPU backend.
    pub fn cold_compile(&self) -> Option<ColdCompile> {
        let sources = NativePrepared::serving_programs(
            &self.device,
            self.module,
            self.entry,
            &self.logical,
            &self.bindings,
            &self.default,
            &self.implementation,
        )
        .ok()?;
        let source = sources.into_iter().max_by_key(|source| source.text.len())?;
        Some(ColdCompile {
            device: self.device.clone(),
            source,
        })
    }

    /// Publish the serving artifact of the unit's served choice: form its
    /// programs exactly as a load requests them, each launch's single
    /// variant, and release them ([`NativePrepared::publish`]). A choice
    /// the search formed among many variants of a launch is served to this
    /// process out of that compile; the artifact a later load requests is
    /// formed here, as the choice is promoted. The canonical default
    /// was formed that way by the mandatory work, and a choice already
    /// published forms nothing. Returns what it took.
    pub fn publish_artifact(&mut self) -> Result<Duration, TuneError> {
        let began = Instant::now();
        NativePrepared::publish(
            &self.device,
            self.module,
            self.entry,
            &self.logical,
            &self.bindings,
            self.served_choice(),
            &self.implementation,
        )
        .map_err(|error| TuneError::Publication(prepare_message(error)))?;
        let elapsed = began.elapsed();
        self.time.forming_seconds += elapsed.as_secs_f64();
        Ok(elapsed)
    }
}

/// The compile of a unit's canonical program ([`UnitSearch::grounding`]).
pub struct Grounding<'m> {
    device: Arc<DeviceInner>,
    module: &'m CheckedModule,
    entry: EntryId,
    logical: Arc<LogicalEntry>,
    bindings: ElementBindings,
    default: NativeSpecialization,
    cpu: Option<&'static CpuNativeKernels>,
    implementation: NativeImplementation,
    served: Vec<BTreeMap<String, u64>>,
}

impl Grounding<'_> {
    /// Form the canonical program, to the end.
    pub fn run(self) -> Grounded {
        let began = Instant::now();
        let kernel = NativePrepared::prepare_implementation(
            &self.device,
            self.module,
            self.entry,
            &self.logical,
            self.bindings,
            self.default,
            self.cpu,
            self.implementation,
        )
        .map_err(|error| TuneError::DefaultUnusable(Exclusion::Formation(prepare_message(error))))
        // Serving runs the defaults wherever no other choice stands.
        .and_then(|kernel| serving(kernel, &self.served).map_err(TuneError::DefaultUnusable));
        Grounded {
            kernel,
            elapsed: began.elapsed(),
        }
    }
}

/// A canonical program that was formed ([`Grounding::run`]), for its unit
/// to take ([`UnitSearch::grounded`]).
pub struct Grounded {
    kernel: Result<Arc<NativePrepared>, TuneError>,
    elapsed: Duration,
}

impl Grounded {
    /// What forming the program took.
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }
}

/// `kernel`, unless a launch of a call it serves at one of `served` exceeds
/// its formed functions' or the device's limits.
fn serving(
    kernel: Arc<NativePrepared>,
    served: &[BTreeMap<String, u64>],
) -> Result<Arc<NativePrepared>, Exclusion> {
    for dimensions in served {
        kernel.admits(dimensions).map_err(|error| {
            Exclusion::Execution(format!("at the served shape {dimensions:?}: {error}"))
        })?;
    }
    Ok(kernel)
}

impl<'m> Formation<'_, 'm> {
    /// What the one call forming `formable` does, read from the device's
    /// live programs and the artifact store before it starts.
    fn class(&self, formable: &Formable) -> Formative {
        self.rendered(formable).class(self.device)
    }

    /// The programs the one call forming `formable` forms, rendered once
    /// for the unit ([`Rendered`]).
    fn rendered(&self, formable: &Formable) -> Rendered {
        let mut price = self.price.lock().expect("formation price lock");
        price
            .rendered
            .entry(formable.clone())
            .or_insert_with(|| match formable {
                Formable::Launches(sources) => NativePrepared::launch_variants_rendered(
                    self.device,
                    self.module,
                    self.entry,
                    self.bindings,
                    &self.default,
                    self.implementation,
                    sources,
                ),
                Formable::Program(specialization) => NativePrepared::implementation_rendered(
                    self.device,
                    self.module,
                    self.entry,
                    self.logical,
                    self.bindings,
                    specialization,
                    self.implementation,
                ),
            })
            .clone()
    }

    /// The quote of the calls forming `formables`: the calls of a class run
    /// [`formation_threads`] at once, so the class takes the price of its
    /// calls for those one thread makes in turn; of nothing when nothing
    /// is formed. Calls of a class nothing prices are one unknown, not one
    /// for each call: they are one kind of work. The one price of
    /// challenger formation, wherever it is quoted.
    fn quote(&self, formables: &[Formable]) -> Quote {
        let mut quote = Quote::default();
        let threads = formation_threads(formables.len());
        let toolchain = self.device.programs.compiled();
        let formatives = formables
            .iter()
            .map(|formable| self.class(formable))
            .collect::<Vec<_>>();
        let price = self.price.lock().expect("formation price lock");
        for class in [FormationClass::Kept, FormationClass::Compiled] {
            let calls = formatives
                .iter()
                .filter(|formative| formative.0 == class)
                .collect::<Vec<_>>();
            if calls.is_empty() {
                continue;
            }
            let in_turn = calls.len().div_ceil(threads) as f64 / calls.len() as f64;
            let seconds = calls
                .iter()
                .map(|formative| price.of(**formative, &toolchain))
                .sum::<Option<f64>>();
            quote.add(seconds.map(|seconds| seconds * in_turn), 1);
        }
        quote
    }

    /// The compile of `formables`, each classed as it is priced, to run on
    /// any thread; none when they are none.
    fn forming(&self, formables: Vec<Formable>) -> Option<Forming<'m>> {
        (!formables.is_empty()).then(|| Forming {
            device: self.device.clone(),
            module: self.module,
            entry: self.entry,
            logical: self.logical.clone(),
            bindings: self.bindings.clone(),
            default: self.default.clone(),
            cpu: self.cpu,
            implementation: self.implementation.clone(),
            served: self.served.to_vec(),
            work: formables
                .into_iter()
                .map(|formable| {
                    let rendered = self.rendered(&formable);
                    (formable, rendered)
                })
                .collect(),
        })
    }
}

fn point_shapes(
    device: &Arc<DeviceInner>,
    logical: &LogicalEntry,
    points: &[PreparedPoint<'_>],
) -> Result<Vec<PointShape>, TuneError> {
    let contract = InvocationContract::compile_entry(logical);
    points
        .iter()
        .map(|point| {
            let mut shape = None;
            for args in point.rotation() {
                let values = validate_invocation(&contract, device.kind.identity(), &args.values())
                    .map_err(|error| {
                        TuneError::DefaultUnusable(Exclusion::Execution(error.to_string()))
                    })?;
                let dimensions = logical
                    .schema()
                    .dimensions()
                    .iter()
                    .map(|dimension| {
                        match values.get(dimension.symbol) {
                            Some(SymbolValue::Nat(value)) => u64::try_from(value).ok(),
                            _ => None,
                        }
                        .map(|value| (dimension.name.clone(), value))
                        .ok_or_else(|| {
                            TuneError::Declaration(format!(
                                "tuning point `{}` has no value for dimension `{}`",
                                point.label, dimension.name,
                            ))
                        })
                    })
                    .collect::<Result<BTreeMap<_, _>, _>>()?;
                if shape.as_ref().is_some_and(|first| first != &dimensions) {
                    return Err(TuneError::Declaration(format!(
                        "tuning point `{}` rotates across different dimensions",
                        point.label,
                    )));
                }
                shape = Some(dimensions);
            }
            Ok(PointShape {
                label: point.label.clone(),
                dimensions: shape.expect("checked nonempty point rotation"),
            })
        })
        .collect()
}

fn launch_key(
    implementation: &NativeImplementation,
    boundary: &[plan::ParameterAddress],
    working: Vec<usize>,
    specialization: &NativeSpecialization,
) -> PointKey {
    let mut values = BTreeMap::new();
    for parameter in &implementation.params {
        let owners = implementation
            .launches
            .iter()
            .enumerate()
            .filter_map(|(ordinal, launch)| {
                launch
                    .parameters()
                    .contains(&parameter.name)
                    .then_some(ordinal)
            })
            .collect::<Vec<_>>();
        // Only condition-only boundaries are represented fully by the active
        // launch set. Other ownerless entry parameters conservatively affect
        // every launch because a source may read them.
        if !boundary.iter().any(|address| {
            matches!(address,
            plan::ParameterAddress::Entry(name) if name == &parameter.name)
        }) && (owners.is_empty() || owners.iter().any(|ordinal| working.contains(ordinal)))
        {
            values.insert(
                parameter.name.clone(),
                specialization
                    .param(&parameter.name)
                    .expect("admissible specialization values the entry parameter"),
            );
        }
    }
    for ((launch, name), value) in specialization.launch_params() {
        if working.contains(launch) {
            values.insert(format!("@{launch}:{name}"), *value);
        }
    }
    PointKey {
        launches: working,
        values,
    }
}

/// The key of a point as the unit's programs are formed: what a
/// configuration's measurement there depends on.
type Key<'k> = &'k dyn Fn(Vec<usize>, &NativeSpecialization) -> PointKey;

/// A challenger formed, with what its comparison with the served choice
/// has shown before this session.
struct Candidate {
    specialization: NativeSpecialization,
    kernel: Arc<NativePrepared>,
    evidence: Tally,
}

/// The served choice as a session samples it beside its challengers: its
/// kernel, and its time at every timed point in the unit's canonical units
/// ([`Matched`]).
struct Incumbent<'s> {
    specialization: &'s NativeSpecialization,
    kernel: &'s Arc<NativePrepared>,
    points: &'s [PointMeasurement],
}

/// What a session of matched rounds is run with.
struct Matching<'s> {
    deadline: Deadline,
    settings: &'s SearchSettings,
    min_sample_seconds: f64,
    /// How the unit's rounds have come out: what its sessions are priced
    /// by.
    rates: &'s mut Rates,
    shuffle: &'s mut Shuffle,
    /// The device's sustained clock, held by the unit from its first
    /// session ([`timing::Clock`]).
    clock: &'s mut Option<Arc<timing::Clock>>,
    /// What the unit's witnesses and rounds have cost and how often each
    /// ended a challenger: the order its challengers' tests are taken in.
    sequence: &'s mut Sequence,
}

/// Bring the device of `placed`'s slots to its sustained clock for the
/// session that follows at once, with the clock its unit holds in `held`:
/// the unit's own work, which is `placed`'s first slot's when the unit
/// holds none yet ([`timing::Clock::hold`]).
fn raise(
    held: &mut Option<Arc<timing::Clock>>,
    placed: &[PointTiming],
    settings: &SearchSettings,
    min_sample_seconds: f64,
    deadline: Deadline,
) -> Result<(), WorkError<CallError>> {
    let Some(first) = placed.first() else {
        return Ok(());
    };
    // The device's state is asked the evidence a lead is.
    let stages = Tally::default().fewest(settings.error_rate);
    let clock = timing::Clock::hold(held, first, min_sample_seconds, stages)?;
    timing::warm(&clock, placed, deadline)
}

/// What a challenger new to the unit takes at the timed points besides its
/// rounds, in the order its two tests are taken ([`Sequence`]): each a
/// class of the unit, observed as one challenger's host seconds over every
/// timed point.
#[derive(Clone, Copy, Debug, Default)]
struct Admission {
    /// Witnessed first: placing it at every timed point, the reference
    /// where it is not yet observed, its witnesses, their readback and
    /// comparison.
    witnessed_first: Observed,
    /// Timed first and left ahead by its rounds: its witnesses at every
    /// timed point after them, in a pass of their own.
    witnessed_after: Observed,
}

impl Admission {
    /// The price of one challenger's witnesses in a pass after its rounds.
    /// The class is observed only when rounds leave a challenger ahead, so
    /// a unit whose challengers are timed first and never lead never
    /// observes it; until it does, the class is priced by the unit's
    /// nearest one, an admission witnessed first: the same witnesses at the
    /// same points, with the placement besides. Left without a price it was
    /// one unknown in every piece of such a unit, each taken as the longest
    /// operation of any kind in the load: eleven pieces in a row of one
    /// unit, quoted 0.05 to 0.37 s, each carried it.
    fn after(&self) -> Option<f64> {
        self.witnessed_after
            .price()
            .or_else(|| self.witnessed_first.price())
    }

    /// The price of what `challengers` new challengers take here in one
    /// session, in the order `sequence` gives. Witnessed first, each is
    /// one admission of that class. Timed first, each is placed at every
    /// timed point as any kernel is (`placements`: each point's placement
    /// class), and witnessed after the rounds only if they leave it ahead:
    /// how many will be is not known before the rounds run, and those
    /// witnesses are priced at their expected number
    /// ([`Sequence::ahead`]).
    fn quote(&self, sequence: &Sequence, placements: &[Option<f64>], challengers: usize) -> Quote {
        let mut quote = Quote::default();
        if !sequence.timed_first() {
            quote.add(self.witnessed_first.price(), challengers);
            return quote;
        }
        for placement in placements {
            quote.add(*placement, challengers);
        }
        quote.expect(self.after(), challengers as f64 * sequence.ahead());
        quote
    }

    /// The price of the session that may resolve the lead of a challenger
    /// measured before, placed again beside the served choice. `points`:
    /// each timed point's classes, and whether the two run different work
    /// there. The served choice is placed where they do, and both are
    /// sampled there: a calibrating pass, and the `rounds` expected of the
    /// session; none before the unit ran a session of this class, one
    /// operation it has not seen.
    ///
    /// A leader `witnessed` at every timed point is placed there as any
    /// kernel is. One that is not (timed first, it took no rounds or was
    /// not left ahead by them) is witnessed in this session before its
    /// rounds, as a challenger witnessed first is: one admission of that
    /// class, its placement at every timed point within it.
    fn conversion(
        &self,
        points: &[(PointPrices, bool)],
        witnessed: bool,
        rounds: Option<f64>,
    ) -> Quote {
        let mut quote = Quote::default();
        quote.add(None, usize::from(rounds.is_none()));
        quote.add(self.witnessed_first.price(), usize::from(!witnessed));
        for (prices, apart) in points {
            let apart = usize::from(*apart);
            quote.add(prices.placement.price(), usize::from(witnessed) + apart);
            quote.add(prices.sample.price(), 2 * apart);
            if let Some(rounds) = rounds {
                quote.expect(prices.sample.price(), 2. * apart as f64 * rounds);
            }
        }
        quote
    }
}

/// One challenger's part of a session.
struct Pairing {
    /// Each point it is sampled at: its slot there, and `w_p · I_p`, the
    /// point's demand times the incumbent's canonical time there. The
    /// incumbent's slot at the point is the slot's partner.
    terms: Vec<(usize, f64)>,
    /// `Σ w_p · τ_p` over the points where it runs other work than the
    /// incumbent and is not sampled here: their measurements are reused
    /// from earlier sessions.
    reused: f64,
    /// `Σ w_p · I_p` over every point where it runs other work than the
    /// incumbent: what its cost there is compared with.
    incumbent: f64,
    /// Whether its rounds are evidence: every point that tells it from the
    /// incumbent is sampled in each. A round that reads a measurement
    /// reused from another session repeats that measurement's error in
    /// every round; such rounds estimate its cost and resolve nothing.
    complete: bool,
    evidence: Tally,
    /// The rounds it is still admitted, and those it has taken.
    admitted: u32,
    taken: u32,
    /// Whether every round it took here was ahead: it could be served.
    ahead: bool,
}

/// The rounds of a session that compares challengers with the served
/// choice ([`super::sampling`]). A round samples, at every point where a
/// challenger still in the session runs other work than the incumbent, that
/// challenger's slot and the incumbent's: the incumbent once at a point,
/// however many challengers are sampled there, and nowhere else.
///
/// What a round yields at a point is the ratio of the two samples, and a
/// challenger's time there is recorded as the median of its ratios times
/// the incumbent's recorded time. The defaults' own measurement is the
/// unit of that chain, so every recorded time of a unit is in one unit,
/// its *canonical* one, whatever state the device was in when each
/// configuration was sampled and whichever choice was served then: a
/// challenger measured beside the third choice served is `ratio₃ × ratio₂ ×
/// ratio₁ × default`. Each link carries the spread of one median of
/// ratios, and holds only between sessions read at one device state: every
/// session is read at the device's sustained clock ([`timing::Clock`]),
/// where the same point and key read in two sessions of one load agreed to
/// 1.3% at the median and 9.5% at the 90th percentile (Qwen3.5-4B on an M4
/// Pro, 2026-10-10; 4.3% and 49% before sessions were raised), so costs measured beside different choices compare within the links
/// between them, and a lead over the served choice is never taken from
/// that comparison: it is resolved in rounds beside the choice itself.
struct Matched<'s> {
    settings: &'s SearchSettings,
    /// By slot: the incumbent's slot at a challenger slot's point; none for
    /// an incumbent's own slot.
    partner: Vec<Option<usize>>,
    /// By slot: the ratio of each round it was sampled in.
    ratios: Vec<Vec<f64>>,
    pairings: Vec<Pairing>,
}

impl Matched<'_> {
    /// `slot`'s sample failed: the challengers sampled there, or beside it,
    /// leave the session. Which they were.
    fn fail(&mut self, slot: usize) -> Vec<usize> {
        let mut left = Vec::new();
        for (index, pairing) in self.pairings.iter_mut().enumerate() {
            let sampled = pairing
                .terms
                .iter()
                .any(|(term, _)| *term == slot || self.partner[*term] == Some(slot));
            if sampled && pairing.admitted > 0 {
                pairing.admitted = 0;
                left.push(index);
            }
        }
        left
    }

    /// The time of the challenger slot `slot` relative to the incumbent's
    /// at its point: the median of its rounds' ratios, and their median
    /// absolute deviation.
    fn ratio(&self, slot: usize) -> (f64, f64) {
        let ratios = &self.ratios[slot];
        let center = median(ratios);
        (
            center,
            median_of(ratios.iter().map(|ratio| (ratio - center).abs()).collect()),
        )
    }
}

impl timing::Rounds for Matched<'_> {
    /// By point: the incumbent's slot there and the slots of the
    /// challengers still in the session that are sampled beside it.
    fn next(&mut self) -> Vec<Vec<usize>> {
        let mut groups = BTreeMap::<usize, BTreeSet<usize>>::new();
        for pairing in self.pairings.iter().filter(|pairing| pairing.admitted > 0) {
            for (slot, _) in &pairing.terms {
                let beside = self.partner[*slot].expect("a challenger's slot has a partner");
                groups.entry(beside).or_default().insert(*slot);
            }
        }
        groups
            .into_iter()
            .map(|(beside, slots)| std::iter::once(beside).chain(slots).collect())
            .collect()
    }

    fn sampled(&mut self, round: &[(usize, f64)]) {
        let mut seconds = vec![None; self.partner.len()];
        for (slot, sample) in round {
            seconds[*slot] = Some(*sample);
        }
        let mut current = vec![None; self.partner.len()];
        for (slot, sample) in round {
            if let Some(partner) = self.partner[*slot] {
                let ratio = sample / seconds[partner].expect("a round samples a slot's partner");
                self.ratios[*slot].push(ratio);
                current[*slot] = Some(ratio);
            }
        }
        let error_rate = self.settings.error_rate;
        for pairing in self.pairings.iter_mut().filter(|pairing| pairing.admitted > 0) {
            let cost = pairing
                .terms
                .iter()
                .map(|(slot, weight)| weight * current[*slot].expect("a round samples its slots"))
                .sum::<f64>()
                + pairing.reused;
            let difference = cost - pairing.incumbent;
            pairing.taken += 1;
            pairing.ahead &= difference < 0.;
            if !pairing.complete {
                // One round estimates its cost.
                pairing.admitted = 0;
                continue;
            }
            pairing.evidence.observe(difference);
            // Its part ends when it is resolved, and with its first round
            // that is not ahead: no round it was admitted can then make it
            // lead.
            pairing.admitted = if difference < 0. && pairing.evidence.verdict(error_rate).is_none()
            {
                pairing.admitted - 1
            } else {
                0
            };
        }
    }
}

/// A challenger as a session measured it: its time at every timed point in
/// the unit's canonical units, and what its comparison with the served
/// choice has shown.
type Compared = (Vec<PointMeasurement>, Tally);

/// Measure `candidates` beside `incumbent` in one session of matched rounds
/// ([`Matched`]). Each starts at `baseline`, the measurement of what it is
/// a change of, and is placed and admitted at every timed point. At a point
/// where the launches in `affected_launches` work (every point when none
/// are named), a point and key measured before is reused from `cache`, and
/// one never measured is sampled, once for all the candidates that run it.
/// Without a cache nothing is reused: every point that tells a candidate
/// from the incumbent is sampled. A failing candidate leaves the session;
/// the others finish their rounds.
///
/// A challenger new to the unit takes its two tests in the order the
/// unit's observations give ([`Sequence`]). Witnessed first, it is admitted
/// at every timed point as it is placed, and one the requirement refuses
/// is never sampled. Timed first, it is placed and sampled unwitnessed,
/// and witnessed after the rounds only if they left it ahead of the served
/// choice: a challenger that could be served. Either way no challenger
/// leaves here ahead without having been admitted at every timed point,
/// and one placed again for more rounds (`admission` none) is witnessed
/// where it has not been before its rounds: nothing is promoted on rounds
/// alone.
#[allow(clippy::too_many_arguments)]
fn measure_matched(
    key: Key<'_>,
    candidates: &[Candidate],
    incumbent: &Incumbent<'_>,
    points: &[PreparedPoint<'_>],
    matching: Matching<'_>,
    affected_launches: Option<&[usize]>,
    baseline: &[PointMeasurement],
    cache: Option<&mut BTreeMap<(usize, PointKey), PointMeasurement>>,
    outputs: &mut OutputPool,
    // The classes a challenger new to the unit is observed in at the timed
    // points; none for one measured before, placed again.
    admission: Option<&mut Admission>,
) -> Vec<Result<Option<Compared>, Exclusion>> {
    let mut cache = cache;
    let mut admission = admission;
    let minimum = matching.min_sample_seconds;
    let mut measured = vec![baseline.to_vec(); candidates.len()];
    let mut failures = vec![None; candidates.len()];
    let mut pending = vec![false; candidates.len()];
    let mut placed = Vec::new();
    // By slot: its point, its key and, for a challenger's slot, the
    // candidates it measures. Candidates that run the same launches with
    // the same values at a point run the same work there: one slot serves
    // all of them.
    let mut slots: Vec<(usize, PointKey, Vec<usize>)> = Vec::new();
    let mut slot_by_key: BTreeMap<(usize, PointKey), usize> = BTreeMap::new();
    let mut sampled = vec![Vec::new(); candidates.len()];
    let timed_first = admission.is_some() && matching.sequence.timed_first();
    for (candidate, formed) in candidates.iter().enumerate() {
        if matching.deadline.expired() {
            pending[candidate] = true;
            continue;
        }
        let mut candidate_placed = Vec::new();
        let mut candidate_slots = Vec::new();
        let admitting = Instant::now();
        for (point, workload) in points.iter().enumerate() {
            let mut timing = match PointTiming::reusing_outputs(
                &formed.kernel,
                workload.workload_rotation(),
                outputs.at(&workload.label),
            ) {
                Ok(timing) => timing,
                Err(error) => {
                    failures[candidate] = Some(measurement_failure(points, point, error));
                    break;
                }
            };
            let admitted = if timed_first {
                workload.price_before(&mut timing, matching.deadline)
            } else {
                workload.admit_before(
                    Some(&mut timing),
                    &formed.kernel,
                    &|working| key(working, &formed.specialization),
                    matching.deadline,
                )
            };
            if let Err(error) = admitted {
                match error {
                    WorkError::Failed(error) => failures[candidate] = Some(error),
                    WorkError::Expired => pending[candidate] = true,
                }
                break;
            }
            let working = timing.working_launches();
            let affected = affected_launches.is_none_or(|launches| {
                launches.iter().any(|launch| {
                    working.contains(launch) || baseline[point].key.launches.contains(launch)
                })
            });
            if !affected {
                continue;
            }
            let key = key(working, &formed.specialization);
            if key == incumbent.points[point].key {
                // The incumbent's own work there: its time is the
                // incumbent's.
                measured[candidate][point] = incumbent.points[point].clone();
            } else if let Some(cached) = cache
                .as_ref()
                .and_then(|cache| cache.get(&(point, key.clone())))
            {
                measured[candidate][point] = cached.clone();
            } else {
                candidate_slots.push((point, key));
                candidate_placed.push(timing);
            }
        }
        if let Some(admission) = admission.as_deref_mut() {
            // Timed first, this was its placement alone: an observation of
            // each point's placement class, made as it was placed.
            if !timed_first {
                let seconds = admitting.elapsed().as_secs_f64();
                admission.witnessed_first.observe(seconds);
                matching.sequence.witnessed(
                    seconds,
                    !matches!(failures[candidate], Some(Exclusion::Validation { .. })),
                );
            }
        }
        if failures[candidate].is_none() && !pending[candidate] {
            for (timing, (point, key)) in candidate_placed.into_iter().zip(candidate_slots) {
                let slot = match slot_by_key.get(&(point, key.clone())) {
                    Some(&slot) => slot,
                    None => {
                        slot_by_key.insert((point, key.clone()), slots.len());
                        slots.push((point, key, Vec::new()));
                        placed.push(timing);
                        slots.len() - 1
                    }
                };
                slots[slot].2.push(candidate);
                sampled[candidate].push(slot);
            }
        }
    }
    // The incumbent, placed at each point a challenger is sampled at.
    let mut partner = vec![None; slots.len()];
    let mut incumbent_slots = BTreeMap::<usize, usize>::new();
    for slot in 0..slots.len() {
        let point = slots[slot].0;
        if let Some(&beside) = incumbent_slots.get(&point) {
            partner[slot] = Some(beside);
            continue;
        }
        let workload = &points[point];
        if matching.deadline.expired() {
            for &candidate in &slots[slot].2 {
                pending[candidate] = true;
            }
            continue;
        }
        let beside = PointTiming::reusing_outputs(
            incumbent.kernel,
            workload.workload_rotation(),
            outputs.at(&workload.label),
        )
        .map_err(|error| WorkError::Failed(measurement_failure(points, point, error)))
        .and_then(|mut timing| {
            // Its admission stands: only its placement is repeated.
            workload.admit_before(
                Some(&mut timing),
                incumbent.kernel,
                &|working| key(working, incumbent.specialization),
                matching.deadline,
            )?;
            Ok(timing)
        });
        match beside {
            Ok(timing) => {
                incumbent_slots.insert(point, placed.len());
                partner[slot] = Some(placed.len());
                partner.push(None);
                placed.push(timing);
            }
            // Nothing is compared where the served choice cannot be
            // sampled.
            Err(exclusion) => {
                for &candidate in &slots[slot].2 {
                    match &exclusion {
                        WorkError::Failed(exclusion) => {
                            failures[candidate].get_or_insert(exclusion.clone());
                        }
                        WorkError::Expired => pending[candidate] = true,
                    }
                }
            }
        }
    }
    let weights = points.iter().map(|point| point.weight).collect::<Vec<_>>();
    let pairings = (0..candidates.len())
        .map(|candidate| {
            let canonical = |point: usize| weights[point] * incumbent.points[point].median_seconds;
            let terms = sampled[candidate]
                .iter()
                .map(|slot| (*slot, canonical(slots[*slot].0)))
                .collect::<Vec<_>>();
            let fresh = |point: usize| sampled[candidate].iter().any(|slot| slots[*slot].0 == point);
            let apart = (0..points.len())
                .filter(|point| measured[candidate][*point].key != incumbent.points[*point].key)
                .filter(|point| !fresh(*point))
                .collect::<Vec<_>>();
            let evidence = candidates[candidate].evidence;
            let complete = apart.is_empty();
            Pairing {
                reused: apart
                    .iter()
                    .map(|point| weights[*point] * measured[candidate][*point].median_seconds)
                    .sum(),
                incumbent: terms.iter().map(|(_, weight)| weight).sum::<f64>()
                    + apart.iter().map(|point| canonical(*point)).sum::<f64>(),
                complete,
                evidence,
                admitted: if failures[candidate].is_some() || pending[candidate] || terms.is_empty()
                {
                    0
                } else if complete {
                    evidence.fewest(matching.settings.error_rate)
                } else {
                    1
                },
                taken: 0,
                ahead: true,
                terms,
            }
        })
        .collect();
    let mut rounds = Matched {
        settings: matching.settings,
        ratios: vec![Vec::new(); partner.len()],
        partner,
        pairings,
    };
    // Every placement is done: from here to the session's last sample the
    // device is given nothing but the session's work. A device that cannot
    // be raised samples nothing.
    let sampling = Instant::now();
    if let Err(error) = raise(
        matching.clock,
        &placed,
        matching.settings,
        minimum,
        matching.deadline,
    ) {
        if let WorkError::Failed(error) = error {
            let exclusion = Exclusion::Measurement {
                point: "device clock".into(),
                detail: error.to_string(),
            };
            for (failure, pairing) in failures.iter_mut().zip(&mut rounds.pairings) {
                failure.get_or_insert(exclusion.clone());
                pairing.admitted = 0;
            }
        }
    }
    while let Err(failure) = timing::sample(
        &mut placed,
        minimum,
        &mut rounds,
        matching.shuffle,
        matching.clock.as_deref(),
        matching.deadline,
    ) {
        let error = match failure.error {
            WorkError::Expired => break,
            WorkError::Failed(error) => error,
        };
        let exclusion = measurement_failure(
            points,
            slots.get(failure.point).map_or_else(
                || incumbent_point(&incumbent_slots, failure.point),
                |slot| slot.0,
            ),
            error,
        );
        for candidate in rounds.fail(failure.point) {
            failures[candidate] = Some(exclusion.clone());
        }
    }
    // What a challenger's part of the rounds took: the session's, shared
    // among the challengers it sampled.
    let took = rounds.pairings.iter().filter(|pairing| pairing.taken > 0).count();
    let each = sampling.elapsed().as_secs_f64() / took.max(1) as f64;
    for (failure, pairing) in failures.iter().zip(&rounds.pairings) {
        if admission.is_some() && pairing.taken > 0 && failure.is_none() {
            matching.sequence.compared(each, pairing.ahead);
        }
    }
    // Timed first, the challengers the rounds left ahead are witnessed
    // now, at every timed point: each in a cleared pass of its own.
    if timed_first {
        for (candidate, formed) in candidates.iter().enumerate() {
            let pairing = &rounds.pairings[candidate];
            if failures[candidate].is_some() || pairing.taken == 0 || !pairing.ahead {
                continue;
            }
            let witnessing = Instant::now();
            let admitted = points.iter().try_for_each(|workload| {
                workload.admit_before(
                    None,
                    &formed.kernel,
                    &|working| key(working, &formed.specialization),
                    matching.deadline,
                )
            });
            let seconds = witnessing.elapsed().as_secs_f64();
            if let Some(admission) = admission.as_deref_mut() {
                admission.witnessed_after.observe(seconds);
            }
            matching.sequence.witnessed(
                seconds,
                !matches!(
                    admitted,
                    Err(WorkError::Failed(Exclusion::Validation { .. }))
                ),
            );
            if let Err(exclusion) = admitted {
                match exclusion {
                    WorkError::Failed(exclusion) => failures[candidate] = Some(exclusion),
                    WorkError::Expired => pending[candidate] = true,
                }
            }
        }
    }
    for (slot, (point, key, measuring)) in slots.into_iter().enumerate() {
        if rounds.ratios[slot].is_empty() {
            continue;
        }
        let (ratio, deviation) = rounds.ratio(slot);
        let beside = incumbent.points[point].median_seconds;
        let result = PointMeasurement {
            median_seconds: beside * ratio,
            deviation_seconds: beside * deviation,
            ..point_measurement(&points[point].label, key.clone(), &placed[slot])
        };
        if let Some(cache) = cache.as_deref_mut() {
            cache.insert((point, key), result.clone());
        }
        for candidate in measuring {
            measured[candidate][point] = result.clone();
        }
    }
    // What each session took of a comparison is its class's observation.
    for (candidate, pairing) in candidates.iter().zip(&rounds.pairings) {
        if pairing.complete && pairing.taken > 0 {
            matching.rates.observe(&candidate.evidence, pairing.taken);
        }
    }
    measured
        .into_iter()
        .zip(failures)
        .zip(rounds.pairings)
        .zip(pending)
        .map(|(((measured, failure), pairing), pending)| {
            failure.map_or_else(
                || {
                    Ok(
                        (!pending && (pairing.terms.is_empty() || pairing.taken > 0))
                            .then_some((measured, pairing.evidence)),
                    )
                },
                Err,
            )
        })
        .collect()
}

/// The point of the incumbent's slot `slot`.
fn incumbent_point(incumbent_slots: &BTreeMap<usize, usize>, slot: usize) -> usize {
    *incumbent_slots
        .iter()
        .find(|(_, placed)| **placed == slot)
        .expect("a slot is a challenger's or the incumbent's")
        .0
}

fn measurement_failure(points: &[PreparedPoint<'_>], point: usize, error: CallError) -> Exclusion {
    Exclusion::Measurement {
        point: points[point].label.clone(),
        detail: error.to_string(),
    }
}

/// Which parameters each launch reads, as the declaration names them: those
/// its `when` condition, groups, group extent or shared bytes read, and on
/// CPU its participant count. A parameter no launch names (read only by
/// scratch sizes or the source, or the CPU tier) is taken to change every
/// launch.
struct Influence {
    launches: Vec<Vec<String>>,
    everywhere: Vec<String>,
}

impl Influence {
    fn of(implementation: &NativeImplementation) -> Self {
        let launches = (0..implementation.launches.len())
            .map(|launch| implementation.launch_parameters(launch))
            .collect::<Vec<_>>();
        let everywhere = implementation
            .params
            .iter()
            .map(|parameter| parameter.name.clone())
            .filter(|name| !launches.iter().any(|read| read.contains(name)))
            .collect();
        Self {
            launches,
            everywhere,
        }
    }

    /// The key of a point where the launches `working` do work, for a
    /// configuration with parameter `values`.
    fn key(&self, working: Vec<usize>, values: &ParameterValues) -> PointKey {
        let values = values
            .iter()
            .filter(|(name, _)| {
                self.everywhere.contains(name)
                    || working
                        .iter()
                        .any(|launch| self.launches[*launch].contains(name))
            })
            .map(|(name, value)| (name.clone(), *value))
            .collect();
        PointKey {
            launches: working,
            values,
        }
    }
}

/// How a unit's points are keyed, which follows how its programs are
/// formed. Launches formed separately each receive the parameters their
/// declaration says they read, and a parameter that only chooses which
/// launch serves is told by the launches that work ([`launch_key`]). A
/// program formed whole hands every entry parameter to its source, so a
/// parameter counts wherever a working launch names it, a launch's
/// condition included ([`Influence`]).
enum Keying {
    /// With the plan's boundary parameters.
    Launches(Vec<plan::ParameterAddress>),
    Whole(Influence),
}

impl Keying {
    /// The key of `specialization` at a point where the launches `working`
    /// do work.
    fn key(
        &self,
        implementation: &NativeImplementation,
        working: Vec<usize>,
        specialization: &NativeSpecialization,
    ) -> PointKey {
        match self {
            Self::Launches(boundary) => {
                launch_key(implementation, boundary, working, specialization)
            }
            Self::Whole(influence) => influence.key(working, specialization.params()),
        }
    }
}

/// The declared domain of a unit: the checked declaration admits its
/// configurations, and a change is forecast to run the launches the
/// configuration it changes ran.
struct Declared<'s> {
    implementation: &'s NativeImplementation,
    keying: &'s Keying,
}

impl plan::Domain for Declared<'_> {
    fn place(
        &self,
        base: &NativeSpecialization,
        parameters: &[plan::ParameterAddress],
        values: &[u64],
    ) -> Result<Option<NativeSpecialization>, plan::PlanError> {
        plan::TuningPartition::place(self.implementation, base, parameters, values)
    }

    fn nearest(
        &self,
        base: &NativeSpecialization,
        parameters: &[plan::ParameterAddress],
        values: &[u64],
    ) -> Result<Option<NativeSpecialization>, plan::PlanError> {
        plan::TuningPartition::nearest(self.implementation, base, parameters, values)
    }

    fn forecast(
        &self,
        _point: usize,
        baseline: &PointKey,
        specialization: &NativeSpecialization,
    ) -> PointKey {
        self.keying
            .key(self.implementation, baseline.launches.clone(), specialization)
    }
}

/// A configuration placed at every point, keyed.
struct Placed<'a> {
    timings: Vec<PointTiming<'a>>,
    keys: Vec<PointKey>,
}

/// Place `kernel`'s calls at every point and key them.
fn place<'a>(
    kernel: &Arc<NativePrepared>,
    points: &[PreparedPoint<'a>],
    influence: &Influence,
    values: &ParameterValues,
    outputs: &mut OutputPool,
    // A challenger is admitted, and its admission observed in its unit's
    // class; the defaults (none) are the reference.
    admission: Option<&mut Observed>,
    // Whether a challenger its unit's requirement does not admit is
    // excluded here; a survey measures it all the same.
    gated: bool,
    deadline: Deadline,
) -> Result<Placed<'a>, WorkError<Exclusion>> {
    let began = Instant::now();
    let qualify = admission.is_some();
    let timings = (|| -> Result<Vec<PointTiming<'a>>, WorkError<Exclusion>> {
        let mut timings = points
            .iter()
            .enumerate()
            .map(
                |(index, point)| -> Result<PointTiming<'a>, WorkError<Exclusion>> {
                    deadline.check(Duration::ZERO)?;
                    Ok(PointTiming::reusing_outputs(
                        kernel,
                        point.workload_rotation(),
                        outputs.at(&point.label),
                    )
                    .map_err(|error| measurement_failure(points, index, error))?)
                },
            )
            .collect::<Result<Vec<_>, _>>()?;
        for (point, timing) in points.iter().zip(&mut timings) {
            if !qualify {
                point.price_before(timing, deadline)?;
                // The defaults' first run is retained as the reference here.
                point.retain_default(timing, deadline)?;
                continue;
            }
            let kernel = timing.prepared().clone();
            let key = |working| influence.key(working, values);
            if gated {
                point.admit_before(Some(timing), &kernel, &key, deadline)?;
            } else {
                point.observe_before(Some(timing), &kernel, &key, deadline)?;
            }
        }
        Ok(timings)
    })();
    if let Some(admission) = admission {
        admission.observe(began.elapsed().as_secs_f64());
    }
    let timings = timings?;
    let keys = timings
        .iter()
        .map(|timing| influence.key(timing.working_launches(), values))
        .collect();
    Ok(Placed { timings, keys })
}

/// Measures configurations at the tuning points, once per point and key.
struct Measurer {
    influence: Influence,
    /// Every point measured so far, by point label and key.
    measured: HashMap<(String, PointKey), PointMeasurement>,
    outputs: OutputPool,
    /// Whether a challenger its unit's requirement does not admit is
    /// excluded before it is timed: always, but in a survey, which records
    /// every configuration's measurements and timing whatever its verdict.
    gated: bool,
    shuffle: Shuffle,
    /// The device's sustained clock, held from the first session
    /// ([`timing::Clock`]).
    clock: Option<Arc<timing::Clock>>,
}

impl Measurer {
    fn new(implementation: &NativeImplementation, gated: bool) -> Self {
        Self {
            influence: Influence::of(implementation),
            measured: HashMap::new(),
            outputs: OutputPool::default(),
            gated,
            shuffle: Shuffle::default(),
            clock: None,
        }
    }

    /// The measurement of `kernel` at every point: points whose key was
    /// measured before reuse that measurement, the others are sampled, on
    /// a device raised as a search with `settings` raises it.
    fn measure(
        &mut self,
        kernel: &Arc<NativePrepared>,
        points: &[PreparedPoint<'_>],
        values: &ParameterValues,
        options: &MeasureOptions,
        settings: &SearchSettings,
        admission: Option<&mut Observed>,
        deadline: Deadline,
    ) -> Result<Vec<PointMeasurement>, WorkError<Exclusion>> {
        let Placed { timings, keys } = place(
            kernel,
            points,
            &self.influence,
            values,
            &mut self.outputs,
            admission,
            self.gated,
            deadline,
        )?;
        let (fresh, mut timings): (Vec<usize>, Vec<PointTiming>) = timings
            .into_iter()
            .enumerate()
            .filter(|(point, _)| {
                !self
                    .measured
                    .contains_key(&(points[*point].label.clone(), keys[*point].clone()))
            })
            .unzip();
        // Each point's own time, the count its caller named, at the
        // device's sustained clock.
        let clock = |error: CallError| Exclusion::Measurement {
            point: "device clock".into(),
            detail: error.to_string(),
        };
        raise(
            &mut self.clock,
            &timings,
            settings,
            options.min_sample_seconds,
            deadline,
        )
        .map_err(|error| error.map(clock))?;
        timing::sample(
            &mut timings,
            options.min_sample_seconds,
            &mut timing::Counted::new(fresh.len(), options.samples),
            &mut self.shuffle,
            self.clock.as_deref(),
            deadline,
        )
        .map_err(|failure| {
            failure
                .error
                .map(|error| measurement_failure(points, fresh[failure.point], error))
        })?;
        for (point, timing) in fresh.into_iter().zip(&timings) {
            let key = keys[point].clone();
            let measurement = point_measurement(&points[point].label, key.clone(), timing);
            self.measured
                .insert((points[point].label.clone(), key), measurement);
        }
        Ok(keys
            .into_iter()
            .enumerate()
            .map(|(point, key)| self.measured[&(points[point].label.clone(), key)].clone())
            .collect())
    }
}

/// Invocation demand: the shared expected elapsed-service objective.
struct Weighing {
    weights: Vec<f64>,
}

impl Weighing {
    fn of(points: &[PreparedPoint<'_>]) -> Self {
        Self {
            weights: points.iter().map(|point| point.weight).collect(),
        }
    }

    /// The cost of `measured` relative to the defaults' measurement at the
    /// same points.
    fn cost(&self, measured: &[PointMeasurement], reference: &[PointMeasurement]) -> Cost {
        Cost::relative(
            measured
                .iter()
                .map(|measurement| measurement.key.clone())
                .collect(),
            &self.weights,
            &medians(measured),
            &medians(reference),
        )
    }
}

fn medians(measured: &[PointMeasurement]) -> Vec<f64> {
    measured
        .iter()
        .map(|measurement| measurement.median_seconds)
        .collect()
}

/// How a measured configuration stands.
#[derive(Clone, Debug, PartialEq)]
enum ContenderStanding {
    /// Measured and admitted at the timed points; not yet through
    /// [`Stage::Promote`].
    Measured,
    /// Its comparison with a choice that was served resolved as a trail
    /// ([`Verdict::Trails`]): it is slower than that choice, and than any
    /// choice that replaced it.
    Trailing,
    /// It could not be formed or sampled again for its conversion, or it
    /// failed a coverage witness.
    Rejected(Exclusion),
    /// Served now or earlier: the defaults, and every challenger promoted.
    Served,
}

/// A configuration measured and admitted at the timed points.
struct Contender {
    specialization: NativeSpecialization,
    /// Its formed kernel, held only while it serves or leads
    /// ([`Ledger::release`]): a measured configuration that does neither
    /// owns nothing on the device. What stays of it is its identity and
    /// its measurements; a later conversion forms it again.
    kernel: Option<Arc<NativePrepared>>,
    /// The identity of its formed artifact.
    artifact: String,
    /// Its time at every timed point in the unit's canonical units
    /// ([`Matched`]), as last measured: in the session that first measured
    /// it, then in each session of its conversion.
    points: Vec<PointMeasurement>,
    /// Its measurement in the last session of its conversion, once it had
    /// one.
    confirmed: Vec<PointMeasurement>,
    /// Its cost relative to the defaults' measurement.
    cost: Cost,
    /// The choice it was compared with in matched rounds, and what those
    /// rounds have shown. A lead is one over that choice: rounds beside a
    /// choice since replaced say nothing of the one served now.
    against: Configuration,
    evidence: Tally,
    standing: ContenderStanding,
}

/// Every configuration a unit's search reached, and the choice it serves.
/// The served choice is always legal: the canonical defaults, or a
/// challenger whose lead over its predecessor was resolved and that was
/// covered.
struct Ledger {
    /// In the order reached.
    order: Vec<Configuration>,
    contenders: BTreeMap<Configuration, Contender>,
    /// Configurations that could not be formed, run or admitted, each with
    /// its artifact when one was formed.
    excluded: BTreeMap<Configuration, (Option<String>, Exclusion)>,
    incumbent: Configuration,
    /// Every choice served, in turn: the defaults, then each promotion.
    served: Vec<Configuration>,
    /// Leaders that failed a coverage witness.
    failed: usize,
    /// Conversions begun, and those that ended in a promotion.
    conversions: usize,
    promotions: usize,
    /// How often a resolved comparison of a challenger with the served
    /// choice may be wrong.
    settings: SearchSettings,
    /// The measured challengers that lead the served choice as it stands,
    /// by their cost: those whose lead is resolved, by matched rounds
    /// beside that very choice ([`Verdict::Leads`]), and those whose lead
    /// is nominal, whose measured cost is below the choice's. A
    /// challenger is ranked when it is measured; all are ranked again only
    /// when what they are ranked against changes ([`Ledger::rerank`]): the
    /// served choice, a leader's measurement or the settings, each the
    /// work of a conversion.
    resolved: BTreeSet<(Total, Configuration)>,
    nominal: BTreeSet<(Total, Configuration)>,
    /// The challengers that hold a formed kernel.
    holding: BTreeSet<Configuration>,
    /// How each challenger was formed before it was first measured.
    formations: BTreeMap<Configuration, FormationRecord>,
}

/// A cost total, ordered.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Total(f64);

impl Eq for Total {}

impl PartialOrd for Total {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Total {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl Ledger {
    fn new(
        default: &NativeSpecialization,
        kernel: Arc<NativePrepared>,
        canonical: Vec<PointMeasurement>,
        cost: Cost,
        settings: SearchSettings,
    ) -> Self {
        let incumbent = Configuration::of(default);
        Self {
            settings,
            resolved: BTreeSet::new(),
            nominal: BTreeSet::new(),
            holding: BTreeSet::new(),
            formations: BTreeMap::new(),
            order: vec![incumbent.clone()],
            contenders: BTreeMap::from([(
                incumbent.clone(),
                Contender {
                    specialization: default.clone(),
                    artifact: kernel.artifact().0.clone(),
                    kernel: Some(kernel),
                    points: canonical,
                    confirmed: Vec::new(),
                    cost,
                    against: incumbent.clone(),
                    evidence: Tally::default(),
                    standing: ContenderStanding::Served,
                },
            )]),
            excluded: BTreeMap::new(),
            served: vec![incumbent.clone()],
            incumbent,
            failed: 0,
            conversions: 0,
            promotions: 0,
        }
    }

    fn incumbent(&self) -> &Contender {
        &self.contenders[&self.incumbent]
    }

    /// What `contender`'s matched rounds beside the choice served now have
    /// shown: nothing, when its rounds were beside another.
    fn evidence(&self, contender: &Contender) -> Tally {
        if contender.against == self.incumbent {
            contender.evidence
        } else {
            Tally::default()
        }
    }

    fn measured(
        &mut self,
        specialization: NativeSpecialization,
        kernel: Arc<NativePrepared>,
        points: Vec<PointMeasurement>,
        cost: Cost,
        evidence: Tally,
    ) {
        let artifact = kernel.artifact().0.clone();
        self.record(specialization, artifact, Some(kernel), points, cost, evidence);
    }

    /// Record a configuration measured and admitted at the timed points
    /// beside the served choice, where its rounds showed `evidence`, formed
    /// as `artifact`, and rank it against that choice.
    fn record(
        &mut self,
        specialization: NativeSpecialization,
        artifact: String,
        kernel: Option<Arc<NativePrepared>>,
        points: Vec<PointMeasurement>,
        cost: Cost,
        evidence: Tally,
    ) {
        let configuration = Configuration::of(&specialization);
        self.order.push(configuration.clone());
        if kernel.is_some() {
            self.holding.insert(configuration.clone());
        }
        self.contenders.insert(
            configuration.clone(),
            Contender {
                specialization,
                artifact,
                kernel,
                points,
                confirmed: Vec::new(),
                cost,
                against: self.incumbent.clone(),
                evidence,
                standing: ContenderStanding::Measured,
            },
        );
        self.rank(&configuration);
    }

    /// Rank the measured challenger `configuration` against the served
    /// choice. Its lead is resolved when its rounds beside that choice say
    /// so, and nominal when its measured cost is below the choice's
    /// ([`Cost::improves_on`]). One whose rounds resolved a trail is not
    /// ranked again.
    fn rank(&mut self, configuration: &Configuration) {
        let incumbent = self.incumbent();
        let contender = &self.contenders[configuration];
        let verdict = self
            .evidence(contender)
            .verdict(self.settings.error_rate);
        let total = Total(contender.cost.total());
        match verdict {
            Some(Verdict::Trails) => {
                self.contenders
                    .get_mut(configuration)
                    .expect("a ranked configuration is a contender")
                    .standing = ContenderStanding::Trailing;
            }
            Some(Verdict::Leads) => {
                self.resolved.insert((total, configuration.clone()));
            }
            None if contender.cost.improves_on(&incumbent.cost) => {
                self.nominal.insert((total, configuration.clone()));
            }
            None => {}
        }
    }

    /// Rank every measured challenger again: what they are ranked against
    /// changed. The one place whose work follows the number measured.
    fn rerank(&mut self) {
        self.resolved.clear();
        self.nominal.clear();
        let measured = self
            .contenders
            .iter()
            .filter(|(_, contender)| contender.standing == ContenderStanding::Measured)
            .map(|(configuration, _)| configuration.clone())
            .collect::<Vec<_>>();
        for configuration in measured {
            self.rank(&configuration);
        }
    }

    /// Hold the settings leads are decided by.
    fn settings(&mut self, settings: &SearchSettings) {
        if self.settings != *settings {
            self.settings = settings.clone();
            self.rerank();
        }
    }

    /// `leader` holds `kernel`, formed again for its conversion.
    fn hold(&mut self, leader: &Configuration, kernel: Arc<NativePrepared>) {
        self.contenders
            .get_mut(leader)
            .expect("the leader is a contender")
            .kernel = Some(kernel);
        self.holding.insert(leader.clone());
    }

    /// Release the kernel of every configuration that neither serves nor
    /// leads: the served incumbent's and the leader's owed a conversion,
    /// resolved or nominal, are the only kernels a unit holds between
    /// pieces of work.
    fn release(&mut self) {
        let leaders = [
            Some(self.incumbent.clone()),
            self.leader().cloned(),
            self.nominal().map(|(leader, _)| leader.clone()),
        ];
        let Self {
            holding,
            contenders,
            ..
        } = self;
        holding.retain(|configuration| {
            let held = leaders.iter().flatten().any(|leader| leader == configuration);
            if !held {
                contenders
                    .get_mut(configuration)
                    .expect("a holder is a contender")
                    .kernel = None;
            }
            held
        });
    }

    /// Record that `configuration`, formed as `kernel` if it was formed,
    /// could not be measured or admitted.
    fn exclude(
        &mut self,
        configuration: Configuration,
        kernel: Option<&NativePrepared>,
        exclusion: Exclusion,
    ) {
        self.order.push(configuration.clone());
        self.excluded.insert(
            configuration,
            (kernel.map(|kernel| kernel.artifact().0.clone()), exclusion),
        );
    }

    /// The leader owed its conversion, if any: the cheapest measured
    /// challenger whose lead is resolved.
    fn leader(&self) -> Option<&Configuration> {
        self.resolved.first().map(|(_, leader)| leader)
    }

    /// The cheapest measured challenger whose lead is nominal, and that
    /// lead: its measured cost is below the served choice's, and its
    /// rounds beside that choice have resolved nothing yet.
    fn nominal(&self) -> Option<(&Configuration, f64)> {
        self.nominal
            .first()
            .map(|(total, leader)| (leader, self.incumbent().cost.total() - total.0))
    }

    /// One record per configuration reached. A measurement is recorded
    /// when it is taken, whether or not its configuration ever serves, and
    /// every observation `validator` holds of a configuration's artifact is
    /// its record's, whatever its verdict; the defaults are the reference
    /// and carry no evidence against themselves.
    fn records(&self, validator: &Validator) -> Vec<ConfigurationRecord> {
        self.order
            .iter()
            .map(|configuration| {
                let (artifact, outcome) = match (
                    self.contenders.get(configuration),
                    self.excluded.get(configuration),
                ) {
                    (Some(contender), _) => (
                        Some(contender.artifact.clone()),
                        match &contender.standing {
                            ContenderStanding::Rejected(exclusion) => {
                                Outcome::Excluded(exclusion.clone())
                            }
                            _ => Outcome::Measured {
                                points: contender.points.clone(),
                                confirmed: contender.confirmed.clone(),
                            },
                        },
                    ),
                    (None, Some((artifact, exclusion))) => {
                        (artifact.clone(), Outcome::Excluded(exclusion.clone()))
                    }
                    (None, None) => unreachable!("every configuration reached has an outcome"),
                };
                ConfigurationRecord {
                    formation: self.formations.get(configuration).copied(),
                    configuration: configuration.clone(),
                    observations: artifact
                        .as_deref()
                        .map(|artifact| validator.held(artifact))
                        .unwrap_or_default(),
                    artifact,
                    outcome,
                }
            })
            .collect()
    }
}

/// Candidate eligibility covers the entire served numerical scope, independently
/// of objective weights. Untimed witnesses are constructed and released one at
/// a time; what is observed at them is the validator's, as at timed points.
struct ScopeQualification<'s, 'a> {
    deadline: Deadline,
    source: &'s mut dyn PointSource<'a>,
    specs: &'s [PointSpec],
    indices: &'s [usize],
    validator: &'s Validator,
    /// The artifacts admitted at every untimed witness.
    covered: &'s mut BTreeSet<String>,
    default: &'s str,
    /// The key of the checked configuration at a point where the given
    /// launches work.
    key: &'s dyn Fn(Vec<usize>) -> PointKey,
    /// The whole check of each untimed point, by point: its class.
    checks: &'s mut [Observed],
    building_seconds: f64,
    validating_seconds: f64,
    reference_seconds: f64,
}

impl ScopeQualification<'_, '_> {
    /// Check `kernel` at every untimed point. The check runs to its end or
    /// to its first failing witness: eligibility is published only after
    /// every witness admits, while what was observed before a failure
    /// stays recorded without making its artifact eligible.
    fn qualify(&mut self, kernel: &Arc<NativePrepared>) -> Result<(), WorkError<Exclusion>> {
        let artifact = &kernel.artifact().0;
        // No check is owed for the reference or an empty witness set.
        if artifact == self.default || self.indices.is_empty() {
            self.covered.insert(artifact.clone());
            return Ok(());
        }
        for &index in self.indices {
            self.deadline.check(Duration::ZERO)?;
            let spec = &self.specs[index];
            let began = Instant::now();
            let built = self.source.build(index);
            self.building_seconds += began.elapsed().as_secs_f64();
            let result = (|| {
                let inputs = built.map_err(Exclusion::Execution)?;
                let mut point = TuningPoint::new(spec, inputs)
                    .and_then(|point| self.validator.point(point))
                    .map_err(|error| Exclusion::Execution(error.to_string()))?;
                let validating = Instant::now();
                let references_before = self.validator.reference_seconds();
                let admitted = point.admit_before(None, kernel, self.key, self.deadline);
                self.validating_seconds += validating.elapsed().as_secs_f64();
                self.reference_seconds +=
                    (self.validator.reference_seconds() - references_before).max(0.0);
                // The witness is held for this check alone.
                point.release();
                admitted
            })();
            self.source.release(index);
            self.checks[index].observe(began.elapsed().as_secs_f64());
            result?;
        }
        self.covered.insert(artifact.clone());
        Ok(())
    }
}

/// Tune one unit within its budget: its canonical defaults formed, then the
/// [`UnitSearch`]'s work started piece by piece while each piece's whole
/// price fits what remains ([`Quote::fits`]), under the original cutoff;
/// or a survey of every admissible configuration. The result is the unit's
/// served choice when no further piece fits.
pub fn tune(request: TuneRequest<'_, '_>) -> Result<TuningResult, TuneError> {
    let began = Instant::now();
    let TuneRequest {
        device,
        module,
        entry,
        bindings,
        statics,
        cpu,
        points,
        validation,
        strategy,
    } = request;
    // The standalone consumer owns its formation's threads as a load's
    // tuner owns the load's: a device nobody lent workers gets its own for
    // this tuning, and they leave with it.
    struct Staffed<'d>(&'d DeviceInner);
    impl Drop for Staffed<'_> {
        fn drop(&mut self) {
            self.0.staff(None);
        }
    }
    let _staffed = device.programs.workers().is_none().then(|| {
        device.staff(Some(Arc::new(Workers::of_the_host())));
        Staffed(device)
    });
    let request = UnitRequest {
        device,
        module,
        entry,
        bindings,
        statics,
        cpu,
        validation,
    };
    match strategy {
        Strategy::Search(plan) => {
            let mut unit = UnitSearch::open(request)?;
            unit.prepare_defaults(points)?;
            unit.plan(UnitPlan {
                settings: plan.settings,
                min_sample_seconds: plan.min_sample_seconds,
                start: plan.start,
            });
            // The standalone consumer admits work by the same rule as the
            // multi-unit allocator.
            while let Some(work) = unit.next() {
                let remaining = plan.allowance.saturating_sub(began.elapsed());
                if !work.quote.fits(remaining, unit.longest_operation()) {
                    break;
                }
                unit.perform(points, Some(began + plan.allowance))?;
            }
            unit.terminal()
        }
        Strategy::Survey(plan) => {
            UnitSearch::open_over(request, &plan.domains)?.survey(points, plan)
        }
    }
}

/// What opening a unit's search needs.
pub struct UnitRequest<'r, 'm> {
    pub device: &'r Arc<DeviceInner>,
    pub module: &'m CheckedModule,
    pub entry: EntryId,
    pub bindings: ElementBindings,
    /// Values of every static dimension.
    pub statics: NativeSpecialization,
    pub cpu: Option<&'static CpuNativeKernels>,
    /// What every candidate is held to against the native default.
    pub validation: NumericalRequirement,
}

/// The defaults formed, and the validator of every point against the
/// reference.
struct Ground {
    kernel: Arc<NativePrepared>,
    validator: Validator,
}

/// A point discovered: built, and the defaults measured there.
struct Timed<'a> {
    point: PreparedPoint<'a>,
    defaults: PointMeasurement,
    /// The dimensions of the point's calls, read when its inputs were
    /// first built: planning the search needs them, not the inputs.
    shape: PointShape,
}

/// How far a unit's optional work is.
enum Progress<'a> {
    /// Measuring the defaults: each point's slot, by its position among
    /// the unit's points.
    Discovering(Vec<Option<Timed<'a>>>),
    Searching(Box<Searching<'a>>),
}

/// A unit searching at its timed points.
struct Searching<'a> {
    /// The timed points. Each keeps what was learned at it; its inputs are
    /// resident only while the unit's work uses them
    /// ([`UnitSearch::release`]).
    points: Vec<PreparedPoint<'a>>,
    /// Each timed point's position among the unit's points.
    timed: Vec<usize>,
    /// The points not timed: a leader is checked there before it serves.
    untimed: Vec<usize>,
    /// The artifacts admitted at every untimed point.
    covered: BTreeSet<String>,
    /// The defaults' measurement at the timed points: the reference of
    /// every cost.
    canonical: Vec<PointMeasurement>,
    ledger: Ledger,
    run: Run,
    /// The next challenger work, planned when the last ended.
    planned: Option<Piece>,
    /// What admitting one challenger at the timed points has taken, in
    /// either order of its tests: the classes every challenger's admission
    /// is priced by.
    admission: Admission,
    /// How the rounds of the unit's comparisons have come out: what the
    /// rounds of its next session are priced by.
    rates: Rates,
    /// What planning the next piece has taken, each planning with the
    /// requests it made of the declared domain
    /// ([`GroupSearch::requests`]): a domain that admits few of the
    /// assignments it declares is asked for many to find the next
    /// challengers, inside the budget.
    planning: Linear,
}

/// The challengers one [`Stage::Form`] or [`Stage::Measure`] forms and
/// measures.
#[derive(Clone, Debug)]
pub enum Piece {
    /// Whole configurations, each measured as itself: every other form's
    /// representative, or every boundary base still unmeasured, or every
    /// coverage candidate, or every generated combination. They are formed
    /// together: one compile per launch source, or their programs at once.
    Whole(Vec<NativeSpecialization>),
    /// Candidates of one group measured together against the group's
    /// `baseline`, with the candidates measured next (`ahead`) formed in
    /// the same compile where one compile of a launch's source forms them.
    Sweep {
        group: usize,
        baseline: Vec<PointMeasurement>,
        ready: Vec<NativeSpecialization>,
        ahead: Vec<NativeSpecialization>,
    },
}

impl Piece {
    /// The challengers the piece measures.
    pub fn challengers(&self) -> &[NativeSpecialization] {
        match self {
            Piece::Whole(specializations) => specializations,
            Piece::Sweep { ready, .. } => ready,
        }
    }

    /// The challengers measured after the piece's whose launch variants
    /// its compile also forms.
    fn ahead(&self) -> &[NativeSpecialization] {
        match self {
            Piece::Whole(_) => &[],
            Piece::Sweep { ahead, .. } => ahead,
        }
    }
}

/// A unit's search as it runs on the device: the order it asks its domain
/// in, and what forming and measuring what it asks for holds.
struct Run {
    search: GroupSearch,
    keying: Keying,
    programs: Programs,
    /// Every configuration placed in this run writes its results here.
    outputs: OutputPool,
    /// Measurements by point and key, in the unit's canonical units: a
    /// point with the same active launches and parameters runs the same
    /// work under every candidate and boundary.
    samples: BTreeMap<(usize, PointKey), PointMeasurement>,
    /// The order of every round's samples, from session to session.
    shuffle: Shuffle,
    /// The device's sustained clock, held as the unit's census held it
    /// ([`timing::Clock`]).
    clock: Option<Arc<timing::Clock>>,
    /// The order the unit's challengers take their two tests in.
    sequence: Sequence,
}

/// One unit's tuning, advanced one piece of work at a time so that a
/// consumer tuning several units decides which of them the next piece
/// belongs to.
///
/// From [`UnitSearch::prepare_defaults`] on, the unit holds a legal served
/// choice and a complete record of it ([`UnitSearch::terminal`]): first its
/// canonical defaults. Everything after is optional and can only replace
/// that choice with a better one, atomically; nothing is owed at the end.
/// [`UnitSearch::next`] names the next piece with its whole price, and
/// [`UnitSearch::perform`] runs it to its end:
///
/// - [`Stage::Discovery`]: one point's inputs built and the defaults
///   measured there, until every discovery point has both;
/// - [`Stage::Form`]: the first representative of each other structural
///   form, formed, measured and admitted like any challenger, ahead of
///   every refinement, whichever way the unit's programs are formed;
/// - [`Stage::Measure`]: a challenger formed, admitted at the timed points
///   and measured there in rounds matched with the served choice, and
///   recorded whatever it measures;
/// - [`Stage::Promote`]: the leader among the challengers whose rounds
///   resolved a lead over the served choice, checked at the untimed points
///   and on success served in its place. A resolved lead is converted
///   before any further challenger is measured;
/// - [`Stage::Confirm`]: more matched rounds for the leader among those
///   whose lead is nominal, and its conversion if they resolve it.
pub struct UnitSearch<'m, 'a> {
    expired: bool,
    device: Arc<DeviceInner>,
    module: &'m CheckedModule,
    entry: EntryId,
    bindings: ElementBindings,
    statics: NativeSpecialization,
    cpu: Option<&'static CpuNativeKernels>,
    backend: seismic_lang::registry::BackendName,
    implementation: NativeImplementation,
    logical: Arc<LogicalEntry>,
    mutable: Vec<(usize, String)>,
    default: NativeSpecialization,
    unit: Unit,
    /// The unit's points, once a step was given their source.
    specs: Vec<PointSpec>,
    /// The served shapes beyond the points', learned with them.
    served: Vec<BTreeMap<String, u64>>,
    ground: Option<Ground>,
    /// Programs formed for the defaults.
    grounded: usize,
    /// Measures the defaults as points are discovered; none once the
    /// search is open, which places into its own result storage.
    measurer: Option<Measurer>,
    plan: Option<UnitPlan>,
    progress: Progress<'a>,
    /// Building each point's inputs, by point: its class.
    builds: Vec<Observed>,
    /// The whole check of each untimed point, by point: its class.
    checks: Vec<Observed>,
    /// The longest measurement of the defaults at one point, and the
    /// longest planning of the search: operations of no repeated class.
    longest: f64,
    /// What making each point's inputs resident added to the bytes held,
    /// by point: its class, observed when the point is discovered and
    /// whenever its inputs are built again.
    resident_bytes: Vec<Observed>,
    /// What measuring one challenger added at its most to the bytes held
    /// with the unit's inputs resident, and what one conversion did: the
    /// classes a challenger piece's and a conversion's bytes are priced by.
    measure_bytes: Observed,
    conversion_bytes: Observed,
    /// The most a single piece of the unit's work added to the bytes held:
    /// what an unpriced footprint is held to ([`Footprint::fits`]).
    largest_bytes: u64,
    /// What compiles of this unit took: the price of every compile of it.
    formation_price: Mutex<FormationPrice>,
    time: TuningTime,
    /// The next piece of work as last derived ([`UnitSearch::next`]); none
    /// since the unit's state last changed. With it, what each formation
    /// its price read does as it was read then.
    offered: std::cell::Cell<Option<Option<Work>>>,
    offered_forming: std::cell::RefCell<Vec<(Formable, Formative)>>,
}

impl<'m, 'a> UnitSearch<'m, 'a> {
    pub fn open(request: UnitRequest<'_, 'm>) -> Result<Self, TuneError> {
        Self::open_over(request, &BTreeMap::new())
    }

    /// Open the unit with `domains` replacing the declared value lists of
    /// the parameters they name (a survey's widened space).
    fn open_over(
        request: UnitRequest<'_, 'm>,
        domains: &BTreeMap<String, Vec<u64>>,
    ) -> Result<Self, TuneError> {
        let UnitRequest {
            device,
            module,
            entry,
            bindings,
            statics,
            cpu,
            validation: requirement,
        } = request;
        let backend = super::backend_name(&device.kind);
        let entry_name = super::entry_name(module, entry);
        let mut implementation = super::on_device(
            device,
            module
                .native_implementation(entry, backend)
                .cloned()
                .ok_or_else(|| {
                    TuneError::Declaration(format!(
                        "`{entry_name}` has no native implementation for `{}`",
                        backend.as_str()
                    ))
                })?,
        );
        widen(&mut implementation, domains)?;
        // Every configuration formed is this one entry at these bindings.
        let logical = Arc::new(
            module
                .entry(entry, &bindings)
                .map_err(|error| TuneError::Declaration(error.to_string()))?,
        );
        let mutable = mutable_parameters(&logical);
        let implementation_identity =
            implementation_digest(device, module, entry, &bindings, &statics, cpu)?;
        let default = implementation
            .default_specialization(&statics)
            .map_err(|error| TuneError::Declaration(error.to_string()))?;
        Ok(Self {
            expired: false,
            unit: Unit {
                tuning_identity: device.tuning_identity(),
                entry: entry_name,
                backend: backend.as_str().to_owned(),
                requirement,
                implementation_identity,
                parameters: entry_parameters(&implementation),
            },
            device: device.clone(),
            module,
            entry,
            bindings,
            statics,
            cpu,
            backend,
            measurer: Some(Measurer::new(&implementation, true)),
            implementation,
            logical,
            mutable,
            default,
            specs: Vec::new(),
            served: Vec::new(),
            ground: None,
            grounded: 0,
            plan: None,
            progress: Progress::Discovering(Vec::new()),
            builds: Vec::new(),
            checks: Vec::new(),
            longest: 0.,
            resident_bytes: Vec::new(),
            measure_bytes: Observed::default(),
            conversion_bytes: Observed::default(),
            largest_bytes: 0,
            formation_price: Mutex::new(FormationPrice::default()),
            time: TuningTime::default(),
            offered: std::cell::Cell::new(None),
            offered_forming: std::cell::RefCell::default(),
        })
    }

    /// Learn the unit's points from `source`, at the first step given one.
    fn describe(&mut self, source: &dyn PointSource<'a>) -> Result<(), TuneError> {
        if self.specs.is_empty() {
            self.specs = source.points();
            if self.specs.is_empty() {
                return Err(TuneError::NoPoints);
            }
            self.served = source.served();
            self.progress = Progress::Discovering(self.specs.iter().map(|_| None).collect());
            self.offered.set(None);
            self.builds = vec![Observed::default(); self.specs.len()];
            self.checks = vec![Observed::default(); self.specs.len()];
            self.resident_bytes = vec![Observed::default(); self.specs.len()];
        }
        Ok(())
    }

    /// The mandatory work of a unit: form the declared canonical program
    /// and hand it to serving. From here the unit holds a legal served
    /// choice ([`UnitSearch::terminal`]); a failure here is the unit's only
    /// preparation failure. Formation alone creates no evidence.
    pub fn prepare_defaults(&mut self, source: &mut dyn PointSource<'a>) -> Result<(), TuneError> {
        match self.grounding(source)? {
            Some(grounding) => self.grounded(grounding.run()),
            None => Ok(()),
        }
    }

    /// The compile of the unit's mandatory work, to run on any thread;
    /// none once the canonical program is formed. Units' canonical
    /// programs form concurrently; each unit then takes its own
    /// ([`UnitSearch::grounded`]).
    pub fn grounding(
        &mut self,
        source: &dyn PointSource<'a>,
    ) -> Result<Option<Grounding<'m>>, TuneError> {
        self.describe(source)?;
        Ok(self.ground.is_none().then(|| Grounding {
            device: self.device.clone(),
            module: self.module,
            entry: self.entry,
            logical: self.logical.clone(),
            bindings: self.bindings.clone(),
            default: self.default.clone(),
            cpu: self.cpu,
            implementation: self.implementation.clone(),
            served: self.served.clone(),
        }))
    }

    /// Take the canonical program formed for this unit and hand it to
    /// serving: the end of the unit's mandatory work.
    pub fn grounded(&mut self, grounded: Grounded) -> Result<(), TuneError> {
        let kernel = grounded.kernel?;
        let validator = Validator::new(
            &self.device,
            &self.logical,
            &self.mutable,
            &self.unit.requirement,
            &kernel,
        );
        kernel
            .hand_off()
            .map_err(|error| TuneError::Publication(error.to_string()))?;
        self.ground = Some(Ground { kernel, validator });
        self.time.forming_seconds += grounded.elapsed.as_secs_f64();
        // Every launch of the canonical program, or the one program.
        self.formation_price
            .get_mut()
            .expect("formation price lock")
            .canonical = grounded.elapsed;
        self.offered.set(None);
        self.grounded = if self.implementation.launch_scoped() {
            self.implementation.launches.len()
        } else {
            1
        };
        Ok(())
    }

    /// What `piece` compiles before it measures, and so what may be
    /// compiled ahead of it: the launch sources its challengers run that no
    /// held program has, or the programs of its challengers not yet
    /// formed.
    fn compiled_ahead(&self, searching: &Searching<'a>, piece: &Piece) -> Vec<Formable> {
        searching
            .run
            .programs
            .formables(&self.implementation, piece.challengers(), piece.ahead())
    }

    /// The compile of the unit's next piece of work
    /// ([`Work::forming`]), to run ahead of it on any thread while the
    /// device does other work; none when the piece compiles nothing ahead.
    /// The piece then forms nothing ([`UnitSearch::formed`]).
    pub fn forming(&self) -> Option<Forming<'m>> {
        let Progress::Searching(searching) = &self.progress else {
            return None;
        };
        if !matches!(self.next()?.stage, Stage::Form | Stage::Measure) {
            return None;
        }
        self.formation()
            .forming(self.compiled_ahead(searching, searching.planned.as_ref()?))
    }

    /// Hold the programs of the next piece's compile, which ran
    /// ([`UnitSearch::forming`]). Returns what the compile took.
    pub fn formed(&mut self, formed: Formed) -> Result<Duration, TuneError> {
        let Progress::Searching(searching) = &mut self.progress else {
            unreachable!("only a search's piece compiles ahead");
        };
        let elapsed = searching.run.programs.hold(
            formed,
            self.formation_price
                .get_mut()
                .expect("formation price lock"),
        )?;
        self.time.forming_seconds += elapsed.as_secs_f64();
        self.offered.set(None);
        Ok(elapsed)
    }

    /// How the unit's optional work measures and searches.
    pub fn plan(&mut self, plan: UnitPlan) {
        // The same plan again changes nothing the unit offers.
        if self.plan.as_ref() == Some(&plan) {
            return;
        }
        if let Progress::Searching(searching) = &mut self.progress {
            searching.ledger.settings(&plan.settings);
        }
        self.plan = Some(plan);
        self.offered.set(None);
    }

    fn planned(&self) -> &UnitPlan {
        self.plan
            .as_ref()
            .expect("a unit is given its plan before its optional work")
    }

    fn formation(&self) -> Formation<'_, 'm> {
        Formation {
            device: &self.device,
            module: self.module,
            entry: self.entry,
            logical: &self.logical,
            bindings: &self.bindings,
            cpu: self.cpu,
            implementation: &self.implementation,
            default: &self.default,
            served: &self.served,
            price: &self.formation_price,
        }
    }

    /// The timed points built so far.
    fn points(&self) -> Box<dyn Iterator<Item = &PreparedPoint<'a>> + '_> {
        match &self.progress {
            Progress::Discovering(slots) => {
                Box::new(slots.iter().flatten().map(|timed| &timed.point))
            }
            Progress::Searching(searching) => Box::new(searching.points.iter()),
        }
    }

    /// What the unit's clock did since this was last asked: its raises and
    /// its establishment ([`timing::Clock::raises`]). The clock is the
    /// census's until a search takes it.
    fn raises(&self) -> Vec<Raise> {
        let searching = match &self.progress {
            Progress::Searching(searching) => searching.run.clock.as_ref(),
            Progress::Discovering(_) => None,
        };
        searching
            .or_else(|| self.measurer.as_ref().and_then(|measurer| measurer.clock.as_ref()))
            .map_or_else(Vec::new, |clock| clock.raises())
    }

    /// Everything the unit's operations have taken so far, by class.
    fn spent(&self) -> Spent {
        let mut spent = Spent {
            compiling: self
                .formation_price
                .lock()
                .expect("formation price lock")
                .total(),
            building: self.builds.iter().map(Observed::total).sum(),
            checking: self.checks.iter().map(Observed::total).sum(),
            ..Spent::default()
        };
        for point in self.points() {
            let prices = point.prices();
            spent.placing += prices.placement.total();
            spent.witnessing += prices.witness.total();
            spent.warming += prices.warm.total();
            spent.sampling += prices.sample.total();
        }
        if let Progress::Searching(searching) = &self.progress {
            spent.planning = searching.planning.total();
        }
        spent
    }

    /// Give up what only running work needs. A unit's inputs are owned by
    /// its work: every point's built inputs, the default's observations of
    /// them and the result storage of its placements leave with this call,
    /// and the source drops what it built ([`PointSource::release`]). The
    /// unit keeps its served choice, the kernels of that choice and of a
    /// leader, its record and the prices of its operation classes: numbers,
    /// identities and what serving takes. Work that follows builds the
    /// inputs again, and is priced with that ([`UnitSearch::next`]). A unit
    /// with no work left gives up its held programs too.
    ///
    /// Whether a unit's inputs stay between two pieces of its work is its
    /// consumer's decision: building them again costs seconds of the
    /// load's budget and holding them costs bytes of the device's plan,
    /// and only the consumer sees both across units.
    pub fn release(&mut self, source: &mut dyn PointSource<'a>) {
        let exhausted = self.next().is_none();
        // A piece that follows builds the inputs again.
        self.offered.set(None);
        if let Some(measurer) = &mut self.measurer {
            measurer.outputs = OutputPool::default();
        }
        match &mut self.progress {
            Progress::Discovering(slots) => {
                for (index, timed) in slots.iter_mut().enumerate() {
                    if let Some(timed) = timed.as_mut().filter(|timed| timed.point.resident()) {
                        timed.point.release();
                        source.release(index);
                    }
                }
            }
            Progress::Searching(searching) => {
                for (point, index) in searching.points.iter_mut().zip(&searching.timed) {
                    if point.resident() {
                        point.release();
                        source.release(*index);
                    }
                }
                searching.run.outputs = OutputPool::default();
                if exhausted {
                    searching.run.programs.clear();
                }
            }
        }
    }

    /// Offer the next piece with one challenger fewer ([`UnitSearch::next`]
    /// then prices the smaller piece); false when it has one challenger or
    /// measures none. The challenger left out is measured next: nothing
    /// asked of the domain is lost. A consumer asks this when the piece's
    /// seconds do not fit what remains, so that the last piece of a budget
    /// is the largest that fits it.
    pub fn narrow(&mut self) -> bool {
        let Progress::Searching(searching) = &mut self.progress else {
            return false;
        };
        let narrowed = match &mut searching.planned {
            Some(Piece::Whole(challengers)) if challengers.len() > 1 => {
                challengers.pop();
                true
            }
            Some(Piece::Sweep { ready, ahead, .. }) if ready.len() > 1 => {
                let next = ready.pop().expect("more than one challenger");
                ahead.insert(0, next);
                true
            }
            _ => false,
        };
        if narrowed {
            self.offered.set(None);
        }
        narrowed
    }

    /// Whether any of the unit's points has its inputs resident.
    pub fn holds_inputs(&self) -> bool {
        self.points().any(|point| point.resident())
    }

    /// What the unit's resident inputs hold, as making them resident was
    /// observed to add: what giving them up is taken to give back.
    pub fn held_inputs(&self) -> u64 {
        let indices: Box<dyn Iterator<Item = (usize, bool)> + '_> = match &self.progress {
            Progress::Discovering(slots) => Box::new(
                slots
                    .iter()
                    .enumerate()
                    .filter_map(|(index, timed)| Some((index, timed.as_ref()?.point.resident()))),
            ),
            Progress::Searching(searching) => Box::new(
                searching
                    .timed
                    .iter()
                    .copied()
                    .zip(searching.points.iter().map(PreparedPoint::resident)),
            ),
        };
        indices
            .filter(|(_, resident)| *resident)
            .filter_map(|(index, _)| self.resident_bytes[index].price())
            .sum::<f64>()
            .ceil() as u64
    }

    /// The price in bytes of one held program of the unit: the mean of
    /// what its compiles added for each program they formed, nothing when
    /// that mean is below nothing; none before a compile of the unit ran.
    fn program_price(&self) -> Option<f64> {
        self.formation_price
            .lock()
            .expect("formation price lock")
            .program_bytes
            .price()
            .map(|bytes| bytes.max(0.))
    }

    /// How many programs the unit's search holds formed: what it gives up
    /// when its consumer asks ([`UnitSearch::give_up_programs`]). The
    /// programs of the served choice and of a leader are their kernels',
    /// not among these.
    pub fn programs_held(&self) -> usize {
        match &self.progress {
            Progress::Discovering(_) => 0,
            Progress::Searching(searching) => searching.run.programs.holding(),
        }
    }

    /// What the unit's held programs hold, as forming them was observed to
    /// add: what giving them up is taken to give back.
    pub fn held_programs(&self) -> u64 {
        self.program_price()
            .map_or(0, |price| (price * self.programs_held() as f64).ceil() as u64)
    }

    /// Give up the programs the unit's search holds. A piece that follows
    /// forms what it runs again and is priced with that, in seconds and in
    /// bytes. Whether a unit's programs stay between two pieces of its work
    /// is its consumer's decision, as with its inputs
    /// ([`UnitSearch::release`]): forming them again costs seconds of the
    /// load's budget and holding them costs bytes, and only the consumer
    /// sees both across units.
    pub fn give_up_programs(&mut self) {
        if let Progress::Searching(searching) = &mut self.progress {
            searching.run.programs.clear();
        }
        self.offered.set(None);
    }

    /// The bytes held now ([`held_bytes`]).
    fn held(&self) -> u64 {
        held_bytes(
            &self.device,
            self.device.memory_usage().charged,
            self.ground
                .as_ref()
                .map_or(0, |ground| ground.validator.retained_bytes()),
            ProcessMemory::footprint_now(),
        )
    }

    /// The most a single piece of this unit's work has added to the bytes
    /// held: what [`Footprint::fits`] holds an unpriced operation to.
    pub fn largest_footprint(&self) -> u64 {
        self.largest_bytes
    }

    /// Build again the inputs of every timed point that gave its own up,
    /// for the piece about to run. Each build is observed in its point's
    /// classes, seconds and bytes: what the piece was quoted. False when
    /// the device's memory limit refused a build ([`build_point`]): the
    /// piece cannot run.
    fn restore(
        &mut self,
        source: &mut dyn PointSource<'a>,
        deadline: Deadline,
    ) -> Result<bool, WorkError<TuneError>> {
        let Self {
            progress,
            ground,
            specs,
            builds,
            resident_bytes,
            time,
            device,
            ..
        } = self;
        let Progress::Searching(searching) = progress else {
            return Ok(true);
        };
        let validator = &ground
            .as_ref()
            .expect("canonical preparation precedes the search")
            .validator;
        let held = || {
            held_bytes(
                device,
                device.memory_usage().charged,
                validator.retained_bytes(),
                ProcessMemory::footprint_now(),
            )
        };
        for (point, &index) in searching.points.iter_mut().zip(&searching.timed) {
            if point.resident() {
                continue;
            }
            deadline.check(Duration::ZERO)?;
            let before = held();
            let building = Instant::now();
            let Some(inputs) = build_point(
                device,
                source,
                &specs[index],
                index,
                &mut resident_bytes[index],
            )?
            else {
                return Ok(false);
            };
            let built = building.elapsed().as_secs_f64();
            builds[index].observe(built);
            time.building_seconds += built;
            validator.restore(point, TuningPoint::new(&specs[index], inputs)?)?;
            resident_bytes[index].observe(held().saturating_sub(before) as f64);
        }
        Ok(true)
    }

    /// What making resident the timed points that are not costs the next
    /// piece: each one's building, in seconds and in bytes.
    fn rebuild(&self, searching: &Searching<'a>) -> (Quote, Footprint) {
        let (mut quote, mut footprint) = (Quote::default(), Footprint::default());
        for (point, index) in searching.points.iter().zip(&searching.timed) {
            if !point.resident() {
                quote.add(self.builds[*index].price(), 1);
                footprint.add(most_bytes(&self.resident_bytes[*index]), 1);
            }
        }
        (quote, footprint)
    }

    /// The longest single operation this unit has completed: what
    /// [`Quote::fits`] holds an unpriced operation to.
    pub fn longest_operation(&self) -> Duration {
        let observed = self
            .points()
            .map(|point| point.prices().longest())
            .chain(self.builds.iter().map(Observed::longest))
            .chain(self.checks.iter().map(Observed::longest))
            .chain(match &self.progress {
                Progress::Searching(searching) => Some(searching.planning.longest()),
                Progress::Discovering(_) => None,
            })
            .fold(self.longest, f64::max);
        Duration::from_secs_f64(observed).max(
            self.formation_price
                .lock()
                .expect("formation price lock")
                .longest(),
        )
    }

    /// The next discovery point: the first whose defaults are unmeasured.
    fn undiscovered(&self) -> Option<usize> {
        let Progress::Discovering(slots) = &self.progress else {
            return None;
        };
        (0..self.specs.len())
            .find(|index| discovery_selected(&self.specs, *index) && slots[*index].is_none())
    }

    /// The unit's next piece of work with its whole price; none when it has
    /// none left. A lead is converted before anything else is measured.
    ///
    /// The piece and its price follow from the unit's own state, and its
    /// compile's from what is live on the device and kept by the store,
    /// which other units' work changes too ([`Rendered`]). So they are
    /// derived once after each change of the unit's state and stand while
    /// every formation the compile's price read does what it did: a
    /// consumer asking every unit before each piece of any pays for the
    /// units whose piece changed.
    pub fn next(&self) -> Option<Work> {
        if self.expired {
            return None;
        }
        if let Some(offered) = self.offered.get() {
            let formation = self.formation();
            let standing = self
                .offered_forming
                .borrow()
                .iter()
                .all(|(formable, class)| formation.class(formable) == *class);
            if standing {
                return offered;
            }
        }
        let offered = self.offer();
        *self.offered_forming.borrow_mut() = match &self.progress {
            Progress::Searching(searching) => searching
                .planned
                .as_ref()
                .map(|piece| {
                    let formation = self.formation();
                    self.compiled_ahead(searching, piece)
                        .into_iter()
                        .map(|formable| {
                            let class = formation.class(&formable);
                            (formable, class)
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Progress::Discovering(_) => Vec::new(),
        };
        self.offered.set(Some(offered));
        offered
    }

    /// Derive the unit's next piece of work and its price.
    fn offer(&self) -> Option<Work> {
        match &self.progress {
            Progress::Discovering(slots) => {
                let index = self.undiscovered()?;
                // Building a point's inputs and measuring the defaults
                // there are operations of classes first observed by this
                // very work; discovering the last point also plans the
                // search.
                let last = (index + 1..self.specs.len()).all(|later| {
                    !discovery_selected(&self.specs, later) || slots[later].is_some()
                });
                let mut footprint = Footprint::default();
                footprint.add(most_bytes(&self.resident_bytes[index]), 1);
                Some(Work {
                    stage: Stage::Discovery,
                    quote: Quote {
                        priced: Duration::ZERO,
                        unpriced: 2 + usize::from(last),
                    },
                    forming: Quote::default(),
                    gain: 0.,
                    footprint,
                })
            }
            Progress::Searching(searching) => {
                let incumbent = searching.ledger.incumbent();
                // A piece of a unit whose inputs were given up builds them
                // again first.
                let (rebuilding, rebuilt) = self.rebuild(searching);
                let mut conversion = rebuilt;
                conversion.add(most_bytes(&self.conversion_bytes), 1);
                if let Some(leader) = searching.ledger.leader() {
                    let leader = &searching.ledger.contenders[leader];
                    return Some(Work {
                        stage: Stage::Promote,
                        quote: self
                            .promotion_quote(searching, incumbent, leader)
                            .and(rebuilding),
                        forming: Quote::default(),
                        gain: incumbent.cost.total() - leader.cost.total(),
                        footprint: conversion,
                    });
                }
                // A nominal lead holds with the chance this unit's
                // conversions have shown: of `n` with `s` promoted, the
                // next succeeds with (s + 1) / (n + 2), were they
                // exchangeable.
                let ledger = &searching.ledger;
                let confirm = ledger.nominal().map(|(leader, lead)| {
                    Work {
                        stage: Stage::Confirm,
                        quote: self
                            .promotion_quote(searching, incumbent, &ledger.contenders[leader])
                            .and(rebuilding),
                        forming: Quote::default(),
                        gain: lead * (ledger.promotions + 1) as f64
                            / (ledger.conversions + 2) as f64,
                        footprint: conversion,
                    }
                });
                // Were the configurations measured exchangeable, each of
                // the piece's challengers would be the fastest of all with
                // the same chance as every one measured before it.
                let structural = searching.run.search.structural;
                let measure = searching.planned.as_ref().map(|piece| {
                    let challengers = piece.challengers().len();
                    let mut footprint = rebuilt;
                    // Measuring the piece's challengers is one class. Once
                    // observed it is priced per challenger; before, it is
                    // one operation the unit has not seen, whatever the
                    // challengers' number: each leaves when it is measured.
                    match most_bytes(&self.measure_bytes) {
                        Some(price) => footprint.add(Some(price), challengers),
                        None => footprint.add(None, 1),
                    }
                    // The programs its compile forms stay held after it:
                    // their class, once a compile of the unit was observed.
                    let forming = self
                        .compiled_ahead(searching, piece)
                        .iter()
                        .map(Formable::programs)
                        .sum::<usize>();
                    if let Some(price) = self.program_price() {
                        footprint.add(Some(price), forming);
                    }
                    Work {
                        stage: if structural {
                            Stage::Form
                        } else {
                            Stage::Measure
                        },
                        quote: self.measurement_quote(searching, piece).and(rebuilding),
                        forming: self
                            .formation()
                            .quote(&self.compiled_ahead(searching, piece)),
                        // Every configuration the unit reached counts,
                        // excluded ones too: each was a challenger that
                        // did not lead, and the next one's chance falls
                        // with their number.
                        gain: incumbent.cost.total() * challengers as f64
                            / (ledger.order.len() + challengers) as f64,
                        footprint,
                    }
                });
                // Other forms are measured before a nominal lead inside
                // one is resolved.
                if structural {
                    return measure;
                }
                // The unit offers whichever is worth more per second of its
                // price, an unpriced operation taken as in its admission.
                let longest = self.longest_operation().as_secs_f64();
                let rate = |work: &Work| {
                    work.gain
                        / (work.quote.priced.as_secs_f64() + longest * work.quote.unpriced as f64)
                            .max(f64::MIN_POSITIVE)
                };
                match (measure, confirm) {
                    (Some(measure), Some(confirm)) if rate(&confirm) > rate(&measure) => {
                        Some(confirm)
                    }
                    (measure, confirm) => measure.or(confirm),
                }
            }
        }
    }

    /// The whole price of forming and measuring `piece`: its compile, each
    /// challenger's admission at the timed points, and the session's
    /// samples. A challenger's active launches are known once it is formed;
    /// its key is forecast from the launches its baseline runs there.
    ///
    /// The session samples every point and key not measured before, and
    /// the served choice once at each point where it samples one: for each
    /// of those slots a calibrating pass and the first round. Every round
    /// after the first is one challenger's: it is still ahead, and its
    /// slots and the served choice's beside them are sampled again. How
    /// many such rounds a challenger takes is not known before they run;
    /// they are priced at their expected number ([`Rates::expected`]),
    /// from how this unit's rounds have come out. Rounds of two
    /// challengers that share a point are priced as if each sampled the
    /// served choice there on its own.
    ///
    /// A challenger's admission (placing it at every timed point, the
    /// reference where it is not yet observed, its own witnesses, their
    /// readback and comparison) is one class of the unit, priced by the
    /// admissions of the unit's challengers so far. The defaults' witnesses
    /// are not of that class: they compare nothing and run only the
    /// workload's rotations, and pricing a challenger by them read 1.5 to
    /// 2 times low. A piece whose challengers are timed first
    /// ([`Sequence`]) is priced by what it does instead: each challenger's
    /// placement, and the witnesses after the rounds of those expected to
    /// be left ahead, from that class's own observations
    /// ([`Admission::quote`]).
    fn measurement_quote(&self, searching: &Searching<'a>, piece: &Piece) -> Quote {
        let formation = self.formation();
        let incumbent = searching.ledger.incumbent();
        let run = &searching.run;
        let mut quote = formation.quote(&self.compiled_ahead(searching, piece));
        // The slots sampled, a challenger's by point and key and the served
        // choice's by point.
        let mut slots = BTreeSet::new();
        let mut beside = BTreeSet::new();
        // The rounds a new comparison takes past its first; none before
        // the unit ran one, and the piece's then are one operation of a
        // class it has not seen.
        let further = searching
            .rates
            .expected(&Tally::default())
            .map(|rounds| rounds - 1.);
        let mut unseen = false;
        // The piece's operations by class, each class added to the quote
        // once: its challengers' admissions, and at each point the first
        // samples and the further ones expected.
        let mut admissions = 0;
        let mut sampled = BTreeMap::<usize, (usize, f64)>::new();
        // `keys`: the challenger's forecast key at each timed point.
        let mut admit = |keys: &dyn Fn(usize) -> PointKey| {
            admissions += 1;
            let apart = (0..searching.points.len())
                .map(|point| (point, keys(point)))
                .filter(|(point, key)| *key != incumbent.points[*point].key)
                .collect::<Vec<_>>();
            let fresh = apart
                .iter()
                .filter(|slot| !run.samples.contains_key(*slot))
                .collect::<Vec<_>>();
            // Rounds that read a reused measurement resolve nothing: such a
            // challenger takes the one round that estimates it.
            let further = if fresh.len() == apart.len() {
                further
            } else {
                Some(0.)
            };
            unseen |= further.is_none() && !fresh.is_empty();
            for (point, key) in fresh {
                let first = usize::from(slots.insert((*point, key.clone())))
                    + usize::from(beside.insert(*point));
                let samples = sampled.entry(*point).or_default();
                samples.0 += 2 * first;
                samples.1 += 2. * further.unwrap_or(0.);
            }
        };
        let forecast = |working: &[usize], specialization: &NativeSpecialization| {
            run.keying
                .key(&self.implementation, working.to_vec(), specialization)
        };
        match piece {
            Piece::Whole(specializations) => {
                let reference = &run.search.reference;
                for specialization in specializations {
                    admit(&|point| forecast(&reference[point].key.launches, specialization));
                }
            }
            Piece::Sweep {
                group,
                baseline,
                ready,
                ..
            } => {
                let launches = &run.search.partition.groups[*group].launches;
                for specialization in ready {
                    admit(&|point| {
                        let working = &baseline[point].key.launches;
                        if launches.iter().any(|launch| working.contains(launch)) {
                            forecast(working, specialization)
                        } else {
                            baseline[point].key.clone()
                        }
                    });
                }
            }
        }
        let placements = searching
            .points
            .iter()
            .map(|point| point.prices().placement.price())
            .collect::<Vec<_>>();
        quote = quote.and(searching.admission.quote(&run.sequence, &placements, admissions));
        for (point, (first, further)) in sampled {
            let price = searching.points[point].prices().sample.price();
            quote.add(price, first);
            if price.is_some() {
                quote.expect(price, further);
            }
        }
        quote.add(None, usize::from(unseen));
        // The planning that follows the piece asks the domain for what
        // comes after its challengers: for each of them, as many requests
        // as the unit's search has made for each configuration it found.
        let found = run.search.generated.len().max(1) as f64;
        let requests = (admissions as f64 * run.search.requests as f64 / found).ceil();
        quote.add(searching.planning.price(requests as usize), 1);
        quote
    }

    /// The whole price of converting `leader`: the check of every untimed
    /// point its artifact has not passed, and for a lead not yet resolved,
    /// the session that may resolve it. That session places the leader at
    /// every timed point and the served choice where they run different
    /// work, and samples both there: a calibrating pass, and the rounds
    /// expected of it ([`Rates::expected`]). A leader not yet witnessed at
    /// every timed point is witnessed there in that session, before its
    /// rounds ([`Admission::conversion`]): whether its work at a point was
    /// observed is the unit's evidence there
    /// ([`PreparedPoint::witnessed`]).
    fn promotion_quote(
        &self,
        searching: &Searching<'a>,
        incumbent: &Contender,
        leader: &Contender,
    ) -> Quote {
        let rate = self.planned().settings.error_rate;
        let mut quote = Quote::default();
        let evidence = searching.ledger.evidence(leader);
        if evidence.verdict(rate).is_none() {
            let points = searching
                .points
                .iter()
                .enumerate()
                .map(|(index, point)| {
                    (
                        point.prices(),
                        incumbent.points[index].key != leader.points[index].key,
                    )
                })
                .collect::<Vec<_>>();
            let witnessed = searching
                .points
                .iter()
                .zip(&leader.points)
                .all(|(point, measured)| point.witnessed(&leader.artifact, &measured.key));
            quote = searching.admission.conversion(
                &points,
                witnessed,
                searching.rates.expected(&evidence),
            );
        }
        // The check of each untimed point is a class, and the unit's checks
        // are one kind of work: a point whose check was not observed is
        // priced by the unit's checks that were, and before the unit has
        // run any they are one thing not known, not one for each point. A
        // unit of twelve untimed points was held to twelve times the
        // longest operation of any kind in the load for a conversion that
        // takes 0.08 s, and served its default with a measured lead.
        if !searching.covered.contains(&leader.artifact) {
            let observed = searching
                .untimed
                .iter()
                .filter_map(|index| self.checks[*index].price())
                .collect::<Vec<_>>();
            let nearest =
                (!observed.is_empty()).then(|| observed.iter().sum::<f64>() / observed.len() as f64);
            let mut unseen = 0;
            for index in &searching.untimed {
                match self.checks[*index].price().or(nearest) {
                    Some(price) => quote.add(Some(price), 1),
                    None => unseen += 1,
                }
            }
            quote.add(None, unseen);
        }
        // A leader that gave up its kernel is formed again: from the
        // programs the unit holds, or by one compile.
        if leader.kernel.is_none() {
            quote = quote.and(self.formation().quote(&searching.run.programs.formables(
                &self.implementation,
                std::slice::from_ref(&leader.specialization),
                &[],
            )));
        }
        quote
    }

    /// Run the unit's next piece ([`UnitSearch::next`]) before `before`,
    /// keeping the served choice and completed evidence on interruption.
    pub fn perform(
        &mut self,
        source: &mut dyn PointSource<'a>,
        before: Option<Instant>,
    ) -> Result<Performed, TuneError> {
        let deadline = Deadline::new(before);
        let work = self
            .next()
            .expect("a unit performs the work it named as its next");
        self.offered.set(None);
        let began = Instant::now();
        let before = self.spent();
        // The piece's raises are those recorded from here to its end.
        self.raises();
        // What the piece holds is read from the ledgers' high-water marks,
        // started again here: what it took and gave back within itself is
        // seen.
        self.device.restart_memory_high_water();
        if let Some(ground) = &self.ground {
            ground.validator.restart_retained_high_water();
        }
        let mut host = ProcessMemory::footprint_now();
        let host_began = host;
        // What the ledgers count as the piece begins: what they rise by
        // within it is over this.
        let ledgers = |unit: &Self| {
            (
                unit.device.memory_usage(),
                unit.ground.as_ref().map_or((0, 0), |ground| {
                    (
                        ground.validator.retained_bytes(),
                        ground.validator.retained_high_water(),
                    )
                }),
            )
        };
        let (usage_began, (retained_began, _)) = ledgers(self);
        let began_bytes = self.held();
        let discovered = self.undiscovered();
        let challengers = match &self.progress {
            Progress::Searching(searching) => searching
                .planned
                .as_ref()
                .map_or(0, |piece| piece.challengers().len()),
            Progress::Discovering(_) => 0,
        };
        // A piece whose inputs the device's memory limit refused ends
        // there, its point priced at what the build demanded: the piece is
        // offered again at that price ([`build_point`]).
        let mut resident = self.held();
        let attempted = (|| -> Result<(bool, bool), WorkError<TuneError>> {
            deadline.check(Duration::ZERO)?;
            let restored = work.stage == Stage::Discovery || self.restore(source, deadline)?;
            host = host.max(ProcessMemory::footprint_now());
            resident = self.held();
            Ok(match work.stage {
                _ if !restored => (false, false),
                Stage::Discovery => (self.discover(source, deadline)?, false),
                Stage::Form | Stage::Measure => {
                    self.measure(deadline)?;
                    (true, false)
                }
                Stage::Promote | Stage::Confirm => {
                    (true, self.promote(source, work.stage, deadline)?)
                }
            })
        })();
        let (built, promoted) = match attempted {
            Ok(performed) => performed,
            Err(WorkError::Expired) => {
                self.expired = true;
                (false, false)
            }
            Err(WorkError::Failed(error)) => return Err(error),
        };
        // What the piece formed and no longer needs leaves the device with
        // it: only the served choice and a leader keep their kernels.
        if built && work.stage != Stage::Discovery {
            if let Progress::Searching(searching) = &mut self.progress {
                // A conversion changed what the challengers are ranked
                // against: its leader's standing, measurement and rounds,
                // or the served choice.
                if matches!(work.stage, Stage::Promote | Stage::Confirm) {
                    searching.ledger.rerank();
                }
                searching.ledger.release();
            }
        }
        // One reading ends the piece: it is the last the peak is taken
        // over and what was held when the piece ended, so the peak is never
        // below the end whatever other threads did between two readings.
        let ended = ProcessMemory::footprint_now();
        host = host.max(ended);
        // Where the device's allocations are in the footprint, what the
        // piece placed and released between those readings is footprint
        // it held: the ledgers' own high-water marks say how much.
        let (usage, (_, retained_most)) = ledgers(self);
        let rise = usage
            .high_water
            .saturating_sub(usage_began.charged)
            .saturating_add(retained_most.saturating_sub(retained_began));
        let host = footprint_peak(&self.device, host_began, host, rise);
        let memory = PieceMemory {
            began_bytes,
            peak_bytes: held_bytes(
                &self.device,
                self.device.memory_usage().high_water,
                self.ground
                    .as_ref()
                    .map_or(0, |ground| ground.validator.retained_high_water()),
                host,
            ),
            ended_bytes: held_bytes(
                &self.device,
                self.device.memory_usage().charged,
                self.ground
                    .as_ref()
                    .map_or(0, |ground| ground.validator.retained_bytes()),
                ended,
            ),
            ledger_peak_bytes: self.device.memory_usage().high_water
                + self
                    .ground
                    .as_ref()
                    .map_or(0, |ground| ground.validator.retained_high_water()),
        };
        let operations = memory.peak_bytes.saturating_sub(resident) as f64;
        match work.stage {
            // The refused point was priced as its build was refused, and
            // the piece's operations did not run.
            _ if !built => {}
            Stage::Discovery => {
                let index = discovered.expect("discovery discovers a point");
                self.resident_bytes[index].observe(memory.added() as f64);
            }
            Stage::Form | Stage::Measure => self
                .measure_bytes
                .observe(operations / challengers.max(1) as f64),
            Stage::Promote | Stage::Confirm => self.conversion_bytes.observe(operations),
        }
        self.largest_bytes = self.largest_bytes.max(memory.added());
        Ok(Performed {
            stage: work.stage,
            quote: work.quote,
            actual: began.elapsed(),
            interrupted: self.expired,
            spent: self.spent().since(before),
            raises: self.raises(),
            promoted,
            footprint: work.footprint,
            memory,
        })
    }

    /// Build the next discovery point and measure the defaults there; with
    /// the last, plan the search. False when the device's memory limit
    /// refused the point's build ([`build_point`]) or the measurement of its
    /// defaults: nothing was measured, and the point is priced at what the
    /// piece demanded.
    fn discover(
        &mut self,
        source: &mut dyn PointSource<'a>,
        deadline: Deadline,
    ) -> Result<bool, WorkError<TuneError>> {
        let index = self
            .undiscovered()
            .expect("discovery has an unmeasured point");
        let before = self.device.memory_usage();
        let building = Instant::now();
        let Some(inputs) = build_point(
            &self.device,
            source,
            &self.specs[index],
            index,
            &mut self.resident_bytes[index],
        )?
        else {
            return Ok(false);
        };
        let spec = &self.specs[index];
        let built = building.elapsed().as_secs_f64();
        self.builds[index].observe(built);
        self.time.building_seconds += built;
        let point = TuningPoint::new(spec, inputs)?;
        let ground = self
            .ground
            .as_ref()
            .expect("canonical preparation precedes discovery");
        for args in &point.rotation {
            ground.kernel.shape(&args.values()).map_err(|error| {
                TuneError::DefaultUnusable(Exclusion::Execution(format!(
                    "at point `{}`: {error}",
                    point.label
                )))
            })?;
        }
        let mut point = ground.validator.point(point)?;
        let shape =
            point_shapes(&self.device, &self.logical, std::slice::from_ref(&point))?.remove(0);
        // The defaults' time here is the unit of every time the unit
        // records and the point's weight in what it optimises: the median
        // of three samples, the fewest whose median one sample cannot
        // move, whatever that sample read.
        let options = MeasureOptions {
            samples: 3,
            min_sample_seconds: self.planned().min_sample_seconds,
        };
        let settings = self.planned().settings.clone();
        let measuring = Instant::now();
        let measured = self
            .measurer
            .as_mut()
            .expect("the measurer serves discovery until a search takes it")
            .measure(
                &ground.kernel,
                std::slice::from_ref(&point),
                self.default.params(),
                &options,
                &settings,
                None,
                deadline,
            );
        let defaults = match measured {
            Ok(mut defaults) => defaults.remove(0),
            // The point's inputs fit, but what measuring its defaults needs
            // beside them does not: as a refused build, the point is priced
            // at the piece's demand and offered again.
            Err(WorkError::Expired) => {
                point.release();
                source.release(index);
                return Err(WorkError::Expired);
            }
            Err(WorkError::Failed(exclusion)) => match self.device.memory_usage().refused_demand {
                Some(demand) if Some(demand) > before.refused_demand => {
                    self.resident_bytes[index]
                        .observe(demand.saturating_sub(before.charged) as f64);
                    return Ok(false);
                }
                _ => return Err(TuneError::DefaultUnusable(exclusion).into()),
            },
        };
        let measured = measuring.elapsed().as_secs_f64();
        self.longest = self.longest.max(measured);
        self.time.measuring_seconds += measured;
        let Progress::Discovering(slots) = &mut self.progress else {
            unreachable!("discovery precedes the search");
        };
        slots[index] = Some(Timed {
            point,
            defaults,
            shape,
        });
        if self.undiscovered().is_none() {
            deadline.check(Duration::ZERO)?;
            let planning = Instant::now();
            let opened = self.open_search();
            self.longest = self.longest.max(planning.elapsed().as_secs_f64());
            opened?;
        }
        Ok(true)
    }

    /// Every discovery point is measured: fix the objective and plan the
    /// search over it. The other served points remain numerical witnesses
    /// without timing demand.
    fn open_search(&mut self) -> Result<(), TuneError> {
        let Progress::Discovering(slots) =
            std::mem::replace(&mut self.progress, Progress::Discovering(Vec::new()))
        else {
            unreachable!("discovery precedes the search");
        };
        let records = point_records(&self.specs);
        let mut timed = Vec::new();
        let mut points = Vec::new();
        let mut canonical = Vec::new();
        let mut shapes = Vec::new();
        let mut untimed = Vec::new();
        for (index, slot) in slots.into_iter().enumerate() {
            match slot {
                Some(Timed {
                    mut point,
                    defaults,
                    shape,
                }) => {
                    point.reweigh(records[index].weight);
                    timed.push(index);
                    points.push(point);
                    canonical.push(defaults);
                    shapes.push(shape);
                }
                None => untimed.push(index),
            }
        }
        let weights = points.iter().map(|point| point.weight).collect::<Vec<_>>();
        let kernel = self
            .ground
            .as_ref()
            .expect("canonical preparation precedes discovery")
            .kernel
            .clone();
        let plan = self.planned().clone();
        match self.open_run(&shapes, &weights, &plan, canonical.clone()) {
            Ok((run, canonical)) => {
                let cost = Weighing::of(&points).cost(&canonical, &canonical);
                let mut searching = Searching {
                    ledger: Ledger::new(
                        &self.default,
                        kernel,
                        canonical.clone(),
                        cost,
                        plan.settings.clone(),
                    ),
                    points,
                    timed,
                    untimed,
                    covered: BTreeSet::new(),
                    canonical,
                    run,
                    planned: None,
                    admission: Admission::default(),
                    rates: Rates::default(),
                    planning: Linear::default(),
                };
                self.replan(&mut searching)?;
                self.progress = Progress::Searching(Box::new(searching));
                Ok(())
            }
            Err(error) => {
                // The defaults stay measured and served.
                let mut slots = self.specs.iter().map(|_| None).collect::<Vec<_>>();
                for (((index, point), defaults), shape) in
                    timed.into_iter().zip(points).zip(canonical).zip(shapes)
                {
                    slots[index] = Some(Timed {
                        point,
                        defaults,
                        shape,
                    });
                }
                self.progress = Progress::Discovering(slots);
                Err(error)
            }
        }
    }

    /// Plan the next challenger work of `searching`.
    fn replan(&self, searching: &mut Searching<'a>) -> Result<(), TuneError> {
        let Run {
            search, keying, ..
        } = &mut searching.run;
        // Planning asks the declared domain for assignments, admissible or
        // not, and what it takes follows how many it asks: observed as one
        // call with its requests, the class the next piece's planning is
        // priced by.
        let (began, asked) = (Instant::now(), search.requests);
        let planned = search.plan(&Declared {
            implementation: &self.implementation,
            keying,
        });
        searching
            .planning
            .observe(search.requests - asked, began.elapsed().as_secs_f64());
        searching.planned = planned?;
        Ok(())
    }

    /// Form the planned challengers and measure and admit them at the
    /// timed points. Each is recorded, whatever it measures.
    fn measure(&mut self, deadline: Deadline) -> Result<(), WorkError<TuneError>> {
        let Progress::Searching(mut searching) =
            std::mem::replace(&mut self.progress, Progress::Discovering(Vec::new()))
        else {
            unreachable!("challengers are measured by a search");
        };
        let performed = self.measure_piece(&mut searching, deadline);
        let replanned = performed.and_then(|()| {
            deadline.check(Duration::ZERO)?;
            Ok(self.replan(&mut searching)?)
        });
        self.progress = Progress::Searching(searching);
        replanned
    }

    fn measure_piece(
        &mut self,
        searching: &mut Searching<'a>,
        deadline: Deadline,
    ) -> Result<(), WorkError<TuneError>> {
        let piece = searching
            .planned
            .take()
            .expect("a measurement performs a planned piece");
        let plan = self.planned();
        let formation = self.formation();
        let weighing = Weighing::of(&searching.points);
        let mut forming = 0.;
        let mut measuring = 0.;
        let Searching {
            points,
            ledger,
            run,
            admission,
            rates,
            ..
        } = searching;
        let performed = run.perform(
            piece,
            &formation,
            points,
            &plan.settings,
            plan.min_sample_seconds,
            rates,
            &weighing,
            ledger,
            admission,
            &mut forming,
            &mut measuring,
            deadline,
        );
        drop(formation);
        self.time.forming_seconds += forming;
        self.time.measuring_seconds += measuring;
        performed
    }

    /// Convert the leader: give its comparison with the served choice the
    /// matched rounds it lacks to resolve, and once its lead is resolved,
    /// check it at the untimed points and serve it. Whether it was
    /// promoted; what the rounds showed is recorded either way.
    fn promote(
        &mut self,
        source: &mut dyn PointSource<'a>,
        stage: Stage,
        deadline: Deadline,
    ) -> Result<bool, WorkError<TuneError>> {
        let plan = self.planned().clone();
        let ground = self
            .ground
            .as_ref()
            .expect("canonical preparation precedes the search");
        let Progress::Searching(searching) = &mut self.progress else {
            unreachable!("a lead is converted by a search");
        };
        let Searching {
            points,
            untimed,
            covered,
            canonical,
            ledger,
            run,
            rates,
            ..
        } = &mut **searching;
        let leader = match stage {
            Stage::Promote => ledger.leader(),
            _ => ledger.nominal().map(|(leader, _)| leader),
        }
        .expect("a conversion converts the leader its work named")
        .clone();
        ledger.conversions += 1;
        // A leader that gave up its kernel is formed again; one that can
        // no longer be formed is recorded so and gives way to the next.
        if ledger.contenders[&leader].kernel.is_none() {
            let formation = Formation {
                device: &self.device,
                module: self.module,
                entry: self.entry,
                logical: &self.logical,
                bindings: &self.bindings,
                cpu: self.cpu,
                implementation: &self.implementation,
                default: &self.default,
                served: &self.served,
                price: &self.formation_price,
            };
            let specialization = ledger.contenders[&leader].specialization.clone();
            let began = Instant::now();
            let formed = run
                .programs
                .form(&formation, &[specialization], &[])?
                .pop()
                .expect("one configuration was formed");
            // Its record states how it was first formed, not this.
            formation
                .price
                .lock()
                .expect("formation price lock")
                .calls
                .clear();
            self.time.forming_seconds += began.elapsed().as_secs_f64();
            match formed {
                Ok(kernel) => ledger.hold(&leader, kernel),
                Err(exclusion) => {
                    ledger
                        .contenders
                        .get_mut(&leader)
                        .expect("the leader is a contender")
                        .standing = ContenderStanding::Rejected(exclusion);
                    return Ok(false);
                }
            }
        }
        let keying = &run.keying;
        let kernel = ledger.contenders[&leader]
            .kernel
            .clone()
            .expect("the leader holds its kernel");
        let leading = ledger.contenders[&leader].specialization.clone();
        let rate = plan.settings.error_rate;
        let before = ledger.evidence(&ledger.contenders[&leader]);
        if before.verdict(rate).is_none() {
            let began = Instant::now();
            let served = ledger.incumbent();
            let (served_kernel, served_points) = (
                served.kernel.clone().expect("the served choice holds its kernel"),
                served.points.clone(),
            );
            // Nothing is reused: every point that tells the two apart is
            // sampled in every round, so the rounds are evidence.
            let compared = measure_matched(
                &|working, specialization| {
                    keying.key(&self.implementation, working, specialization)
                },
                &[Candidate {
                    specialization: leading.clone(),
                    kernel: kernel.clone(),
                    evidence: before,
                }],
                &Incumbent {
                    specialization: &served.specialization,
                    kernel: &served_kernel,
                    points: &served_points,
                },
                points,
                Matching {
                    deadline,
                    settings: &plan.settings,
                    min_sample_seconds: plan.min_sample_seconds,
                    rates,
                    shuffle: &mut run.shuffle,
                    clock: &mut run.clock,
                    sequence: &mut run.sequence,
                },
                None,
                &served_points,
                None,
                &mut run.outputs,
                // It is no longer new to the unit: it is witnessed where it
                // has not been, before its rounds, and its placement is
                // repeated.
                None,
            )
            .pop()
            .expect("one challenger was compared");
            self.time.measuring_seconds += began.elapsed().as_secs_f64();
            let served = ledger.incumbent.clone();
            let contender = ledger
                .contenders
                .get_mut(&leader)
                .expect("the leader is a contender");
            match compared {
                Err(exclusion) => {
                    contender.standing = ContenderStanding::Rejected(exclusion);
                    return Ok(false);
                }
                Ok(None) => return Err(WorkError::Expired),
                Ok(Some((measured, evidence))) => {
                    contender.cost = Weighing::of(points).cost(&measured, canonical);
                    contender.points = measured.clone();
                    contender.confirmed = measured;
                    contender.against = served;
                    contender.evidence = evidence;
                    // A trail is recorded when the challengers are ranked
                    // again, as this piece ends; an unresolved lead stays
                    // as it is.
                    if evidence.verdict(rate) != Some(Verdict::Leads) {
                        return Ok(false);
                    }
                }
            }
        }
        let key = |working| keying.key(&self.implementation, working, &leading);
        let mut scope = ScopeQualification {
            deadline,
            source,
            specs: &self.specs,
            indices: untimed,
            validator: &ground.validator,
            covered,
            default: &ground.kernel.artifact().0,
            key: &key,
            checks: &mut self.checks,
            building_seconds: 0.,
            validating_seconds: 0.,
            reference_seconds: 0.,
        };
        let covered = scope.qualify(&kernel);
        self.time.building_seconds += scope.building_seconds;
        self.time.validating_seconds += scope.validating_seconds;
        self.time.scope_validation_seconds +=
            (scope.validating_seconds - scope.reference_seconds).max(0.0);
        if let Err(exclusion) = covered {
            let exclusion = match exclusion {
                WorkError::Expired => return Err(WorkError::Expired),
                WorkError::Failed(exclusion) => exclusion,
            };
            ledger.failed += 1;
            ledger
                .contenders
                .get_mut(&leader)
                .expect("the leader is a contender")
                .standing = ContenderStanding::Rejected(exclusion);
            return Ok(false);
        }
        // Serving prepares the chosen kernel as formed here. The unit's
        // handoff is its one served choice: the leader's replaces the
        // choice's it supersedes, which serving will never take.
        deadline.check(Duration::ZERO)?;
        kernel
            .hand_off_replacing(ledger.incumbent().kernel.as_deref())
            .map_err(|error| TuneError::Publication(error.to_string()))?;
        ledger
            .contenders
            .get_mut(&leader)
            .expect("the leader is a contender")
            .standing = ContenderStanding::Served;
        ledger.served.push(leader.clone());
        // The choice it supersedes holds its kernel as any challenger does:
        // until the piece ends.
        let superseded = std::mem::replace(&mut ledger.incumbent, leader);
        ledger.holding.insert(superseded);
        ledger.promotions += 1;
        Ok(true)
    }

    /// The unit's complete record as it stands: its served choice, every
    /// configuration reached with its measurements, and the evidence. Legal
    /// at every instant from [`UnitSearch::prepare_defaults`] on.
    pub fn terminal(&self) -> Result<TuningResult, TuneError> {
        let ground = self.ground.as_ref().ok_or_else(|| {
            TuneError::Declaration("canonical defaults were not prepared before their record".into())
        })?;
        let mut time = self.time.clone();
        time.reference_seconds = ground.validator.reference_seconds();
        let searching = match &self.progress {
            Progress::Discovering(slots) => {
                return Ok(self.unit.census(
                    point_records(&self.specs),
                    &self.default,
                    ground,
                    slots
                        .iter()
                        .flatten()
                        .map(|timed| timed.defaults.clone())
                        .collect(),
                    time,
                ));
            }
            Progress::Searching(searching) => searching,
        };
        time.validating_seconds += searching
            .points
            .iter()
            .map(|point| point.validation_seconds())
            .sum::<f64>();
        let ledger = &searching.ledger;
        let incumbent = ledger.incumbent();
        let search = &searching.run.search;
        let method = TuningMethod::Search {
            groups: search.partition.groups.len(),
            candidates: search.generated.len(),
            // The search's own word: a unit with no piece planned that has
            // not exhausted its domain stopped short, and is not complete.
            complete: search.exhausted(),
        };
        let mut result = self.unit.result(
            // Every completed observation is retained, nonwinners' and
            // partial untimed scopes' too. Their presence makes no
            // configuration eligible.
            Records {
                points: point_records(&self.specs),
                records: ledger.records(&ground.validator),
                served: ledger.served.clone(),
                evidence: Evidence::of(&ground.validator),
            },
            &incumbent.specialization,
            method,
            time,
        );
        // The native default is the reference: serving it qualifies nothing.
        let default = &ground.kernel.artifact().0;
        let promoted = incumbent.artifact != *default;
        if !promoted {
            result.disposition = TuningDisposition::CanonicalFallback {
                artifact: default.clone(),
            };
        }
        result.coverage = (promoted || ledger.failed > 0).then(|| CoverageCheck {
            outcome: if promoted {
                CoverageOutcome::Passed
            } else {
                CoverageOutcome::Failed
            },
            failed: ledger.failed,
        });
        Ok(result)
    }

    /// The defaults' measurements at the points discovered so far: what
    /// the unit's default service is known to take.
    pub fn service_measurements(&self) -> Vec<PointMeasurement> {
        match &self.progress {
            Progress::Discovering(slots) => slots
                .iter()
                .flatten()
                .map(|timed| timed.defaults.clone())
                .collect(),
            Progress::Searching(searching) => searching.canonical.clone(),
        }
    }

    /// The search of a unit whose defaults are measured, and their
    /// measurement as the search keys it: over the groups of launches that
    /// run together where launches are formed separately, over the one
    /// group of the whole program where it is formed whole. Stored results
    /// are not search inputs.
    fn open_run(
        &mut self,
        shapes: &[PointShape],
        weights: &[f64],
        plan: &UnitPlan,
        mut canonical: Vec<PointMeasurement>,
    ) -> Result<(Run, Vec<PointMeasurement>), TuneError> {
        let implementation = &self.implementation;
        let (partition, keying, programs) = if implementation.launch_scoped() {
            if !matches!(
                self.backend,
                seismic_lang::registry::BackendName::Metal
                    | seismic_lang::registry::BackendName::Cuda
            ) {
                return Err(TuneError::Declaration(format!(
                    "launch-scoped native tuning is not yet available on `{}`",
                    self.backend.as_str(),
                )));
            }
            let partition =
                plan::partition(implementation, &self.statics, shapes).map_err(planning)?;
            let keying = Keying::Launches(partition.boundary.clone());
            (partition, keying, Programs::Launches(LaunchPrograms::default()))
        } else {
            let (unserved, points) = unserved_parameters(implementation, &self.statics, shapes)?;
            let parameters = implementation
                .params
                .iter()
                .filter(|parameter| !unserved.contains(&parameter.name))
                .map(|parameter| {
                    (
                        parameter.name.clone(),
                        parameter.values.clone(),
                        parameter.form,
                    )
                })
                .collect::<Vec<_>>();
            (
                plan::whole(&parameters, implementation.launches.len(), points),
                Keying::Whole(Influence::of(implementation)),
                Programs::Whole(WholePrograms::default()),
            )
        };
        // The defaults' measurement, keyed as the search keys its own.
        for measured in &mut canonical {
            measured.key =
                keying.key(implementation, measured.key.launches.clone(), &self.default);
        }
        // A point with the same active launches and parameters runs the
        // same work under every candidate: its measurement is reused. A
        // conversion's session reuses none.
        let samples = canonical
            .iter()
            .enumerate()
            .map(|(point, measurement)| ((point, measurement.key.clone()), measurement.clone()))
            .collect();
        // The run places into its own result storage, and holds the clock
        // the census held.
        let clock = self.measurer.take().and_then(|measurer| measurer.clock);
        Ok((
            Run {
                search: GroupSearch::new(
                    partition,
                    self.default.clone(),
                    canonical.clone(),
                    weights.to_vec(),
                    plan.start.clone(),
                ),
                keying,
                programs,
                outputs: OutputPool::default(),
                samples,
                shuffle: Shuffle::default(),
                clock,
                sequence: Sequence::default(),
            },
            canonical,
        ))
    }

    /// Where the unit's tuning stands.
    pub fn standing(&self) -> Standing {
        match &self.progress {
            Progress::Discovering(_) => Standing {
                cost: 1.,
                measured: 0,
                generated: 1,
                requests: 0,
                declared: None,
                exhausted: false,
                programs: 0,
                forming_seconds: self.time.forming_seconds,
            },
            Progress::Searching(searching) => Standing {
                cost: searching.ledger.incumbent().cost.total().min(1.),
                measured: searching.ledger.order.len(),
                generated: 1 + searching.run.search.generated.len(),
                requests: searching.run.search.requests,
                declared: Some(searching.run.search.declared()),
                exhausted: searching.planned.is_none() && searching.run.search.exhausted(),
                programs: searching.run.programs.formed(),
                forming_seconds: self.time.forming_seconds,
            },
        }
    }

    /// Measure and validate every admissible configuration at every point
    /// (development): the domain the caller named ([`SurveyPlan`]).
    pub fn survey(
        mut self,
        source: &mut dyn PointSource<'a>,
        plan: SurveyPlan,
    ) -> Result<TuningResult, TuneError> {
        self.prepare_defaults(source)?;
        let ground = self
            .ground
            .take()
            .expect("the defaults were just prepared");
        // A survey builds and times every point whatever it costs.
        let records = point_records(&self.specs);
        let mut points = Vec::with_capacity(self.specs.len());
        for (index, spec) in self.specs.iter().enumerate() {
            let began = Instant::now();
            let inputs = source.build(index).map_err(|detail| TuneError::Inputs {
                point: spec.label.clone(),
                detail,
            })?;
            self.time.building_seconds += began.elapsed().as_secs_f64();
            points.push(ground.validator.point(TuningPoint::new(spec, inputs)?)?);
        }
        if self.implementation.launch_scoped() {
            return Err(TuneError::Domain(
                "a survey measures programs formed whole; launches formed separately are searched"
                    .into(),
            ));
        }
        let formation = Formation {
            device: &self.device,
            module: self.module,
            entry: self.entry,
            logical: &self.logical,
            bindings: &self.bindings,
            cpu: self.cpu,
            implementation: &self.implementation,
            default: &self.default,
            served: &self.served,
            price: &self.formation_price,
        };
        let tuned = Tuned {
            formation: &formation,
            statics: &self.statics,
            default: &self.default,
            ground: &ground,
        };
        let (configurations, overall, method, time) = tuned.survey(&points, plan)?;
        let mut result = self.unit.result(configurations, &overall, method, time);
        result.points = records;
        result.time.building_seconds += self.time.building_seconds;
        result.time.forming_seconds += self.time.forming_seconds;
        result.time.reference_seconds += ground.validator.reference_seconds();
        Ok(result)
    }
}

/// What every result of one unit records about it.
struct Unit {
    tuning_identity: String,
    entry: String,
    backend: String,
    requirement: NumericalRequirement,
    implementation_identity: String,
    parameters: Vec<DeclaredParameter>,
}

impl Unit {
    fn result(
        &self,
        configurations: Records,
        overall: &NativeSpecialization,
        method: TuningMethod,
        time: TuningTime,
    ) -> TuningResult {
        TuningResult {
            disposition: TuningDisposition::QualifiedCandidate,
            tuning_identity: self.tuning_identity.clone(),
            entry: self.entry.clone(),
            backend: self.backend.clone(),
            points: configurations.points,
            requirement: self.requirement.clone(),
            measurement_identity: configurations.evidence.identity,
            observations: configurations.evidence.observations,
            served: configurations.served,
            reselection: None,
            implementation_identity: self.implementation_identity.clone(),
            parameters: self.parameters.clone(),
            configurations: configurations.records,
            overall: Configuration::of(overall),
            method,
            time,
            coverage: None,
        }
    }

    /// The canonical defaults as the served choice of a unit not yet
    /// searching, measured at `defaults`, the points discovered so far:
    /// none for a unit whose mandatory work alone is done. The unmeasured
    /// scope is explicit in what the record lacks.
    fn census(
        &self,
        records: Vec<PointRecord>,
        default: &NativeSpecialization,
        ground: &Ground,
        defaults: Vec<PointMeasurement>,
        time: TuningTime,
    ) -> TuningResult {
        let kernel = &ground.kernel;
        // The canonical default is the reference itself: measuring it
        // qualifies nothing.
        let configurations = if defaults.is_empty() {
            Vec::new()
        } else {
            vec![ConfigurationRecord {
                formation: None,
                configuration: Configuration::of(default),
                artifact: Some(kernel.artifact().0.clone()),
                observations: Vec::new(),
                outcome: Outcome::Measured {
                    points: defaults,
                    confirmed: Vec::new(),
                },
            }]
        };
        let mut result = self.result(
            Records {
                points: records,
                records: configurations,
                served: vec![Configuration::of(default)],
                evidence: Evidence::of(&ground.validator),
            },
            default,
            TuningMethod::Census,
            time,
        );
        result.disposition = TuningDisposition::CanonicalFallback {
            artifact: kernel.artifact().0.clone(),
        };
        result
    }
}

/// The entry-wide parameters of `implementation`, as recorded.
fn entry_parameters(implementation: &NativeImplementation) -> Vec<DeclaredParameter> {
    implementation
        .params
        .iter()
        .map(|parameter| DeclaredParameter {
            name: parameter.name.clone(),
            launch: None,
            arithmetic: parameter.arithmetic,
            form: parameter.form,
            values: parameter.values.clone(),
        })
        .collect()
}

/// How many candidates of a sweep are formed and measured together: one
/// piece of work.
///
/// NOTE (sampling and pricing rewrite): this and [`SWEEP_AHEAD`] are
/// constants where a sweep's batch should be what its consumer admitted
/// for the piece by seconds and bytes; [`GroupSearch::advance`] asks on
/// until their sum awaits measurement.
const SWEEP_CHUNK: usize = 16;
/// How many candidates after a chunk have their programs formed with it
/// when the chunk needs a compile: three chunks more, so that a pass costs a
/// quarter of the compiles while at most three chunks' programs are formed
/// and not yet measured.
const SWEEP_AHEAD: usize = 3 * SWEEP_CHUNK;

/// The programs a search has formed and holds for the challengers it is
/// about to measure, as its unit's implementation is formed.
enum Programs {
    Launches(LaunchPrograms),
    Whole(WholePrograms),
}

/// The programs formed whole. Those of challengers not yet measured are
/// `ready`: each is taken by the piece that measures it, and the ledger
/// then holds its kernel only while it serves or leads
/// ([`Ledger::release`]). Its programs stay `held` apart from the kernel,
/// as launch programs do ([`LaunchPrograms`]), so a configuration converted
/// after its kernel was given up is assembled from them and compiles
/// nothing.
#[derive(Default)]
struct WholePrograms {
    ready: BTreeMap<Configuration, Result<Arc<NativePrepared>, Exclusion>>,
    held: BTreeMap<Configuration, HeldPrograms>,
    /// Programs formed so far.
    formed: usize,
}

impl Programs {
    /// The formation calls forming `specializations` makes: per launch
    /// whose held program lacks a variant they run, one compile of its
    /// source that also forms what `ahead` runs
    /// ([`LaunchPrograms::sources`]); or the program of each of them not
    /// formed yet, and nothing of `ahead`, whose programs share no compile
    /// with theirs. None when everything is held.
    fn formables(
        &self,
        implementation: &NativeImplementation,
        specializations: &[NativeSpecialization],
        ahead: &[NativeSpecialization],
    ) -> Vec<Formable> {
        match self {
            Self::Launches(programs) => {
                let sources = programs.sources(implementation, specializations, ahead);
                if sources.is_empty() {
                    Vec::new()
                } else {
                    vec![Formable::Launches(sources)]
                }
            }
            Self::Whole(programs) => specializations
                .iter()
                .filter(|specialization| {
                    !programs
                        .ready
                        .contains_key(&Configuration::of(specialization))
                })
                .cloned()
                .map(Formable::Program)
                .collect(),
        }
    }

    /// Hold the programs of a compile that ran, each call observed in the
    /// class it was priced by. Returns what the compile took.
    fn hold(&mut self, formed: Formed, price: &mut FormationPrice) -> Result<Duration, TuneError> {
        let programs = formed
            .calls
            .iter()
            .map(|call| call.formable.programs())
            .sum::<usize>();
        if programs > 0 {
            price
                .program_bytes
                .observe(formed.added_bytes / programs as f64);
        }
        for call in formed.calls {
            price.observe(call.class, call.elapsed);
            price.calls.push((
                call.formable.clone(),
                FormationRecord {
                    seconds: call.elapsed.as_secs_f64(),
                    programs: call.formable.programs(),
                    reused: call.class.0 == FormationClass::Kept,
                },
            ));
            match (&mut *self, call.formable, call.product) {
                (Self::Launches(programs), Formable::Launches(sources), Product::Launches(held)) => {
                    programs.hold(sources, held)?
                }
                (
                    Self::Whole(programs),
                    Formable::Program(specialization),
                    Product::Program(kernel),
                ) => {
                    let configuration = Configuration::of(&specialization);
                    // Assembled again from its held programs, it was
                    // formed before.
                    if !programs.held.contains_key(&configuration) {
                        programs.formed += 1;
                    }
                    programs.ready.insert(configuration, kernel);
                }
                _ => unreachable!("a unit's compile forms what its programs asked for"),
            }
        }
        Ok(formed.elapsed)
    }

    /// How each of `specializations`, formed for the piece about to
    /// measure them, was formed: the formation calls made since the last
    /// piece's ([`FormationPrice::calls`]), which this takes. Launch
    /// variants of a piece's challengers are formed by the same calls, so
    /// each challenger states all of them; a whole program states its own
    /// call. One for which no call was made ran held programs.
    fn records(
        &self,
        price: &mut FormationPrice,
        specializations: &[NativeSpecialization],
    ) -> Vec<FormationRecord> {
        let calls = std::mem::take(&mut price.calls);
        let held = FormationRecord {
            seconds: 0.,
            programs: 0,
            reused: true,
        };
        specializations
            .iter()
            .map(|specialization| {
                calls
                    .iter()
                    .filter(|(formable, _)| match (self, formable) {
                        (Self::Launches(_), _) => true,
                        (Self::Whole(_), Formable::Program(formed)) => formed == specialization,
                        (Self::Whole(_), Formable::Launches(_)) => false,
                    })
                    .fold(held, |sum, (_, call)| FormationRecord {
                        seconds: sum.seconds + call.seconds,
                        programs: sum.programs + call.programs,
                        reused: sum.reused && call.reused,
                    })
            })
            .collect()
    }

    /// Form `specializations`: first, in one compile, whatever of them is
    /// not held ([`Programs::formables`]); then each of them from what is
    /// held.
    fn form(
        &mut self,
        formation: &Formation<'_, '_>,
        specializations: &[NativeSpecialization],
        ahead: &[NativeSpecialization],
    ) -> Result<Vec<Result<Arc<NativePrepared>, Exclusion>>, TuneError> {
        let implementation = formation.implementation;
        if let Some(forming) =
            formation.forming(self.formables(implementation, specializations, ahead))
        {
            self.hold(
                forming.run(),
                &mut formation.price.lock().expect("formation price lock"),
            )?;
        }
        Ok(specializations
            .iter()
            .map(|specialization| match self {
                // Assembled from the held launch programs: nothing compiles.
                Self::Launches(_) => NativePrepared::prepare_implementation(
                    formation.device,
                    formation.module,
                    formation.entry,
                    formation.logical,
                    formation.bindings.clone(),
                    specialization.clone(),
                    formation.cpu,
                    implementation.clone(),
                )
                .map_err(|error| Exclusion::Formation(prepare_message(error)))
                .and_then(|kernel| serving(kernel, formation.served)),
                Self::Whole(programs) => {
                    let configuration = Configuration::of(specialization);
                    let kernel = programs
                        .ready
                        .remove(&configuration)
                        .expect("the program was formed above or ahead");
                    if let Ok(kernel) = &kernel {
                        programs.held.insert(configuration, kernel.programs());
                    }
                    kernel
                }
            })
            .collect())
    }

    /// How many programs are held: launch variants, or whole programs.
    fn holding(&self) -> usize {
        match self {
            Self::Launches(programs) => programs.holding(),
            Self::Whole(programs) => programs.held.len() + programs.ready.len(),
        }
    }

    /// Programs formed so far: launch variants, or whole programs.
    fn formed(&self) -> usize {
        match self {
            Self::Launches(programs) => programs.formed.len(),
            Self::Whole(programs) => programs.formed,
        }
    }

    /// Give up every held program: the unit has no work left for them, or
    /// its consumer needs their bytes. What was formed stays counted.
    fn clear(&mut self) {
        match self {
            Self::Launches(programs) => programs.held.clear(),
            Self::Whole(programs) => {
                programs.ready.clear();
                programs.held.clear();
            }
        }
    }
}

/// The launch programs a search holds for the configurations it is about
/// to measure. A launch's variants share one source, so those about to be
/// measured are formed in one compile and the configurations assembled
/// from them form nothing; a variant is formed only when a configuration
/// about to be measured runs it.
///
/// What was formed stays held for as long as the unit has work and its
/// consumer has room for it: every compile call's programs are kept beside
/// the earlier ones, so a variant is formed once however often a
/// configuration that runs it is measured, compared again or converted. A
/// unit gives its programs up only when its consumer asks, for the bytes
/// ([`UnitSearch::give_up_programs`]), or has no work left; a variant given
/// up is formed again when a configuration that runs it is measured again.
/// Releasing by role instead (keeping only what was about to be measured)
/// made a compile call in 861 of one load's 873 measuring pieces where
/// holding made one in 172 of 848, and took 40 s of compiles for 21 s.
#[derive(Default)]
struct LaunchPrograms {
    /// Per launch, each compile call's variants and the programs that hold
    /// them.
    held: BTreeMap<usize, Vec<(BTreeSet<plan::LaunchVariant>, Rc<HeldPrograms>)>>,
    /// Every launch variant formed so far.
    formed: BTreeSet<(usize, plan::LaunchVariant)>,
}

impl LaunchPrograms {
    /// Whether a held program of launch `ordinal` has `variant`.
    fn holds(&self, ordinal: usize, variant: &plan::LaunchVariant) -> bool {
        self.held
            .get(&ordinal)
            .is_some_and(|held| held.iter().any(|(variants, _)| variants.contains(variant)))
    }

    /// The launch sources forming `specializations` compiles: per launch
    /// with a variant they run that no held program has, those variants
    /// and the variants `ahead` (the configurations measured next) run of
    /// that launch that none has either. A compile call has a cost of its
    /// own besides what each variant it forms costs, so a call that must
    /// happen also forms what is measured next. None when every variant is
    /// held.
    fn sources(
        &self,
        implementation: &NativeImplementation,
        specializations: &[NativeSpecialization],
        ahead: &[NativeSpecialization],
    ) -> Vec<plan::LaunchSource> {
        let variants = |specializations: &[NativeSpecialization]| {
            let mut variants = BTreeMap::<usize, BTreeSet<plan::LaunchVariant>>::new();
            for specialization in specializations {
                let specialization = implementation.with_owned_defaults(specialization.clone());
                for ordinal in 0..implementation.launches.len() {
                    variants
                        .entry(ordinal)
                        .or_default()
                        .insert(plan::LaunchVariant {
                            code: plan::code_values(implementation, &specialization, ordinal),
                            group_size: implementation.static_group_size(&specialization, ordinal),
                        });
                }
            }
            variants
        };
        let mut next = variants(ahead);
        variants(specializations)
            .into_iter()
            .filter_map(|(ordinal, mut variants)| {
                variants.retain(|variant| !self.holds(ordinal, variant));
                if variants.is_empty() {
                    return None;
                }
                variants.extend(
                    next.remove(&ordinal)
                        .unwrap_or_default()
                        .into_iter()
                        .filter(|variant| !self.holds(ordinal, variant)),
                );
                Some(plan::LaunchSource {
                    ordinal,
                    variants: variants.into_iter().collect(),
                })
            })
            .collect()
    }

    /// How many launch variants the held programs have.
    fn holding(&self) -> usize {
        self.held
            .values()
            .flatten()
            .map(|(variants, _)| variants.len())
            .sum()
    }

    /// Hold the programs `held` of a compile of `sources` that ran.
    fn hold(
        &mut self,
        sources: Vec<plan::LaunchSource>,
        held: Result<HeldPrograms, String>,
    ) -> Result<(), TuneError> {
        let held = Rc::new(
            held.map_err(|error| TuneError::DefaultUnusable(Exclusion::Formation(error)))?,
        );
        for source in sources {
            self.formed.extend(
                source
                    .variants
                    .iter()
                    .cloned()
                    .map(|variant| (source.ordinal, variant)),
            );
            self.held
                .entry(source.ordinal)
                .or_default()
                .push((source.variants.into_iter().collect(), held.clone()));
        }
        Ok(())
    }
}

/// One boundary assignment as the sweeps left it: its score, its values
/// and each group's best assignment under it. A boundary no sweep reached
/// stands at its base: the seed's group assignments under it.
type Boundary = (f64, Vec<u64>, Vec<Vec<u64>>);

/// The nearby admissible change of each coordinate of `group` from its
/// assignment in `base`, in each direction: the nearest declared value the
/// domain admits there, with the other coordinates as `base` has them
/// wherever it permits ([`plan::Domain::nearest`]). Coupled restrictions
/// can require more than one coordinate to change; those are still real
/// starts.
fn coordinate_starts(
    domain: &dyn plan::Domain,
    group: &plan::Group,
    base: &NativeSpecialization,
) -> Result<Vec<NativeSpecialization>, TuneError> {
    let seed = group.assignment(base);
    let mut starts = Vec::<NativeSpecialization>::new();
    for coordinate in 0..group.parameters.len() {
        for ladder in group.steps(&seed, coordinate) {
            for assignment in ladder {
                // The moved coordinate alone is held: the others give way
                // where a restriction couples them to it.
                let admitted = domain
                    .nearest(
                        base,
                        &group.parameters[coordinate..=coordinate],
                        &assignment[coordinate..=coordinate],
                    )
                    .map_err(planning)?;
                if let Some(start) = admitted {
                    if !starts.contains(&start) {
                        starts.push(start);
                    }
                    break;
                }
            }
        }
    }
    Ok(starts)
}

/// Give each group one nearby change before giving any group its second.
/// Declaration order cannot let a large first group consume the complete
/// search slice before later launches have a rival.
fn coverage_order(
    domain: &dyn plan::Domain,
    groups: &[plan::Group],
    base: &NativeSpecialization,
) -> Result<Vec<NativeSpecialization>, TuneError> {
    let starts = groups
        .iter()
        .map(|group| coordinate_starts(domain, group, base))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((0..starts.iter().map(Vec::len).max().unwrap_or(0))
        .flat_map(|round| starts.iter().filter_map(move |starts| starts.get(round).cloned()))
        .collect())
}

/// One group's sweep under one base, as far as planning has taken it. A
/// sweep asks its group's domain in one order; planning continues it from
/// here each time, so its work follows what was measured since, never
/// everything measured before.
struct Sweep {
    /// The group's assignment under the base: where the sweep starts.
    seed: Vec<u64>,
    /// Every assignment asked for so far, the seed among them. An
    /// assignment's place in the order asked is how many were asked before
    /// it.
    asked: BTreeSet<Vec<u64>>,
    /// Asked for and not yet measured, in the order asked: each one's
    /// place in that order, and its configuration.
    pending: Vec<(usize, Configuration, NativeSpecialization)>,
    /// How many of the group's form starts were asked for.
    starts: usize,
    /// Each form's leader: the cheapest measured assignment of that form,
    /// the first asked among equals, with its cost and its place in the
    /// order asked.
    leaders: BTreeMap<Vec<u64>, (Vec<u64>, f64, usize)>,
    /// The leaders whose neighbours a round took up.
    expanded: BTreeSet<Vec<u64>>,
    /// The round in progress: enough awaited measurement before all its
    /// leaders' neighbours were asked for.
    round: Option<Round>,
    /// The group's choice under the base: the cheapest measured
    /// assignment, the seed's own among equals, so an unobserved group
    /// keeps it.
    best: (Vec<u64>, f64),
    /// What the leaders' neighbourhoods do not reach, asked for once they
    /// are measured ([`Rest`]), and whether all of it was asked for: the
    /// group's declared assignments are then all measured, passed over as
    /// inadmissible or inert, or awaiting measurement.
    rest: Rest,
    exhausted: bool,
    /// What awaits measurement was asked for from the rest, not by a
    /// leader's round.
    resting: bool,
}

/// The order in which a search asks for what its leaders' neighbourhoods
/// did not reach: the one place that order is decided. Given where a sweep
/// (or the boundaries) started, it yields every other assignment of the
/// declared value lists, each once, and nothing else of the search reads
/// how: a policy that orders them otherwise replaces this and
/// [`Rest::next`] alone.
///
/// The order here is the declared one nearest the start first: the
/// assignments that change one coordinate, then two, and so on, each
/// number's in the declaration's order of coordinates and values. Nothing
/// is produced before it is asked for: the walk holds one position, and
/// stepping it costs the number of coordinates. An assignment the domain
/// does not admit is passed over by whoever asks ([`plan::Domain::place`]),
/// so reaching the next admissible one costs one such check for each
/// declared assignment between them, and a walk to the end costs the
/// declared product of them, once.
struct Rest {
    start: Vec<u64>,
    /// Each coordinate's other values, in declaration order.
    others: Vec<Vec<u64>>,
    /// The coordinates changed, ascending, and which other value each has.
    changed: Vec<usize>,
    choice: Vec<usize>,
    ended: bool,
}

impl Rest {
    fn new(values: &[Vec<u64>], start: Vec<u64>) -> Self {
        let others = values
            .iter()
            .zip(&start)
            .map(|(values, at)| {
                values
                    .iter()
                    .copied()
                    .filter(|value| value != at)
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut rest = Self {
            start,
            others,
            changed: Vec::new(),
            choice: Vec::new(),
            ended: false,
        };
        rest.widen();
        rest
    }

    /// Whether every changed coordinate has another value to take.
    fn stands(&self) -> bool {
        self.changed
            .iter()
            .all(|coordinate| !self.others[*coordinate].is_empty())
    }

    /// Move to the first set of one more changed coordinates that stands;
    /// the walk ends when there is none.
    fn widen(&mut self) {
        let count = self.changed.len() + 1;
        self.changed = (0..count).collect();
        self.choice = vec![0; count];
        if count > self.others.len() {
            self.ended = true;
        } else if !self.stands() {
            self.shift();
        }
    }

    /// Move to the next set of as many changed coordinates that stands,
    /// widening past the last.
    fn shift(&mut self) {
        let (count, coordinates) = (self.changed.len(), self.others.len());
        loop {
            // The last position that can still move right.
            let Some(position) = (0..count)
                .rev()
                .find(|position| self.changed[*position] < coordinates - (count - position))
            else {
                return self.widen();
            };
            let next = self.changed[position] + 1;
            for (offset, slot) in self.changed[position..].iter_mut().enumerate() {
                *slot = next + offset;
            }
            self.choice = vec![0; count];
            if self.stands() {
                return;
            }
        }
    }
}

impl Iterator for Rest {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Vec<u64>> {
        if self.ended {
            return None;
        }
        let mut assignment = self.start.clone();
        for (coordinate, choice) in self.changed.iter().zip(&self.choice) {
            assignment[*coordinate] = self.others[*coordinate][*choice];
        }
        // The next choice of values for these coordinates, the last
        // coordinate's fastest; past the last, the next coordinates.
        let advanced = (0..self.choice.len()).rev().find(|position| {
            self.choice[*position] + 1 < self.others[self.changed[*position]].len()
        });
        match advanced {
            Some(position) => {
                self.choice[position] += 1;
                for later in &mut self.choice[position + 1..] {
                    *later = 0;
                }
            }
            None => self.shift(),
        }
        Some(assignment)
    }
}

/// A round of a sweep: its leaders, the cheapest form's first, and the
/// next step from the leader in turn to ask for.
struct Round {
    leaders: Vec<Vec<u64>>,
    leader: usize,
    neighbour: usize,
}

/// What a sweep asks a leader for: a form one form parameter from the
/// leader's (the coordinate and its value), or an assignment of the
/// leader's form one other parameter from it.
enum Step {
    Form((usize, u64)),
    Neighbour(Vec<u64>),
}

impl Sweep {
    /// A sweep of `group` from `seed`, which costs `cost`.
    fn new(group: &plan::Group, seed: Vec<u64>, cost: f64) -> Self {
        Self {
            asked: BTreeSet::from([seed.clone()]),
            pending: Vec::new(),
            starts: 0,
            leaders: BTreeMap::from([(group.form(&seed), (seed.clone(), cost, 0))]),
            expanded: BTreeSet::new(),
            round: None,
            best: (seed.clone(), cost),
            rest: Rest::new(&group.values, seed.clone()),
            exhausted: false,
            resting: false,
            seed,
        }
    }

    /// `assignment`, asked for `sequence`th, measured at `cost`.
    fn rank(&mut self, group: &plan::Group, assignment: Vec<u64>, cost: f64, sequence: usize) {
        let leader = self
            .leaders
            .entry(group.form(&assignment))
            .or_insert_with(|| (assignment.clone(), cost, sequence));
        if cost
            .total_cmp(&leader.1)
            .then_with(|| sequence.cmp(&leader.2))
            .is_lt()
        {
            *leader = (assignment.clone(), cost, sequence);
        }
        let seed = &self.seed;
        if cost
            .total_cmp(&self.best.1)
            .then_with(|| (assignment != *seed).cmp(&(self.best.0 != *seed)))
            .then_with(|| assignment.cmp(&self.best.0))
            .is_lt()
        {
            self.best = (assignment, cost);
        }
    }
}

/// The search of one unit's planned domain, as far as it got: the one
/// order in which a search asks its domain for configurations, whichever
/// way the unit's programs are formed and whoever measures them. It forms
/// and measures nothing itself: [`GroupSearch::plan`] names the next
/// challengers from everything recorded so far, and whoever measures them
/// says what they measured ([`GroupSearch::record`]). A unit's search
/// measures on its device ([`Run`]); a replay answers from a survey's
/// record ([`super::replay`]).
pub struct GroupSearch {
    partition: Rc<plan::TuningPartition>,
    /// Whole combinations the sweeps' group choices generate, each measured
    /// as a configuration of its own before it can lead.
    pending: BTreeMap<Configuration, NativeSpecialization>,
    /// The reference of every cost: the defaults.
    seed: NativeSpecialization,
    reference: Vec<PointMeasurement>,
    /// The consumer's hints ([`UnitPlan::start`]).
    start: Vec<ParameterValues>,
    /// Every configuration measured, with its measurement; none for one
    /// that could not be formed or measured. Planning names only what is
    /// missing of these.
    swept: HashMap<Configuration, Option<Vec<PointMeasurement>>>,
    /// Each group's sweep under each base, by the base's boundary and the
    /// group, as far as planning has taken it.
    sweeps: BTreeMap<(Vec<u64>, usize), Sweep>,
    /// Requests the sweeps have made of the declared domain: one for each
    /// assignment asked for, admitted or not.
    requests: usize,
    default_boundary: Vec<u64>,
    /// The timed points' weights in the objective.
    weights: Vec<f64>,
    /// Every configuration the search has asked its domain for, measured
    /// or not: the declared domain is never produced, and this is all of
    /// it that exists.
    generated: BTreeSet<Configuration>,
    /// The last planning went through its whole order and named nothing:
    /// the one way a search ends ([`GroupSearch::exhausted`]).
    ended: bool,
    /// The piece planned last measures other forms' representatives: the
    /// unit's structural exploration, ahead of every refinement.
    structural: bool,
    /// The piece the planning in progress names: the first work its pass
    /// finds missing.
    proposed: Option<Piece>,
    /// Passes of the sweeps so far: whose turn it is to ask for the rest of
    /// its group ([`Rest`]).
    rest_turn: usize,
}

impl GroupSearch {
    /// Open the search of `partition` from `seed`, the defaults, measured
    /// at `reference` (keyed as every later measurement is) at timed points
    /// that weigh `weights` in the objective, with the consumer's hints
    /// `start`. Nothing is asked of the domain here.
    pub fn new(
        partition: plan::TuningPartition,
        seed: NativeSpecialization,
        reference: Vec<PointMeasurement>,
        weights: Vec<f64>,
        start: Vec<ParameterValues>,
    ) -> Self {
        Self {
            default_boundary: partition.boundary_of(&seed),
            partition: Rc::new(partition),
            pending: BTreeMap::new(),
            seed,
            reference,
            start,
            swept: HashMap::new(),
            sweeps: BTreeMap::new(),
            requests: 0,
            weights,
            generated: BTreeSet::new(),
            ended: false,
            structural: false,
            proposed: None,
            rest_turn: 0,
        }
    }

    /// How many configurations the declaration's value lists name: the
    /// product of every parameter's values, which `where` can only reduce.
    /// A number alone: nothing is produced to count it.
    pub fn declared(&self) -> u128 {
        let boundary = self
            .partition
            .boundary_values
            .iter()
            .fold(1u128, |product, values| product.saturating_mul(values.len() as u128));
        self.partition
            .groups
            .iter()
            .fold(boundary, |product, group| product.saturating_mul(group.declared()))
    }

    /// Whether the search has asked for everything its order reaches and
    /// measured it: every sweep begun has asked for all its group declares,
    /// and nothing asked for awaits measurement. The sweeps are those of
    /// the groups under every boundary that was a base: the boundaries one
    /// parameter from each base that led.
    ///
    /// This is exactly when [`GroupSearch::plan`] names nothing: planning
    /// that names no piece has gone through its whole order, every form,
    /// base and coverage of the present leader measured and every sweep at
    /// its end, and planning that has anything left names it. A search
    /// that stopped any other way is not exhausted.
    pub fn exhausted(&self) -> bool {
        self.ended
    }

    /// Whether what a sweep has awaiting measurement is of the rest of its
    /// group: every leader's neighbourhood under that sweep is measured.
    pub fn resting(&self) -> bool {
        self.sweeps
            .values()
            .any(|sweep| sweep.resting && !sweep.pending.is_empty())
    }

    /// `specialization` was measured at `measured`, or could not be formed
    /// or measured (none): planning asks for it no more.
    pub fn record(
        &mut self,
        specialization: &NativeSpecialization,
        measured: Option<Vec<PointMeasurement>>,
    ) {
        let configuration = Configuration::of(specialization);
        self.pending.remove(&configuration);
        self.swept.insert(configuration, measured);
    }

    /// The one rule for what is not worth forming: a change of `base`,
    /// measured at `baseline`, that under the same boundary is expected to
    /// run the work `base` runs at every point with serving demand
    /// ([`plan::Domain::forecast`]). It cannot change the service cost, so
    /// it has no expected improvement and nothing is formed or measured
    /// for it. A group none of whose launches works where there is demand
    /// is the case where this holds of every change of it. The served
    /// choice is still covered at the points without demand, by its
    /// conversion alone.
    fn inert(
        &self,
        domain: &dyn plan::Domain,
        candidate: &NativeSpecialization,
        base: &NativeSpecialization,
        baseline: &[PointMeasurement],
    ) -> bool {
        self.partition.boundary_of(candidate) == self.partition.boundary_of(base)
            && baseline
                .iter()
                .zip(&self.weights)
                .enumerate()
                .all(|(point, (measured, weight))| {
                    *weight <= 0. || domain.forecast(point, &measured.key, candidate) == measured.key
                })
    }

    /// The consumer's hints the domain admits, as whole configurations:
    /// the defaults with the hinted values of the parameters the plan
    /// moves. A hint that moves a parameter the plan holds, or that the
    /// domain does not admit, is skipped.
    fn hints(&self, domain: &dyn plan::Domain) -> Result<Vec<NativeSpecialization>, TuneError> {
        let planned = self
            .partition
            .boundary
            .iter()
            .chain(self.partition.groups.iter().flat_map(|group| &group.parameters))
            .collect::<Vec<_>>();
        let mut hints = Vec::new();
        for hint in &self.start {
            let mut moved = (Vec::new(), Vec::new());
            let held = hint.iter().all(|(name, value)| {
                let address = plan::ParameterAddress::Entry(name.clone());
                if planned.contains(&&address) {
                    moved.0.push(address);
                    moved.1.push(*value);
                    true
                } else {
                    self.seed.param(name) == Some(*value)
                }
            });
            if held {
                hints.extend(
                    domain
                        .place(&self.seed, &moved.0, &moved.1)
                        .map_err(planning)?,
                );
            }
        }
        Ok(hints)
    }

    /// The representative of the form that has the form parameter at
    /// `address` at `value`: the first of `hints` of that form, which its
    /// consumer named; else the configuration of the form nearest the
    /// defaults ([`plan::Domain::nearest`]), where a parameter the form
    /// needs moved has given way. None when the domain admits none.
    fn representative(
        &self,
        domain: &dyn plan::Domain,
        hints: &[NativeSpecialization],
        address: &plan::ParameterAddress,
        value: u64,
    ) -> Result<Option<NativeSpecialization>, TuneError> {
        match hints.iter().find(|hint| address.value(hint) == value) {
            Some(hint) => Ok(Some(hint.clone())),
            None => domain
                .nearest(&self.seed, std::slice::from_ref(address), &[value])
                .map_err(planning),
        }
    }

    /// Planning named its piece: the rest of its pass names nothing.
    fn named(&self) -> bool {
        self.proposed.is_some()
    }

    /// Add a whole configuration to the piece planning names, unless it
    /// named a sweep already.
    fn propose_whole(&mut self, configuration: &NativeSpecialization) {
        match &mut self.proposed {
            None => self.proposed = Some(Piece::Whole(vec![configuration.clone()])),
            Some(Piece::Whole(configurations)) => configurations.push(configuration.clone()),
            Some(_) => {}
        }
    }

    /// Name the candidates `ready` of `group` as the piece to measure
    /// together, unless planning named one already. The whole chunk is
    /// formed together: one compile per launch also forms the variants
    /// measured next (`ahead`), so a compile's fixed cost is paid once for
    /// many candidates.
    fn propose_sweep(
        &mut self,
        group: usize,
        baseline: &[PointMeasurement],
        ready: &[NativeSpecialization],
        ahead: &[NativeSpecialization],
    ) {
        if ready.is_empty() || self.named() {
            return;
        }
        self.proposed = Some(Piece::Sweep {
            group,
            baseline: baseline.to_vec(),
            ready: ready.to_vec(),
            ahead: ahead.to_vec(),
        });
    }

}

impl Run {
    /// Form the challengers of `piece` and measure and admit them at the
    /// timed points. The timed run compares at the timed points; untimed
    /// served points are a leader's coverage check alone.
    #[allow(clippy::too_many_arguments)]
    fn perform(
        &mut self,
        piece: Piece,
        formation: &Formation<'_, '_>,
        points: &[PreparedPoint<'_>],
        settings: &SearchSettings,
        min_sample_seconds: f64,
        rates: &mut Rates,
        weighing: &Weighing,
        ledger: &mut Ledger,
        admission: &mut Admission,
        forming_seconds: &mut f64,
        measuring_seconds: &mut f64,
        deadline: Deadline,
    ) -> Result<(), WorkError<TuneError>> {
        deadline.check(Duration::ZERO)?;
        let implementation = formation.implementation;
        let (specializations, ahead, group) = match piece {
            Piece::Whole(configurations) => (configurations, Vec::new(), None),
            Piece::Sweep {
                group,
                baseline,
                ready,
                ahead,
            } => (ready, ahead, Some((group, baseline))),
        };
        let began = Instant::now();
        let formed = self.programs.form(formation, &specializations, &ahead)?;
        *forming_seconds += began.elapsed().as_secs_f64();
        let formations = self.programs.records(
            &mut formation.price.lock().expect("formation price lock"),
            &specializations,
        );
        for (specialization, formation) in specializations.iter().zip(formations) {
            ledger
                .formations
                .insert(Configuration::of(specialization), formation);
        }
        let mut candidates = Vec::with_capacity(specializations.len());
        for (specialization, kernel) in specializations.into_iter().zip(formed) {
            match kernel {
                Ok(kernel) => candidates.push(Candidate {
                    specialization,
                    kernel,
                    evidence: Tally::default(),
                }),
                Err(exclusion) => {
                    self.search.record(&specialization, None);
                    ledger.exclude(Configuration::of(&specialization), None, exclusion);
                }
            }
        }
        let began = Instant::now();
        let keying = &self.keying;
        let key: Key<'_> =
            &|working, specialization| keying.key(implementation, working, specialization);
        // Every challenger of the piece is measured beside the served
        // choice in one session: a sweep's candidates where their group's
        // launches work, from the measurement of what they change; whole
        // configurations at every point, from the defaults'.
        let served = ledger.incumbent();
        let (served_kernel, served_points) = (
            served.kernel.clone().expect("the served choice holds its kernel"),
            served.points.clone(),
        );
        let (affected, baseline) = match &group {
            Some((group, baseline)) => (
                Some(self.search.partition.groups[*group].launches.as_slice()),
                baseline.as_slice(),
            ),
            None => (None, self.search.reference.as_slice()),
        };
        let measured = measure_matched(
            key,
            &candidates,
            &Incumbent {
                specialization: &served.specialization,
                kernel: &served_kernel,
                points: &served_points,
            },
            points,
            Matching {
                deadline,
                settings,
                min_sample_seconds,
                rates,
                shuffle: &mut self.shuffle,
                clock: &mut self.clock,
                sequence: &mut self.sequence,
            },
            affected,
            baseline,
            Some(&mut self.samples),
            &mut self.outputs,
            Some(admission),
        );
        *measuring_seconds += began.elapsed().as_secs_f64();
        let mut interrupted = false;
        for (candidate, outcome) in candidates.into_iter().zip(measured) {
            match outcome {
                Ok(None) => interrupted = true,
                Ok(Some((measured, evidence))) => {
                    self.search
                        .record(&candidate.specialization, Some(measured.clone()));
                    let cost = weighing.cost(&measured, &self.search.reference);
                    ledger.measured(
                        candidate.specialization,
                        candidate.kernel,
                        measured,
                        cost,
                        evidence,
                    );
                }
                Err(exclusion) => {
                    self.search.record(&candidate.specialization, None);
                    ledger.exclude(
                        Configuration::of(&candidate.specialization),
                        Some(&candidate.kernel),
                        exclusion,
                    );
                }
            }
        }
        if interrupted {
            Err(WorkError::Expired)
        } else {
            Ok(())
        }
    }
}

impl GroupSearch {
    /// Ask `sweep`'s group for `specialization`, a change of `base`
    /// (measured at `baseline`), once: a configuration measured before is
    /// ranked at once, one never measured awaits its measurement, and one
    /// that could not be measured, or that cannot change the service cost
    /// ([`GroupSearch::inert`]), is neither.
    #[allow(clippy::too_many_arguments)]
    fn ask(
        &mut self,
        sweep: &mut Sweep,
        domain: &dyn plan::Domain,
        group: &plan::Group,
        weighing: &Weighing,
        base: &NativeSpecialization,
        baseline: &[PointMeasurement],
        specialization: Option<NativeSpecialization>,
    ) {
        self.requests += 1;
        let Some(specialization) = specialization else {
            return;
        };
        let assignment = group.assignment(&specialization);
        if !sweep.asked.insert(assignment.clone()) {
            return;
        }
        if self.inert(domain, &specialization, base, baseline) {
            return;
        }
        let sequence = sweep.asked.len() - 1;
        let configuration = Configuration::of(&specialization);
        self.generated.insert(configuration.clone());
        match self.swept.get(&configuration) {
            Some(Some(measured)) => sweep.rank(
                group,
                assignment,
                weighing.cost(measured, &self.reference).total(),
                sequence,
            ),
            Some(None) => {}
            None => sweep.pending.push((sequence, configuration, specialization)),
        }
    }

    /// Take `sweep` of `group` under `base` (measured at `baseline`) as far
    /// as what was measured since it was last advanced allows: rank what
    /// was asked and has been measured, then ask on in the sweep's one
    /// order until enough awaits measurement. Each other value of each
    /// form parameter seeds a start of its own, so every form one form
    /// parameter from the base's is measured first, at the configuration
    /// of it nearest the base. Then each form's leader, the cheapest
    /// measured of its form, is asked for what is one parameter from it,
    /// the cheapest form's first, in rounds: the forms one form parameter
    /// from the leader's, each at the configuration nearest the leader,
    /// and then the leader's neighbours inside its own form. A round's
    /// leaders are those not asked before, and the next round begins when
    /// everything asked is measured. A neighbour that measures cheaper
    /// leads the next round; a form further from the base than one
    /// parameter is reached from the form between once that one has a
    /// leader; a form whose leader has nothing one parameter from it left
    /// unmeasured asks nothing more.
    fn advance(
        &mut self,
        sweep: &mut Sweep,
        domain: &dyn plan::Domain,
        group: &plan::Group,
        base: &NativeSpecialization,
        baseline: &[PointMeasurement],
        weighing: &Weighing,
        rest: bool,
    ) -> Result<(), TuneError> {
        for (sequence, configuration, specialization) in std::mem::take(&mut sweep.pending) {
            match self.swept.get(&configuration) {
                Some(Some(measured)) => sweep.rank(
                    group,
                    group.assignment(&specialization),
                    weighing.cost(measured, &self.reference).total(),
                    sequence,
                ),
                Some(None) => {}
                None => sweep.pending.push((sequence, configuration, specialization)),
            }
        }
        let wanted = SWEEP_CHUNK + SWEEP_AHEAD;
        // The form `coordinate` at `value` selects, at the configuration of
        // it nearest `from`.
        let form = |from: &NativeSpecialization, (coordinate, value): (usize, u64)| {
            domain
                .nearest(from, &group.parameters[coordinate..=coordinate], &[value])
                .map_err(planning)
        };
        let seed = sweep.seed.clone();
        for start in group.form_starts(&seed).skip(sweep.starts) {
            if sweep.pending.len() >= wanted {
                break;
            }
            sweep.starts += 1;
            let admitted = form(base, start)?;
            self.ask(sweep, domain, group, weighing, base, baseline, admitted);
        }
        loop {
            let mut round = match sweep.round.take() {
                Some(round) => round,
                // A round begins when everything asked is measured.
                None if !sweep.pending.is_empty() => break,
                None => {
                    let mut leaders = sweep
                        .leaders
                        .values()
                        .map(|(leader, cost, _)| (leader.clone(), *cost))
                        .collect::<Vec<_>>();
                    leaders.sort_by(|left, right| {
                        left.1.total_cmp(&right.1).then_with(|| left.0.cmp(&right.0))
                    });
                    let leaders = leaders
                        .into_iter()
                        .map(|(leader, _)| leader)
                        .filter(|leader| sweep.expanded.insert(leader.clone()))
                        .collect::<Vec<_>>();
                    if leaders.is_empty() {
                        // Every leader's neighbourhood is measured: the
                        // rest of the group's assignments, a chunk at a
                        // time. One of them that measures cheapest of its
                        // form leads the next round, and its neighbourhood
                        // is asked for before the rest goes on. The groups
                        // take the rest in turn (`rest`), so none has its
                        // second chunk of it before every other its first.
                        if !rest {
                            break;
                        }
                        sweep.resting = true;
                        let asked = sweep.asked.len();
                        while sweep.pending.len() < wanted {
                            let Some(assignment) = sweep.rest.next() else {
                                sweep.exhausted = true;
                                break;
                            };
                            if sweep.asked.contains(&assignment) {
                                continue;
                            }
                            let admitted = domain
                                .place(base, &group.parameters, &assignment)
                                .map_err(planning)?;
                            self.ask(sweep, domain, group, weighing, base, baseline, admitted);
                        }
                        // What was asked and had been measured under
                        // another base is ranked already and may lead.
                        if sweep.pending.is_empty() && sweep.asked.len() > asked {
                            continue;
                        }
                        break;
                    }
                    sweep.resting = false;
                    Round {
                        leaders,
                        leader: 0,
                        neighbour: 0,
                    }
                }
            };
            while round.leader < round.leaders.len() && sweep.pending.len() < wanted {
                let leader = &round.leaders[round.leader];
                // The leader as a whole configuration: what the forms one
                // parameter from it are nearest to. None when the leader
                // stands only with a parameter outside the group moved: it
                // has no configuration under this base to be near.
                let leading = domain
                    .place(base, &group.parameters, leader)
                    .map_err(planning)?;
                let mut steps = group
                    .form_starts(leader)
                    .map(Step::Form)
                    .chain(group.neighbours(leader).map(Step::Neighbour))
                    .skip(round.neighbour);
                let mut asked_all = false;
                while sweep.pending.len() < wanted {
                    let Some(step) = steps.next() else {
                        asked_all = true;
                        break;
                    };
                    round.neighbour += 1;
                    let admitted = match step {
                        Step::Form(start) => match &leading {
                            Some(leading) => form(leading, start)?,
                            None => None,
                        },
                        Step::Neighbour(neighbour) => domain
                            .place(base, &group.parameters, &neighbour)
                            .map_err(planning)?,
                    };
                    self.ask(sweep, domain, group, weighing, base, baseline, admitted);
                }
                if asked_all {
                    round.leader += 1;
                    round.neighbour = 0;
                }
            }
            if round.leader < round.leaders.len() {
                // Enough awaits measurement: the round goes on from here.
                sweep.round = Some(round);
                break;
            }
        }
        Ok(())
    }

    /// Name the next piece of the sweeps, coarse to fine, from everything
    /// measured so far; none when nothing is left.
    /// First the base of every boundary: the seed's group choices under
    /// each value of the parameters that choose which launch serves which
    /// rows, so every such value has a measurement at the rows it moves
    /// before any launch's mapping is refined. The boundaries' groups are
    /// then swept, the boundary with the cheapest base first (the default
    /// boundary when its base is within one percent of the cheapest). That
    /// order is fixed by the bases. Nearby changes of the leading
    /// boundary's groups come round-robin before deeper sweeps, so the
    /// pieces follow the order one uninterrupted search does. Every
    /// boundary with a measured base stays ranked by what was measured
    /// under it.
    ///
    /// Nothing of the declared domain exists before it is asked for here.
    /// A boundary is asked for one parameter from the leading one, so the
    /// bases follow the leader instead of the boundary parameters'
    /// product; a group is asked for the forms one form parameter from its
    /// seed's and then, of the cheapest measured of each form, the
    /// cheapest form first, for the forms one form parameter from it and
    /// the assignments one other parameter from it, so a group's sweep
    /// follows its leaders downhill and through the forms, and ends where
    /// everything one parameter from every form's leader is measured. The
    /// forms, the bases and the leading base's coverage are derived here
    /// each time, at a cost the declaration's structure fixes (the sum of
    /// its parameters' value counts); each group's sweep continues from
    /// where the last planning left it ([`Sweep`]), so what planning a
    /// piece costs follows what was measured since the last, never
    /// everything measured before, and never the domain's size.
    pub fn plan(&mut self, domain: &dyn plan::Domain) -> Result<Option<Piece>, TuneError> {
        self.proposed = None;
        self.ended = false;
        if !self.pending.is_empty() {
            return Ok(Some(Piece::Whole(self.pending.values().cloned().collect())));
        }
        let partition = self.partition.clone();
        let weighing = Weighing {
            weights: self.weights.clone(),
        };
        // Structurally different forms come before any refinement: one
        // representative of each form one form parameter from the
        // defaults, whether that parameter selects which launch serves
        // (a boundary) or the shape of a group's launches, and the
        // consumer's other hints with them. Their best settings of the
        // other parameters are unrelated to the defaults', so nothing
        // measured inside the default form says whether another form is
        // faster. A form two form parameters from the defaults is not
        // among them: it is reached from the form between, by the sweeps.
        self.structural = false;
        let hints = self.hints(domain)?;
        let mut forms = Vec::new();
        for coordinate in 0..partition.boundary.len() {
            if !partition.boundary_forms[coordinate] {
                continue;
            }
            for value in &partition.boundary_values[coordinate] {
                if *value != self.default_boundary[coordinate] {
                    forms.extend(self.representative(
                        domain,
                        &hints,
                        &partition.boundary[coordinate],
                        *value,
                    )?);
                }
            }
        }
        for group in &partition.groups {
            for (coordinate, value) in group.form_starts(&group.assignment(&self.seed)) {
                forms.extend(self.representative(
                    domain,
                    &hints,
                    &group.parameters[coordinate],
                    value,
                )?);
            }
        }
        forms.extend(hints);
        let mut named = BTreeSet::from([Configuration::of(&self.seed)]);
        for form in forms {
            let configuration = Configuration::of(&form);
            if !named.insert(configuration.clone())
                || self.inert(domain, &form, &self.seed, &self.reference)
            {
                continue;
            }
            self.generated.insert(configuration.clone());
            if !self.swept.contains_key(&configuration) {
                self.propose_whole(&form);
            }
        }
        if self.named() {
            self.structural = true;
            return Ok(self.proposed.take());
        }
        let mut boundaries: Vec<Boundary> = Vec::new();
        let mut bases_ready = true;
        // Each base: its cost, its boundary, its measurement and its
        // configuration. The default boundary's is the seed.
        let seed_cost = weighing.cost(&self.reference, &self.reference).total();
        let mut bases = vec![(
            seed_cost,
            self.default_boundary.clone(),
            self.reference.clone(),
            self.seed.clone(),
        )];
        let lead = |bases: &mut Vec<(f64, Vec<u64>, Vec<PointMeasurement>, NativeSpecialization)>,
                    default_boundary: &[u64]| {
            bases.sort_by(|left, right| left.0.total_cmp(&right.0));
            if let Some(default_index) = bases
                .iter()
                .position(|(_, boundary, _, _)| boundary == default_boundary)
            {
                if bases[default_index].0 <= bases[0].0 * 1.01 {
                    let preferred = bases.remove(default_index);
                    bases.insert(0, preferred);
                }
            }
        };
        // The boundaries one parameter from the leading base, again from
        // each base that takes the lead, until the leader's are all asked.
        let mut expanded = BTreeSet::<Vec<u64>>::new();
        loop {
            lead(&mut bases, &self.default_boundary);
            let leader = bases[0].1.clone();
            if !expanded.insert(leader.clone()) {
                break;
            }
            for boundary in partition.boundary_neighbours(&leader) {
                if bases.iter().any(|(_, known, _, _)| *known == boundary) {
                    continue;
                }
                let admitted = domain
                    .nearest(&self.seed, &partition.boundary, &boundary)
                    .map_err(planning)?;
                let Some(base) = admitted else {
                    continue;
                };
                let configuration = Configuration::of(&base);
                self.generated.insert(configuration.clone());
                match self.swept.get(&configuration).cloned() {
                    Some(Some(measured)) => {
                        let cost = weighing.cost(&measured, &self.reference).total();
                        bases.push((cost, boundary, measured, base));
                    }
                    Some(None) => {}
                    None => {
                        bases_ready = false;
                        self.propose_whole(&base);
                    }
                }
            }
        }
        // Cover the coordinates of the leading boundary before a deep sweep
        // of any group, skipping every configuration already measured or
        // excluded and every change that cannot change the service cost.
        // The coverage is the present leader's, derived here each time and
        // owed only while it leads: until every base asked for is measured
        // the leader is not settled and nothing is asked under it, and
        // what was covered under a base that led before owes nothing once
        // another leads (its assignments are still its own sweeps'). The
        // coverage candidates are one piece: they are formed together.
        if bases_ready {
            let (_, _, baseline, base) = &bases[0];
            for specialization in coverage_order(domain, &partition.groups, base)? {
                if self.inert(domain, &specialization, base, baseline) {
                    continue;
                }
                let configuration = Configuration::of(&specialization);
                self.generated.insert(configuration.clone());
                if !self.swept.contains_key(&configuration) {
                    self.propose_whole(&specialization);
                }
            }
        }
        if self.named() {
            // Do not assemble every downhill group while a base or the
            // leader's coverage awaits measurement.
            return Ok(self.proposed.take());
        }
        // The boundaries the bases stand at: a combination is asked for
        // under each of them.
        let based = bases
            .iter()
            .map(|(_, boundary, _, _)| boundary.clone())
            .collect::<Vec<_>>();
        // The sweeps are advanced until one names work. What the leaders'
        // neighbourhoods do not reach is asked of one group at a time, the
        // groups in turn: a pass that names nothing gives the turn to the
        // next group, and when every group has had it and named nothing,
        // every sweep has asked for all its group declares.
        let turns = partition.groups.len().max(1);
        let mut passes = 0;
        loop {
            boundaries.clear();
            let turn = self.rest_turn % turns;
            self.rest_turn += 1;
            passes += 1;
            for (base_cost, boundary, baseline, base) in bases.iter().cloned() {
                let mut choices = partition
                    .groups
                    .iter()
                    .map(|group| group.assignment(&base))
                    .collect::<Vec<_>>();
                let mut score = base_cost;
                for (group_index, group) in partition.groups.iter().enumerate() {
                    let seed = group.assignment(&base);
                    // The group's sweep under this base continues from
                    // where the last planning left it: what was asked then
                    // is not asked again.
                    let key = (boundary.clone(), group_index);
                    let mut sweep = self
                        .sweeps
                        .remove(&key)
                        .unwrap_or_else(|| Sweep::new(group, seed.clone(), base_cost));
                    let advanced = self.advance(
                        &mut sweep,
                        domain,
                        group,
                        &base,
                        &baseline,
                        &weighing,
                        group_index == turn,
                    );
                    if advanced.is_ok() && !sweep.pending.is_empty() {
                        // One chunk is a piece; what follows it is asked
                        // for from what it measures.
                        let pending = sweep
                            .pending
                            .iter()
                            .map(|(_, _, specialization)| specialization.clone())
                            .collect::<Vec<_>>();
                        let (ready, rest) =
                            pending.split_at(pending.len().min(SWEEP_CHUNK));
                        self.propose_sweep(group_index, &baseline, ready, rest);
                    }
                    choices[group_index] = sweep.best.0.clone();
                    score += sweep.best.1 - base_cost;
                    self.sweeps.insert(key, sweep);
                    advanced?;
                    // Later groups are always advanced, whatever was named.
                }
                boundaries.push((score, boundary, choices));
            }
            if self.proposed.is_some() || passes == turns {
                break;
            }
        }
        boundaries.sort_by(|left, right| left.0.total_cmp(&right.0));
        // A boundary moves work between launches. Prefer the default band split
        // when its apparent loss is within one percent of the leading score.
        if let Some(default_index) = boundaries
            .iter()
            .position(|(_, boundary, _)| *boundary == self.default_boundary)
        {
            if boundaries[default_index].0 <= boundaries[0].0 * 1.01 {
                let preferred = boundaries.remove(default_index);
                boundaries.insert(0, preferred);
            }
        }
        // Inside a group's sweep the candidates measured so far are ranked,
        // while later groups retain their defaults, and the boundaries no
        // sweep reached stand at their bases. Group scores generate
        // combinations; only a whole configuration actually measured can
        // lead.
        let seed = Configuration::of(&self.seed);
        for (_, _, choices) in &boundaries {
            for candidate_boundary in &based {
                let (parameters, values) = partition.assembled(choices, candidate_boundary);
                if let Some(candidate) = domain
                    .place(&self.seed, &parameters, &values)
                    .map_err(planning)?
                {
                    let configuration = Configuration::of(&candidate);
                    if !self.swept.contains_key(&configuration) && configuration != seed {
                        self.generated.insert(configuration.clone());
                        self.pending.insert(configuration, candidate);
                    }
                }
            }
        }
        // A sweep named above comes before the combinations it may change.
        let piece = self.proposed.take().or_else(|| {
            (!self.pending.is_empty())
                .then(|| Piece::Whole(self.pending.values().cloned().collect()))
        });
        // The one place planning names nothing: every return above names a
        // piece, and here every group has had its turn at the rest of its
        // domain under every base and none has anything awaiting
        // measurement. The search is exhausted exactly then.
        self.ended = piece.is_none();
        debug_assert!(
            !self.ended
                || self
                    .sweeps
                    .values()
                    .all(|sweep| sweep.exhausted && sweep.pending.is_empty()),
            "a search that names nothing has a sweep with work left"
        );
        Ok(piece)
    }
}

/// Digest of everything about an entry's implementation on `device`'s
/// backend that its tuning depends on besides the device: the declaration
/// (parameters and their domains, `where`, launches, scratch) and, for Metal,
/// CUDA and Vulkan, the source rendered for the defaults at these bindings
/// and static values (the asset with its inlined includes, and the ABI
/// prefix); for CPU, the compiled implementation's digest (its asset, the
/// CPU library files of its source root and the Seismic CPU library version).
/// Embedders key stored tuning results with it.
pub fn implementation_digest(
    device: &Arc<DeviceInner>,
    module: &CheckedModule,
    entry: EntryId,
    bindings: &ElementBindings,
    statics: &NativeSpecialization,
    cpu: Option<&'static CpuNativeKernels>,
) -> Result<String, TuneError> {
    let backend = super::backend_name(&device.kind);
    let entry_name = super::entry_name(module, entry);
    let implementation = module
        .native_implementation(entry, backend)
        .ok_or_else(|| {
            TuneError::Declaration(format!(
                "`{entry_name}` has no native implementation for `{}`",
                backend.as_str()
            ))
        })?;
    let default = implementation
        .default_specialization(statics)
        .map_err(|error| TuneError::Declaration(error.to_string()))?;
    let mut digest = Sha256::new();
    update_declaration_digest(&mut digest, module, &entry_name, implementation);
    // Backends whose implementations are rendered source; CPU implementations
    // are compiled into the binary and identified by their compiled digest.
    let dialect = match &device.kind {
        crate::backends::OpenedKind::Cpu(_) => {
            let kernels = cpu.ok_or_else(|| {
                TuneError::Declaration(format!(
                    "`{entry_name}` has no compiled CPU native functions in this build"
                ))
            })?;
            digest.update(kernels.digest.as_bytes());
            None
        }
        #[cfg(target_os = "macos")]
        crate::backends::OpenedKind::Metal(opened) => Some(super::metal_dialect(opened)),
        crate::backends::OpenedKind::Cuda(_) => Some(super::abi::Dialect::Cuda),
        #[cfg(not(target_os = "macos"))]
        crate::backends::OpenedKind::Vulkan(opened) => {
            Some(super::abi::Dialect::Vulkan(opened.features()))
        }
    };
    if let Some(dialect) = dialect {
        let logical = module
            .entry(entry, bindings)
            .map_err(|error| TuneError::Declaration(error.to_string()))?;
        let asset = module.native_asset(entry, backend).ok_or_else(|| {
            TuneError::Declaration(format!(
                "native asset of `{entry_name}` is absent from the module"
            ))
        })?;
        digest.update(
            super::abi::render_source(dialect, &logical, bindings, implementation, &default, asset)
                .as_bytes(),
        );
    }
    Ok(crate::telemetry::hex(&digest.finalize()))
}

/// Checked entry IDs carry a process-local owner. A tuning result names the
/// declaration's values and stable source identity, never that owner. Where
/// the declaring file was read from is not the implementation either: a build
/// canonicalizes its sources to absolute paths, so the same sources built in
/// another directory must keep their stored results. The native source the
/// declaration names is digested by content, rendered for the entry.
fn update_declaration_digest(
    digest: &mut Sha256,
    module: &CheckedModule,
    entry_name: &str,
    implementation: &NativeImplementation,
) {
    digest.update(b"native-declaration-v2");
    digest.update(format!("{entry_name:?}").as_bytes());
    digest.update(
        format!(
            "{:?}",
            (
                implementation.backend,
                &implementation.source_path,
                &implementation.statics,
                &implementation.params,
                &implementation.elements,
                &implementation.constraint,
                &implementation.scratch,
                &implementation.launches,
            )
        )
        .as_bytes(),
    );
    // The declaration is named by entry within the module's source text. A
    // stable entry identity also hashes the paths the sources were built
    // from, which never identify an implementation.
    digest.update(b"checked-native-source-v4");
    for file in module.sources().files() {
        digest.update((file.text.len() as u64).to_le_bytes());
        digest.update(file.text.as_bytes());
    }
    let name = super::entry_name(module, implementation.entry);
    digest.update((name.len() as u64).to_le_bytes());
    digest.update(name.as_bytes());
}

/// Replace parameter domains for a survey. Each keeps its default first.
fn widen(
    implementation: &mut NativeImplementation,
    domains: &BTreeMap<String, Vec<u64>>,
) -> Result<(), TuneError> {
    for (name, values) in domains {
        let parameter = implementation
            .params
            .iter_mut()
            .find(|parameter| parameter.name == *name)
            .ok_or_else(|| TuneError::Domain(format!("no parameter `{name}` is declared")))?;
        if values.first() != parameter.values.first() {
            return Err(TuneError::Domain(format!(
                "widened values of `{name}` must keep its default {:?} first",
                parameter.values.first()
            )));
        }
        parameter.values = values.clone();
    }
    Ok(())
}

/// The configuration records of a tuning run and the points they were
/// measured at.
struct Records {
    points: Vec<PointRecord>,
    records: Vec<ConfigurationRecord>,
    /// The choices served in turn ([`TuningResult::served`]).
    served: Vec<Configuration>,
    evidence: Evidence,
}

/// What a unit observed of its candidates against its default.
struct Evidence {
    identity: String,
    observations: Vec<NumericalObservation>,
}

impl Evidence {
    fn of(validator: &Validator) -> Self {
        Self {
            identity: validator.identity().to_owned(),
            observations: validator.observations(),
        }
    }
}

/// One entry's survey.
struct Tuned<'s> {
    formation: &'s Formation<'s, 's>,
    statics: &'s NativeSpecialization,
    default: &'s NativeSpecialization,
    ground: &'s Ground,
}

type Tuning = (Records, NativeSpecialization, TuningMethod, TuningTime);

impl Tuned<'_> {
    fn survey(&self, points: &[PreparedPoint<'_>], plan: SurveyPlan) -> Result<Tuning, TuneError> {
        let options = MeasureOptions {
            samples: plan.samples,
            min_sample_seconds: plan.min_sample_seconds,
        };
        let mut time = TuningTime::default();
        // A survey measures and times every configuration whatever its
        // verdict, and prices nothing.
        let mut measurer = Measurer::new(self.formation.implementation, false);
        // A survey resolves no comparison and names no error rate. Its
        // device is raised at the rate one of its samples stands for: what
        // the samples of a point can tell apart at all.
        let settings = SearchSettings {
            error_rate: 1. / plan.samples.max(2) as f64,
        };
        let mut admission = Observed::default();
        let validator = &self.ground.validator;
        let mut records = Vec::new();
        // One batch of the walk: its programs formed together, then each
        // measured. A batch is as many as form at once.
        let mut survey = |batch: Vec<NativeSpecialization>| {
            let began = Instant::now();
            let formed = self
                .formation
                .forming(batch.into_iter().map(Formable::Program).collect())
                .map(Forming::run);
            time.forming_seconds += began.elapsed().as_secs_f64();
            for call in formed.into_iter().flat_map(|formed| formed.calls) {
                let (Formable::Program(specialization), Product::Program(kernel)) =
                    (call.formable, call.product)
                else {
                    unreachable!("a survey forms whole programs");
                };
                let began = Instant::now();
                let (artifact, outcome) = match kernel {
                    Err(exclusion) => (None, Outcome::Excluded(exclusion)),
                    Ok(kernel) => (
                        Some(kernel.artifact().0.clone()),
                        match measurer.measure(
                            &kernel,
                            points,
                            specialization.params(),
                            &options,
                            &settings,
                            Some(&mut admission),
                            Deadline::default(),
                        ) {
                            Ok(measured) => Outcome::Measured {
                                points: measured,
                                confirmed: Vec::new(),
                            },
                            Err(WorkError::Failed(exclusion)) => Outcome::Excluded(exclusion),
                            Err(WorkError::Expired) => unreachable!("a survey has no deadline"),
                        },
                    ),
                };
                time.measuring_seconds += began.elapsed().as_secs_f64();
                records.push(ConfigurationRecord {
                    formation: None,
                    configuration: Configuration::of(&specialization),
                    observations: artifact
                        .as_deref()
                        .map(|artifact| validator.held(artifact))
                        .unwrap_or_default(),
                    artifact,
                    outcome,
                });
            }
        };
        // The admissible configurations in the declaration's order, the
        // defaults first, a batch at a time: the walk holds one batch, and
        // the survey the records it returns.
        let threads = formation_threads(usize::MAX);
        let mut batch = Vec::with_capacity(threads);
        self.formation
            .implementation
            .walk_admissible(self.statics, |specialization| {
                batch.push(specialization.clone());
                if batch.len() == threads {
                    survey(std::mem::take(&mut batch));
                }
                true
            })
            .map_err(|error| TuneError::Declaration(error.to_string()))?;
        survey(batch);
        let measured = |record: &ConfigurationRecord| match &record.outcome {
            Outcome::Measured { points, .. } => Some(points.clone()),
            Outcome::Excluded(_) => None,
        };
        let reference = records.iter().find_map(measured).ok_or_else(|| {
            TuneError::NoValidatedCandidate(
                records
                    .iter()
                    .filter_map(|record| match &record.outcome {
                        Outcome::Excluded(error) => Some(error.clone()),
                        _ => None,
                    })
                    .collect(),
            )
        })?;
        // The survey serves the fastest configuration its requirement
        // admits; the others' measurements and times are its record.
        let weighing = Weighing::of(points);
        let chosen = records
            .iter()
            .enumerate()
            .filter(|(_, record)| {
                record
                    .artifact
                    .as_deref()
                    .is_some_and(|artifact| validator.admits(artifact))
            })
            .filter_map(|(index, record)| {
                measured(record).map(|points| (index, weighing.cost(&points, &reference).total()))
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| records[index].configuration.specialization())
            .ok_or_else(|| {
                TuneError::Reference("no measured configuration is admitted".into())
            })?;
        time.validating_seconds += points.iter().map(|p| p.validation_seconds()).sum::<f64>();
        let mut served = vec![Configuration::of(self.default)];
        if Configuration::of(&chosen) != served[0] {
            served.push(Configuration::of(&chosen));
        }
        Ok((
            Records {
                points: labels(points),
                records,
                served,
                evidence: Evidence::of(validator),
            },
            chosen,
            TuningMethod::Survey {
                samples: plan.samples,
            },
            time,
        ))
    }
}

/// The declared invocation demand never changes when discovery is incomplete.
fn point_records(points: &[PointSpec]) -> Vec<PointRecord> {
    points
        .iter()
        .map(|point| PointRecord {
            label: point.label.clone(),
            weight: point.weight,
            class: point.class.clone(),
        })
        .collect()
}

/// A plan that failed: its declaration cannot be planned.
fn planning(error: plan::PlanError) -> TuneError {
    TuneError::Declaration(format!("native launch plan: {error:?}"))
}

/// Entry parameters read only by launches that no admissible configuration
/// activates at any of `shapes`: nothing measures them and no validation
/// runs their code, so a search keeps their defaults. A parameter no launch
/// reads affects every launch and is never one of them. With them, the
/// launches each of `shapes` can run.
fn unserved_parameters(
    implementation: &NativeImplementation,
    statics: &NativeSpecialization,
    shapes: &[PointShape],
) -> Result<(Vec<String>, Vec<plan::PlannedPoint>), TuneError> {
    let points = plan::partition(implementation, statics, shapes)
        .map_err(planning)?
        .points;
    let active = points
        .iter()
        .flat_map(|point| point.active.iter().copied())
        .collect::<std::collections::BTreeSet<_>>();
    let influence = Influence::of(implementation);
    let unserved = implementation
        .params
        .iter()
        .map(|parameter| &parameter.name)
        .filter(|name| {
            !influence.everywhere.contains(name)
                && influence
                    .launches
                    .iter()
                    .enumerate()
                    .all(|(launch, reads)| !reads.contains(name) || !active.contains(&launch))
        })
        .cloned()
        .collect();
    Ok((unserved, points))
}

fn labels(points: &[PreparedPoint<'_>]) -> Vec<PointRecord> {
    points
        .iter()
        .map(|point| PointRecord {
            label: point.label.clone(),
            weight: point.weight,
            class: point.class.clone(),
        })
        .collect()
}

/// Ordinal and name of every `&mut` tensor parameter.
fn mutable_parameters(logical: &LogicalEntry) -> Vec<(usize, String)> {
    logical
        .schema()
        .parameters()
        .iter()
        .enumerate()
        .filter(|(_, parameter)| {
            matches!(
                parameter.kind,
                ParameterKind::Tensor {
                    access: TensorAccess::Mutable | TensorAccess::Owned,
                    ..
                }
            )
        })
        .map(|(ordinal, parameter)| (ordinal, parameter.name.clone()))
        .collect()
}

pub(super) fn prepare_message(error: PrepareError) -> String {
    match error {
        PrepareError::Source(error) => error.to_string(),
        PrepareError::Preparation(error) => error.to_string(),
    }
}

fn point_measurement(label: &str, key: PointKey, timing: &PointTiming<'_>) -> PointMeasurement {
    let measurement = timing.measurement();
    PointMeasurement {
        point: label.to_owned(),
        key,
        median_seconds: measurement.median,
        deviation_seconds: measurement.deviation,
        samples: measurement.samples,
        repetitions: measurement.repetitions,
        rotation_bytes: measurement.rotation_bytes,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn workload_rotation_requires_distinct_present_witnesses() {
        assert_eq!(WorkloadRotation::All.indices(4).unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(
            WorkloadRotation::Selected(vec![2, 0]).indices(4).unwrap(),
            vec![2, 0]
        );
        assert!(WorkloadRotation::Selected(vec![]).indices(4).is_err());
        assert!(WorkloadRotation::Selected(vec![1, 1]).indices(4).is_err());
        assert!(WorkloadRotation::Selected(vec![4]).indices(4).is_err());
        assert!(WorkloadRotation::All.indices(0).is_err());
    }

    use super::*;

    fn priced(seconds: f64, unpriced: usize) -> Quote {
        Quote {
            priced: Duration::from_secs_f64(seconds),
            unpriced,
        }
    }

    /// Work starts only when its whole price fits what remains; an
    /// operation of a class never observed is late once what remains after
    /// the priced operations is less than the longest operation the load
    /// has completed, for each such operation.
    #[test]
    fn work_starts_only_when_its_whole_price_fits_and_unpriced_work_is_never_late() {
        let second = Duration::from_secs(1);
        assert!(priced(1., 0).fits(second, second * 30));
        assert!(!priced(1.5, 0).fits(second, Duration::ZERO));
        // One unpriced operation: the longest completed operation must fit.
        assert!(priced(1., 1).fits(second * 3, second * 2));
        assert!(!priced(1., 1).fits(second * 3, second * 3));
        assert!(!priced(0., 2).fits(second * 3, second * 2));
        // Before anything was observed the whole budget remains.
        assert!(priced(0., 3).fits(second, Duration::ZERO));
        // A class is priced by the mean of its own observations.
        let mut class = Observed::default();
        assert_eq!(class.price(), None);
        class.observe(1.);
        class.observe(3.);
        assert_eq!((class.price(), class.longest()), (Some(2.), 3.));
        let mut quote = Quote::default();
        quote.add(class.price(), 2);
        quote.add(None, 1);
        assert_eq!(quote, priced(4., 1));
        assert_eq!(quote.forecast(), None);
        assert_eq!(priced(4., 0).forecast(), Some(second * 4));
    }

    /// The rounds of a comparison of which `ahead` were ahead and `behind`
    /// behind.
    fn tally(ahead: u32, behind: u32) -> Tally {
        let mut tally = Tally::default();
        for _ in 0..ahead {
            tally.observe(-1.);
        }
        for _ in 0..behind {
            tally.observe(1.);
        }
        tally
    }

    /// There is one notion of a resolved lead: a challenger's matched
    /// rounds beside the choice served now say it leads. However fast it
    /// read, a challenger whose rounds have resolved nothing is a nominal
    /// lead, owed rounds and not a promotion; one whose rounds resolved a
    /// trail is dropped however fast it read; and rounds beside a choice
    /// since replaced resolve nothing about the one that replaced it.
    #[test]
    fn a_round_groups_each_point_with_the_served_choice_there() {
        use timing::Rounds;
        let settings = SearchSettings { error_rate: 0.05 };
        let pairing = |terms: Vec<(usize, f64)>| Pairing {
            incumbent: terms.iter().map(|(_, weight)| weight).sum(),
            terms,
            reused: 0.,
            complete: true,
            evidence: Tally::default(),
            admitted: 7,
            taken: 0,
            ahead: true,
        };
        // Slots 0 and 1: two challengers at one point, beside slot 3.
        // Slot 2: the first challenger at another point, beside slot 4.
        let mut rounds = Matched {
            settings: &settings,
            partner: vec![Some(3), Some(3), Some(4), None, None],
            ratios: vec![Vec::new(); 5],
            pairings: vec![pairing(vec![(0, 1.), (2, 1.)]), pairing(vec![(1, 1.)])],
        };
        // The served choice is sampled once at a point, in one group with
        // every challenger sampled beside it there.
        assert_eq!(rounds.next(), vec![vec![3, 0, 1], vec![4, 2]]);
        // The first challenger reads 10% faster at both points, the second
        // 10% slower: the second's part ends with this round, and only the
        // first could be served.
        rounds.sampled(&[(3, 1.), (0, 0.9), (1, 1.1), (4, 2.), (2, 1.8)]);
        assert_eq!(rounds.ratio(0).0, 0.9);
        assert_eq!(rounds.ratio(2).0, 0.9);
        assert!(rounds.pairings[0].ahead && !rounds.pairings[1].ahead);
        assert_eq!(rounds.next(), vec![vec![3, 0], vec![4, 2]]);
    }

    #[test]
    fn a_lead_is_resolved_by_matched_rounds_beside_the_served_choice() {
        let mut recorded = recorded(1);
        let at = |challenger: usize, seconds: f64| timed(1, |_| challenger as u64, |_| seconds);
        // Read 40% faster in its one round.
        let read_fast = recorded.compared(1, at(1, 0.6), tally(1, 0));
        assert_eq!(recorded.ledger.leader(), None);
        assert_eq!(
            recorded.ledger.nominal().map(|(nominal, _)| nominal.clone()),
            Some(read_fast.clone())
        );
        // Six rounds ahead resolve nothing at one in twenty; seven do.
        let six = recorded.compared(2, at(2, 0.7), tally(6, 0));
        assert_eq!(recorded.ledger.leader(), None);
        let seven = recorded.compared(3, at(3, 0.7), tally(7, 0));
        assert_eq!(recorded.ledger.leader(), Some(&seven));
        // Read twice as fast, and its rounds resolved a trail.
        let trailing = recorded.compared(4, at(4, 0.5), tally(0, 7));
        assert_eq!(
            recorded.ledger.contenders[&trailing].standing,
            ContenderStanding::Trailing
        );
        assert_eq!(
            recorded.ledger.nominal().map(|(nominal, _)| nominal.clone()),
            Some(read_fast.clone())
        );
        // The resolved leader is promoted. The rounds of the others were
        // beside the defaults: none of them leads the new choice by them.
        let stale = recorded.compared(5, at(5, 0.5), tally(7, 0));
        assert_eq!(recorded.ledger.leader(), Some(&stale));
        let ledger = &mut recorded.ledger;
        ledger.contenders.get_mut(&seven).unwrap().standing = ContenderStanding::Served;
        ledger.served.push(seven.clone());
        ledger.incumbent = seven;
        ledger.rerank();
        assert_eq!(ledger.leader(), None);
        // Their measured costs still rank them: the cheapest is the
        // nominal lead, and the one that trailed stays dropped.
        assert_eq!(ledger.nominal().map(|(nominal, _)| nominal.clone()), Some(stale));
        assert_eq!(ledger.contenders[&six].standing, ContenderStanding::Measured);
        assert_eq!(ledger.contenders[&trailing].standing, ContenderStanding::Trailing);
    }

    /// The leader owed its conversion is the cheapest measured challenger
    /// whose cost is below the served choice's, by however little. One
    /// that fails its conversion gives way to the next; the defaults are
    /// the reference and are never a challenger; a challenger that costs
    /// as much as the served choice, or more, leads nothing.
    #[test]
    fn only_the_winner_is_covered_and_a_failure_gives_way_to_the_next() {
        let cost = |form: u64, total: f64| {
            Cost::new(vec![(
                PointKey {
                    launches: vec![0],
                    values: [("FORM".to_owned(), form)].into(),
                },
                total,
            )])
        };
        let served = cost(0, 1.);
        let measured = [
            (1usize, cost(1, 0.8)),
            (2, cost(2, 0.7)),
            (3, cost(3, 0.99)),
            (4, cost(4, 1.)),
            (5, cost(5, 1.01)),
        ];
        let leader = |standing: &[usize]| {
            leading(
                &served,
                measured
                    .iter()
                    .filter(|(index, _)| standing.contains(index))
                    .map(|(index, cost)| (index, cost)),
            )
            .copied()
        };
        // The cheapest leads: it alone is confirmed and covered.
        assert_eq!(leader(&[1, 2, 3, 4, 5]), Some(2));
        // It failed and no longer stands as measured: the next leads.
        assert_eq!(leader(&[1, 3, 4, 5]), Some(1));
        // One percent below the served choice leads: no margin is asked.
        assert_eq!(leader(&[3, 4, 5]), Some(3));
        // At the served choice's cost, or above it, nothing leads.
        assert_eq!(leader(&[4, 5]), None);
        assert_eq!(leader(&[]), None);
        // Once a leader serves, the rest must beat it, not the defaults.
        assert_eq!(leading(&cost(2, 0.7), [(&1usize, &cost(1, 0.8))]), None);
    }

    /// No margin is asked of a challenger: one that is faster than the
    /// served choice by less than 2% in every round is a nominal lead from
    /// its first round, leads once the sign test resolves, in the fewest
    /// rounds there are, and is then the leader owed its conversion, which
    /// promotes it.
    #[test]
    fn a_challenger_faster_by_less_than_two_percent_in_every_round_is_promoted() {
        use timing::Rounds;
        let settings = SearchSettings { error_rate: 0.05 };
        let fewest = Tally::default().fewest(settings.error_rate);
        assert_eq!(fewest, 7);
        // One challenger at one point, in slot 0, beside the served choice
        // in slot 1.
        let mut rounds = Matched {
            settings: &settings,
            partner: vec![Some(1), None],
            ratios: vec![Vec::new(); 2],
            pairings: vec![Pairing {
                terms: vec![(0, 1.)],
                reused: 0.,
                incumbent: 1.,
                complete: true,
                evidence: Tally::default(),
                admitted: fewest,
                taken: 0,
                ahead: true,
            }],
        };
        // Between 0.5% and 1.9% faster in each round, never 2%.
        let gains = [0.019, 0.005, 0.012, 0.008, 0.015, 0.01, 0.006];
        for (round, gain) in gains.iter().enumerate() {
            assert_eq!(rounds.next(), vec![vec![1, 0]], "admitted round {round}");
            assert_eq!(rounds.pairings[0].evidence.verdict(settings.error_rate), None);
            rounds.sampled(&[(1, 2e-3), (0, 2e-3 * (1. - gain))]);
        }
        let pairing = &rounds.pairings[0];
        assert!(pairing.ahead);
        assert_eq!(pairing.taken, fewest);
        let evidence = pairing.evidence;
        assert_eq!(evidence.verdict(settings.error_rate), Some(Verdict::Leads));
        // Resolved: its part of the session is over.
        assert!(rounds.next().is_empty());
        let ratio = rounds.ratio(0).0;
        assert!(ratio > 0.98 && ratio < 1., "{ratio}");

        // Its one-round reading is a nominal lead, owed rounds; the same
        // reading with its resolved rounds is the leader owed a conversion.
        let mut recorded = recorded(1);
        let at = |challenger: usize| timed(1, |_| challenger as u64, |_| ratio);
        let read = recorded.compared(1, at(1), tally(1, 0));
        assert_eq!(recorded.ledger.leader(), None);
        assert_eq!(
            recorded.ledger.nominal().map(|(nominal, _)| nominal.clone()),
            Some(read)
        );
        let resolved = recorded.compared(2, at(2), evidence);
        assert_eq!(recorded.ledger.leader(), Some(&resolved));
        // The conversion's promotion: it serves, and nothing measured
        // beside the defaults leads it.
        let ledger = &mut recorded.ledger;
        ledger.contenders.get_mut(&resolved).unwrap().standing = ContenderStanding::Served;
        ledger.served.push(resolved.clone());
        ledger.incumbent = resolved;
        ledger.rerank();
        assert_eq!(ledger.leader(), None);
        assert!(ledger.nominal().is_none());
    }

    /// The rule the ledger's ranking keeps: the cheapest of the `measured`
    /// configurations whose cost is below `incumbent`'s is the leader owed
    /// its conversion. A leader that fails gives way to the next; one that
    /// is promoted becomes the incumbent the rest must beat. Computed over
    /// all of them: what the ledger's ranking is checked against.
    fn leading<'c, 'm, C>(
        incumbent: &Cost,
        measured: impl IntoIterator<Item = (&'c C, &'m Cost)>,
    ) -> Option<&'c C> {
        measured
            .into_iter()
            .filter(|(_, cost)| cost.improves_on(incumbent))
            .min_by(|left, right| left.1.total().total_cmp(&right.1.total()))
            .map(|(configuration, _)| configuration)
    }

    use seismic_lang::checked::{SourceFile, SourceSet, check_source};
    use seismic_lang::registry::BackendName;

    /// The Metal implementation of a one-entry module whose native
    /// declaration body is `body`.
    fn declared(body: &str) -> NativeImplementation {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "scale.seismic".into(),
            text: format!(
                "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n{body}"
            ),
        }]))
        .unwrap();
        module
            .native_implementation(module.entries()[0].id, BackendName::Metal)
            .unwrap()
            .clone()
    }

    fn shapes(rows: &[u64]) -> Vec<PointShape> {
        rows.iter()
            .map(|rows| PointShape {
                label: format!("m{rows}"),
                dimensions: BTreeMap::from([("N".into(), *rows)]),
            })
            .collect()
    }

    /// The declared domain of `implementation` keyed as launches formed
    /// separately are, with the boundary of `partition`.
    fn launches(partition: &plan::TuningPartition) -> Keying {
        Keying::Launches(partition.boundary.clone())
    }

    #[test]
    fn coverage_reaches_later_groups_before_refining_the_first_and_skips_what_cannot_matter() {
        let bands = (1..=3)
            .map(|rows| {
                format!(
                    "    launch band{rows} when N == {rows}:\n        params (WIDTH in [8, 4, 16], ROWS in [1, 2])\n        threadgroups (ceil_div(N, WIDTH * ROWS), 1, 1)\n        threads_per_threadgroup (WIDTH * 32, 1, 1)\n"
                )
            })
            .collect::<String>();
        let implementation = declared(&bands);
        let statics = NativeSpecialization::new();
        let default = implementation.default_specialization(&statics).unwrap();
        let partition = plan::partition(&implementation, &statics, &shapes(&[1, 2, 3])).unwrap();
        let keying = launches(&partition);
        let domain = Declared {
            implementation: &implementation,
            keying: &keying,
        };
        let order = coverage_order(&domain, &partition.groups, &default).unwrap();
        // Which launch a start changes, and to what.
        let changed = |start: &NativeSpecialization| {
            (0..3)
                .find(|launch| {
                    ["ROWS", "WIDTH"].iter().any(|name| {
                        start.launch_param(*launch, name) != default.launch_param(*launch, name)
                    })
                })
                .unwrap()
        };
        // Each group's first nearby change before any group's second; ROWS
        // has no lower value, WIDTH one on each side.
        assert_eq!(
            order.iter().map(changed).collect::<Vec<_>>(),
            [0, 1, 2, 0, 1, 2, 0, 1, 2]
        );
        assert_eq!(order[0].launch_param(0, "ROWS"), Some(2));
        assert_eq!(order[4].launch_param(1, "WIDTH"), Some(4));
        assert_eq!(order[7].launch_param(1, "WIDTH"), Some(16));
        // Asked again, the order is the same: a continuation that skips
        // what was measured preserves the uninterrupted order.
        assert_eq!(
            coverage_order(&domain, &partition.groups, &default).unwrap(),
            order
        );
        // Each band works at its own point, and the third point has no
        // serving demand: no change of the third band can change the
        // service cost, and none is worth forming. The one rule says so of
        // each candidate by the key it is forecast to run under.
        let reference = (0..3)
            .map(|point| PointMeasurement {
                point: format!("m{}", point + 1),
                key: keying.key(&implementation, vec![point], &default),
                ..measured(1., 0.)
            })
            .collect::<Vec<_>>();
        let mut search = GroupSearch::new(
            partition,
            default.clone(),
            reference.clone(),
            vec![1., 1., 0.],
            Vec::new(),
        );
        assert_eq!(
            order
                .iter()
                .filter(|start| !search.inert(&domain, start, &default, &reference))
                .map(changed)
                .collect::<Vec<_>>(),
            [0, 1, 0, 1, 0, 1]
        );
        let Some(Piece::Whole(first)) = search.plan(&domain).unwrap() else {
            panic!("the first piece is the coverage");
        };
        assert_eq!(first.iter().map(changed).collect::<Vec<_>>(), [0, 1, 0, 1, 0, 1]);
    }

    /// The rule for what is not worth forming is one rule under either
    /// keying: a program formed whole whose parameter only a launch that
    /// does no work where there is demand reads is skipped exactly as a
    /// separately formed launch's parameter is.
    #[test]
    fn a_change_that_cannot_change_served_work_is_not_formed_under_either_keying() {
        let implementation = declared(
            "    params (NEAR in [1, 2], FAR in [1, 2])\n    launch small when N < 8:\n        threadgroups (ceil_div(N, NEAR), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch large when N >= 8:\n        threadgroups (ceil_div(N, FAR), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
        );
        assert!(!implementation.launch_scoped());
        let statics = NativeSpecialization::new();
        let default = implementation.default_specialization(&statics).unwrap();
        let shapes = shapes(&[4, 16]);
        let (unserved, points) = unserved_parameters(&implementation, &statics, &shapes).unwrap();
        assert!(unserved.is_empty());
        let keying = Keying::Whole(Influence::of(&implementation));
        let domain = Declared {
            implementation: &implementation,
            keying: &keying,
        };
        let parameters = implementation
            .params
            .iter()
            .map(|parameter| (parameter.name.clone(), parameter.values.clone(), parameter.form))
            .collect::<Vec<_>>();
        let reference = [0usize, 1]
            .map(|launch| PointMeasurement {
                point: format!("m{launch}"),
                key: keying.key(&implementation, vec![launch], &default),
                ..measured(1., 0.)
            })
            .to_vec();
        // Demand at the small rows alone.
        let mut search = GroupSearch::new(
            plan::whole(&parameters, 2, points),
            default.clone(),
            reference.clone(),
            vec![1., 0.],
            Vec::new(),
        );
        let near = default.clone().with_param("NEAR", 2);
        let far = default.clone().with_param("FAR", 2);
        assert!(!search.inert(&domain, &near, &default, &reference));
        assert!(search.inert(&domain, &far, &default, &reference));
        let mut asked = Vec::new();
        // Up to where the leaders' neighbourhoods are measured: what is
        // asked after that is the rest of the domain.
        while let Some(piece) = search.plan(&domain).unwrap().filter(|_| !search.resting()) {
            for challenger in piece.challengers() {
                asked.push((challenger.param("NEAR"), challenger.param("FAR")));
                let key = keying.key(&implementation, vec![0], challenger);
                search.record(
                    challenger,
                    Some(vec![
                        PointMeasurement {
                            key,
                            ..reference[0].clone()
                        },
                        reference[1].clone(),
                    ]),
                );
            }
        }
        assert_eq!(asked, [(Some(2), Some(1))]);
    }

    #[test]
    fn coverage_respects_coupled_admissibility_and_deduplicates() {
        // Only diagonal assignments are admissible: neither coordinate can
        // change alone. A coupled start covers both coordinates once.
        let implementation = declared(
            "    params (LEFT in [2, 1, 4], RIGHT in [2, 1, 4])\n    where LEFT == RIGHT\n    launch gemv:\n        threadgroups (ceil_div(N, LEFT * RIGHT), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
        );
        let statics = NativeSpecialization::new();
        let default = implementation.default_specialization(&statics).unwrap();
        let partition = plan::partition(&implementation, &statics, &shapes(&[4])).unwrap();
        let keying = launches(&partition);
        let domain = Declared {
            implementation: &implementation,
            keying: &keying,
        };
        let starts = coordinate_starts(&domain, &partition.groups[0], &default).unwrap();
        assert_eq!(
            starts
                .iter()
                .map(|start| (start.param("LEFT"), start.param("RIGHT")))
                .collect::<Vec<_>>(),
            [(Some(1), Some(1)), (Some(4), Some(4))]
        );
    }

    /// What the leaders' neighbourhoods do not reach is every other
    /// declared assignment, each once, those that change fewer coordinates
    /// first, produced one at a time.
    #[test]
    fn the_rest_of_a_domain_is_every_other_assignment_once_nearest_first() {
        let values = [vec![1, 2, 3], vec![7], vec![4, 5]];
        assert_eq!(
            Rest::new(&values, vec![1, 7, 4]).collect::<Vec<_>>(),
            [
                // One coordinate changed, then two; the second coordinate
                // has no other value.
                vec![2, 7, 4],
                vec![3, 7, 4],
                vec![1, 7, 5],
                vec![2, 7, 5],
                vec![3, 7, 5],
            ]
        );
        assert_eq!(Rest::new(&[vec![1]], vec![1]).count(), 0);
        assert_eq!(Rest::new(&[], Vec::new()).count(), 0);
        // A start that is not the first declared value, and every
        // assignment of a larger product exactly once.
        let values = vec![vec![1, 2, 3, 4]; 5];
        let rest = Rest::new(&values, vec![3; 5]).collect::<BTreeSet<_>>();
        assert_eq!(rest.len(), 4usize.pow(5) - 1);
        assert!(!rest.contains(&vec![3; 5]));
    }

    /// A search ends only when its domain is exhausted. The fastest
    /// configuration here differs from the defaults in two parameters, each
    /// of which changes nothing alone: no leader's neighbourhood holds it,
    /// and the search measures on through the rest of the group until it
    /// is found and everything declared is measured.
    #[test]
    fn a_search_ends_when_its_domain_is_exhausted_and_finds_what_no_neighbourhood_holds() {
        let implementation = declared(
            "    launch gemv:\n        params (A in [1, 2, 3], B in [1, 2, 3])\n        threadgroups (ceil_div(N, A * B), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
        );
        let statics = NativeSpecialization::new();
        let seed = implementation.default_specialization(&statics).unwrap();
        let partition = plan::partition(&implementation, &statics, &shapes(&[4])).unwrap();
        let keying = launches(&partition);
        let group = partition.groups[0].clone();
        let measurement = |specialization: &NativeSpecialization| PointMeasurement {
            key: keying.key(&implementation, vec![0], specialization),
            ..measured(
                if group.assignment(specialization) == [3, 3] {
                    0.5
                } else {
                    1.
                },
                0.,
            )
        };
        let domain = Declared {
            implementation: &implementation,
            keying: &keying,
        };
        let mut search = GroupSearch::new(
            partition,
            seed.clone(),
            vec![measurement(&seed)],
            vec![1.],
            Vec::new(),
        );
        assert_eq!(search.declared(), 9);
        let (mut neighbourhood, mut found) = (0, None);
        while let Some(piece) = search.plan(&domain).unwrap() {
            assert!(!search.exhausted());
            for challenger in piece.challengers() {
                if !search.resting() {
                    neighbourhood += 1;
                }
                if group.assignment(challenger) == [3, 3] {
                    found = Some(search.resting());
                }
                search.record(challenger, Some(vec![measurement(challenger)]));
            }
        }
        // The four single changes are the seed's neighbourhood; the pair
        // was reached past it.
        assert_eq!(neighbourhood, 4);
        assert_eq!(found, Some(true));
        assert!(search.exhausted());
        // Everything declared but the seed was measured.
        assert_eq!(search.swept.len() as u128, search.declared() - 1);
        let (boundary, _) = search.sweeps.keys().next().unwrap().clone();
        assert_eq!(search.sweeps[&(boundary, 0)].best, (vec![3, 3], 0.5));
    }

    /// A search whose leading boundary moves off the default goes on to its
    /// groups' sweeps and ends only when its domain is exhausted. The row
    /// band's edge is cheaper one value from the default, so that boundary
    /// leads once the bases are measured; the fastest mapping of the small
    /// launch (`T == 2`) is one no coverage of either leader holds. What
    /// was derived under the default base while it still led owes nothing:
    /// planning names a piece every time until nothing is left, and names
    /// nothing exactly when the search is exhausted.
    #[test]
    fn a_boundary_that_moves_off_the_default_still_sweeps_its_groups_to_exhaustion() {
        let implementation = declared(
            "    params (FROM in [3, 2, 5])\n    launch small when N < FROM:\n        params (A in [1, 2, 3], T in [0, 1, 2])\n        threadgroups (ceil_div(N, A + T), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch large when N >= FROM:\n        params (B in [1, 2, 3])\n        threadgroups (ceil_div(N, B), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
        );
        let statics = NativeSpecialization::new();
        let seed = implementation.default_specialization(&statics).unwrap();
        let rows = [1u64, 2, 4];
        let partition = plan::partition(&implementation, &statics, &shapes(&rows)).unwrap();
        assert_eq!(partition.boundary.len(), 1, "FROM only moves the row band's edge");
        assert_eq!(partition.groups.len(), 2);
        let keying = launches(&partition);
        let measurements = |specialization: &NativeSpecialization| {
            let from = specialization.param("FROM").unwrap();
            rows.iter()
                .map(|rows| {
                    let small = *rows < from;
                    let seconds = match (small, *rows) {
                        // The small launch's one faster mapping.
                        (true, 1 | 2) if specialization.launch_param(0, "T") == Some(2) => 0.9,
                        (true, 1 | 2) => 1.,
                        // Four rows are the large launch's.
                        (true, _) => 3.,
                        // Two rows are cheaper on the large launch.
                        (false, 2) => 0.5,
                        (false, _) => 1.,
                    };
                    PointMeasurement {
                        point: format!("m{rows}"),
                        key: keying.key(&implementation, vec![usize::from(!small)], specialization),
                        ..measured(seconds, 0.)
                    }
                })
                .collect::<Vec<_>>()
        };
        let domain = Declared {
            implementation: &implementation,
            keying: &keying,
        };
        let mut search = GroupSearch::new(
            partition,
            seed.clone(),
            measurements(&seed),
            vec![1.; rows.len()],
            Vec::new(),
        );
        let mut pieces = 0;
        while let Some(piece) = search.plan(&domain).unwrap() {
            assert!(!search.exhausted(), "a search that names a piece has work left");
            pieces += 1;
            assert!(pieces < 1000, "the search names pieces without end");
            for challenger in piece.challengers() {
                search.record(challenger, Some(measurements(challenger)));
            }
        }
        assert!(search.exhausted(), "planning names nothing only at the search's end");
        // The sweeps ran: the domain was asked past the bases and coverage.
        assert!(search.requests > 0);
        assert!(
            search
                .sweeps
                .values()
                .all(|sweep| sweep.exhausted && sweep.pending.is_empty())
        );
        // The boundary that leads is not the default, and under it every
        // mapping of the small launch was measured, the fastest among them.
        let moved = seed.clone().with_param("FROM", 2);
        let leading = search.partition.boundary_of(&moved);
        assert_ne!(leading, search.default_boundary);
        let small = search.partition.groups[0].clone();
        let sweep = &search.sweeps[&(leading, 0)];
        assert_eq!(sweep.best.0, [1, 2]);
        assert!(sweep.best.1 < 2.5 / 3.);
        assert_eq!(sweep.asked.len() as u128, small.declared());
        // Nothing generated is left unmeasured.
        assert!(
            search
                .generated
                .iter()
                .all(|configuration| search.swept.contains_key(configuration))
        );
    }

    #[test]
    fn unobserved_group_keeps_seed_on_equal_cost() {
        let implementation = declared(
            "    launch gemv:\n        params (P in [2, 0, 1, 3])\n        threadgroups (ceil_div(N, P + 1), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
        );
        let statics = NativeSpecialization::new();
        let partition = plan::partition(&implementation, &statics, &shapes(&[4])).unwrap();
        let group = &partition.groups[0];
        let mut sweep = Sweep::new(group, vec![2], 1.0);
        sweep.rank(group, vec![0], 1.0, 1);
        sweep.rank(group, vec![1], 1.0, 2);
        assert_eq!(sweep.best.0, [2]);
        sweep.rank(group, vec![3], 0.9, 3);
        assert_eq!(sweep.best.0, [3], "a measured improvement still wins");
        // Among equals a form's leader is the first asked for.
        assert_eq!(sweep.leaders[&group.form(&[3])].0, [3]);
        sweep.rank(group, vec![1], 0.9, 4);
        assert_eq!(sweep.leaders[&group.form(&[3])].0, [3]);
        assert_eq!(sweep.best.0, [1], "the lower assignment of equals other than the seed");
    }

    /// What a search of [`searched`] did.
    struct Searched {
        search: GroupSearch,
        /// The cheapest configuration measured, by its parameters' values
        /// in order.
        reached: Vec<u64>,
        /// How many configurations were measured before it.
        reached_after: usize,
        /// Whether each piece measured other forms' representatives.
        structural: Vec<bool>,
    }

    /// A search over `parameters` two-valued parameters, every piece
    /// measured at once at a cost that falls with each parameter of
    /// `better` a configuration has at its second value. The parameters
    /// are one launch's own, or (`whole`) those of a program formed whole,
    /// the first `forms` of them selecting forms: the same domain planned
    /// as launches formed separately and as the one group of a whole
    /// program.
    fn searched(parameters: usize, better: &[usize], whole: Option<usize>) -> Searched {
        let names = (0..parameters)
            .map(|parameter| format!("P{parameter}"))
            .collect::<Vec<_>>();
        let domains = |forms: usize| {
            names
                .iter()
                .enumerate()
                .map(|(parameter, name)| {
                    format!(
                        "{}{name} in [1, 2]",
                        if parameter < forms { "form " } else { "" }
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        let product = names.join(" * ");
        let implementation = declared(&match whole {
            Some(forms) => format!(
                "    params ({})\n    launch gemv:\n        threadgroups (ceil_div(N, {product}), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
                domains(forms),
            ),
            None => format!(
                "    launch gemv:\n        params ({})\n        threadgroups (ceil_div(N, {product}), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
                domains(0),
            ),
        });
        assert_eq!(implementation.launch_scoped(), whole.is_none());
        let statics = NativeSpecialization::new();
        let seed = implementation.default_specialization(&statics).unwrap();
        let (partition, keying) = match whole {
            Some(_) => {
                let (unserved, points) =
                    unserved_parameters(&implementation, &statics, &shapes(&[4])).unwrap();
                assert!(unserved.is_empty());
                let entries = implementation
                    .params
                    .iter()
                    .map(|parameter| {
                        (parameter.name.clone(), parameter.values.clone(), parameter.form)
                    })
                    .collect::<Vec<_>>();
                (
                    plan::whole(&entries, 1, points),
                    Keying::Whole(Influence::of(&implementation)),
                )
            }
            None => {
                let partition =
                    plan::partition(&implementation, &statics, &shapes(&[4])).unwrap();
                let keying = launches(&partition);
                (partition, keying)
            }
        };
        let addresses = partition.groups[0].parameters.clone();
        let values = |specialization: &NativeSpecialization| {
            let mut values = addresses
                .iter()
                .map(|address| match address {
                    plan::ParameterAddress::Entry(name)
                    | plan::ParameterAddress::Launch { name, .. } => {
                        (name[1..].parse::<usize>().unwrap(), address.value(specialization))
                    }
                })
                .collect::<Vec<_>>();
            values.sort_unstable();
            values.into_iter().map(|(_, value)| value).collect::<Vec<_>>()
        };
        let measurement = |specialization: &NativeSpecialization| {
            let values = values(specialization);
            let off = (0..parameters)
                .filter(|parameter| (values[*parameter] == 2) != better.contains(parameter))
                .count();
            PointMeasurement {
                key: keying.key(&implementation, vec![0], specialization),
                ..measured(1. + off as f64, 0.)
            }
        };
        let reference = vec![measurement(&seed)];
        let domain = Declared {
            implementation: &implementation,
            keying: &keying,
        };
        let mut search =
            GroupSearch::new(partition, seed.clone(), reference.clone(), vec![1.], Vec::new());
        let mut best = (values(&seed), reference[0].median_seconds, 0);
        let mut structural = Vec::new();
        let mut count = 0;
        // Up to where the leaders' neighbourhoods are measured: what is
        // asked after that is the rest of the domain.
        while let Some(piece) = search.plan(&domain).unwrap().filter(|_| !search.resting()) {
            structural.push(search.structural);
            for challenger in piece.challengers() {
                count += 1;
                let measured = measurement(challenger);
                if measured.median_seconds < best.1 {
                    best = (values(challenger), measured.median_seconds, count);
                }
                search.record(challenger, Some(vec![measured]));
            }
        }
        Searched {
            search,
            reached: best.0,
            reached_after: best.2,
            structural,
        }
    }

    /// On the engine's dense output projection as it is declared, the
    /// search's first piece is the other forms' representatives, the
    /// token-packing form among them: it is reached within the first
    /// handful of configurations measured, before any row band's mapping
    /// is refined, and with a cost model in which only packing is faster
    /// it is what the sweeps rank first.
    #[test]
    fn the_dense_output_search_reaches_the_packing_form_first() {
        // The engine's kernel sources as its build loads them.
        let kernels = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../engine/kernels/kernels");
        let module = seismic_lang::source::load(&[kernels], seismic_std::sources())
            .unwrap_or_else(|_| panic!("the engine's kernel sources load"))
            .module;
        let entry = module
            .entries()
            .iter()
            .find(|entry| entry.name == "dense_output")
            .expect("the dense rows declare dense_output")
            .id;
        let implementation = module
            .native_implementation(entry, BackendName::Metal)
            .unwrap()
            .clone();
        let statics = NativeSpecialization::new()
            .with_static("H", 1024)
            .with_static("F", 4096)
            .with_static("DS", 0);
        let seed = implementation.default_specialization(&statics).unwrap();
        let rows = [1u64, 8, 128];
        let shapes = rows
            .iter()
            .map(|rows| PointShape {
                label: format!("m{rows}"),
                dimensions: BTreeMap::from([("M".into(), *rows), ("O".into(), *rows)]),
            })
            .collect::<Vec<_>>();
        let partition = Rc::new(plan::partition(&implementation, &statics, &shapes).unwrap());
        // The launches a configuration runs at a point, and a time there
        // that only the packing form improves, at prefill rows.
        let measurement = |specialization: &NativeSpecialization| {
            shapes
                .iter()
                .map(|shape| {
                    let working = (0..implementation.launches.len())
                        .filter(|&ordinal| {
                            implementation.launches[ordinal]
                                .when
                                .as_ref()
                                .is_none_or(|when| {
                                    when.holds(
                                        &|name| {
                                            shape
                                                .dimensions
                                                .get(name)
                                                .copied()
                                                .or_else(|| statics.static_value(name))
                                        },
                                        &|name| {
                                            specialization
                                                .launch_param(ordinal, name)
                                                .or_else(|| specialization.param(name))
                                        },
                                    )
                                    .unwrap()
                                })
                        })
                        .collect::<Vec<_>>();
                    let packed =
                        shape.dimensions["O"] > 64 && specialization.param("PACK") == Some(1);
                    PointMeasurement {
                        point: shape.label.clone(),
                        key: launch_key(
                            &implementation,
                            &partition.boundary,
                            working,
                            specialization,
                        ),
                        ..measured(if packed { 0.5 } else { 1. }, 0.)
                    }
                })
                .collect::<Vec<_>>()
        };
        let reference = measurement(&seed);
        let keying = launches(&partition);
        let domain = Declared {
            implementation: &implementation,
            keying: &keying,
        };
        let mut run = GroupSearch::new(
            (*partition).clone(),
            seed.clone(),
            reference.clone(),
            vec![1.; rows.len()],
            Vec::new(),
        );
        let Some(Piece::Whole(first)) = run.plan(&domain).unwrap() else {
            panic!("the first piece is whole configurations");
        };
        assert!(run.structural, "the first piece is the other forms'");
        assert!(first.len() <= 4, "{}", first.len());
        let packing = first
            .iter()
            .find(|form| form.param("PACK") == Some(1))
            .expect("the packing form is among the first measured");
        // Each differs from the defaults in its form, and in what that
        // form's restrictions couple to it alone.
        assert!(first.iter().all(|form| {
            ["TALL", "INT8", "PACK"]
                .iter()
                .filter(|name| form.param(name) != seed.param(name))
                .count()
                >= 1
                && form.param("BATCH_FROM") == seed.param("BATCH_FROM")
        }));
        for form in &first {
            run.record(form, Some(measurement(form)));
        }
        // What follows is refinement, and it starts from the packing form.
        let next = run.plan(&domain).unwrap();
        assert!(!run.structural);
        assert!(next.is_some());
        let weighing = Weighing {
            weights: run.weights.clone(),
        };
        let cheapest = run
            .swept
            .iter()
            .filter_map(|(configuration, measured)| {
                Some((configuration, weighing.cost(measured.as_ref()?, &reference).total()))
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap()
            .0;
        assert_eq!(*cheapest, Configuration::of(packing));
    }

    /// A group of 2^20 assignments is searched as a group of 2^10 is: what
    /// planning asks for and holds follows what is measured, and the search
    /// still reaches the configuration known to be better, seven parameters
    /// from the defaults.
    #[test]
    fn a_group_of_a_million_assignments_is_searched_by_what_is_measured() {
        let better = [0, 3, 6, 9];
        let expected = |parameters: usize| {
            (0..parameters)
                .map(|parameter| if better.contains(&parameter) { 2 } else { 1 })
                .collect::<Vec<u64>>()
        };
        // The same domain as one launch's own parameters and as a program
        // formed whole: one order searches both.
        for whole in [None, Some(0)] {
            let small = searched(10, &better, whole);
            let began = Instant::now();
            let large = searched(20, &better, whole);
            let elapsed = began.elapsed();
            assert_eq!(large.reached, expected(20), "{whole:?}");
            assert_eq!(small.reached, expected(10), "{whole:?}");
            let (small, large) = (small.search, large.search);
            // Everything the leaders' neighbourhoods asked for was measured,
            // and nothing else exists but the first of the rest, asked for
            // and awaiting measurement.
            let awaiting = |search: &GroupSearch| {
                search
                    .sweeps
                    .values()
                    .map(|sweep| sweep.pending.len())
                    .sum::<usize>()
            };
            assert_eq!(large.generated.len(), large.swept.len() + awaiting(&large));
            // Each step downhill asks one leader for its neighbours, one
            // per parameter: the search is linear in the parameters, where
            // the group's assignments are exponential in them.
            let steps = better.len() + 1;
            assert!(large.swept.len() <= steps * 20, "{}", large.swept.len());
            assert!(small.swept.len() <= steps * 10, "{}", small.swept.len());
            assert!(
                large.swept.len() <= 2 * small.swept.len() + 20,
                "{} against {}",
                large.swept.len(),
                small.swept.len()
            );
            // A search that produced the group's assignments would hold a
            // million of them; this one ends in the time of its
            // measurements.
            assert!(elapsed < Duration::from_secs(20), "{elapsed:?}");
        }
    }

    /// A program formed whole whose first six parameters select forms: the
    /// other forms one form parameter from the defaults are its first
    /// piece, six of them and not the sixty-three there are; a form two
    /// form parameters away is reached from the form between; and the
    /// configuration known to be better, two forms and two other
    /// parameters from the defaults, is reached while the forms within as
    /// many steps are being asked for, of a domain of 2^20.
    #[test]
    fn a_whole_program_measures_other_forms_first_and_reaches_further_forms_from_their_leaders() {
        let better = [0, 3, 6, 9];
        let forms = 6;
        let began = Instant::now();
        let searched = searched(20, &better, Some(forms));
        assert!(began.elapsed() < Duration::from_secs(60), "{:?}", began.elapsed());
        assert_eq!(
            searched.reached,
            (0..20)
                .map(|parameter| if better.contains(&parameter) { 2 } else { 1 })
                .collect::<Vec<u64>>()
        );
        // The first piece is the other forms' representatives, and no
        // later one is.
        assert!(searched.structural[0]);
        assert!(searched.structural[1..].iter().all(|structural| !structural));
        let search = &searched.search;
        let group = &search.partition.groups[0];
        let seed = group.assignment(&search.seed);
        assert_eq!(group.form_starts(&seed).count(), forms);
        // A leader is asked for the forms one form parameter from its own
        // and its neighbours, one per parameter; each of the better
        // configuration's four steps from the defaults is a round, and a
        // round's leaders are at most the forms there are.
        let rounds = better.len();
        assert!(
            searched.reached_after <= rounds * (1 << forms) * 20,
            "{}",
            searched.reached_after
        );
        // Nothing exists but what was measured and the first of the rest,
        // asked for and awaiting measurement.
        let awaiting = search
            .sweeps
            .values()
            .map(|sweep| sweep.pending.len())
            .sum::<usize>();
        assert_eq!(search.generated.len(), search.swept.len() + awaiting);
    }

    /// A unit's ledger over a canonical kernel that is formed and never
    /// run, with its validator: what a unit's record is made from.
    struct Recorded {
        ledger: Ledger,
        validator: Validator,
        unit: Unit,
        default: NativeSpecialization,
        canonical: Vec<PointMeasurement>,
    }

    /// A measurement at `points` points shaped as the engine's units
    /// record theirs: eight keyed parameter values and three samples a
    /// point. `median(point)` is its time there, and it runs the work
    /// keyed `key(point)`.
    fn timed(
        points: usize,
        key: impl Fn(usize) -> u64,
        median: impl Fn(usize) -> f64,
    ) -> Vec<PointMeasurement> {
        (0..points)
            .map(|point| PointMeasurement {
                point: format!("m{point}"),
                key: PointKey {
                    launches: vec![0, 1],
                    values: (0..8)
                        .map(|value| (format!("@0:PARAMETER_{value}"), key(point)))
                        .collect(),
                },
                median_seconds: median(point),
                deviation_seconds: median(point) / 100.,
                samples: vec![median(point); 3],
                repetitions: 16,
                rotation_bytes: 1 << 20,
            })
            .collect()
    }

    impl Recorded {
        /// Record challenger `challenger`, measured at `measured`, whose
        /// rounds beside the served choice resolved nothing.
        fn challenge(&mut self, challenger: usize, measured: Vec<PointMeasurement>) -> Configuration {
            self.compared(challenger, measured, Tally::default())
        }

        /// Record challenger `challenger`, measured at `measured` in rounds
        /// beside the served choice that showed `evidence`.
        fn compared(
            &mut self,
            challenger: usize,
            measured: Vec<PointMeasurement>,
            evidence: Tally,
        ) -> Configuration {
            let specialization = (0..8).fold(self.default.clone(), |specialization, value| {
                specialization.with_launch_param(
                    0,
                    format!("PARAMETER_{value}"),
                    (challenger + value) as u64,
                )
            });
            let cost = Weighing {
                weights: vec![1.; self.canonical.len()],
            }
            .cost(&measured, &self.canonical);
            let configuration = Configuration::of(&specialization);
            self.ledger.record(
                specialization,
                format!("cpu;project;;challenger {challenger};{challenger:064x};toolchain"),
                None,
                measured,
                cost,
                evidence,
            );
            configuration
        }

        /// The unit's record as it stands.
        fn result(&self) -> TuningResult {
            self.unit.result(
                Records {
                    points: Vec::new(),
                    records: self.ledger.records(&self.validator),
                    served: self.ledger.served.clone(),
                    evidence: Evidence::of(&self.validator),
                },
                &self.ledger.incumbent().specialization,
                TuningMethod::Census,
                TuningTime::default(),
            )
        }
    }

    /// The ledger of a unit timed at `points` points, its defaults measured
    /// at one second each.
    fn recorded(points: usize) -> Recorded {
        use crate::native::cpu::{CpuInvocation, CpuLaunchVariants, CpuVariant};
        use seismic_native_cpu::Tier;
        unsafe fn unused(_: &CpuInvocation<'_>, _: [u64; 3], _: &mut [u8]) {}
        const fn variant(tier: Tier) -> CpuVariant {
            CpuVariant {
                tier,
                elements: &[],
                function: unused,
            }
        }
        static KERNELS: CpuNativeKernels = CpuNativeKernels {
            digest: "ledger-test",
            elements: &[],
            launches: &[CpuLaunchVariants {
                kernel: "project",
                variants: &[
                    variant(Tier::X86V2),
                    variant(Tier::X86V3),
                    variant(Tier::X86V4),
                    variant(Tier::X86V4Vnni),
                    variant(Tier::Neon),
                ],
            }],
        };
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "ledger.seismic".into(),
            text: "fn project[N](x: &tensor[N] f32) -> tensor[N] f32:\n    let mut y = tensor[N] f32\n    for i in 0..N:\n        y[i] = x[i]\n    return y\n\nnative project for cpu from \"project.rs\":\n    static (N)\n    launch project:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n".into(),
        }]))
        .unwrap();
        let device = crate::devices::Catalog::discover()
            .unwrap()
            .open_backend(BackendName::Cpu)
            .unwrap();
        let entry = module.entry_named("project").unwrap();
        let default = NativeSpecialization::new().with_static("N", 8);
        let kernel = NativePrepared::prepare(
            &device,
            &module,
            entry,
            ElementBindings::new(),
            default.clone(),
            Some(&KERNELS),
        )
        .unwrap();
        let logical = Arc::new(module.entry(entry, &ElementBindings::new()).unwrap());
        let requirement = NumericalRequirement::BitExact;
        let validator = Validator::new(&device, &logical, &[], &requirement, &kernel);
        let canonical = timed(points, |_| 0, |_| 1.);
        let cost = Weighing {
            weights: vec![1.; points],
        }
        .cost(&canonical, &canonical);
        let settings = SearchSettings { error_rate: 0.05 };
        Recorded {
            ledger: Ledger::new(&default, kernel, canonical.clone(), cost, settings),
            validator,
            unit: Unit {
                tuning_identity: "device".into(),
                entry: "project".into(),
                backend: "cpu".into(),
                requirement,
                implementation_identity: "implementation".into(),
                parameters: Vec::new(),
            },
            default,
            canonical,
        }
    }

    /// The ledger ranks a challenger when it is measured and ranks them
    /// all again only after a conversion: at every step its leaders are
    /// those ranking every measured challenger afresh names.
    #[test]
    fn the_ledger_ranks_a_challenger_as_it_is_measured_and_agrees_with_ranking_them_all() {
        let mut recorded = recorded(3);
        // A deterministic noise in [-0.5, 0.5].
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut noise = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        };
        // What ranking every measured challenger names.
        let afresh = |ledger: &Ledger| {
            let incumbent = ledger.incumbent();
            let measured = ledger
                .contenders
                .iter()
                .filter(|(_, contender)| contender.standing == ContenderStanding::Measured)
                .collect::<Vec<_>>();
            let verdict = |contender: &Contender| {
                ledger
                    .evidence(contender)
                    .verdict(ledger.settings.error_rate)
            };
            (
                // Resolved by its rounds beside the served choice: the
                // cheapest.
                measured
                    .iter()
                    .filter(|(_, contender)| verdict(contender) == Some(Verdict::Leads))
                    .min_by(|left, right| left.1.cost.total().total_cmp(&right.1.cost.total()))
                    .map(|(configuration, _)| (*configuration).clone()),
                leading(
                    &incumbent.cost,
                    measured
                        .iter()
                        .filter(|(_, contender)| verdict(contender).is_none())
                        .map(|(configuration, contender)| (*configuration, &contender.cost)),
                )
                .cloned(),
                incumbent.artifact.clone(),
            )
        };
        let agrees = |ledger: &Ledger| {
            let (leader, nominal, _) = afresh(ledger);
            assert_eq!(ledger.leader().cloned(), leader);
            assert_eq!(ledger.nominal().map(|(nominal, _)| nominal.clone()), nominal);
        };
        let (mut conversions, mut promotions, mut leads) = (0, 0, 0);
        for challenger in 1..=400 {
            // Each challenger shares the served work at some points and
            // reads within a few percent of the defaults at the others.
            let shared = [noise() > 0., noise() > 0.3, false];
            let medians = [0, 1, 2].map(|_| 1. + noise() * 0.12);
            let measured = timed(
                3,
                |point| if shared[point] { 0 } else { challenger as u64 },
                |point| if shared[point] { 1. } else { medians[point] },
            );
            // Its rounds: one, which resolves nothing, or now and then
            // enough ahead, or behind, to resolve.
            let rounds = match challenger % 7 {
                3 => tally(7, 0),
                5 => tally(0, 7),
                _ => tally(1, 0),
            };
            recorded.compared(challenger, measured, rounds);
            agrees(&recorded.ledger);
            leads += usize::from(recorded.ledger.nominal().is_some());
            // Every tenth piece converts the nominal leader there is, as a
            // conversion does: it is given rounds beside the served
            // choice, which resolve its lead, and it is promoted, or
            // resolve a trail.
            if challenger % 10 != 0 {
                continue;
            }
            let ledger = &mut recorded.ledger;
            let Some(leader) = ledger.nominal().map(|(leader, _)| leader.clone()) else {
                continue;
            };
            conversions += 1;
            let promoted = conversions % 2 == 1;
            let served = ledger.incumbent.clone();
            let contender = ledger.contenders.get_mut(&leader).unwrap();
            contender.against = served;
            contender.evidence = if promoted { tally(8, 0) } else { tally(0, 8) };
            if promoted {
                promotions += 1;
                contender.standing = ContenderStanding::Served;
                ledger.served.push(leader.clone());
                let superseded = std::mem::replace(&mut ledger.incumbent, leader.clone());
                ledger.holding.insert(superseded);
            }
            ledger.rerank();
            ledger.release();
            agrees(ledger);
            if !promoted {
                assert_eq!(ledger.contenders[&leader].standing, ContenderStanding::Trailing);
            }
        }
        assert!(leads > 20, "the challengers do lead: {leads}");
        assert!(conversions >= 3 && promotions >= 2, "{conversions} {promotions}");
        // Changed settings rank them again.
        recorded.ledger.settings(&SearchSettings { error_rate: 0.01 });
        agrees(&recorded.ledger);
    }

    /// A launch of `parameters` parameters of `values` values each, and a
    /// search of it whose every measurement is cheaper the larger
    /// its parameter values: a search that keeps finding a better
    /// neighbour until every parameter is at its last value.
    fn climbing(
        parameters: usize,
        values: u64,
    ) -> (
        NativeImplementation,
        GroupSearch,
        impl Fn(&NativeImplementation, &NativeSpecialization) -> PointMeasurement,
    ) {
        let names = (0..parameters)
            .map(|parameter| format!("P{parameter}"))
            .collect::<Vec<_>>();
        let domain = (1..=values)
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let implementation = declared(&format!(
            "    launch gemv:\n        params ({})\n        threadgroups (ceil_div(N, {}), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n",
            names
                .iter()
                .map(|name| format!("{name} in [{domain}]"))
                .collect::<Vec<_>>()
                .join(", "),
            names.join(" * "),
        ));
        let measurement =
            move |implementation: &NativeImplementation, specialization: &NativeSpecialization| {
                let sum = names
                    .iter()
                    .map(|name| specialization.launch_param(0, name).unwrap())
                    .sum::<u64>();
                PointMeasurement {
                    key: launch_key(implementation, &[], vec![0], specialization),
                    ..measured((parameters as u64 * values + 1 - sum) as f64, 0.)
                }
            };
        let statics = NativeSpecialization::new();
        let seed = implementation.default_specialization(&statics).unwrap();
        let partition = plan::partition(&implementation, &statics, &shapes(&[4])).unwrap();
        let reference = vec![measurement(&implementation, &seed)];
        let run = GroupSearch::new(partition, seed, reference, vec![1.], Vec::new());
        (implementation, run, measurement)
    }

    /// Plan `run`'s next piece and measure it; none when nothing is left of
    /// the leaders' neighbourhoods (what is asked after them is the rest of
    /// the domain). Returns the requests the planning made of the domain.
    fn planned_and_measured(
        implementation: &NativeImplementation,
        run: &mut GroupSearch,
        measurement: &impl Fn(&NativeImplementation, &NativeSpecialization) -> PointMeasurement,
    ) -> Option<usize> {
        let before = run.requests;
        let piece = run
            .plan(&Declared {
                implementation,
                keying: &Keying::Launches(Vec::new()),
            })
            .unwrap()
            .filter(|_| !run.resting())?;
        for challenger in piece.challengers() {
            run.record(challenger, Some(vec![measurement(implementation, challenger)]));
        }
        Some(run.requests - before)
    }

    /// Planning a piece asks the domain for what the last piece's
    /// measurements lead to, never again for what was asked before: what
    /// one planning asks is the same after two thousand configurations
    /// measured as after two hundred, and over the whole search every
    /// assignment a leader neighbours is asked for once.
    #[test]
    fn planning_a_piece_asks_the_same_however_much_was_measured() {
        let (implementation, mut run, measurement) = climbing(12, 16);
        let mut asked = Vec::new();
        while let Some(requests) = planned_and_measured(&implementation, &mut run, &measurement)
        {
            asked.push((run.swept.len(), requests));
        }
        let measured = run.swept.len();
        assert!(measured > 2000, "{measured}");
        // The most one planning asked while fewer than two hundred were
        // measured, and while more than two thousand were.
        let most = |range: std::ops::Range<usize>| {
            asked
                .iter()
                .filter(|(measured, _)| range.contains(measured))
                .map(|(_, requests)| *requests)
                .max()
                .unwrap()
        };
        let (early, late) = (most(0..200), most(2000..usize::MAX));
        // A leader has at most 12 × 15 neighbours, and a planning that
        // finds everything asked measured takes up one more leader.
        assert!(early <= 2 * 12 * 15 && late <= 2 * 12 * 15, "{early} {late}");
        assert!(late <= early, "{late} asked late against {early} early");
        // Each of the 12 × 15 steps uphill has one leader, asked for its
        // neighbours once.
        assert!(run.requests <= (12 * 15 + 1) * 12 * 15, "{}", run.requests);
        // Nothing exists but what was measured and the first of the rest,
        // asked for and awaiting measurement.
        let awaiting = run
            .sweeps
            .values()
            .map(|sweep| sweep.pending.len())
            .sum::<usize>();
        assert_eq!(run.generated.len(), measured + awaiting);
        // Planning again with nothing measured since asks for nothing.
        let before = run.requests;
        assert!(planned_and_measured(&implementation, &mut run, &measurement).is_none());
        assert_eq!(run.requests, before);
    }

    /// PROFILE: what the host work of a piece costs with `count`
    /// configurations measured (`cargo test --release -- --ignored
    /// profile_per_piece_host_work --nocapture`). Ranking a new challenger
    /// and planning the next piece do not follow `count`; ranking all of
    /// them again (a conversion) and the record's publication do.
    #[test]
    #[ignore]
    fn profile_per_piece_host_work() {
        let time = |work: &mut dyn FnMut()| {
            let began = Instant::now();
            for _ in 0..5 {
                work();
            }
            began.elapsed().as_secs_f64() / 5. * 1e3
        };
        for count in [200usize, 2000] {
            let mut recorded = recorded(6);
            // No challenger leads.
            for challenger in 1..=count {
                let median = 1. + challenger as f64 * 1e-6;
                recorded.challenge(challenger, timed(6, |_| challenger as u64, |_| median));
            }
            let mut next = count;
            let rank = time(&mut || {
                next += 1;
                recorded.challenge(next, timed(6, |_| next as u64, |_| 2.));
                std::hint::black_box((
                    recorded.ledger.leader().is_some(),
                    recorded.ledger.nominal().is_some(),
                ));
                recorded.ledger.release();
            });
            let rerank = time(&mut || recorded.ledger.rerank());
            let mut result = None;
            let record = time(&mut || result = Some(recorded.result()));
            let result = result.unwrap();
            let mut bytes = Vec::new();
            let serialise = time(&mut || bytes = serde_json::to_vec(&result).unwrap());
            let directory =
                std::env::temp_dir().join(format!("seismic-profile-{}", std::process::id()));
            std::fs::create_dir_all(&directory).unwrap();
            let store = time(&mut || {
                use std::io::Write;
                let temporary = directory.join(".record.tmp");
                let mut file = std::fs::File::create(&temporary).unwrap();
                file.write_all(&bytes).unwrap();
                file.sync_all().unwrap();
                std::fs::rename(&temporary, directory.join("record.json")).unwrap();
            });
            std::fs::remove_dir_all(&directory).unwrap();
            eprintln!(
                "measured {count}: per piece: rank a challenger, read the leaders, release \
                 {rank:.3} ms; per conversion: rank all again {rerank:.3} ms; per publication: \
                 record {record:.3} ms, serialise {serialise:.3} ms ({} bytes), \
                 write+fsync+rename {store:.3} ms",
                bytes.len()
            );
        }
        let (implementation, mut run, measurement) = climbing(12, 16);
        let mut reported = 0;
        loop {
            let began = Instant::now();
            let planned = planned_and_measured(&implementation, &mut run, &measurement);
            let seconds = began.elapsed().as_secs_f64() * 1e3;
            if run.swept.len() >= reported + 200 {
                reported = run.swept.len();
                eprintln!(
                    "measured {}: planning and recording a piece {seconds:.3} ms",
                    run.swept.len()
                );
            }
            if planned.is_none() {
                break;
            }
        }
    }

    fn measured(median: f64, deviation: f64) -> PointMeasurement {
        PointMeasurement {
            point: "m1".into(),
            key: PointKey {
                launches: vec![0],
                values: ParameterValues::new(),
            },
            median_seconds: median,
            deviation_seconds: deviation,
            samples: Vec::new(),
            repetitions: 1,
            rotation_bytes: 0,
        }
    }

    #[test]
    fn declaration_digest_uses_stable_entry_identity() {
        let source = |domain: &str| {
            format!(
                "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    launch scale:\n        params (code ROWS in {domain})\n        threadgroups (ceil_div(N, ROWS), 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n"
            )
        };
        let check = |path: &str, text: String| {
            check_source(SourceSet::new(vec![SourceFile {
                path: path.into(),
                text,
            }]))
            .unwrap()
        };
        // The same sources built in two directories are one implementation.
        let first = check("/build/one/kernels/scale.seismic", source("[1, 2]"));
        let second = check("/build/two/kernels/scale.seismic", source("[1, 2]"));
        let changed = check("/build/one/kernels/scale.seismic", source("[1, 3]"));
        assert_ne!(
            first
                .native_implementation(first.entry_named("scale").unwrap(), BackendName::Metal)
                .unwrap()
                .declared_in,
            second
                .native_implementation(second.entry_named("scale").unwrap(), BackendName::Metal)
                .unwrap()
                .declared_in
        );
        let first_entry = first.entry_named("scale").unwrap();
        let second_entry = second.entry_named("scale").unwrap();
        assert_ne!(first_entry, second_entry);
        let digest = |module: &CheckedModule, entry| {
            use sha2::Digest;
            let mut hasher = Sha256::new();
            update_declaration_digest(
                &mut hasher,
                module,
                "scale",
                module
                    .native_implementation(entry, BackendName::Metal)
                    .unwrap(),
            );
            hasher.finalize().to_vec()
        };
        assert_eq!(digest(&first, first_entry), digest(&second, second_entry));
        assert_ne!(
            digest(&first, first_entry),
            digest(&changed, changed.entry_named("scale").unwrap())
        );
    }

    #[test]
    fn launch_keys_ignore_parameters_of_inactive_launches() {
        let text = "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (SPLIT in [1, 2])\n    launch gemv when N < 16:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch gemm when N >= 16:\n        threadgroups (N, 1, SPLIT)\n        threads_per_threadgroup (32, 1, 1)\n";
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "scale.seismic".into(),
            text: text.into(),
        }]))
        .unwrap();
        let implementation = module
            .native_implementation(module.entry_named("scale").unwrap(), BackendName::Metal)
            .unwrap();
        let default = implementation
            .default_specialization(&NativeSpecialization::new())
            .unwrap();
        let split = default.clone().with_param("SPLIT", 2);
        assert_eq!(
            launch_key(implementation, &[], vec![0], &default),
            launch_key(implementation, &[], vec![0], &split)
        );
        assert_ne!(
            launch_key(implementation, &[], vec![1], &default),
            launch_key(implementation, &[], vec![1], &split)
        );
    }

    /// What a search formed stays held whatever was formed after it: a
    /// variant is asked of the toolchain once, a later compile call asks
    /// only for what no held program has, and the programs leave only when
    /// they are given up.
    #[test]
    fn a_formed_variant_is_held_until_given_up_and_never_asked_for_twice() {
        let implementation = declared(
            "    params (ROWS in [16, 32, 64])\n    launch gemv when N < 16:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (ROWS, 1, 1)\n    launch gemm when N >= 16:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (ROWS, 1, 1)\n",
        );
        let default = implementation
            .default_specialization(&NativeSpecialization::new())
            .unwrap();
        let rows = |rows| default.clone().with_param("ROWS", rows);
        let mut programs = Programs::Launches(LaunchPrograms::default());
        let form = |programs: &mut Programs,
                        specializations: &[NativeSpecialization],
                        ahead: &[NativeSpecialization]| {
            let Programs::Launches(launches) = programs else {
                unreachable!()
            };
            let sources = launches.sources(&implementation, specializations, ahead);
            let asked = sources
                .iter()
                .map(|source| source.variants.len())
                .collect::<Vec<_>>();
            if !sources.is_empty() {
                launches
                    .hold(sources, Ok(HeldPrograms(Box::new(()))))
                    .unwrap();
            }
            asked
        };
        // The defaults' variant of each launch, then one more with the one
        // measured next.
        assert_eq!(form(&mut programs, &[default.clone()], &[]), [1, 1]);
        assert_eq!(form(&mut programs, &[rows(32)], &[rows(64)]), [2, 2]);
        // Everything formed is held: nothing is asked for again, whichever
        // call formed it.
        for held in [default.clone(), rows(32), rows(64)] {
            assert!(form(&mut programs, &[held], &[]).is_empty());
        }
        assert_eq!((programs.holding(), programs.formed()), (6, 6));
        // Given up, a variant is formed again when it is measured again;
        // what was formed stays counted.
        programs.clear();
        assert_eq!((programs.holding(), programs.formed()), (0, 6));
        assert_eq!(form(&mut programs, &[rows(32)], &[]), [1, 1]);
    }

    /// A configuration's record states the formation calls made for its
    /// piece: launch variants formed together share their calls, each
    /// stating the calls' seconds and all the programs they formed; a
    /// whole program states its own call; one for which no call was made
    /// reused what was held. The calls are taken by the piece they were
    /// made for.
    #[test]
    fn a_configuration_records_the_formation_calls_made_for_its_piece() {
        let a = NativeSpecialization::new().with_param("ROWS", 16);
        let b = NativeSpecialization::new().with_param("ROWS", 32);
        let call = |formable: Formable, seconds, reused| {
            (
                formable.clone(),
                FormationRecord {
                    seconds,
                    programs: formable.programs(),
                    reused,
                },
            )
        };
        let source = |ordinal, variants: usize| plan::LaunchSource {
            ordinal,
            variants: (0..variants)
                .map(|variant| plan::LaunchVariant {
                    code: vec![variant as u64],
                    group_size: None,
                })
                .collect(),
        };
        let mut price = FormationPrice::default();
        price.calls = vec![
            call(Formable::Launches(vec![source(0, 3), source(1, 2)]), 1.5, false),
            call(Formable::Launches(vec![source(0, 1)]), 0.25, true),
        ];
        let launches = Programs::Launches(LaunchPrograms::default());
        let shared = FormationRecord {
            seconds: 1.75,
            programs: 6,
            reused: false,
        };
        assert_eq!(
            launches.records(&mut price, &[a.clone(), b.clone()]),
            [shared, shared]
        );
        let held = FormationRecord {
            seconds: 0.,
            programs: 0,
            reused: true,
        };
        assert_eq!(launches.records(&mut price, &[a.clone()]), [held]);
        price.calls = vec![call(Formable::Program(b.clone()), 0.5, true)];
        let whole = Programs::Whole(WholePrograms::default());
        assert_eq!(
            whole.records(&mut price, &[a, b]),
            [
                held,
                FormationRecord {
                    seconds: 0.5,
                    programs: 1,
                    reused: true,
                }
            ]
        );
    }

    #[test]
    fn launch_keys_ignore_boundary_values_when_the_active_launch_is_unchanged() {
        let text = "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative scale for metal from \"scale.metal\":\n    params (arithmetic BATCH_FROM in [5, 9])\n    launch gemv when N < BATCH_FROM:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n    launch batch when N >= BATCH_FROM:\n        threadgroups (N, 1, 1)\n        threads_per_threadgroup (32, 1, 1)\n";
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "scale.seismic".into(),
            text: text.into(),
        }]))
        .unwrap();
        let implementation = module
            .native_implementation(module.entry_named("scale").unwrap(), BackendName::Metal)
            .unwrap();
        let default = implementation
            .default_specialization(&NativeSpecialization::new())
            .unwrap();
        let changed = default.clone().with_param("BATCH_FROM", 9);
        let boundary = [plan::ParameterAddress::Entry("BATCH_FROM".into())];
        assert_eq!(
            launch_key(implementation, &boundary, vec![1], &default),
            launch_key(implementation, &boundary, vec![1], &changed)
        );
    }

    /// Formation cannot be interrupted, so no program starts that its price
    /// (from the formations observed for the unit) says cannot end within
    /// the budget: its work is not admitted, and the device compiles
    /// nothing for it. The same compile has the same price wherever it is
    /// quoted.
    #[cfg(target_os = "macos")]
    #[test]
    fn no_formation_starts_that_cannot_end_before_the_deadline() {
        use seismic_lang::checked::{SourceFile, SourceSet, check_source};
        use seismic_lang::registry::BackendName;
        let Ok(device) = crate::devices::Catalog::discover()
            .expect("device discovery")
            .open_backend(BackendName::Metal)
        else {
            return;
        };
        let mut module = check_source(SourceSet::new(vec![SourceFile {
            path: "copy.seismic".into(),
            text: "fn copy[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\nnative copy for metal from \"copy.metal\":\n    launch copy:\n        threadgroups (1, 1, 1)\n        threads_per_threadgroup (1, 1, 1)\n".into(),
        }]))
        .expect("copy source checks");
        let entry = module.entry_named("copy").expect("copy entry");
        module
            .capture_native_asset(entry, BackendName::Metal, "kernel void copy() {}\n".into())
            .expect("asset capture");
        let bindings = ElementBindings::default();
        let logical = Arc::new(module.entry(entry, &bindings).expect("copy entry binds"));
        let implementation = super::super::on_device(
            &device,
            module
                .native_implementation(entry, BackendName::Metal)
                .cloned()
                .expect("a Metal implementation"),
        );
        let default = implementation
            .default_specialization(&NativeSpecialization::new())
            .expect("defaults");
        let mut observed = FormationPrice::default();
        observed.observe((FormationClass::Compiled, 1), Duration::from_secs(5));
        observed.observe((FormationClass::Kept, 0), Duration::from_millis(1));
        let price = Mutex::new(observed);
        let formation = Formation {
            device: &device,
            module: &module,
            entry,
            logical: &logical,
            bindings: &bindings,
            cpu: None,
            implementation: &implementation,
            default: &default,
            served: &[],
            price: &price,
        };
        // The device was opened without a store and holds no program: the
        // toolchain compiles this one.
        let program = Formable::Program(default.clone());
        assert_eq!(formation.class(&program), (FormationClass::Compiled, 1));
        let quote = formation.quote(std::slice::from_ref(&program));
        assert_eq!(quote, priced(5., 0));
        assert!(
            !quote.fits(Duration::from_secs(1), Duration::ZERO),
            "the program's work is refused before it is formed"
        );
        assert!(!quote.fits(Duration::from_secs(4), Duration::ZERO));
        assert!(quote.fits(Duration::from_secs(6), Duration::ZERO));
        // Nothing ran, so nothing was observed.
        assert_eq!(price.lock().unwrap().total(), 5.001);
        // Forming nothing compiles nothing and costs nothing.
        assert_eq!(formation.quote(&[]), Quote::default());
        assert!(formation.forming(Vec::new()).is_none());
        // A search of launches formed separately quotes the same compile
        // call from the same observations: one price per class, wherever
        // it is quoted.
        let launches = LaunchPrograms::default();
        let sources = launches.sources(&implementation, std::slice::from_ref(&default), &[]);
        assert_eq!(sources.len(), 1);
        assert_eq!(formation.quote(&[Formable::Launches(sources.clone())]), quote);
        assert_eq!(
            launches.sources(&implementation, &[default.clone()], &[default.clone()]),
            sources,
            "the same variants measured next add nothing to the compile"
        );
        // A piece of a program formed whole compiles its challengers'
        // programs, and they compile ahead of it on another thread, as a
        // consumer's worker runs them.
        let challengers = std::slice::from_ref(&default);
        let mut programs = Programs::Whole(WholePrograms::default());
        let ahead = programs.formables(&implementation, challengers, &[]);
        assert_eq!(ahead, [program.clone()]);
        let forming = formation.forming(ahead).expect("the piece's compile");
        let formed = std::thread::scope(|scope| scope.spawn(|| forming.run()).join())
            .expect("the compile runs to its end");
        programs
            .hold(formed, &mut price.lock().unwrap())
            .expect("the unit holds what was formed");
        // The compile was observed in the class it was priced by, and the
        // piece then forms nothing: nothing is left to compile, and
        // forming its challengers observes no formation.
        let compiled = price
            .lock()
            .unwrap()
            .of((FormationClass::Compiled, 1), &Linear::default());
        assert!(compiled.is_some_and(|mean| mean < 5.), "{compiled:?}");
        assert!(programs.formables(&implementation, challengers, &[]).is_empty());
        assert_eq!(formation.quote(&programs.formables(&implementation, challengers, &[])), Quote::default());
        let before = price.lock().unwrap().total();
        let held = programs
            .form(&formation, challengers, &[])
            .expect("the piece forms its challengers")
            .pop()
            .expect("one challenger")
            .expect("the program forms");
        assert_eq!(price.lock().unwrap().total(), before);
        assert_eq!(programs.formed(), 1);
        // Formed and held, the program is live on the device: forming it
        // again runs no compiler, and is priced by the formations that ran
        // none.
        assert_eq!(formation.class(&program), (FormationClass::Kept, 0));
        let again = formation.quote(std::slice::from_ref(&program));
        assert_eq!(
            again.priced,
            Duration::from_secs_f64(
                price
                    .lock()
                    .unwrap()
                    .of((FormationClass::Kept, 0), &Linear::default())
                    .unwrap()
            )
        );
        assert!(again.priced < Duration::from_millis(2));
        // Given up by whatever held it, as another unit whose launches
        // render the same text gives up its compiles, the program is live
        // no longer: forming it compiles again, and is classed so though
        // this unit formed and released nothing since it was last asked.
        drop(held);
        drop(programs);
        assert_eq!(formation.class(&program), (FormationClass::Compiled, 1));
    }

    /// A challenger formation is priced by the formations of its class
    /// observed. The canonical program's formation prices none; it is only
    /// an operation the load completed.
    #[test]
    fn formation_is_priced_by_the_formations_of_its_class() {
        use FormationClass::{Compiled, Kept};
        let none = Linear::default();
        let mut price = FormationPrice::default();
        price.canonical = Duration::from_millis(180);
        assert_eq!(price.of((Compiled, 4), &none), None, "a class never observed has no price");
        assert_eq!(price.longest(), Duration::from_millis(180));
        price.observe((Compiled, 4), Duration::from_secs(1));
        price.observe((Compiled, 4), Duration::from_secs(3));
        assert_eq!(price.of((Kept, 0), &none), None, "a compile says nothing of a kept formation");
        price.observe((Kept, 0), Duration::ZERO);
        price.observe((Kept, 0), Duration::ZERO);
        // Calls of one size: their mean, whatever is asked.
        assert_eq!(price.of((Compiled, 4), &none), Some(2.));
        assert_eq!(price.of((Compiled, 1), &none), Some(2.));
        assert_eq!(price.of((Kept, 0), &none), Some(0.));
        assert_eq!(price.longest(), Duration::from_secs(3));
        assert_eq!(price.total(), 4.18);
    }

    /// A class nothing prices is one unknown in a piece, however many of
    /// its operations the piece has: a piece of seven challengers whose
    /// admission the unit has not observed is held to the longest
    /// operation once, and starts in what is left of a budget where seven
    /// times a multi-second compile would not fit.
    #[test]
    fn an_unobserved_class_is_one_unknown_in_a_piece() {
        let mut quote = Quote::default();
        quote.add(None, 7);
        quote.add(Some(0.01), 7);
        quote.expect(None, 2.5);
        quote.add(None, 0);
        quote.expect(None, 0.);
        assert_eq!(quote, priced(0.07, 2));
        let longest = Duration::from_secs_f64(4.3);
        assert!(quote.fits(Duration::from_secs(23), longest));
        assert!(!quote.fits(Duration::from_secs(8), longest));
    }

    /// A new challenger's tests are priced in the order they are taken:
    /// witnessed first, by the unit's admissions; timed first, by each
    /// timed point's placement and, for the challengers expected to be
    /// left ahead, by the witnesses taken after rounds, a class of their
    /// own that the unit's admissions witnessed first price until one was
    /// taken.
    #[test]
    fn a_witness_after_rounds_is_priced_in_its_piece() {
        let mut admission = Admission::default();
        let mut sequence = Sequence::default();
        let placements = [Some(0.001), Some(0.002)];
        // Before either test was observed the witness is first, and its
        // class is not yet priced.
        assert_eq!(admission.quote(&sequence, &placements, 4), priced(0., 1));
        assert_eq!(admission.quote(&sequence, &placements, 0), priced(0., 0));
        // Eight challengers witnessed first at 90 ms each, none refused,
        // every one behind after rounds of 12 ms: the next are timed first.
        for _ in 0..8 {
            admission.witnessed_first.observe(0.09);
            sequence.witnessed(0.09, true);
            sequence.compared(0.012, false);
        }
        assert!(sequence.timed_first());
        assert_eq!(sequence.ahead(), 0.1);
        // Their placements are priced; their witnesses after rounds are a
        // class the unit has not observed, priced by its nearest: the
        // admissions witnessed first. Of five, half a witness is expected.
        let quote = admission.quote(&sequence, &placements, 5);
        assert_eq!(quote.unpriced, 0);
        assert!((quote.priced.as_secs_f64() - (0.015 + 0.5 * 0.09)).abs() < 1e-9);
        // One was left ahead and witnessed in 80 ms. Of five, one in ten
        // is expected ahead: half a witness.
        admission.witnessed_after.observe(0.08);
        let quote = admission.quote(&sequence, &placements, 5);
        assert_eq!(quote.unpriced, 0);
        assert!((quote.priced.as_secs_f64() - (0.015 + 0.5 * 0.08)).abs() < 1e-9);
    }

    /// A leader that reaches its conversion unwitnessed is witnessed in
    /// that session before its rounds, and the session is priced so: one
    /// admission of the witnessed-first class in place of its placements,
    /// an unknown while that class is unobserved. A witnessed leader's
    /// session is its placements and samples alone.
    #[test]
    fn a_witness_owed_at_a_conversion_is_priced_in_its_session() {
        let mut admission = Admission::default();
        let mut prices = PointPrices::default();
        prices.placement.observe(0.002);
        prices.sample.observe(0.01);
        // Two timed points; the leader runs other work than the served
        // choice at the first. One round is expected past the calibrating
        // pass and the first.
        let points = [(prices, true), (prices, false)];
        let seconds = |quote: Quote| quote.priced.as_secs_f64();
        // Witnessed: three placements, and two slots sampled twice and
        // once more.
        let witnessed = admission.conversion(&points, true, Some(1.));
        assert_eq!(witnessed.unpriced, 0);
        assert!((seconds(witnessed) - (3. * 0.002 + 4. * 0.01)).abs() < 1e-9);
        // Unwitnessed, before any challenger was witnessed first: its
        // witness is one operation of a class the unit has not observed,
        // and only the served choice's placement is priced by the point.
        let owed = admission.conversion(&points, false, Some(1.));
        assert_eq!(owed.unpriced, 1);
        assert!((seconds(owed) - (0.002 + 4. * 0.01)).abs() < 1e-9);
        // Once observed, the class prices it.
        admission.witnessed_first.observe(0.09);
        let owed = admission.conversion(&points, false, Some(1.));
        assert_eq!(owed.unpriced, 0);
        assert!((seconds(owed) - (0.09 + 0.002 + 4. * 0.01)).abs() < 1e-9);
        // The rounds of a class of session the unit has not run are an
        // unknown of their own.
        assert_eq!(admission.conversion(&points, false, None).unpriced, 1);
        assert_eq!(admission.conversion(&points, true, None).unpriced, 1);
    }

    /// A compile call costs what the call costs and what each entry it
    /// forms costs, from the most specific observations the load has: the
    /// unit's calls of two sizes; its calls of one size moved along the
    /// toolchain's line; with none of its own, the toolchain's compiles.
    #[test]
    fn a_compile_call_is_priced_by_its_call_and_its_entries() {
        use FormationClass::Compiled;
        let close = |price: Option<f64>, expected: f64| {
            assert!(price.is_some_and(|price| (price - expected).abs() < 1e-9), "{price:?}");
        };
        // A toolchain whose calls cost 1 s and 0.05 s an entry (an NVRTC
        // call of 48 entries takes 3.4 s, one of 4 entries 1.2 s).
        let mut toolchain = Linear::default();
        toolchain.observe(48, 3.4);
        toolchain.observe(4, 1.2);
        toolchain.observe(20, 2.0);
        close(toolchain.price(1), 1.05);
        // A unit that has compiled nothing is priced by the toolchain's
        // compiles: its first piece is not unpriced.
        let mut unit = FormationPrice::default();
        close(unit.of((Compiled, 4), &toolchain), 1.2);
        assert_eq!(unit.of((Compiled, 4), &Linear::default()), None);
        // One call of its own, twice what the toolchain's line says of its
        // size: the unit's level, the toolchain's shape.
        unit.observe((Compiled, 20), Duration::from_secs_f64(4.0));
        close(unit.of((Compiled, 20), &toolchain), 4.0);
        close(unit.of((Compiled, 4), &toolchain), 2.4);
        // A mean over its calls would price the small call at 4 s.
        // Calls of two sizes: the unit's own line, 3 s a call and 0.05 s
        // an entry.
        unit.observe((Compiled, 40), Duration::from_secs_f64(5.0));
        close(unit.of((Compiled, 4), &toolchain), 3.2);
        close(unit.of((Compiled, 60), &toolchain), 6.0);
        // Neither term is negative: sizes whose times fall as they grow
        // are priced at their mean.
        let mut falling = Linear::default();
        falling.observe(2, 3.);
        falling.observe(8, 1.);
        close(falling.price(100), 2.);
    }

    /// Formations found kept and formations compiled, in turn, each
    /// admitted only while its price fits what remains: priced by class,
    /// the sequence ends within the budget. One mean over both, as each
    /// compile was priced before, admits a compile that runs past it.
    #[test]
    fn a_mixed_sequence_of_kept_and_compiled_formations_does_not_overrun() {
        use FormationClass::{Compiled, Kept};
        let sequence = [
            (Kept, 0.),
            (Kept, 0.),
            (Kept, 0.),
            (Compiled, 1.),
            (Compiled, 1.),
            (Kept, 0.),
            (Compiled, 1.),
            (Compiled, 1.),
            (Compiled, 1.),
            (Compiled, 1.),
        ];
        let budget = 3.5;
        // What the sequence takes when each formation is admitted on
        // `price`, and each admitted one observed by `observe`.
        fn run(
            sequence: &[(FormationClass, f64)],
            budget: f64,
            mut price: impl FnMut(FormationClass) -> Option<f64>,
            mut observe: impl FnMut(FormationClass, f64),
        ) -> f64 {
            let (mut clock, mut longest) = (0., 0f64);
            for &(class, took) in sequence {
                let mut quote = Quote::default();
                quote.add(price(class), 1);
                if quote.fits(
                    Duration::from_secs_f64((budget - clock).max(0.)),
                    Duration::from_secs_f64(longest),
                ) {
                    clock += took;
                    longest = longest.max(took);
                    observe(class, took);
                }
            }
            clock
        }
        let classed = std::cell::RefCell::new(FormationPrice::default());
        let ended = run(
            &sequence,
            budget,
            |class| classed.borrow().of((class, 1), &Linear::default()),
            |class, took| {
                classed
                    .borrow_mut()
                    .observe((class, 1), Duration::from_secs_f64(took))
            },
        );
        assert_eq!(ended, 3., "three compiles fit; the fourth is refused");
        let mixed = std::cell::RefCell::new(Observed::default());
        let ended = run(
            &sequence,
            budget,
            |_| mixed.borrow().price(),
            |_, took| mixed.borrow_mut().observe(took),
        );
        assert!(ended > budget, "one mean over both classes overruns: {ended}");
    }

}
