//! Native tuning validates every candidate against the selected reference before
//! repeated timing. The first initialized timed invocation supplies its outputs
//! and writable state; host comparison remains outside the device interval.
//! Ordinary, factored and survey paths use the same evidence. Timing
//! normalization and reuse do not confer numerical eligibility.

use super::plan::{self, PointShape};
use super::search::{
    self, Cost, Evaluator, ParameterValues, PointKey, SearchParameter, SearchSettings, SearchSpace,
    SearchSpaceError, SearchStop,
};
use super::timing::{self, OutputPool, PointTiming};
pub use super::validation::{NumericalEvidence, NumericalMetrics};
use super::validation::{PreparedPoint, Validator};
use super::{HeldPrograms, MeasureOptions, Measurement, NativePrepared};
use crate::api::device::DeviceInner;
use crate::api::kernel::{EncodedArgs, PrepareError};
use crate::api::{CallError, TensorError};
use seismic_compiler::prepared::{validate_invocation, InvocationContract};
use seismic_lang::checked::{CheckedModule, NativeImplementation, NativeSpecialization};
use seismic_lang::entry::{ElementBindings, LogicalEntry, ParameterKind, TensorAccess};
use seismic_lang::expr::SymbolValue;
use seismic_lang::ids::EntryId;
use seismic_lang::precision::{PrecisionPolicy, TuningPrecision};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::cpu::CpuNativeKernels;

/// Restores a point's `&mut` tensors to their initial contents.
pub type TuningInitializer<'a> = Box<dyn FnMut() -> Result<(), TensorError> + 'a>;

/// One workload the tuned implementation serves, described before its
/// inputs exist: they are built only when the tuner first needs the point.
#[derive(Clone, Debug)]
pub struct PointSpec {
    pub label: String,
    /// Share of step time spent in this workload: the objective weighs a
    /// configuration's time here, relative to the defaults', by it.
    pub weight: f64,
    /// Points naming the same class are variants of one workload (the same
    /// rows at different history lengths), each occurring as often as its
    /// weight says: together they carry their summed weight, split by the
    /// defaults' real time at each ([`Cost::relative`]). `None`: a class of
    /// its own.
    pub class: Option<String>,
    /// The point's cost relative to the unit's other points. Points come in
    /// ascending cost; the tuner predicts a point's time from the previous
    /// one's by the ratio of their costs.
    pub cost: f64,
    /// A candidate may be chosen only if it was validated here: the point
    /// exercises a code path a candidate may run in serving. When the
    /// required points do not fit the unit's time, the unit keeps its
    /// defaults.
    pub required: bool,
    /// The census measures the defaults here: the point's time stands for
    /// its workload when the consumer divides its tuning time among units.
    /// Every required point is measured by the census whatever this says;
    /// a point that is only this is left for the start when it does not fit.
    pub census: bool,
}

/// A point's inputs.
pub struct PointInputs<'a> {
    /// Argument sets cycled through by measurement and numerical validation.
    pub rotation: Vec<EncodedArgs>,
    /// Required when the entry has `&mut` parameters: called before every
    /// reference and validated invocation, it restores all writable tensors
    /// in every rotation. Timed passes after validation run without it.
    pub initialize: Option<TuningInitializer<'a>>,
    /// The leading-axis rows each named `&mut` parameter's entry writes:
    /// validation observes exactly those rows. A `&mut` parameter absent
    /// here is observed whole.
    pub written: BTreeMap<String, Range<u64>>,
}

/// Why a point's inputs were not built.
#[derive(Clone, Debug)]
pub enum PointUnavailable {
    /// Building them would not fit the time given.
    Unaffordable,
    Failed(String),
}

/// The points of one tuning unit, whose inputs are built on request.
pub trait PointSource<'a> {
    /// Every point, in ascending cost.
    fn points(&self) -> Vec<PointSpec>;
    /// Build point `point`'s inputs, refusing before any step predicted not
    /// to fit within `limit`.
    fn build(&mut self, point: usize, limit: Duration)
        -> Result<PointInputs<'a>, PointUnavailable>;
}

/// A point with its inputs built.
pub(crate) struct TuningPoint<'a> {
    pub label: String,
    /// Share of step time spent in this workload: the objective weighs a
    /// configuration's time here, relative to the defaults', by it.
    pub weight: f64,
    /// Points naming the same class are variants of one workload (the same
    /// rows at different history lengths), each occurring as often as its
    /// weight says: together they carry their summed weight, split by the
    /// defaults' real time at each ([`Cost::relative`]). `None`: a class of
    /// its own.
    pub class: Option<String>,
    pub rotation: Vec<EncodedArgs>,
    pub initialize: Option<TuningInitializer<'a>>,
    pub written: BTreeMap<String, Range<u64>>,
}

impl<'a> TuningPoint<'a> {
    fn new(spec: &PointSpec, inputs: PointInputs<'a>) -> Result<Self, TuneError> {
        if inputs.rotation.is_empty() {
            return Err(TuneError::NoPoints);
        }
        Ok(Self {
            label: spec.label.clone(),
            weight: spec.weight,
            class: spec.class.clone(),
            rotation: inputs.rotation,
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
    fn of(specialization: &NativeSpecialization) -> Self {
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
    pub median_seconds: f64,
    pub deviation_seconds: f64,
    pub samples: Vec<f64>,
    pub repetitions: usize,
    pub rotation_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Outcome {
    Measured {
        artifact: String,
        /// Candidate measurements at every point measured.
        points: Vec<PointMeasurement>,
        /// The finalists' re-measurement (empty for every other
        /// configuration).
        confirmed: Vec<PointMeasurement>,
        /// Whether the complete results and writable state passed the
        /// precision policy against the selected reference on every required case.
        validated: bool,
    },
    Excluded(Exclusion),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfigurationRecord {
    pub configuration: Configuration,
    pub outcome: Outcome,
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
    Search {
        /// The search's time allowance.
        allowance_seconds: f64,
        settings: SearchSettings,
        stop: SearchStop,
    },
    /// The defaults measured at the points a census admitted.
    Census,
    /// Every admissible configuration, `samples` per point.
    Survey { samples: usize },
    /// A factored search of each independent launch group; `complete` when
    /// every candidate was measured within the allowance.
    Factored {
        /// The search's time allowance.
        allowance_seconds: f64,
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
}

/// A tuning point as recorded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointRecord {
    pub label: String,
    pub weight: f64,
    #[serde(default)]
    pub class: Option<String>,
}

/// The complete, serializable outcome of tuning one implementation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuningResult {
    pub tuning_identity: String,
    pub entry: String,
    pub backend: String,
    /// Points in the order given, with their weights and classes.
    pub points: Vec<PointRecord>,
    pub validation: PrecisionPolicy,
    pub numerical_evidence: Vec<NumericalEvidence>,
    pub implementation_identity: String,
    /// The parameters tuned, in declaration order.
    pub parameters: Vec<DeclaredParameter>,
    /// Every configuration reached, in the order reached.
    pub configurations: Vec<ConfigurationRecord>,
    /// The chosen configuration, for the consumer to prepare.
    pub overall: Configuration,
    pub method: TuningMethod,
    pub time: TuningTime,
}

impl TuningResult {
    /// Configurations rejected during formation or numerical qualification.
    /// Exceeding the caller's precision bound does not itself prove a kernel defect.
    pub fn rejections(&self) -> impl Iterator<Item = &ConfigurationRecord> {
        self.configurations.iter().filter(|record| {
            matches!(
                record.outcome,
                Outcome::Excluded(Exclusion::Formation(_) | Exclusion::Validation { .. })
            )
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
    /// The declared parameters cannot be searched.
    Space(SearchSpaceError),
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
}

impl std::fmt::Display for TuneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Declaration(message) | Self::Domain(message) | Self::Reference(message) => write!(f, "{message}"),
            Self::NoValidatedCandidate(failures) => write!(f, "no candidate passed numerical validation and measurement: {failures:?}"),
            Self::NoPoints => f.write_str("tuning needs at least one point"),
            Self::Space(error) => write!(f, "{error}"),
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
    /// The unit's whole time: building and measuring its points, the
    /// search, confirming its finalists and validating its choice at the
    /// points it did not time. Exploration ends when what remains only
    /// covers the last two.
    pub allowance: Duration,
    /// What building the timed points and measuring the defaults there may
    /// take, each point predicted from the last before it starts. Points
    /// beyond it are not timed; the choice is validated there.
    pub admission: Duration,
    /// What the required points may take; when they cannot fit, the unit
    /// keeps its defaults.
    pub required: Duration,
    pub settings: SearchSettings,
    /// Minimum device time of one sample; sets repetitions per sample.
    pub min_sample_seconds: f64,
    /// Consumer hints for the search. An admissible hint matching a structural
    /// form becomes that form's first measurement; other hints follow the
    /// form starts. Inadmissible hints are skipped.
    pub start: Vec<ParameterValues>,
}

/// Every admissible configuration, measured and validated (development).
#[derive(Clone, Debug)]
pub struct SurveyPlan {
    /// Samples per point of every configuration.
    pub samples: usize,
    pub min_sample_seconds: f64,
    /// Replacement value lists for some parameters, widening the declared
    /// space. Each keeps its declared default first.
    pub domains: BTreeMap<String, Vec<u64>>,
}

/// Measuring the defaults at the points every candidate must pass and at the
/// unit's census points (the cheapest point when there is neither): the
/// fixed cost of searching the unit, and the defaults' time there, from
/// which the consumer divides its tuning time among units.
#[derive(Clone, Debug)]
pub struct CensusPlan {
    /// The tuning time left. When the required points cannot fit it, the
    /// unit keeps its defaults.
    pub limit: Duration,
    pub min_sample_seconds: f64,
}

#[derive(Clone, Debug)]
pub enum Strategy {
    Search(SearchPlan),
    Survey(SurveyPlan),
}

/// How a [`UnitSearch`] searches once its points are admitted.
#[derive(Clone, Debug)]
pub struct StartPlan {
    /// What the required points not yet admitted may take; when they cannot
    /// fit, the unit keeps its defaults. A census admitted them all.
    pub required: Duration,
    /// What building the further timed points and measuring the defaults
    /// there may take, each point predicted from the last before it starts.
    /// Points beyond it are not timed; the choice is validated there.
    pub admission: Duration,
    pub settings: SearchSettings,
    /// Minimum device time of one sample; sets repetitions per sample.
    pub min_sample_seconds: f64,
    /// Consumer hints, as [`SearchPlan::start`].
    pub start: Vec<ParameterValues>,
    /// Whether the start measures every form's start besides the defaults,
    /// and active-coordinate starts for a factored declaration.
    /// A consumer whose time does not cover that for every unit starts some
    /// with their defaults alone; their first refinement measures the form
    /// starts.
    pub forms: bool,
}

/// Where a unit's search stands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Standing {
    /// The search ran to its own end: no further time would measure anything.
    pub finished: bool,
    /// The cheapest configuration measured so far relative to the defaults:
    /// 1 until something measures cheaper.
    pub cost: f64,
    /// Configurations measured so far.
    pub measured: usize,
    /// Configurations the search can reach at its points.
    pub admissible: usize,
    /// Programs formed so far: launch variants of a launch-scoped
    /// implementation, configurations of any other.
    pub programs: usize,
    /// Programs the search would form if it measured everything it can
    /// reach.
    pub declared_programs: usize,
    /// What forming programs has taken so far.
    pub forming_seconds: f64,
    /// The step that returned this reached its hard limit before the work
    /// planned for it was done.
    pub cut: bool,
}

/// Numerical reference for empirical native tuning. Neither choice proves compiler applicability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TuningReference {
    /// Independently execute the checked portable operation (development cases).
    Portable,
    /// Compare with the declaration's default native specialization.
    /// Shared defects are invisible; model/kernel regressions remain necessary.
    NativeDefault,
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
    /// The policy every configuration in no error class is validated under,
    /// and the admitted error classes with their envelopes.
    pub validation: TuningPrecision,
    pub strategy: Strategy,
    pub reference: TuningReference,
}

/// Forms configurations of one entry's implementation.
struct Formation<'s> {
    device: &'s Arc<DeviceInner>,
    module: &'s CheckedModule,
    entry: EntryId,
    logical: &'s Arc<LogicalEntry>,
    bindings: &'s ElementBindings,
    cpu: Option<&'static CpuNativeKernels>,
    implementation: &'s NativeImplementation,
}

impl Formation<'_> {
    /// Form every configuration concurrently.
    fn form_all(
        &self,
        configurations: &[NativeSpecialization],
    ) -> Vec<Result<Arc<NativePrepared>, Exclusion>> {
        let workers = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
            .min(configurations.len().max(1));
        let chunk = configurations.len().div_ceil(workers).max(1);
        std::thread::scope(|scope| {
            let handles = configurations
                .chunks(chunk)
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|specialization| {
                                NativePrepared::prepare_implementation(
                                    self.device,
                                    self.module,
                                    self.entry,
                                    self.logical,
                                    self.bindings.clone(),
                                    specialization.clone(),
                                    self.cpu,
                                    self.implementation.clone(),
                                )
                                .map_err(|error| Exclusion::Formation(prepare_message(error)))
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("native formation thread panicked"))
                .collect()
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
            for args in &point.rotation {
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

fn boundary_assignments(
    partition: &plan::TuningPartition,
    implementation: &NativeImplementation,
) -> Vec<Vec<u64>> {
    partition
        .boundary
        .iter()
        .fold(vec![Vec::new()], |choices, address| {
            let plan::ParameterAddress::Entry(name) = address else {
                unreachable!("only entry parameters choose launch bands")
            };
            let values = &implementation
                .params
                .iter()
                .find(|parameter| &parameter.name == name)
                .expect("boundary is a declared entry parameter")
                .values;
            choices
                .into_iter()
                .flat_map(|choice| {
                    values.iter().map(move |value| {
                        let mut next = choice.clone();
                        next.push(*value);
                        next
                    })
                })
                .collect()
        })
}

fn factored_key(
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

fn measure_factored(
    implementation: &NativeImplementation,
    boundary: &[plan::ParameterAddress],
    kernel: &Arc<NativePrepared>,
    specialization: &NativeSpecialization,
    points: &[PreparedPoint<'_>],
    options: &MeasureOptions,
    affected_launches: Option<&[usize]>,
    baseline: Option<&[PointMeasurement]>,
    cache: Option<&mut BTreeMap<(usize, PointKey), PointMeasurement>>,
    outputs: &mut OutputPool,
) -> Result<Vec<PointMeasurement>, Exclusion> {
    debug_assert!(cache.is_none() || baseline.is_some());
    let mut cache = cache;
    let placed = points
        .iter()
        .enumerate()
        .map(|(point, workload)| {
            PointTiming::reusing_outputs(
                kernel,
                workload.rotation.clone(),
                outputs.at(&workload.label),
            )
            .map_err(|error| measurement_failure(points, point, error))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut placed = placed;
    for (point, timing) in points.iter().zip(&mut placed) {
        point.validate(timing, options.min_sample_seconds)?;
    }
    let mut sampled = Vec::new();
    let mut indices = Vec::new();
    let mut reused = baseline
        .map(|baseline| baseline.to_vec())
        .unwrap_or_else(|| Vec::with_capacity(points.len()));
    for (point, timing) in placed.into_iter().enumerate() {
        let working = timing.working_launches();
        let affected = affected_launches.is_none_or(|launches| {
            launches.iter().any(|launch| {
                working.contains(launch)
                    || baseline
                        .is_some_and(|baseline| baseline[point].key.launches.contains(launch))
            })
        });
        if affected {
            let key = factored_key(implementation, boundary, working, specialization);
            if let Some(cached) = cache
                .as_ref()
                .and_then(|cache| cache.get(&(point, key.clone())))
            {
                reused[point] = cached.clone();
            } else {
                indices.push((point, key));
                sampled.push(timing);
            }
        }
    }
    if !sampled.is_empty() {
        timing::sample(&mut sampled, options).map_err(|failure| {
            measurement_failure(points, indices[failure.point].0, failure.error)
        })?;
    }
    if baseline.is_none() {
        for ((point, key), timing) in indices.into_iter().zip(&sampled) {
            debug_assert_eq!(point, reused.len());
            reused.push(point_measurement(
                &points[point].label,
                key,
                timing.measurement(),
            ));
        }
    } else {
        for ((point, key), timing) in indices.into_iter().zip(&sampled) {
            let result = point_measurement(&points[point].label, key.clone(), timing.measurement());
            if let Some(cache) = cache.as_deref_mut() {
                cache.insert((point, key), result.clone());
            }
            reused[point] = result;
        }
    }
    Ok(reused)
}

struct FactoredCandidate {
    index: usize,
    specialization: NativeSpecialization,
    kernel: Arc<NativePrepared>,
}

/// One sampled point/key of a sweep and the candidates it measures.
type Slot = (usize, PointKey, Vec<usize>);

/// Bring every placed slot to `options.samples` samples. A failing slot's
/// candidates leave the sweep; the other slots keep their samples and finish.
fn sample_slots(
    placed: &mut Vec<PointTiming>,
    slots: &mut Vec<Slot>,
    failures: &mut [Option<Exclusion>],
    points: &[PreparedPoint<'_>],
    options: &MeasureOptions,
) {
    while let Err(failure) = timing::sample(placed, options) {
        let point = slots[failure.point].0;
        for &candidate in &slots[failure.point].2 {
            failures[candidate] = Some(measurement_failure(points, point, failure.error.clone()));
        }
        let (kept_timings, kept_slots) = std::mem::take(placed)
            .into_iter()
            .zip(std::mem::take(slots))
            .filter_map(|(timing, mut slot)| {
                slot.2.retain(|candidate| failures[*candidate].is_none());
                (!slot.2.is_empty()).then_some((timing, slot))
            })
            .unzip();
        *placed = kept_timings;
        *slots = kept_slots;
    }
}

/// Measure one launch group's candidate-point pairs in shared rounds. A
/// failing candidate leaves the sweep; the others keep samples already taken
/// and finish their remaining rounds.
fn measure_factored_group(
    implementation: &NativeImplementation,
    boundary: &[plan::ParameterAddress],
    candidates: &[FactoredCandidate],
    points: &[PreparedPoint<'_>],
    options: &MeasureOptions,
    affected_launches: &[usize],
    baseline: &[PointMeasurement],
    cache: Option<&mut BTreeMap<(usize, PointKey), PointMeasurement>>,
    outputs: &mut OutputPool,
) -> Vec<Result<Vec<PointMeasurement>, Exclusion>> {
    let mut cache = cache;
    let mut measured = vec![baseline.to_vec(); candidates.len()];
    let mut failures = vec![None; candidates.len()];
    let mut placed = Vec::new();
    // A sweep can encounter the same point/key through several candidates.
    // Those candidates execute the same active launches with the same values,
    // so one device timing serves all of them. The cross-sweep cache is only
    // populated after sampling and cannot catch duplicates within this sweep.
    let mut slots: Vec<Slot> = Vec::new();
    let mut slot_by_key: BTreeMap<(usize, PointKey), usize> = BTreeMap::new();
    for (candidate, formed) in candidates.iter().enumerate() {
        let mut candidate_placed = Vec::new();
        let mut candidate_slots = Vec::new();
        for (point, workload) in points.iter().enumerate() {
            let mut timing = match PointTiming::reusing_outputs(
                &formed.kernel,
                workload.rotation.clone(),
                outputs.at(&workload.label),
            ) {
                Ok(timing) => timing,
                Err(error) => {
                    failures[candidate] = Some(measurement_failure(points, point, error));
                    break;
                }
            };
            if let Err(error) = workload.validate(&mut timing, options.min_sample_seconds) {
                failures[candidate] = Some(error);
                break;
            }
            let working = timing.working_launches();
            let affected = affected_launches.iter().any(|launch| {
                working.contains(launch) || baseline[point].key.launches.contains(launch)
            });
            if affected {
                let key = factored_key(implementation, boundary, working, &formed.specialization);
                if let Some(cached) = cache
                    .as_ref()
                    .and_then(|cache| cache.get(&(point, key.clone())))
                {
                    measured[candidate][point] = cached.clone();
                } else {
                    candidate_slots.push((point, key));
                    candidate_placed.push(timing);
                }
            }
        }
        if failures[candidate].is_none() {
            for (timing, (point, key)) in candidate_placed.into_iter().zip(candidate_slots) {
                if let Some(&slot) = slot_by_key.get(&(point, key.clone())) {
                    slots[slot].2.push(candidate);
                } else {
                    slot_by_key.insert((point, key.clone()), slots.len());
                    slots.push((point, key, vec![candidate]));
                    placed.push(timing);
                }
            }
        }
    }
    sample_slots(&mut placed, &mut slots, &mut failures, points, options);
    for (timing, (point, key, candidates)) in placed.iter().zip(slots) {
        let result = point_measurement(&points[point].label, key.clone(), timing.measurement());
        if let Some(cache) = cache.as_deref_mut() {
            cache.insert((point, key), result.clone());
        }
        for candidate in candidates {
            measured[candidate][point] = result.clone();
        }
    }
    measured
        .into_iter()
        .zip(failures)
        .map(|(measured, failure)| failure.map_or(Ok(measured), Err))
        .collect()
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
    minimum_seconds: f64,
) -> Result<Placed<'a>, Exclusion> {
    let timings = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            PointTiming::reusing_outputs(kernel, point.rotation.clone(), outputs.at(&point.label))
                .map_err(|error| measurement_failure(points, index, error))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut timings = timings;
    for (point, timing) in points.iter().zip(&mut timings) {
        point.validate(timing, minimum_seconds)?;
    }
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
}

impl Measurer {
    fn new(implementation: &NativeImplementation) -> Self {
        Self {
            influence: Influence::of(implementation),
            measured: HashMap::new(),
            outputs: OutputPool::default(),
        }
    }

    /// The measurement of `kernel` at every point: points whose key was
    /// measured before reuse that measurement, the others are sampled.
    fn measure(
        &mut self,
        kernel: &Arc<NativePrepared>,
        points: &[PreparedPoint<'_>],
        values: &ParameterValues,
        options: &MeasureOptions,
    ) -> Result<Vec<PointMeasurement>, Exclusion> {
        let Placed { timings, keys } = place(
            kernel,
            points,
            &self.influence,
            values,
            &mut self.outputs,
            options.min_sample_seconds,
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
        timing::sample(&mut timings, options)
            .map_err(|failure| measurement_failure(points, fresh[failure.point], failure.error))?;
        for (point, timing) in fresh.into_iter().zip(&timings) {
            let key = keys[point].clone();
            let measurement =
                point_measurement(&points[point].label, key.clone(), timing.measurement());
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

/// The largest median absolute deviation, relative to the median, of a
/// confirmed finalist's samples at any point.
const CONFIRMED_DEVIATION: f64 = 0.10;

/// Why a finalist's confirmation cannot be trusted, if it cannot: at some
/// point its samples spread widely, so a sample saw something other than
/// the kernel (another process on the device, a clock change) and its cost
/// may be an outlier that would win the ranking.
///
/// The confirmation, not the search, is the measurement of record: it
/// samples the defaults and the finalists alternately, so a slow device
/// change affects all of them alike, and a sample cannot read faster than
/// the kernel runs. A search measurement taken while the device was still
/// reaching its clock reads high, and one point's measurement is shared by
/// every configuration with its key, so the two may disagree while the
/// confirmation is right.
fn unstable(confirmed: &[PointMeasurement]) -> Option<Exclusion> {
    confirmed.iter().find_map(|confirmed| {
        let deviation = confirmed.deviation_seconds / confirmed.median_seconds;
        (deviation > CONFIRMED_DEVIATION).then(|| Exclusion::Measurement {
            point: confirmed.point.clone(),
            detail: format!(
                "confirmed samples deviate {:.0}% from their median",
                deviation * 100.0
            ),
        })
    })
}

/// What a finalist's confirmed cost is taken from. Samples that spread
/// widely leave a rival out of the ranking, but not the defaults: they are
/// costed at their fastest sample at each point, a time they did reach (a
/// sample cannot read faster than the kernel runs), so a leader must beat the
/// defaults at their best by the margin to replace them, and noise in the
/// defaults' own timing neither removes them nor makes them win.
fn confirmed_measurement(
    defaults: bool,
    confirmed: Vec<PointMeasurement>,
) -> Result<Vec<PointMeasurement>, Exclusion> {
    match unstable(&confirmed) {
        None => Ok(confirmed),
        Some(exclusion) if !defaults => Err(exclusion),
        Some(_) => Ok(confirmed
            .into_iter()
            .map(|point| PointMeasurement {
                median_seconds: point
                    .samples
                    .iter()
                    .copied()
                    .fold(point.median_seconds, f64::min),
                ..point
            })
            .collect()),
    }
}

/// The points' weights and classes: how the objective weighs them.
struct Weighing {
    weights: Vec<f64>,
    classes: Vec<usize>,
}

impl Weighing {
    fn of(points: &[PreparedPoint<'_>]) -> Self {
        Self {
            weights: points.iter().map(|point| point.weight).collect(),
            classes: search::classes(points.iter().map(|point| point.class.as_deref())),
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
            &self.classes,
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

fn specialization(
    statics: &NativeSpecialization,
    values: &ParameterValues,
) -> NativeSpecialization {
    values
        .iter()
        .fold(statics.clone(), |specialization, (name, value)| {
            specialization.with_param(name.clone(), *value)
        })
}

/// A measured configuration of the live search.
struct Evaluated {
    kernel: Arc<NativePrepared>,
    points: Vec<PointMeasurement>,
    confirmed: Vec<PointMeasurement>,
}

/// What the live search of a unit has formed and measured so far.
struct Lived {
    measurer: Measurer,
    search: MeasureOptions,
    confirmation: MeasureOptions,
    /// Finalists confirmed besides the defaults.
    confirmed: usize,
    evaluated: HashMap<usize, Evaluated>,
    anchor: Option<usize>,
    /// Configurations formed.
    formed: usize,
    time: TuningTime,
}

impl Lived {
    /// What confirming the finalists found so far will take: the defaults
    /// re-measured with the cheapest rivals within noise of the leader, each
    /// sample costing what that configuration's measured samples did; the
    /// defaults alone are not re-measured.
    fn reserve(&self, default: usize) -> Duration {
        let sample = |evaluated: &Evaluated| {
            evaluated
                .points
                .iter()
                .map(|point| point.median_seconds * point.repetitions as f64)
                .sum::<f64>()
        };
        let mut rivals = self
            .evaluated
            .iter()
            .filter(|(index, _)| **index != default)
            .map(|(_, evaluated)| sample(evaluated))
            .collect::<Vec<_>>();
        rivals.sort_by(f64::total_cmp);
        rivals.truncate(self.confirmed);
        let reserve = match rivals.first() {
            Some(leader) => {
                let contending = rivals
                    .iter()
                    .filter(|sample| **sample <= leader * (1. + search::CONTENDING))
                    .sum::<f64>();
                let defaults = self.evaluated.get(&default).map_or(0., sample);
                (contending + defaults) * (self.confirmation.samples + 1) as f64
            }
            None => 0.,
        };
        Duration::from_secs_f64(reserve)
    }
}

/// When a step of a unit's search must stop measuring: a refinement at
/// `slice` (a start has none: it measures its starts whatever the slice),
/// and every step when what remains before `until` only covers concluding
/// the search. `until` is the guarantee under the consumer's plan: a step
/// that reaches it returns with what it has.
#[derive(Clone, Copy)]
struct Limits {
    slice: Option<Instant>,
    until: Instant,
}

/// Forms and measures configurations on the device for [`search::explore`]
/// and [`search::rank`].
struct Live<'s, 'a> {
    formation: &'s Formation<'s>,
    space: &'s SearchSpace,
    statics: &'s NativeSpecialization,
    points: &'s [PreparedPoint<'a>],
    state: &'s mut Lived,
    /// `None`: the finalists are confirmed whatever the time.
    limits: Option<Limits>,
    /// What concluding takes besides confirmation: validating the choice at
    /// the points not timed.
    untimed: Duration,
}

impl Live<'_, '_> {
    /// Measure configuration `index` at every point; a point and key
    /// measured before (the defaults', as the unit admitted its points)
    /// reuses that measurement.
    fn cost(
        &mut self,
        index: usize,
        kernel: Result<Arc<NativePrepared>, Exclusion>,
        weighing: &Weighing,
    ) -> Result<Cost, Exclusion> {
        let kernel = kernel?;
        let points = self.state.measurer.measure(
            &kernel,
            self.points,
            &self.space.values(index),
            &self.state.search,
        )?;
        self.state.anchor.get_or_insert(index);
        self.state.evaluated.insert(
            index,
            Evaluated {
                kernel,
                points: points.clone(),
                confirmed: Vec::new(),
            },
        );
        Ok(weighing.cost(&points, self.reference()?))
    }

    /// The first passing candidate supplies the timing normalization.
    fn reference(&self) -> Result<&[PointMeasurement], Exclusion> {
        self.state
            .evaluated
            .get(
                &self
                    .state
                    .anchor
                    .expect("a passing candidate establishes the timing anchor"),
            )
            .map(|defaults| defaults.points.as_slice())
            .ok_or_else(|| Exclusion::Measurement {
                point: String::new(),
                detail: "the defaults, the reference of every cost, were not measured".into(),
            })
    }
}

impl Evaluator for Live<'_, '_> {
    fn evaluate(&mut self, batch: &[usize]) -> Vec<Result<Cost, Exclusion>> {
        let specializations = batch
            .iter()
            .map(|index| specialization(self.statics, &self.space.values(*index)))
            .collect::<Vec<_>>();
        let began = Instant::now();
        let formed = self.formation.form_all(&specializations);
        let measuring = Instant::now();
        self.state.time.forming_seconds += (measuring - began).as_secs_f64();
        self.state.formed += specializations.len();

        let weighing = Weighing::of(self.points);
        let mut costs = Vec::with_capacity(batch.len());
        for (position, (index, kernel)) in batch.iter().zip(formed).enumerate() {
            // The first configuration is always answered; the rest while the
            // time lasts.
            if position > 0 && self.expired() {
                break;
            }
            costs.push(self.cost(*index, kernel, &weighing));
        }
        self.state.time.measuring_seconds += measuring.elapsed().as_secs_f64();
        costs
    }

    fn confirm(&mut self, finalists: &[usize]) -> Vec<Result<Cost, Exclusion>> {
        let began = Instant::now();
        // Every finalist placed at every point; one timing per distinct point
        // and key, all sampled round by round, so sampling alternates between
        // the finalists and finalists sharing a key share its measurement.
        let mut ids: Vec<(usize, PointKey)> = Vec::new();
        let mut timings = Vec::new();
        let keys = finalists
            .iter()
            .map(|index| {
                let Placed {
                    timings: placed,
                    keys,
                } = place(
                    &self.state.evaluated[index].kernel,
                    self.points,
                    &self.state.measurer.influence,
                    &self.space.values(*index),
                    &mut self.state.measurer.outputs,
                    self.state.confirmation.min_sample_seconds,
                )?;
                for (point, (timing, key)) in placed.into_iter().zip(&keys).enumerate() {
                    let id = (point, key.clone());
                    if !ids.contains(&id) {
                        ids.push(id);
                        timings.push(timing);
                    }
                }
                Ok(keys)
            })
            .collect::<Vec<Result<Vec<PointKey>, Exclusion>>>();
        let points = self.points;
        let mut failed = HashMap::new();
        while let Err(failure) = timing::sample(&mut timings, &self.state.confirmation) {
            let (point, key) = ids.remove(failure.point);
            timings.remove(failure.point);
            failed.insert(
                (point, key),
                measurement_failure(points, point, failure.error),
            );
        }
        let measured = ids
            .into_iter()
            .zip(&timings)
            .map(|((point, key), timing)| {
                let measurement =
                    point_measurement(&points[point].label, key.clone(), timing.measurement());
                ((point, key), measurement)
            })
            .collect::<HashMap<_, _>>();
        let default = self.space.default_index();
        let confirmed = keys
            .into_iter()
            .zip(finalists)
            .map(|(keys, index)| {
                let confirmed = keys?
                    .into_iter()
                    .enumerate()
                    .map(|(point, key)| {
                        let id = (point, key);
                        match failed.get(&id) {
                            Some(exclusion) => Err(exclusion.clone()),
                            None => Ok(measured[&id].clone()),
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                confirmed_measurement(*index == default, confirmed)
            })
            .collect::<Vec<_>>();
        self.state.time.measuring_seconds += began.elapsed().as_secs_f64();
        let weighing = Weighing::of(points);
        // Any successfully confirmed finalist can anchor timing normalization.
        let reference = confirmed
            .iter()
            .find_map(|result| result.as_ref().ok())
            .cloned();
        finalists
            .iter()
            .zip(confirmed)
            .map(|(index, measured)| {
                let measured = measured?;
                let reference = reference
                    .as_ref()
                    .expect("a measured finalist establishes the anchor");
                let cost = weighing.cost(&measured, reference);
                self.state
                    .evaluated
                    .get_mut(index)
                    .expect("finalists were evaluated")
                    .confirmed = measured;
                Ok(cost)
            })
            .collect()
    }

    fn expired(&self) -> bool {
        self.limits.is_some_and(|limits| {
            let now = Instant::now();
            limits.slice.is_some_and(|slice| now >= slice)
                || now + self.state.reserve(self.space.default_index()) + self.untimed
                    >= limits.until
        })
    }
}

/// Builds points' inputs from a [`PointSource`], timing it.
struct Inputs<'r, 'a> {
    source: &'r mut dyn PointSource<'a>,
    specs: &'r [PointSpec],
    seconds: f64,
    /// The last point's building time.
    last: f64,
}

impl<'a> Inputs<'_, 'a> {
    /// Point `index` built within `limit`; `None` when building it would not
    /// fit.
    fn build(
        &mut self,
        index: usize,
        limit: Duration,
    ) -> Result<Option<TuningPoint<'a>>, TuneError> {
        let started = Instant::now();
        let built = self.source.build(index, limit);
        self.last = started.elapsed().as_secs_f64();
        self.seconds += self.last;
        match built {
            Ok(inputs) => TuningPoint::new(&self.specs[index], inputs).map(Some),
            Err(PointUnavailable::Unaffordable) => Ok(None),
            Err(PointUnavailable::Failed(detail)) => Err(TuneError::Inputs {
                point: self.specs[index].label.clone(),
                detail,
            }),
        }
    }
}

/// The last admitted point, from which later points' times are predicted.
/// Only a point's invocations scale with its cost: building its inputs and
/// the floor a calibrated sample measures do not.
#[derive(Clone)]
struct Admitted {
    cost: f64,
    /// Building the point, running its reference, validating and measuring
    /// the defaults there.
    step: f64,
    /// Building the point's inputs.
    build: f64,
    /// The defaults' device time of one invocation there, when measured.
    kernel: Option<f64>,
    /// Argument sets in the point's rotation.
    rotation: usize,
}

/// Invocations of each argument set while admitting a point: the
/// reference, the validated invocation and a sampled pass.
const ADMISSION_INVOCATIONS: f64 = 3.;

impl Admitted {
    /// One invocation's device time at a point of `cost`.
    fn kernel_at(&self, cost: f64) -> Option<f64> {
        self.kernel
            .map(|kernel| kernel * cost / self.cost.max(f64::MIN_POSITIVE))
    }

    /// Admitting a point of `cost`: this point's time with its invocations
    /// at the other point's cost. Without a measurement of the defaults,
    /// this point's time scaled by cost.
    fn step_at(&self, cost: f64) -> f64 {
        match (self.kernel, self.kernel_at(cost)) {
            (Some(here), Some(there)) => {
                self.step + ADMISSION_INVOCATIONS * self.rotation as f64 * (there - here).max(0.)
            }
            _ => self.step * cost / self.cost.max(f64::MIN_POSITIVE),
        }
    }

    /// Validating a choice at a point of `cost`: building it, and a reference
    /// and a candidate invocation of each argument set.
    fn validation_at(&self, cost: f64) -> f64 {
        match self.kernel_at(cost) {
            Some(kernel) => self.build + 2. * self.rotation as f64 * kernel,
            None => self.step * cost / self.cost.max(f64::MIN_POSITIVE),
        }
    }
}

/// `instant` less `duration`, or now when that has passed.
fn before(instant: Instant, duration: Duration) -> Instant {
    instant
        .checked_sub(duration)
        .filter(|earlier| *earlier > Instant::now())
        .unwrap_or_else(Instant::now)
}

/// Tune one unit within its time: a [`UnitSearch`] started, refined until
/// what remains of the plan's allowance only covers its conclusion, and
/// concluded; or a survey of every admissible configuration.
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
        reference,
    } = request;
    let request = UnitRequest {
        device,
        module,
        entry,
        bindings,
        statics,
        cpu,
        validation,
        reference,
    };
    match strategy {
        Strategy::Search(plan) => {
            let mut unit = UnitSearch::open(request)?;
            let until = began + plan.allowance;
            let allowance = plan.allowance;
            // One call measures its defaults and form starts whatever its
            // allowance: there is no other unit whose time they could take.
            let unbounded = until + Duration::from_secs(24 * 60 * 60);
            unit.start(
                points,
                StartPlan {
                    required: plan.required,
                    admission: plan.admission,
                    settings: plan.settings,
                    min_sample_seconds: plan.min_sample_seconds,
                    start: plan.start,
                    forms: true,
                },
                unbounded,
            )?;
            unit.refine(until, until)?;
            unit.conclude(points, until, allowance)
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
    /// The policy every configuration in no error class is validated under,
    /// and the admitted error classes with their envelopes.
    pub validation: TuningPrecision,
    pub reference: TuningReference,
}

/// The defaults formed, and the validator of every point against the
/// reference.
struct Ground {
    kernel: Arc<NativePrepared>,
    validator: Validator,
}

/// A point admitted to the objective: built, its reference run, and the
/// defaults measured there.
struct Timed<'a> {
    point: PreparedPoint<'a>,
    defaults: Option<PointMeasurement>,
    admitted: Admitted,
}

/// How far a unit's search is.
enum Stage<'a> {
    /// Admitting points: each point's slot, by its position among the
    /// unit's points.
    Admitting(Vec<Option<Timed<'a>>>),
    /// The points every candidate must pass did not fit: the unit keeps its
    /// defaults. The settings of the search that found so; none when its
    /// census did.
    Unaffordable(Option<SearchSettings>),
    Searching(Box<Searching<'a>>),
}

/// A unit searching at its timed points.
struct Searching<'a> {
    ground: Ground,
    points: Vec<PreparedPoint<'a>>,
    /// Every point's record, the untimed points' weights folded into timed
    /// ones.
    records: Vec<PointRecord>,
    /// The points not timed: the choice is validated there.
    untimed: Vec<usize>,
    /// What validating a choice at the untimed points will take, each
    /// predicted from the last admitted point.
    untimed_reserve: Duration,
    plan: StartPlan,
    run: Run,
}

enum Run {
    Search(SearchRun),
    Factored(FactoredRun),
}

/// A local search of a unit's admissible configurations, as far as it got.
struct SearchRun {
    space: SearchSpace,
    start: Vec<usize>,
    lived: Lived,
    explored: search::Explored,
}

/// The timed points of a unit, weighted for the objective.
struct Timing<'a> {
    points: Vec<PreparedPoint<'a>>,
    records: Vec<PointRecord>,
    untimed: Vec<usize>,
    untimed_reserve: Duration,
}

/// One unit's search, advanced in steps so that a consumer tuning several
/// units decides which of them the next time goes to.
///
/// A census admits the points every candidate must pass and the unit's
/// census points, and measures the defaults there. Starting admits further points within the plan's
/// admission, then measures the defaults and every form's start whatever the
/// time: every form has a measurement before any is searched further.
/// Refining continues the search until a given instant, and may be repeated:
/// the steps together measure what one uninterrupted search measures, in the
/// same order. Concluding confirms the finalists found so far and validates
/// the choice at the points not timed, so a search concluded at any step
/// yields the best configuration confirmed by the same rules.
pub struct UnitSearch<'m, 'a> {
    device: Arc<DeviceInner>,
    module: &'m CheckedModule,
    entry: EntryId,
    bindings: ElementBindings,
    statics: NativeSpecialization,
    cpu: Option<&'static CpuNativeKernels>,
    reference: TuningReference,
    backend: seismic_lang::registry::BackendName,
    admitted: BTreeMap<String, seismic_lang::precision::ErrorEnvelope>,
    implementation: NativeImplementation,
    logical: Arc<LogicalEntry>,
    mutable: Vec<(usize, String)>,
    default: NativeSpecialization,
    unit: Unit,
    /// The unit's points, once a step was given their source.
    specs: Vec<PointSpec>,
    ground: Option<Ground>,
    /// Programs formed for the defaults.
    grounded: usize,
    /// Measures the defaults as points are admitted; a local search takes
    /// it, with those measurements.
    measurer: Option<Measurer>,
    /// Why the defaults could not be measured at an admitted point.
    defaults_failure: Option<Exclusion>,
    min_sample_seconds: f64,
    stage: Stage<'a>,
    building: f64,
    measuring: f64,
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
            validation: TuningPrecision {
                policy: validation,
                admitted,
            },
            reference,
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
        // Configurations of an error class that is not admitted are outside the
        // domain: never formed, timed or chosen.
        let implementation = implementation.admitting(|class| admitted.contains_key(class));
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
        // The default is the validation reference: it changes no numerics.
        if let Some(class) = implementation
            .error_classes_of(&default)
            .map_err(|error| TuneError::Declaration(error.to_string()))?
            .first()
        {
            return Err(TuneError::Declaration(format!(
                "the default configuration of `{entry_name}` is in error class `{class}`"
            )));
        }
        Ok(Self {
            unit: Unit {
                tuning_identity: device.tuning_identity(),
                entry: entry_name,
                backend: backend.as_str().to_owned(),
                validation,
                implementation_identity,
                parameters: entry_parameters(&implementation),
            },
            device: device.clone(),
            module,
            entry,
            bindings,
            statics,
            cpu,
            reference,
            backend,
            admitted,
            measurer: Some(Measurer::new(&implementation)),
            implementation,
            logical,
            mutable,
            default,
            specs: Vec::new(),
            ground: None,
            grounded: 0,
            defaults_failure: None,
            min_sample_seconds: 0.,
            stage: Stage::Admitting(Vec::new()),
            building: 0.,
            measuring: 0.,
        })
    }

    /// Learn the unit's points from `source`, at the first step given one.
    fn describe(&mut self, source: &dyn PointSource<'a>) -> Result<(), TuneError> {
        if self.specs.is_empty() {
            self.specs = source.points();
            if self.specs.is_empty() {
                return Err(TuneError::NoPoints);
            }
            self.stage = Stage::Admitting(self.specs.iter().map(|_| None).collect());
        }
        Ok(())
    }

    /// Form the defaults and the validator.
    fn form_ground(&self) -> Result<Ground, TuneError> {
        let kernel = NativePrepared::prepare_implementation(
            &self.device,
            self.module,
            self.entry,
            &self.logical,
            self.bindings.clone(),
            self.default.clone(),
            self.cpu,
            self.implementation.clone(),
        )
        .map_err(|error| {
            TuneError::DefaultUnusable(Exclusion::Formation(prepare_message(error)))
        })?;
        let validator = Validator::new(
            &self.device,
            self.module,
            self.entry,
            &self.bindings,
            &self.logical,
            &self.mutable,
            &self.unit.validation,
            &self.admitted,
            self.reference,
            &self.default,
            self.cpu,
        )?;
        Ok(Ground { kernel, validator })
    }

    /// Admit points not yet admitted, in ascending cost: build each, run its
    /// reference and, when `measure`, measure the defaults there. A required
    /// point may take what is left of `required`, any other what is left of
    /// `admission`, each predicted from the point before it; a `census`
    /// admits only required points and census points (the cheapest point
    /// when there is neither), and passes over the others. The first point
    /// that did not fit closes admission to the points after it that are
    /// not required. Whether every required point was admitted: when
    /// not, the unit keeps its defaults.
    fn admit(
        &mut self,
        source: &mut dyn PointSource<'a>,
        census: bool,
        measure: bool,
        required: Duration,
        admission: Duration,
        until: Option<Instant>,
    ) -> Result<bool, TuneError> {
        let began = Instant::now();
        let specs = self.specs.clone();
        let mut inputs = Inputs {
            source,
            specs: &specs,
            seconds: 0.,
            last: 0.,
        };
        let options = MeasureOptions {
            samples: 1,
            min_sample_seconds: self.min_sample_seconds,
        };
        let mut last: Option<Admitted> = None;
        let mut closed = false;
        let mut affordable = true;
        for (index, spec) in specs.iter().enumerate() {
            let Stage::Admitting(slots) = &self.stage else {
                unreachable!("points are admitted before the search starts");
            };
            if let Some(timed) = &slots[index] {
                last = Some(timed.admitted.clone());
                continue;
            }
            let available =
                if spec.required { required } else { admission }.saturating_sub(began.elapsed());
            let predicted_fits = last
                .as_ref()
                .is_none_or(|last| last.step_at(spec.cost) <= available.as_secs_f64());
            // The cheapest point first: the defaults are formed against it.
            let first = index == 0 && self.ground.is_none();
            // Past the step's hard limit nothing more is admitted.
            let late = until.is_some_and(|until| Instant::now() >= until);
            // A census passes over the points it does not measure: they are
            // admitted by the start, and close nothing.
            if census && !first && !spec.required && !spec.census {
                continue;
            }
            let attempted = !late
                && (first
                    || ((spec.required || !closed) && predicted_fits && !available.is_zero()));
            let point = if attempted {
                inputs.build(index, available)?
            } else {
                None
            };
            let built = if attempted { inputs.last } else { 0. };
            let Some(point) = point else {
                if spec.required || first {
                    affordable = false;
                    break;
                }
                closed = true;
                continue;
            };
            if self.ground.is_none() {
                self.ground = Some(self.form_ground()?);
                self.grounded = if self.implementation.launch_scoped() {
                    self.implementation.launches.len()
                } else {
                    1
                };
            }
            let ground = self
                .ground
                .as_ref()
                .expect("the defaults were formed above");
            let step = Instant::now();
            // Serving runs the defaults at every point, so one they cannot launch
            // at is an error.
            for args in &point.rotation {
                ground.kernel.shape(&args.values()).map_err(|error| {
                    TuneError::DefaultUnusable(Exclusion::Execution(format!(
                        "at point `{}`: {error}",
                        point.label
                    )))
                })?;
            }
            let point = ground.validator.point(point)?;
            point.ensure_reference()?;
            let mut defaults = None;
            if measure && self.defaults_failure.is_none() {
                let started = Instant::now();
                match self
                    .measurer
                    .as_mut()
                    .expect("the measurer serves admission until a search takes it")
                    .measure(
                        &ground.kernel,
                        std::slice::from_ref(&point),
                        self.default.params(),
                        &options,
                    ) {
                    Ok(mut at) => defaults = Some(at.remove(0)),
                    Err(exclusion) => self.defaults_failure = Some(exclusion),
                }
                self.measuring += started.elapsed().as_secs_f64();
            }
            let admitted = Admitted {
                cost: spec.cost,
                step: built + step.elapsed().as_secs_f64(),
                build: built,
                kernel: defaults
                    .as_ref()
                    .map(|measured: &PointMeasurement| measured.median_seconds),
                rotation: point.rotation.len(),
            };
            let Stage::Admitting(slots) = &mut self.stage else {
                unreachable!("points are admitted before the search starts");
            };
            slots[index] = Some(Timed {
                point,
                defaults,
                admitted: admitted.clone(),
            });
            last = Some(admitted);
        }
        self.building += inputs.seconds;
        Ok(affordable)
    }

    /// The admitted points as the objective's timed points: each reweighed
    /// with the weights of the points not admitted folded in.
    fn timed(&mut self) -> Timing<'a> {
        let Stage::Admitting(slots) = std::mem::replace(&mut self.stage, Stage::Unaffordable(None))
        else {
            unreachable!("points are admitted before the search starts");
        };
        let mut admitted = Vec::new();
        let mut points = Vec::new();
        let mut untimed = Vec::new();
        let mut last = None;
        for (index, slot) in slots.into_iter().enumerate() {
            match slot {
                Some(timed) => {
                    admitted.push(index);
                    points.push(timed.point);
                    last = Some(timed.admitted);
                }
                None => untimed.push(index),
            }
        }
        let records = folded(&self.specs, &admitted);
        for (point, index) in points.iter_mut().zip(&admitted) {
            point.reweigh(records[*index].weight);
        }
        let untimed_reserve = last.map_or(Duration::ZERO, |last| {
            Duration::from_secs_f64(
                untimed
                    .iter()
                    .map(|index| last.validation_at(self.specs[*index].cost))
                    .sum(),
            )
        });
        Timing {
            points,
            records,
            untimed,
            untimed_reserve,
        }
    }

    /// The defaults, chosen without searching.
    fn kept(&self, method: TuningMethod) -> TuningResult {
        let mut result = self.unit.result(
            Records {
                points: folded(&self.specs, &[]),
                evidence: Vec::new(),
                records: Vec::new(),
            },
            &self.default,
            method,
            TuningTime::default(),
        );
        result.time.building_seconds = self.building;
        result
    }

    /// The census of a unit still admitting points: the defaults measured at
    /// the points admitted, with their evidence there.
    fn censused(&self) -> Result<TuningResult, TuneError> {
        let (Stage::Admitting(slots), Some(ground)) = (&self.stage, &self.ground) else {
            return Ok(self.kept(TuningMethod::Census));
        };
        let admitted = slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| slot.as_ref().map(|_| index))
            .collect::<Vec<_>>();
        let defaults = match &self.defaults_failure {
            Some(exclusion) => Err(exclusion.clone()),
            None => Ok(slots
                .iter()
                .flatten()
                .filter_map(|timed| timed.defaults.clone())
                .collect()),
        };
        self.unit.census(
            slots.iter().flatten().map(|timed| &timed.point),
            folded(&self.specs, &admitted),
            &self.default,
            &ground.kernel,
            defaults,
            TuningTime {
                building_seconds: self.building,
                reference_seconds: ground.validator.reference_seconds(),
                measuring_seconds: self.measuring,
                ..TuningTime::default()
            },
        )
    }

    /// Measure the defaults at the points every candidate must pass and at
    /// the unit's census points (the cheapest point when there is neither)
    /// within the plan's limit: the fixed
    /// cost of searching the unit, and the defaults' time there. A result
    /// without configurations says the required points did not fit: the
    /// unit keeps its defaults.
    pub fn census(
        &mut self,
        source: &mut dyn PointSource<'a>,
        plan: CensusPlan,
    ) -> Result<TuningResult, TuneError> {
        self.describe(source)?;
        self.min_sample_seconds = plan.min_sample_seconds;
        let until = Instant::now().checked_add(plan.limit);
        if !self.admit(source, true, true, plan.limit, plan.limit, until)? {
            self.stage = Stage::Unaffordable(None);
        }
        self.censused()
    }

    /// Admit the further points the plan's admission covers, then measure
    /// the defaults and every form's start whatever the time (`until` bounds
    /// only the search for a passing configuration when the defaults fail).
    pub fn start(
        &mut self,
        source: &mut dyn PointSource<'a>,
        plan: StartPlan,
        until: Instant,
    ) -> Result<Standing, TuneError> {
        self.describe(source)?;
        if !matches!(self.stage, Stage::Admitting(_)) {
            return Ok(self.standing());
        }
        self.min_sample_seconds = plan.min_sample_seconds;
        if !self.admit(
            source,
            false,
            true,
            plan.required,
            plan.admission,
            Some(until),
        )? {
            self.stage = Stage::Unaffordable(Some(plan.settings));
            return Ok(self.standing());
        }
        // The hard limit was reached admitting points: the unit stays as its
        // census left it, on its defaults.
        if Instant::now() >= until {
            return Ok(Standing {
                cut: true,
                ..self.standing()
            });
        }
        let Timing {
            points,
            records,
            untimed,
            untimed_reserve,
        } = self.timed();
        let ground = self
            .ground
            .take()
            .expect("admitting a point forms the defaults");
        let run = if self.implementation.launch_scoped() {
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
            let formation = Formation {
                device: &self.device,
                module: self.module,
                entry: self.entry,
                logical: &self.logical,
                bindings: &self.bindings,
                cpu: self.cpu,
                implementation: &self.implementation,
            };
            Run::Factored(FactoredRun::open(
                &formation,
                &self.statics,
                &self.default,
                &points,
                &plan,
                before(until, untimed_reserve),
            )?)
        } else {
            let space = search_space(
                &self.device,
                &self.logical,
                &self.implementation,
                &self.statics,
                &self.default,
                &points,
            )?;
            let start = plan
                .start
                .iter()
                .filter_map(|values| space.index_of(values))
                .collect();
            Run::Search(SearchRun {
                space,
                start,
                lived: Lived {
                    measurer: self
                        .measurer
                        .take()
                        .expect("the measurer serves admission until a search takes it"),
                    search: MeasureOptions {
                        samples: plan.settings.samples,
                        min_sample_seconds: plan.min_sample_seconds,
                    },
                    confirmation: MeasureOptions {
                        samples: plan.settings.confirmation_samples,
                        min_sample_seconds: plan.min_sample_seconds,
                    },
                    confirmed: plan.settings.confirmed,
                    evaluated: HashMap::new(),
                    anchor: None,
                    formed: 0,
                    time: TuningTime::default(),
                },
                explored: search::Explored {
                    evaluated: Vec::new(),
                    stop: SearchStop::Expired,
                },
            })
        };
        self.stage = Stage::Searching(Box::new(Searching {
            ground,
            points,
            records,
            untimed,
            untimed_reserve,
            plan,
            run,
        }));
        let standing = self.explore(Limits { slice: None, until })?;
        Ok(Standing {
            // The guarantee was reached: the planned starts may not all
            // have been measured.
            cut: Instant::now() + self.reserve() >= until,
            ..standing
        })
    }

    /// Continue the search until `slice`, or until what remains before
    /// `until` only covers concluding it ([`Self::reserve`]).
    pub fn refine(&mut self, slice: Instant, until: Instant) -> Result<Standing, TuneError> {
        self.explore(Limits {
            slice: Some(slice),
            until,
        })
    }

    /// One step of the search: the starts without a slice, else whatever
    /// the search asks for next within the limits.
    fn explore(&mut self, limits: Limits) -> Result<Standing, TuneError> {
        if let Stage::Searching(searching) = &mut self.stage {
            let formation = Formation {
                device: &self.device,
                module: self.module,
                entry: self.entry,
                logical: &self.logical,
                bindings: &self.bindings,
                cpu: self.cpu,
                implementation: &self.implementation,
            };
            let Searching {
                points,
                plan,
                run,
                untimed_reserve,
                ..
            } = &mut **searching;
            match run {
                Run::Search(run) => {
                    let mut live = Live {
                        formation: &formation,
                        space: &run.space,
                        statics: &self.statics,
                        points,
                        state: &mut run.lived,
                        limits: Some(limits),
                        untimed: *untimed_reserve,
                    };
                    let explored = search::explore(
                        &run.space,
                        &run.start,
                        &plan.settings,
                        &run.explored.evaluated,
                        match limits.slice {
                            Some(_) => search::Reach::End,
                            None if plan.forms => search::Reach::Starts,
                            None => search::Reach::Defaults,
                        },
                        &mut live,
                    );
                    run.explored = explored;
                }
                Run::Factored(run) => {
                    let reserve = run.reserve(&plan.settings) + *untimed_reserve;
                    let forms = plan.forms;
                    // The form starts are measured whatever the slice, unless
                    // the start was planned without them; anything else
                    // within the slice; nothing once the time left only
                    // covers the conclusion.
                    let expired = move |start: bool| {
                        let now = Instant::now();
                        if now + reserve >= limits.until {
                            true
                        } else if start {
                            limits.slice.is_none() && !forms
                        } else {
                            limits.slice.is_none_or(|slice| now >= slice)
                        }
                    };
                    run.explore(&formation, &self.statics, &self.default, points, plan, &expired)?;
                }
            }
        }
        Ok(self.standing())
    }

    /// Where the search stands.
    pub fn standing(&self) -> Standing {
        match &self.stage {
            Stage::Admitting(_) | Stage::Unaffordable(_) => Standing {
                finished: matches!(self.stage, Stage::Unaffordable(_)),
                cost: 1.,
                measured: 0,
                admissible: 0,
                programs: self.grounded,
                declared_programs: self.grounded,
                forming_seconds: 0.,
                cut: false,
            },
            Stage::Searching(searching) => {
                let standing = match &searching.run {
                    Run::Search(run) => {
                        let default = run.space.default_index();
                        let totals = run
                            .explored
                            .evaluated
                            .iter()
                            .filter_map(|(index, result)| {
                                result.as_ref().ok().map(|cost| (*index, cost.total()))
                            })
                            .collect::<Vec<_>>();
                        let defaults = totals
                            .iter()
                            .find(|(index, _)| *index == default)
                            .map(|(_, total)| *total);
                        let best = totals.iter().map(|(_, total)| *total).fold(f64::INFINITY, f64::min);
                        Standing {
                            finished: run.explored.stop != SearchStop::Expired,
                            cost: defaults.map_or(1., |defaults| (best / defaults).min(1.)),
                            measured: run.explored.evaluated.len(),
                            admissible: run.space.len(),
                            programs: run.lived.formed,
                            declared_programs: run.space.len(),
                            forming_seconds: run.lived.time.forming_seconds,
                            cut: false,
                        }
                    }
                    Run::Factored(run) => run.standing(&searching.points),
                };
                Standing {
                    programs: standing.programs + self.grounded,
                    ..standing
                }
            }
        }
    }

    /// What concluding the search will take: confirming the finalists found
    /// so far and validating the choice at the points not timed.
    pub fn reserve(&self) -> Duration {
        match &self.stage {
            Stage::Admitting(_) | Stage::Unaffordable(_) => Duration::ZERO,
            Stage::Searching(searching) => {
                searching.untimed_reserve
                    + match &searching.run {
                        Run::Search(run) => run.lived.reserve(run.space.default_index()),
                        Run::Factored(run) => run.reserve(&searching.plan.settings),
                    }
            }
        }
    }

    /// Conclude the search with what it has reached: confirm its finalists,
    /// choose by the confirmed costs, and validate a choice other than the
    /// defaults at the points not timed, building those within what is left
    /// before `until`. `allowance` is the time the unit was given, for the
    /// record. A unit that never started returns its census.
    pub fn conclude(
        mut self,
        source: &mut dyn PointSource<'a>,
        until: Instant,
        allowance: Duration,
    ) -> Result<TuningResult, TuneError> {
        let searching = match std::mem::replace(&mut self.stage, Stage::Unaffordable(None)) {
            Stage::Admitting(slots) => {
                self.stage = Stage::Admitting(slots);
                return self.censused();
            }
            Stage::Unaffordable(None) => return Ok(self.kept(TuningMethod::Census)),
            Stage::Unaffordable(Some(settings)) => {
                return Ok(self.kept(TuningMethod::Search {
                    allowance_seconds: allowance.as_secs_f64(),
                    settings,
                    stop: SearchStop::Unaffordable,
                }))
            }
            Stage::Searching(searching) => *searching,
        };
        let Searching {
            ground,
            points,
            records,
            untimed,
            untimed_reserve,
            plan,
            run,
        } = searching;
        let formation = Formation {
            device: &self.device,
            module: self.module,
            entry: self.entry,
            logical: &self.logical,
            bindings: &self.bindings,
            cpu: self.cpu,
            implementation: &self.implementation,
        };
        let mut result = match run {
            Run::Search(mut run) => {
                let mut live = Live {
                    formation: &formation,
                    space: &run.space,
                    statics: &self.statics,
                    points: &points,
                    state: &mut run.lived,
                    limits: None,
                    untimed: Duration::ZERO,
                };
                let trace = search::rank(&run.space, &plan.settings, run.explored, &mut live);
                let (configurations, overall, method, time) = searched(
                    &run.space,
                    &self.statics,
                    &points,
                    plan.settings.clone(),
                    allowance,
                    run.lived,
                    trace,
                )?;
                self.unit.result(configurations, &overall, method, time)
            }
            Run::Factored(run) => run.conclude(
                &formation,
                &self.statics,
                &self.default,
                &points,
                &plan,
                before(until, untimed_reserve),
                allowance,
                &self.unit,
            )?,
        };
        result.points = records;
        let mut inputs = Inputs {
            source,
            specs: &self.specs,
            seconds: 0.,
            last: 0.,
        };
        if !untimed.is_empty() {
            let trusted = (self.reference == TuningReference::NativeDefault)
                .then(|| Configuration::of(&self.default));
            if trusted.as_ref() != Some(&result.overall) {
                let mut built = Vec::with_capacity(untimed.len());
                for index in &untimed {
                    let limit = until.saturating_duration_since(Instant::now());
                    match inputs.build(*index, limit)? {
                        Some(point) => built.push(ground.validator.point(point)?),
                        None => break,
                    }
                }
                if built.len() == untimed.len() {
                    validate_everywhere(
                        &formation,
                        &self.default,
                        &points,
                        &built,
                        plan.min_sample_seconds,
                        &mut result,
                    )?;
                } else {
                    give_way(
                        &mut result,
                        &self.default,
                        &points,
                        Exclusion::Execution(
                            "validating it where the search did not time it would not fit the unit's time"
                                .into(),
                        ),
                    )?;
                }
            }
        }
        result.time.building_seconds += self.building + inputs.seconds;
        result.time.measuring_seconds += self.measuring;
        result.time.reference_seconds += ground.validator.reference_seconds();
        Ok(result)
    }

    /// Measure and validate every admissible configuration at every point
    /// (development).
    pub fn survey(
        mut self,
        source: &mut dyn PointSource<'a>,
        plan: SurveyPlan,
    ) -> Result<TuningResult, TuneError> {
        self.describe(source)?;
        self.min_sample_seconds = plan.min_sample_seconds;
        // A survey builds and times every point whatever it costs.
        if !self.admit(source, false, false, Duration::MAX, Duration::MAX, None)? {
            return Err(TuneError::Inputs {
                point: self.specs[0].label.clone(),
                detail: "a survey's point was refused for its time".into(),
            });
        }
        let Timing {
            points, records, ..
        } = self.timed();
        let ground = self
            .ground
            .take()
            .expect("admitting a point forms the defaults");
        if self.implementation.launch_scoped() {
            return Err(TuneError::Domain(
                "launch-scoped surveys require the factored search".into(),
            ));
        }
        let declared = search_parameters(&self.implementation);
        let space = SearchSpace::new(
            &declared,
            &self
                .implementation
                .admissible(&self.statics)
                .map_err(|error| TuneError::Declaration(error.to_string()))?
                .iter()
                .map(|specialization| specialization.params().clone())
                .collect::<Vec<_>>(),
        )
        .map_err(TuneError::Space)?;
        let formation = Formation {
            device: &self.device,
            module: self.module,
            entry: self.entry,
            logical: &self.logical,
            bindings: &self.bindings,
            cpu: self.cpu,
            implementation: &self.implementation,
        };
        let tuned = Tuned {
            formation: &formation,
            space: &space,
            statics: &self.statics,
        };
        let (configurations, overall, method, time) = tuned.survey(&points, plan)?;
        let mut result = self.unit.result(configurations, &overall, method, time);
        result.points = records;
        result.time.building_seconds += self.building;
        result.time.measuring_seconds += self.measuring;
        result.time.reference_seconds += ground.validator.reference_seconds();
        Ok(result)
    }
}

/// What every result of one unit records about it.
struct Unit {
    tuning_identity: String,
    entry: String,
    backend: String,
    validation: PrecisionPolicy,
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
            tuning_identity: self.tuning_identity.clone(),
            entry: self.entry.clone(),
            backend: self.backend.clone(),
            points: configurations.points,
            validation: self.validation.clone(),
            numerical_evidence: configurations.evidence,
            implementation_identity: self.implementation_identity.clone(),
            parameters: self.parameters.clone(),
            configurations: configurations.records,
            overall: Configuration::of(overall),
            method,
            time,
        }
    }

    /// A census: the defaults measured at the points it admitted, with their
    /// evidence there.
    fn census<'p, 'a: 'p>(
        &self,
        points: impl IntoIterator<Item = &'p PreparedPoint<'a>>,
        records: Vec<PointRecord>,
        default: &NativeSpecialization,
        kernel: &NativePrepared,
        defaults: Result<Vec<PointMeasurement>, Exclusion>,
        time: TuningTime,
    ) -> Result<TuningResult, TuneError> {
        let configuration = Configuration::of(default);
        let measured = defaults.is_ok();
        let configurations = vec![ConfigurationRecord {
            configuration: configuration.clone(),
            outcome: match defaults {
                Ok(points) => Outcome::Measured {
                    artifact: kernel.artifact().0.clone(),
                    points,
                    confirmed: Vec::new(),
                    validated: true,
                },
                Err(exclusion) => Outcome::Excluded(exclusion),
            },
        }];
        let evidence = if measured {
            winner_evidence(points, &configurations, &configuration)?
        } else {
            Vec::new()
        };
        Ok(self.result(
            Records {
                points: records,
                evidence,
                records: configurations,
            },
            default,
            TuningMethod::Census,
            time,
        ))
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

fn search_parameters(implementation: &NativeImplementation) -> Vec<SearchParameter> {
    implementation
        .params
        .iter()
        .map(|parameter| SearchParameter {
            name: parameter.name.clone(),
            values: parameter.values.clone(),
            form: parameter.form,
        })
        .collect()
}

/// The search's space at `points`: a parameter read only by launches
/// inactive at every point keeps its default, since nothing measures it and
/// no validation runs its code.
fn search_space(
    device: &Arc<DeviceInner>,
    logical: &Arc<LogicalEntry>,
    implementation: &NativeImplementation,
    statics: &NativeSpecialization,
    default: &NativeSpecialization,
    points: &[PreparedPoint<'_>],
) -> Result<SearchSpace, TuneError> {
    let unserved = unserved_parameters(
        implementation,
        statics,
        &point_shapes(device, logical, points)?,
    )?;
    let admissible = implementation
        .admissible(statics)
        .map_err(|error| TuneError::Declaration(error.to_string()))?
        .into_iter()
        .filter(|candidate| {
            unserved
                .iter()
                .all(|name| candidate.param(name) == default.param(name))
        })
        .map(|specialization| specialization.params().clone())
        .collect::<Vec<_>>();
    SearchSpace::new(&search_parameters(implementation), &admissible).map_err(TuneError::Space)
}

/// Validate a search's choice at the points it was not timed at, so every
/// choice passes at every point. A choice that fails there gives way to the
/// defaults, which must pass there too unless they are themselves the
/// reference (a `NativeDefault` reference, against which they pass by
/// construction).
fn validate_everywhere(
    formation: &Formation<'_>,
    default: &NativeSpecialization,
    weighted: &[PreparedPoint<'_>],
    unweighted: &[PreparedPoint<'_>],
    min_sample_seconds: f64,
    result: &mut TuningResult,
) -> Result<(), TuneError> {
    let Some(first) = unweighted.first() else {
        return Ok(());
    };
    let defaults = Configuration::of(default);
    let trusted = first.reference_kind() == TuningReference::NativeDefault;
    let began = Instant::now();
    let mut failures = Vec::new();
    let mut chosen = None;
    for candidate in std::iter::once(result.overall.clone())
        .chain((result.overall != defaults).then(|| defaults.clone()))
    {
        if trusted && candidate == defaults {
            chosen = Some((candidate, Vec::new()));
            break;
        }
        match validate_at(formation, &candidate, unweighted, min_sample_seconds) {
            Ok(evidence) => {
                chosen = Some((candidate, evidence));
                break;
            }
            Err(exclusion) => {
                exclude(result, &candidate, exclusion.clone());
                failures.push(exclusion);
            }
        }
    }
    result.time.validating_seconds += began.elapsed().as_secs_f64();
    let (chosen, evidence) = chosen.ok_or(TuneError::NoValidatedCandidate(failures))?;
    if chosen != result.overall {
        result.overall = chosen;
        result.numerical_evidence =
            winner_evidence(weighted, &result.configurations, &result.overall)?;
    }
    result.numerical_evidence.extend(evidence);
    Ok(())
}

/// Exclude the choice for `exclusion`, giving way to the defaults.
fn give_way(
    result: &mut TuningResult,
    default: &NativeSpecialization,
    weighted: &[PreparedPoint<'_>],
    exclusion: Exclusion,
) -> Result<(), TuneError> {
    let overall = result.overall.clone();
    exclude(result, &overall, exclusion);
    result.overall = Configuration::of(default);
    result.numerical_evidence = winner_evidence(weighted, &result.configurations, &result.overall)?;
    Ok(())
}

/// Record `configuration` as excluded for `exclusion`.
fn exclude(result: &mut TuningResult, configuration: &Configuration, exclusion: Exclusion) {
    if let Some(record) = result
        .configurations
        .iter_mut()
        .find(|record| record.configuration == *configuration)
    {
        record.outcome = Outcome::Excluded(exclusion);
    }
}

/// Validate `configuration` at `points`; its evidence at each.
fn validate_at(
    formation: &Formation<'_>,
    configuration: &Configuration,
    points: &[PreparedPoint<'_>],
    min_sample_seconds: f64,
) -> Result<Vec<NumericalEvidence>, Exclusion> {
    let kernel = formation
        .form_all(&[configuration.specialization()])
        .remove(0)?;
    points
        .iter()
        .map(|point| {
            point
                .ensure_reference()
                .map_err(|error| Exclusion::Execution(error.to_string()))?;
            let mut timing = PointTiming::new(&kernel, point.rotation.clone())
                .map_err(|error| Exclusion::Execution(error.to_string()))?;
            point.validate(&mut timing, min_sample_seconds)?;
            point
                .evidence(&kernel.artifact().0)
                .ok_or_else(|| Exclusion::Execution("validation recorded no evidence".into()))
        })
        .collect()
}

/// How many candidates of a sweep are formed and measured together: a step
/// of the search can end between chunks, and only a chunk's programs are
/// formed before it is measured.
const SWEEP_CHUNK: usize = 16;
/// How many candidates after a chunk have their programs formed with it
/// when the chunk needs a compile: three chunks more, so that a pass costs a
/// quarter of the compiles while at most three chunks' programs are formed
/// and not measured when the time ends.
const SWEEP_AHEAD: usize = 3 * SWEEP_CHUNK;

/// The launch programs a factored search has formed. A launch's variants
/// share one source, so those about to be measured are formed in one compile
/// and the configurations assembled from them form nothing; a variant is
/// formed only when a configuration about to be measured runs it.
#[derive(Default)]
struct LaunchPrograms {
    /// Per launch, the program last formed for it and the variants it holds.
    held: BTreeMap<usize, (BTreeSet<plan::LaunchVariant>, Rc<HeldPrograms>)>,
    /// Every launch variant formed so far.
    formed: BTreeSet<(usize, plan::LaunchVariant)>,
}

impl LaunchPrograms {
    /// Form `specializations`: first, in one compile per launch, the
    /// variants they run that no held program has; then each of them from
    /// the held programs. A compile of a launch's source costs much the
    /// same whatever it instantiates, so one that must happen also forms
    /// the variants `ahead`, the configurations measured next, run of that
    /// launch. `default` is the unit's defaults.
    fn form(
        &mut self,
        formation: &Formation<'_>,
        default: &NativeSpecialization,
        specializations: &[NativeSpecialization],
        ahead: &[NativeSpecialization],
    ) -> Result<Vec<Result<Arc<NativePrepared>, Exclusion>>, TuneError> {
        let implementation = formation.implementation;
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
                            group_size: implementation
                                .static_group_size(&specialization, ordinal),
                        });
                }
            }
            variants
        };
        let mut next = variants(ahead);
        let sources = variants(specializations)
            .into_iter()
            .filter(|(ordinal, variants)| {
                self.held
                    .get(ordinal)
                    .is_none_or(|(held, _)| !variants.is_subset(held))
            })
            .map(|(ordinal, mut variants)| {
                variants.extend(next.remove(&ordinal).unwrap_or_default());
                // The program replaces the launch's held one: it keeps what
                // that one held, so a configuration formed again later (a
                // finalist, another group's candidate) forms nothing.
                if let Some((held, _)) = self.held.get(&ordinal) {
                    variants.extend(held.iter().cloned());
                }
                plan::LaunchSource {
                    ordinal,
                    variants: variants.into_iter().collect(),
                    truncated: false,
                }
            })
            .collect::<Vec<_>>();
        if !sources.is_empty() {
            let held = Rc::new(
                NativePrepared::hold_launch_variants(
                    formation.device,
                    formation.module,
                    formation.entry,
                    formation.bindings,
                    default,
                    implementation,
                    &sources,
                )
                .map_err(|error| {
                    TuneError::DefaultUnusable(Exclusion::Formation(prepare_message(error)))
                })?,
            );
            for source in sources {
                self.formed.extend(
                    source
                        .variants
                        .iter()
                        .cloned()
                        .map(|variant| (source.ordinal, variant)),
                );
                self.held.insert(
                    source.ordinal,
                    (source.variants.into_iter().collect(), held.clone()),
                );
            }
        }
        Ok(specializations
            .iter()
            .map(|specialization| {
                NativePrepared::prepare_implementation(
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
            })
            .collect())
    }

    /// Form one specialization.
    fn form_one(
        &mut self,
        formation: &Formation<'_>,
        default: &NativeSpecialization,
        specialization: &NativeSpecialization,
    ) -> Result<Result<Arc<NativePrepared>, Exclusion>, TuneError> {
        Ok(self
            .form(formation, default, std::slice::from_ref(specialization), &[])?
            .remove(0))
    }
}

/// One boundary assignment as the sweeps left it: its score, its values,
/// each group's best candidate and each group's ranking. A boundary no
/// sweep reached stands at its base: the seed's group choices under it.
type Boundary = (f64, Vec<u64>, Vec<usize>, Vec<Vec<(usize, f64)>>);

/// Nearby admissible changes of each coordinate, keeping other coordinates
/// at the seed whenever the declaration permits. Coupled restrictions can
/// require more than one coordinate to change; those are still real starts.
fn coordinate_starts(candidates: &[Vec<u64>], seed: usize) -> Vec<usize> {
    let defaults = &candidates[seed];
    let mut starts = Vec::new();
    for (coordinate, &default) in defaults.iter().enumerate() {
        let lower = candidates
            .iter()
            .map(|values| values[coordinate])
            .filter(|&value| value < default)
            .max();
        let upper = candidates
            .iter()
            .map(|values| values[coordinate])
            .filter(|&value| value > default)
            .min();
        for value in [lower, upper].into_iter().flatten() {
            let candidate = candidates
                .iter()
                .enumerate()
                .filter(|(_, values)| values[coordinate] == value)
                .min_by_key(|(index, values)| {
                    (
                        values
                            .iter()
                            .zip(defaults)
                            .filter(|(value, default)| value != default)
                            .count(),
                        *index,
                    )
                })
                .map(|(index, _)| index)
                .expect("coordinate value came from a candidate");
            if !starts.contains(&candidate) {
                starts.push(candidate);
            }
        }
    }
    starts
}

/// Give each working group one nearby change before giving any group its
/// second. Declaration order cannot let a large first group consume the
/// complete search slice before later active launches have a rival.
fn coverage_order(
    groups: &[plan::Group],
    defaults: &[usize],
    active: &[usize],
) -> Vec<(usize, usize)> {
    let starts = groups
        .iter()
        .zip(defaults)
        .map(|(group, &default)| {
            if group.launches.iter().any(|launch| active.contains(launch)) {
                coordinate_starts(&group.candidates, default)
            } else {
                Vec::new()
            }
        })
        .collect::<Vec<_>>();
    (0..starts.iter().map(Vec::len).max().unwrap_or(0))
        .flat_map(|round| {
            starts
                .iter()
                .enumerate()
                .filter_map(move |(group, starts)| {
                    starts.get(round).map(|&candidate| (group, candidate))
                })
        })
        .collect()
}

fn rank_group(ranking: &mut [(usize, f64)], seed: usize) {
    ranking.sort_by(|left, right| {
        left.1.total_cmp(&right.1)
            .then_with(|| (left.0 != seed).cmp(&(right.0 != seed)))
            .then_with(|| left.0.cmp(&right.0))
    });
}

/// Replay all existing evidence before considering more device work. A
/// missing form start or an expired budget cannot hide a measured candidate
/// from a later pass of this group.
fn replay_group(
    seed: usize,
    seed_cost: f64,
    costs: impl IntoIterator<Item = Option<f64>>,
) -> Vec<(usize, f64)> {
    std::iter::once((seed, seed_cost))
        .chain(
            costs
                .into_iter()
                .enumerate()
                .filter_map(|(candidate, cost)| {
                    (candidate != seed)
                        .then_some(cost)
                        .flatten()
                        .map(|cost| (candidate, cost))
                }),
        )
        .collect()
}

/// A factored search of one unit's independent launch groups, as far as it
/// got.
struct FactoredRun {
    partition: Rc<plan::TuningPartition>,
    programs: LaunchPrograms,
    /// Every configuration placed in this run writes its results here.
    outputs: OutputPool,
    records: Vec<ConfigurationRecord>,
    /// The reference of every cost: the defaults, or when they fail the
    /// first configuration that passes.
    seed: NativeSpecialization,
    seed_kernel: Arc<NativePrepared>,
    reference: Vec<PointMeasurement>,
    /// Sweep samples by point and key: a point with the same active launches
    /// and parameters runs the same work under every candidate and boundary.
    sweep_cache: BTreeMap<(usize, PointKey), PointMeasurement>,
    /// Every configuration a sweep has measured, with its measurement; none
    /// for one that could not be formed or measured. A step of the search
    /// replays the sweeps over these and measures only what is missing.
    swept: HashMap<Configuration, Option<Vec<PointMeasurement>>>,
    default_choices: Vec<usize>,
    default_boundary: Vec<u64>,
    /// Configurations the sweeps can reach.
    reachable: usize,
    /// Every boundary whose base was measured, best first by what was
    /// measured under it: swept as far as the sweeps got, else at its base.
    boundaries: Vec<Boundary>,
    /// Whether every candidate of every boundary was measured.
    complete: bool,
    time: TuningTime,
}

impl FactoredRun {
    /// Form and measure the seed at `points`. `window` bounds the search
    /// for a passing seed when the defaults fail.
    fn open(
        formation: &Formation<'_>,
        statics: &NativeSpecialization,
        default: &NativeSpecialization,
        points: &[PreparedPoint<'_>],
        search: &StartPlan,
        window: Instant,
    ) -> Result<Self, TuneError> {
        let implementation = formation.implementation;
        let deadline = window;
        let mut time = TuningTime::default();
        let shapes = point_shapes(formation.device, formation.logical, points)?;
        let partition = plan::partition(implementation, statics, &shapes)
            .map_err(|error| TuneError::Declaration(format!("factored native plan: {error:?}")))?;
        for group in partition.groups.iter().filter(|group| group.truncated) {
            eprintln!(
                "seismic: tuning group {:?} candidate set capped at {}",
                group.launches,
                plan::MAX_GROUP_CANDIDATES
            );
        }
        for source in partition.sources.iter().filter(|source| source.truncated) {
            eprintln!(
                "seismic: launch {} variant set capped at {}",
                source.ordinal,
                plan::MAX_GROUP_CANDIDATES
            );
        }
        let measuring = MeasureOptions {
            samples: search.settings.samples,
            min_sample_seconds: search.min_sample_seconds,
        };
        let mut programs = LaunchPrograms::default();
        let mut outputs = OutputPool::default();
        let mut records = Vec::new();
        let mut swept = HashMap::new();
        let mut measure_seed = |specialization: &NativeSpecialization,
                                outputs: &mut OutputPool|
         -> Result<_, TuneError> {
            let kernel = match programs.form_one(formation, default, specialization)? {
                Ok(kernel) => kernel,
                Err(exclusion) => return Ok(Err(exclusion)),
            };
            Ok(measure_factored(
                implementation,
                &partition.boundary,
                &kernel,
                specialization,
                points,
                &measuring,
                None,
                None,
                None,
                outputs,
            )
            .map(|measured| (kernel, measured)))
        };
        let began = Instant::now();
    let mut seed = match measure_seed(default, &mut outputs)? {
        Ok((kernel, measured)) => Some((default.clone(), kernel, measured)),
        Err(exclusion) => {
            swept.insert(Configuration::of(default), None);
            records.push(ConfigurationRecord {
                configuration: Configuration::of(default),
                outcome: Outcome::Excluded(exclusion),
            });
            None
        }
    };
    // No group is an eligible baseline until the complete invocation passes.
    // Enumerate alternative seeds lazily; never materialize the Cartesian product.
    if seed.is_none() {
        'seeds: for boundary in boundary_assignments(&partition, implementation) {
            let mut choices = vec![0; partition.groups.len()];
            loop {
                if Instant::now() >= deadline {
                    break 'seeds;
                }
                if let Ok(candidate) =
                    partition.assemble(implementation, statics, &choices, &boundary)
                {
                    if candidate != *default {
                        match measure_seed(&candidate, &mut outputs)? {
                            Ok((kernel, measured)) => {
                                seed = Some((candidate, kernel, measured));
                                break 'seeds;
                            }
                            Err(exclusion) => records.push(ConfigurationRecord {
                                configuration: Configuration::of(&candidate),
                                outcome: Outcome::Excluded(exclusion),
                            }),
                        }
                    }
                }
                let mut advanced = false;
                for group in (0..choices.len()).rev() {
                    choices[group] += 1;
                    if choices[group] < partition.groups[group].candidates.len() {
                        advanced = true;
                        break;
                    }
                    choices[group] = 0;
                }
                if !advanced {
                    break;
                }
            }
        }
    }
    let (seed, seed_kernel, reference) =
        seed.ok_or_else(|| TuneError::NoValidatedCandidate(excluded(&records)))?;
    let default = &seed;
    time.measuring_seconds += began.elapsed().as_secs_f64();
    // A point with the same active launches and parameters runs the same work
    // under every boundary assignment. Reuse its sweep sample across those
    // assignments; confirmation still takes fresh samples.
    let sweep_cache = reference
        .iter()
        .enumerate()
        .map(|(point, measurement)| ((point, measurement.key.clone()), measurement.clone()))
        .collect::<BTreeMap<_, _>>();
    let default_choices = partition
        .groups
        .iter()
        .map(|group| {
            group
                .candidates
                .iter()
                .position(|candidate| {
                    group
                        .parameters
                        .iter()
                        .map(|address| address.value(default))
                        .eq(candidate.iter().copied())
                })
                .expect("admissible defaults belong to every group")
        })
        .collect::<Vec<_>>();
    let default_boundary = partition
        .boundary
        .iter()
        .map(|address| address.value(default))
        .collect::<Vec<_>>();
    records.push(ConfigurationRecord {
        configuration: Configuration::of(default),
        outcome: Outcome::Measured {
            artifact: seed_kernel.artifact().0.clone(),
            points: reference.clone(),
            confirmed: Vec::new(),
            validated: true,
        },
    });
    let reachable = boundary_assignments(&partition, implementation).len()
        * (1 + partition
            .groups
            .iter()
            .map(|group| group.candidates.len() - 1)
            .sum::<usize>());
    Ok(Self {
        partition: Rc::new(partition),
        programs,
        outputs,
        records,
        seed,
        seed_kernel,
        reference,
        sweep_cache,
        swept,
        default_choices,
        default_boundary,
        reachable,
        boundaries: Vec::new(),
        complete: false,
        time,
    })
    }

    /// What confirming the choice against the seed will take.
    fn reserve(&self, settings: &SearchSettings) -> Duration {
        confirmation_time(
            &self.reference,
            settings.confirmed + 1,
            settings.confirmation_samples,
        )
    }

    fn standing(&self, points: &[PreparedPoint<'_>]) -> Standing {
        let seed = Weighing::of(points)
            .cost(&self.reference, &self.reference)
            .total();
        let best = self
            .boundaries
            .iter()
            .map(|(score, ..)| *score)
            .fold(seed, f64::min);
        Standing {
            finished: self.complete,
            cost: best / seed,
            measured: 1 + self.swept.values().flatten().count(),
            admissible: self.reachable,
            programs: self.programs.formed.len(),
            declared_programs: self
                .partition
                .sources
                .iter()
                .map(|source| source.variants.len())
                .sum(),
            forming_seconds: self.time.forming_seconds,
            cut: false,
        }
    }

    /// Form and measure the base of a boundary: the seed's group choices
    /// under it. None when it cannot be formed or measured.
    fn base(
        &mut self,
        formation: &Formation<'_>,
        default: &NativeSpecialization,
        base: &NativeSpecialization,
        points: &[PreparedPoint<'_>],
        measuring: &MeasureOptions,
    ) -> Result<Option<Vec<PointMeasurement>>, TuneError> {
        let configuration = Configuration::of(base);
        let began = Instant::now();
        let formed = self.programs.form_one(formation, default, base)?;
        self.time.forming_seconds += began.elapsed().as_secs_f64();
        let began = Instant::now();
        let measured = formed.and_then(|kernel| {
            measure_factored(
                formation.implementation,
                &self.partition.boundary,
                &kernel,
                base,
                points,
                measuring,
                None,
                Some(&self.reference),
                Some(&mut self.sweep_cache),
                &mut self.outputs,
            )
            .map(|measured| (kernel, measured))
        });
        self.time.measuring_seconds += began.elapsed().as_secs_f64();
        Ok(match measured {
            Ok((kernel, measured)) => {
                self.records.push(ConfigurationRecord {
                    configuration: configuration.clone(),
                    outcome: Outcome::Measured {
                        artifact: kernel.artifact().0.clone(),
                        points: measured.clone(),
                        confirmed: Vec::new(),
                        validated: true,
                    },
                });
                self.swept.insert(configuration, Some(measured.clone()));
                Some(measured)
            }
            Err(exclusion) => {
                self.records.push(ConfigurationRecord {
                    configuration: configuration.clone(),
                    outcome: Outcome::Excluded(exclusion),
                });
                self.swept.insert(configuration, None);
                None
            }
        })
    }

    /// Form and measure the candidates `ready` of `group` together, and
    /// rank those that pass.
    #[allow(clippy::too_many_arguments)]
    fn sweep(
        &mut self,
        formation: &Formation<'_>,
        default: &NativeSpecialization,
        group: &plan::Group,
        baseline: &[PointMeasurement],
        ready: &mut Vec<(usize, NativeSpecialization)>,
        ahead: &[(usize, NativeSpecialization)],
        points: &[PreparedPoint<'_>],
        measuring: &MeasureOptions,
        weighing: &Weighing,
        ranking: &mut Vec<(usize, f64)>,
    ) -> Result<(), TuneError> {
        if ready.is_empty() {
            return Ok(());
        }
        let ready = std::mem::take(ready);
        let began = Instant::now();
        let specializations = |candidates: &[(usize, NativeSpecialization)]| {
            candidates
                .iter()
                .map(|(_, specialization)| specialization.clone())
                .collect::<Vec<_>>()
        };
        let formed = self.programs.form(
            formation,
            default,
            &specializations(&ready),
            &specializations(ahead),
        )?;
        self.time.forming_seconds += began.elapsed().as_secs_f64();
        let mut candidates = Vec::with_capacity(ready.len());
        for ((index, specialization), kernel) in ready.into_iter().zip(formed) {
            match kernel {
                Ok(kernel) => candidates.push(FactoredCandidate {
                    index,
                    specialization,
                    kernel,
                }),
                Err(exclusion) => {
                    let configuration = Configuration::of(&specialization);
                    self.swept.insert(configuration.clone(), None);
                    self.records.push(ConfigurationRecord {
                        configuration,
                        outcome: Outcome::Excluded(exclusion),
                    });
                }
            }
        }
        let began = Instant::now();
        let measured = measure_factored_group(
            formation.implementation,
            &self.partition.boundary,
            &candidates,
            points,
            measuring,
            &group.launches,
            baseline,
            Some(&mut self.sweep_cache),
            &mut self.outputs,
        );
        self.time.measuring_seconds += began.elapsed().as_secs_f64();
        for (candidate, outcome) in candidates.into_iter().zip(measured) {
            let configuration = Configuration::of(&candidate.specialization);
            match outcome {
                Ok(measured) => {
                    ranking.push((
                        candidate.index,
                        weighing.cost(&measured, &self.reference).total(),
                    ));
                    self.swept.insert(configuration.clone(), Some(measured.clone()));
                    self.records.push(ConfigurationRecord {
                        configuration,
                        outcome: Outcome::Measured {
                            artifact: candidate.kernel.artifact().0.clone(),
                            points: measured,
                            confirmed: Vec::new(),
                            validated: true,
                        },
                    });
                }
                Err(exclusion) => {
                    self.swept.insert(configuration.clone(), None);
                    self.records.push(ConfigurationRecord {
                        configuration,
                        outcome: Outcome::Excluded(exclusion),
                    });
                }
            }
        }
        Ok(())
    }

    /// One step of the sweeps, coarse to fine. First the base of every
    /// boundary is measured: the seed's group choices under each value of
    /// the parameters that choose which launch serves which rows, so every
    /// such value has a measurement at the rows it moves before any launch's
    /// mapping is refined. The boundaries' groups are then swept, the
    /// boundary with the cheapest base first (the default boundary when its
    /// base is within one percent of the cheapest). That order is fixed by
    /// the bases. Nearby changes of the leading boundary's active groups are
    /// measured round-robin before deeper sweeps, so every step follows the
    /// order one uninterrupted search
    /// does; a configuration an earlier step measured is taken from it, and
    /// one not yet measured is formed and measured unless `expired` says
    /// the step's time is out for it, which ends the step's measuring.
    /// `expired` is asked with whether the configuration is a start: a
    /// boundary's base or one of the first boundary's form starts. A
    /// boundary or a form nothing measured cannot be chosen, so those are
    /// held back only by a start planned without them or by the step's hard
    /// limit. Every boundary with a measured base stays ranked by what was
    /// measured under it, so a search that ends early still chooses among
    /// all of them.
    fn explore(
        &mut self,
        formation: &Formation<'_>,
        statics: &NativeSpecialization,
        default: &NativeSpecialization,
        points: &[PreparedPoint<'_>],
        search: &StartPlan,
        expired: &dyn Fn(bool) -> bool,
    ) -> Result<(), TuneError> {
        let implementation = formation.implementation;
        let partition = self.partition.clone();
        let measuring = MeasureOptions {
            samples: search.settings.samples,
            min_sample_seconds: search.min_sample_seconds,
        };
        let weighing = Weighing::of(points);
        let default_choices = self.default_choices.clone();
        let mut boundaries: Vec<Boundary> = Vec::new();
        let mut complete = !partition.groups.iter().any(|group| group.truncated)
            && !partition.sources.iter().any(|source| source.truncated);
        let mut bases = Vec::new();
        for boundary in boundary_assignments(&partition, implementation) {
            let base = partition
                .assemble(implementation, statics, &default_choices, &boundary)
                .map_err(|error| {
                    TuneError::Declaration(format!("factored native boundary: {error:?}"))
                })?;
            let baseline = if base == self.seed {
                self.reference.clone()
            } else {
                match self.swept.get(&Configuration::of(&base)).cloned() {
                    Some(Some(measured)) => measured,
                    Some(None) => continue,
                    None => {
                        if expired(true) {
                            complete = false;
                            continue;
                        }
                        match self.base(formation, default, &base, points, &measuring)? {
                            Some(measured) => measured,
                            None => continue,
                        }
                    }
                }
            };
            let base_cost = weighing.cost(&baseline, &self.reference).total();
            bases.push((base_cost, boundary, baseline));
        }
        bases.sort_by(|left, right| left.0.total_cmp(&right.0));
        if let Some(default_index) = bases
            .iter()
            .position(|(_, boundary, _)| *boundary == self.default_boundary)
        {
            if bases[default_index].0 <= bases[0].0 * 1.01 {
                let preferred = bases.remove(default_index);
                bases.insert(0, preferred);
            }
        }
        // Once a sweep runs out of time the later boundaries measure nothing
        // more: they are ranked by what earlier steps measured under them.
        let stopped = std::cell::Cell::new(false);
        let expired = |start: bool| stopped.get() || expired(start);
        // Cover the active coordinates of the leading boundary before a
        // deep sweep of any group. These starts share the form-start budget:
        // a short refinement slice cannot starve them, but the hard deadline
        // and conclusion reserve still apply. Resume uses the same order and
        // skips every configuration already measured or excluded.
        if let Some((_, boundary, baseline)) = bases.first() {
            let active = baseline
                .iter()
                .flat_map(|point| point.key.launches.iter().copied())
                .collect::<Vec<_>>();
            let mut coverage = Vec::new();
            for (group_index, candidate) in
                coverage_order(&partition.groups, &default_choices, &active)
            {
                let mut selected = default_choices.clone();
                selected[group_index] = candidate;
                let specialization = partition
                    .assemble(implementation, statics, &selected, boundary)
                    .map_err(|error| {
                        TuneError::Declaration(format!("factored native coverage: {error:?}"))
                    })?;
                if self.swept.contains_key(&Configuration::of(&specialization)) {
                    continue;
                }
                coverage.push((group_index, candidate, specialization));
            }
            for (position, (group_index, candidate, specialization)) in coverage.iter().enumerate()
            {
                if expired(true) {
                    complete = false;
                    stopped.set(true);
                    break;
                }
                // Compile this group's nearby variants together when a
                // compile is needed, but keep measurements round-robin.
                let ahead = coverage[position + 1..]
                    .iter()
                    .filter(|(group, _, _)| group == group_index)
                    .map(|(_, candidate, specialization)| (*candidate, specialization.clone()))
                    .collect::<Vec<_>>();
                self.sweep(
                    formation,
                    default,
                    &partition.groups[*group_index],
                    baseline,
                    &mut vec![(*candidate, specialization.clone())],
                    &ahead,
                    points,
                    &measuring,
                    &weighing,
                    &mut Vec::new(),
                )?;
            }
        }
        for (base_cost, boundary, baseline) in bases {
            let reserved = boundaries.is_empty();
            let mut choices = default_choices.clone();
            let mut group_rankings = Vec::with_capacity(partition.groups.len());
            let mut score = base_cost;
            let mut interrupted = false;
            for (group_index, group) in partition.groups.iter().enumerate() {
                let specializations = (0..group.candidates.len())
                    .map(|candidate| {
                        let mut selected = default_choices.clone();
                        selected[group_index] = candidate;
                        partition
                            .assemble(implementation, statics, &selected, &boundary)
                            .map_err(|error| {
                                TuneError::Declaration(format!(
                                    "factored native candidate: {error:?}"
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut ranking = replay_group(
                    default_choices[group_index],
                    base_cost,
                    specializations.iter().map(|specialization| {
                        self.swept
                            .get(&Configuration::of(specialization))
                            .and_then(Option::as_ref)
                            .map(|measured| weighing.cost(measured, &self.reference).total())
                    }),
                );
                // Each form parameter's values seed a start of their own: the
                // candidate of every other form nearest the defaults is measured
                // first, then the forms' other candidates, the cheapest start's
                // first. A group without form parameters is swept in order.
                let form_of = |candidate: usize| {
                    group
                        .parameters
                        .iter()
                        .zip(&group.candidates[candidate])
                        .filter(|(address, _)| match address {
                            plan::ParameterAddress::Entry(name) => implementation
                                .params
                                .iter()
                                .any(|parameter| parameter.form && &parameter.name == name),
                            plan::ParameterAddress::Launch { .. } => false,
                        })
                        .map(|(_, value)| *value)
                        .collect::<Vec<_>>()
                };
                let default_candidate = default_choices[group_index];
                let mut starts = BTreeMap::<Vec<u64>, (usize, usize)>::new();
                for candidate in 0..group.candidates.len() {
                    let form = form_of(candidate);
                    if form == form_of(default_candidate) {
                        continue;
                    }
                    let distance = group.candidates[candidate]
                        .iter()
                        .zip(&group.candidates[default_candidate])
                        .filter(|(value, default)| value != default)
                        .count();
                    let start = starts.entry(form).or_insert((distance, candidate));
                    if distance < start.0 {
                        *start = (distance, candidate);
                    }
                }
                let starts = starts
                    .into_values()
                    .map(|(_, candidate)| candidate)
                    .collect::<Vec<_>>();
                for pass in 0..2 {
                    let sweep = if pass == 0 {
                        starts.clone()
                    } else {
                        let start_cost = |candidate: usize| {
                            let form = form_of(candidate);
                            ranking
                                .iter()
                                .filter(|(ranked, _)| form_of(*ranked) == form)
                                .map(|(_, cost)| *cost)
                                .fold(f64::INFINITY, f64::min)
                        };
                        let mut rest = (0..group.candidates.len())
                            .filter(|candidate| !starts.contains(candidate))
                            .map(|candidate| (start_cost(candidate), candidate))
                            .collect::<Vec<_>>();
                        rest.sort_by(|left, right| {
                            left.0.total_cmp(&right.0).then(left.1.cmp(&right.1))
                        });
                        rest.into_iter().map(|(_, candidate)| candidate).collect()
                    };
                    // The first boundary's form starts whatever the slice;
                    // anything else while the step's slice lasts; a chunk
                    // at a time, the step's limit checked before each
                    // candidate.
                    let always = reserved && pass == 0;
                    // The pass's candidates no earlier step measured, in
                    // sweep order.
                    let mut pending = Vec::new();
                    for candidate in sweep {
                        if candidate == default_candidate {
                            continue;
                        }
                        let specialization = &specializations[candidate];
                        if !self.swept.contains_key(&Configuration::of(specialization)) {
                            pending.push((candidate, specialization.clone()));
                        }
                    }
                    let mut ready = Vec::new();
                    for (position, candidate) in pending.iter().enumerate() {
                        if expired(always) {
                            complete = false;
                            interrupted = true;
                            break;
                        }
                        ready.push(candidate.clone());
                        if ready.len() == SWEEP_CHUNK {
                            let ahead = &pending
                                [position + 1..pending.len().min(position + 1 + SWEEP_AHEAD)];
                            self.sweep(
                                formation, default, group, &baseline, &mut ready, ahead, points,
                                &measuring, &weighing, &mut ranking,
                            )?;
                        }
                    }
                    self.sweep(
                        formation, default, group, &baseline, &mut ready, &[], points,
                        &measuring, &weighing, &mut ranking,
                    )?;
                    if interrupted {
                        break;
                    }
                }
                // An inactive group reuses precisely the baseline cost for
                // every candidate. Keep its seed instead of silently choosing
                // the lexically first, unobserved mapping on that tie.
                rank_group(&mut ranking, default_choices[group_index]);
                choices[group_index] = ranking[0].0;
                score += ranking[0].1 - base_cost;
                group_rankings.push(ranking);
                // Always replay later groups, even after the deadline. New
                // device work remains guarded by `expired` above.
            }
            boundaries.push((score, boundary, choices, group_rankings));
            if interrupted {
                stopped.set(true);
            }
        }
        boundaries.sort_by(|left, right| left.0.total_cmp(&right.0));
        // A boundary moves work between launches. Prefer the default band split
        // when its apparent loss is within one percent of the leading score.
        if let Some(default_index) = boundaries
            .iter()
            .position(|(_, boundary, _, _)| *boundary == self.default_boundary)
        {
            if boundaries[default_index].0 <= boundaries[0].0 * 1.01 {
                let preferred = boundaries.remove(default_index);
                boundaries.insert(0, preferred);
            }
        }
        // A step can end inside a group's sweep: the candidates measured so
        // far are ranked, while later groups retain their defaults, and the
        // boundaries no sweep reached stand at their bases. All of them are
        // costs at the same points against the same reference, so the
        // conclusion chooses among them, never as a complete search.
        self.boundaries = boundaries;
        self.complete = complete;
        Ok(())
    }

    /// Conclude the search with what its sweeps reached: when they are
    /// complete and `window` allows, confirm each group's finalists; then
    /// confirm the assembled choice against the seed. The boundaries are
    /// taken best first. When the sweeps are not complete, the best
    /// boundaries' assembled choices are confirmed in turn until one
    /// improves on the seed, as many as the unit's reserve covers.
    #[allow(clippy::too_many_arguments)]
    fn conclude(
        self,
        formation: &Formation<'_>,
        statics: &NativeSpecialization,
        unit_default: &NativeSpecialization,
        points: &[PreparedPoint<'_>],
        search: &StartPlan,
        window: Instant,
        allowance: Duration,
        unit: &Unit,
    ) -> Result<TuningResult, TuneError> {
        let implementation = formation.implementation;
        let deadline = before(window, self.reserve(&search.settings));
        let Self {
            partition,
            mut programs,
            mut outputs,
            mut records,
            seed,
            seed_kernel: default_kernel,
            reference,
            default_choices,
            boundaries,
            mut complete,
            mut time,
            ..
        } = self;
        let default = &seed;
        let weighing = Weighing::of(points);
        let confirmation = MeasureOptions {
            samples: search.settings.confirmation_samples,
            min_sample_seconds: search.min_sample_seconds,
        };
        let mut form = |specialization: &NativeSpecialization| {
            programs.form_one(formation, unit_default, specialization)
        };
    let mut overall = default.clone();
    // The reserve covers the seed and the settings' finalists: each
    // boundary confirmed without its groups' finalists samples the seed
    // and its assembled choice.
    let unswept = search.settings.confirmed.div_ceil(2).max(1);
    let mut attempted = 0;
    'confirm_boundaries: for (_, best_boundary, mut best_choices, best_group_rankings) in boundaries
    {
        let mut confirmed_choices = Vec::with_capacity(partition.groups.len());
        if Instant::now() >= deadline {
            complete = false;
        }
        if complete && !best_group_rankings.is_empty() {
            // Confirm each group's finalists against the defaults under the
            // chosen boundary. Points outside the group reuse the same baseline
            // and therefore cannot influence this choice.
            let base = partition
                .assemble(implementation, statics, &default_choices, &best_boundary)
                .map_err(|error| {
                    TuneError::Declaration(format!("factored confirmation baseline: {error:?}"))
                })?;
            let began = Instant::now();
            let kernel = match form(&base)? {
                Ok(kernel) => kernel,
                Err(exclusion) => {
                    records.push(ConfigurationRecord {
                        configuration: Configuration::of(&base),
                        outcome: Outcome::Excluded(exclusion),
                    });
                    continue;
                }
            };
            time.forming_seconds += began.elapsed().as_secs_f64();
            let began = Instant::now();
            let baseline = match measure_factored(
                implementation,
                &partition.boundary,
                &kernel,
                &base,
                points,
                &confirmation,
                None,
                None,
                None,
                &mut outputs,
            ) {
                Ok(baseline) => baseline,
                Err(exclusion) => {
                    records.push(ConfigurationRecord {
                        configuration: Configuration::of(&base),
                        outcome: Outcome::Excluded(exclusion),
                    });
                    continue;
                }
            };
            time.measuring_seconds += began.elapsed().as_secs_f64();
            for (group_index, ranking) in best_group_rankings.iter().enumerate() {
                if Instant::now() >= deadline {
                    complete = false;
                    break;
                }
                let mut finalists = ranking
                    .iter()
                    .take(search.settings.confirmed)
                    .map(|(index, _)| *index)
                    .collect::<Vec<_>>();
                if !finalists.contains(&default_choices[group_index]) {
                    finalists.push(default_choices[group_index]);
                }
                let mut ready = vec![FactoredCandidate {
                    index: default_choices[group_index],
                    specialization: base.clone(),
                    kernel: kernel.clone(),
                }];
                for candidate in finalists {
                    if candidate == default_choices[group_index] {
                        continue;
                    }
                    let mut choices = default_choices.clone();
                    choices[group_index] = candidate;
                    let specialization = partition
                        .assemble(implementation, statics, &choices, &best_boundary)
                        .map_err(|error| {
                            TuneError::Declaration(format!("factored finalist: {error:?}"))
                        })?;
                    let began = Instant::now();
                    let kernel = match form(&specialization)? {
                        Ok(kernel) => kernel,
                        Err(exclusion) => {
                            records.push(ConfigurationRecord {
                                configuration: Configuration::of(&specialization),
                                outcome: Outcome::Excluded(exclusion),
                            });
                            continue;
                        }
                    };
                    time.forming_seconds += began.elapsed().as_secs_f64();
                    ready.push(FactoredCandidate {
                        index: candidate,
                        specialization,
                        kernel,
                    });
                }
                let began = Instant::now();
                let confirmed = measure_factored_group(
                    implementation,
                    &partition.boundary,
                    &ready,
                    points,
                    &confirmation,
                    &partition.groups[group_index].launches,
                    &baseline,
                    None,
                    &mut outputs,
                );
                time.measuring_seconds += began.elapsed().as_secs_f64();
                let group_baseline = match &confirmed[0] {
                    Ok(baseline) => baseline.clone(),
                    Err(exclusion) => {
                        records.push(ConfigurationRecord {
                            configuration: Configuration::of(&base),
                            outcome: Outcome::Excluded(exclusion.clone()),
                        });
                        continue 'confirm_boundaries;
                    }
                };
                let baseline_cost = weighing.cost(&group_baseline, &reference);
                let mut winner = default_choices[group_index];
                let mut winner_cost = baseline_cost.total();
                let mut ranked = vec![(winner, winner_cost)];
                for (candidate, confirmed) in ready.into_iter().zip(confirmed).skip(1) {
                    let outcome = match confirmed {
                        Ok(confirmed) => {
                            let changed = confirmed
                                .iter()
                                .zip(&group_baseline)
                                .filter_map(|(point, base)| {
                                    (point.key != base.key).then_some(point.clone())
                                })
                                .collect::<Vec<_>>();
                            if let Some(exclusion) = unstable(&changed) {
                                Outcome::Excluded(exclusion)
                            } else {
                                let cost = weighing.cost(&confirmed, &reference);
                                ranked.push((candidate.index, cost.total()));
                                if cost.improves_on(&baseline_cost, 0.02)
                                    && cost.total() < winner_cost
                                {
                                    winner = candidate.index;
                                    winner_cost = cost.total();
                                }
                                Outcome::Measured {
                                    artifact: candidate.kernel.artifact().0.clone(),
                                    points: confirmed.clone(),
                                    confirmed,
                                    validated: true,
                                }
                            }
                        }
                        Err(exclusion) => Outcome::Excluded(exclusion),
                    };
                    records.push(ConfigurationRecord {
                        configuration: Configuration::of(&candidate.specialization),
                        outcome,
                    });
                }
                best_choices[group_index] = winner;
                ranked.sort_by(|left, right| {
                    left.1
                        .total_cmp(&right.1)
                        .then_with(|| left.0.cmp(&right.0))
                });
                let mut order = vec![winner];
                order.extend(
                    ranked
                        .into_iter()
                        .map(|(candidate, _)| candidate)
                        .filter(|candidate| *candidate != winner),
                );
                confirmed_choices.push(order);
            }
        }
        loop {
            let selected = partition
                .assemble(implementation, statics, &best_choices, &best_boundary)
                .map_err(|error| {
                    TuneError::Declaration(format!("factored native selection: {error:?}"))
                })?;
            if selected == *default {
                break;
            }
            let began = Instant::now();
            let kernel = form(&selected)?;
            time.forming_seconds += began.elapsed().as_secs_f64();
            let outcome = match kernel {
                Err(exclusion) => Outcome::Excluded(exclusion),
                Ok(kernel) => {
                    let began = Instant::now();
                    let finalists = [
                        FactoredCandidate {
                            index: 0,
                            specialization: default.clone(),
                            kernel: default_kernel.clone(),
                        },
                        FactoredCandidate {
                            index: 1,
                            specialization: selected.clone(),
                            kernel: kernel.clone(),
                        },
                    ];
                    let all_launches = (0..implementation.launches.len()).collect::<Vec<_>>();
                    let mut confirmed = measure_factored_group(
                        implementation,
                        &partition.boundary,
                        &finalists,
                        points,
                        &confirmation,
                        &all_launches,
                        &reference,
                        None,
                        &mut outputs,
                    );
                    time.measuring_seconds += began.elapsed().as_secs_f64();
                    let confirmed_reference =
                        confirmed.remove(0).map_err(TuneError::DefaultUnusable)?;
                    match confirmed.remove(0) {
                        Err(exclusion) => Outcome::Excluded(exclusion),
                        Ok(confirmed) => {
                            let improved = weighing.cost(&confirmed, &confirmed_reference).total()
                                < weighing
                                    .cost(&confirmed_reference, &confirmed_reference)
                                    .total()
                                    * 0.98;
                            if let Some(exclusion) = unstable(&confirmed) {
                                Outcome::Excluded(exclusion)
                            } else if !improved {
                                Outcome::Measured {
                                    artifact: kernel.artifact().0.clone(),
                                    points: confirmed.clone(),
                                    confirmed,
                                    validated: true,
                                }
                            } else {
                                overall = selected.clone();
                                Outcome::Measured {
                                    artifact: kernel.artifact().0.clone(),
                                    points: confirmed.clone(),
                                    confirmed,
                                    validated: true,
                                }
                            }
                        }
                    }
                }
            };
            records.push(ConfigurationRecord {
                configuration: Configuration::of(&selected),
                outcome,
            });
            if overall != *default {
                break;
            }
            if Instant::now() >= deadline {
                complete = false;
                break;
            }
            break;
        }
        if Instant::now() >= deadline {
            complete = false;
        }
        if overall != *default {
            break;
        }
        if !complete {
            attempted += 1;
            if attempted >= unswept || Instant::now() >= deadline {
                break;
            }
        }
    }
    let parameters = implementation
        .params
        .iter()
        .map(|parameter| DeclaredParameter {
            name: parameter.name.clone(),
            launch: None,
            arithmetic: parameter.arithmetic,
            form: parameter.form,
            values: parameter.values.clone(),
        })
        .chain(
            implementation
                .launches
                .iter()
                .enumerate()
                .flat_map(|(ordinal, launch)| {
                    launch
                        .params
                        .iter()
                        .map(move |parameter| DeclaredParameter {
                            name: parameter.name.clone(),
                            launch: Some(ordinal),
                            arithmetic: parameter.arithmetic,
                            form: parameter.form,
                            values: parameter.values.clone(),
                        })
                }),
        )
        .collect();
    time.validating_seconds = points.iter().map(|p| p.validation_seconds()).sum::<f64>();
    time.measuring_seconds = (time.measuring_seconds - time.validating_seconds).max(0.);
    Ok(TuningResult {
        tuning_identity: unit.tuning_identity.clone(),
        entry: unit.entry.clone(),
        backend: "metal".into(),
        points: labels(points),
        validation: unit.validation.clone(),
        numerical_evidence: winner_evidence(points, &records, &Configuration::of(&overall))?,
        implementation_identity: unit.implementation_identity.clone(),
        parameters,
        configurations: records,
        overall: Configuration::of(&overall),
        method: TuningMethod::Factored {
            allowance_seconds: allowance.as_secs_f64(),
            groups: partition.groups.len(),
            candidates: partition
                .groups
                .iter()
                .map(|group| group.candidates.len())
                .sum(),
            complete,
        },
        time,
    })
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
    update_declaration_digest(&mut digest, &entry_name, implementation);
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
    // Only a declaration with error classes digests them, so one without
    // keeps the digest its stored results were keyed by.
    if !implementation.error_classes.is_empty() {
        digest.update(format!("{:?}", implementation.error_classes).as_bytes());
    }
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
    evidence: Vec<NumericalEvidence>,
}

/// One entry's tuning, shared by both strategies.
struct Tuned<'s> {
    formation: &'s Formation<'s>,
    space: &'s SearchSpace,
    statics: &'s NativeSpecialization,
}

type Tuning = (Records, NativeSpecialization, TuningMethod, TuningTime);

/// The outcome of a local search whose finalists `trace` confirmed: its
/// records, its choice and where its time went. `lived` is what the search
/// formed and measured.
fn searched(
    space: &SearchSpace,
    statics: &NativeSpecialization,
    points: &[PreparedPoint<'_>],
    settings: SearchSettings,
    allowance: Duration,
    lived: Lived,
    trace: search::SearchTrace,
) -> Result<Tuning, TuneError> {
    let Lived {
        evaluated,
        mut time,
        ..
    } = lived;
    let default = space.default_index();
    let chosen = *trace.ranking.first().ok_or_else(|| {
        TuneError::NoValidatedCandidate(
            trace
                .evaluated
                .iter()
                .chain(&trace.confirmed)
                .filter_map(|(_, outcome)| outcome.as_ref().err().cloned())
                .collect(),
        )
    })?;
    time.validating_seconds = points.iter().map(|p| p.validation_seconds()).sum::<f64>();
    time.measuring_seconds = (time.measuring_seconds - time.validating_seconds).max(0.);
    let records: Vec<_> = trace
        .evaluated
        .iter()
        .map(|(index, result)| {
            let configuration = Configuration::of(&specialization(statics, &space.values(*index)));
            // Failed confirmation excludes a finalist; the validated
            // defaults stay the fallback whatever their confirmation showed.
            let unconfirmed = trace
                .confirmed
                .iter()
                .find(|(finalist, _)| finalist == index && *index != default)
                .and_then(|(_, confirmed)| confirmed.as_ref().err());
            let outcome = match (result, unconfirmed) {
                (Err(exclusion), _) | (Ok(_), Some(exclusion)) => {
                    Outcome::Excluded(exclusion.clone())
                }
                (Ok(_), None) => {
                    let measured = &evaluated[index];
                    Outcome::Measured {
                        artifact: measured.kernel.artifact().0.clone(),
                        points: measured.points.clone(),
                        confirmed: measured.confirmed.clone(),
                        validated: true,
                    }
                }
            };
            ConfigurationRecord {
                configuration,
                outcome,
            }
        })
        .collect();
    let overall = specialization(statics, &space.values(chosen));
    Ok((
        Records {
            points: labels(points),
            evidence: winner_evidence(points, &records, &Configuration::of(&overall))?,
            records,
        },
        overall,
        TuningMethod::Search {
            allowance_seconds: allowance.as_secs_f64(),
            settings,
            stop: trace.stop,
        },
        time,
    ))
}

impl Tuned<'_> {
    fn survey(&self, points: &[PreparedPoint<'_>], plan: SurveyPlan) -> Result<Tuning, TuneError> {
        let options = MeasureOptions {
            samples: plan.samples,
            min_sample_seconds: plan.min_sample_seconds,
        };
        let mut time = TuningTime::default();
        let workers = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let mut measurer = Measurer::new(self.formation.implementation);
        let mut outcomes: Vec<Option<Outcome>> = vec![None; self.space.len()];
        let order = (0..self.space.len()).collect::<Vec<_>>();
        for batch in order.chunks(workers) {
            let specializations = batch
                .iter()
                .map(|index| specialization(self.statics, &self.space.values(*index)))
                .collect::<Vec<_>>();
            let began = Instant::now();
            let formed = self.formation.form_all(&specializations);
            time.forming_seconds += began.elapsed().as_secs_f64();

            for (index, kernel) in batch.iter().zip(formed) {
                let began = Instant::now();
                let measured = kernel.and_then(|kernel| {
                    measurer
                        .measure(&kernel, &points, &self.space.values(*index), &options)
                        .map(|measured| (kernel, measured))
                });
                time.measuring_seconds += began.elapsed().as_secs_f64();
                let began = Instant::now();
                let outcome = match measured {
                    Err(exclusion) => Outcome::Excluded(exclusion),
                    Ok((kernel, measured)) => Outcome::Measured {
                        artifact: kernel.artifact().0.clone(),
                        points: measured,
                        confirmed: Vec::new(),
                        validated: true,
                    },
                };
                time.validating_seconds += began.elapsed().as_secs_f64();
                outcomes[*index] = Some(outcome);
            }
        }
        let outcomes = outcomes
            .into_iter()
            .map(|outcome| outcome.expect("every configuration was surveyed"))
            .collect::<Vec<_>>();
        let measured = |outcome: &Outcome| match outcome {
            Outcome::Measured { points, .. } => Some(points.clone()),
            Outcome::Excluded(_) => None,
        };
        let reference = outcomes.iter().find_map(measured).ok_or_else(|| {
            TuneError::NoValidatedCandidate(
                outcomes
                    .iter()
                    .filter_map(|outcome| match outcome {
                        Outcome::Excluded(error) => Some(error.clone()),
                        _ => None,
                    })
                    .collect(),
            )
        })?;
        let weighing = Weighing::of(&points);
        let chosen = (0..outcomes.len())
            .filter_map(|index| {
                measured(&outcomes[index])
                    .map(|points| (index, weighing.cost(&points, &reference).total()))
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .ok_or_else(|| TuneError::Reference("winner lacks numerical evidence".into()))?;
        time.validating_seconds = points.iter().map(|p| p.validation_seconds()).sum::<f64>();
        time.measuring_seconds = (time.measuring_seconds - time.validating_seconds).max(0.);
        let records: Vec<_> = outcomes
            .into_iter()
            .enumerate()
            .map(|(index, outcome)| ConfigurationRecord {
                configuration: Configuration::of(&specialization(
                    self.statics,
                    &self.space.values(index),
                )),
                outcome,
            })
            .collect();
        Ok((
            Records {
                points: labels(&points),
                evidence: winner_evidence(
                    points,
                    &records,
                    &Configuration::of(&specialization(self.statics, &self.space.values(chosen))),
                )?,
                records,
            },
            specialization(self.statics, &self.space.values(chosen)),
            TuningMethod::Survey {
                samples: plan.samples,
            },
            time,
        ))
    }
}

fn excluded(records: &[ConfigurationRecord]) -> Vec<Exclusion> {
    records
        .iter()
        .filter_map(|record| match &record.outcome {
            Outcome::Excluded(error) => Some(error.clone()),
            _ => None,
        })
        .collect()
}

fn winner_evidence<'p, 'a: 'p>(
    points: impl IntoIterator<Item = &'p PreparedPoint<'a>>,
    records: &[ConfigurationRecord],
    winner: &Configuration,
) -> Result<Vec<NumericalEvidence>, TuneError> {
    let artifact = records
        .iter()
        .find_map(|record| {
            if &record.configuration != winner {
                return None;
            }
            match &record.outcome {
                Outcome::Measured { artifact, .. } => Some(artifact),
                _ => None,
            }
        })
        .ok_or_else(|| TuneError::Reference("winner lacks numerical evidence".into()))?;
    points
        .into_iter()
        .map(|point| {
            point
                .evidence(artifact)
                .ok_or_else(|| TuneError::Reference("winner lacks numerical evidence".into()))
        })
        .collect()
}

/// Every point's record, the weight of each point not `admitted` folded
/// into the largest admitted point of its class, or into the largest
/// admitted point when its class has none. Points come in ascending cost, so
/// the largest is the last. Without admitted points the weights stand.
fn folded(points: &[PointSpec], admitted: &[usize]) -> Vec<PointRecord> {
    let mut weights = points.iter().map(|point| point.weight).collect::<Vec<_>>();
    if let Some(&largest) = admitted.last() {
        for position in (0..points.len()).filter(|position| !admitted.contains(position)) {
            let target = admitted
                .iter()
                .rev()
                .copied()
                .find(|candidate| {
                    points[position].class.is_some()
                        && points[*candidate].class == points[position].class
                })
                .unwrap_or(largest);
            weights[target] += weights[position];
            weights[position] = 0.0;
        }
    }
    points
        .iter()
        .zip(weights)
        .map(|(point, weight)| PointRecord {
            label: point.label.clone(),
            weight,
            class: point.class.clone(),
        })
        .collect()
}

/// What confirming `finalists` configurations, `samples` each (after a
/// calibrating pass), costs when each measures like `measured`.
fn confirmation_time(measured: &[PointMeasurement], finalists: usize, samples: usize) -> Duration {
    let sample = measured
        .iter()
        .map(|point| point.median_seconds * point.repetitions as f64)
        .sum::<f64>();
    Duration::from_secs_f64(sample * (finalists * (samples + 1)) as f64)
}

/// Entry parameters read only by launches that no admissible configuration
/// activates at any of `shapes`: nothing measures them and no validation
/// runs their code, so a search keeps their defaults. A parameter no launch
/// reads affects every launch and is never one of them.
fn unserved_parameters(
    implementation: &NativeImplementation,
    statics: &NativeSpecialization,
    shapes: &[PointShape],
) -> Result<Vec<String>, TuneError> {
    if implementation
        .launches
        .iter()
        .all(|launch| launch.when.is_none())
    {
        return Ok(Vec::new());
    }
    let partition = plan::partition(implementation, statics, shapes)
        .map_err(|error| TuneError::Declaration(format!("native launch activity: {error:?}")))?;
    let active = partition
        .points
        .iter()
        .flat_map(|point| point.active_sets.iter().flatten().copied())
        .collect::<std::collections::BTreeSet<_>>();
    let influence = Influence::of(implementation);
    Ok(implementation
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
        .collect())
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

fn point_measurement(label: &str, key: PointKey, measurement: Measurement) -> PointMeasurement {
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
    use super::*;

    #[test]
    fn factored_coverage_reaches_later_active_groups_before_refining_first() {
        let group = |launch, candidates| plan::Group {
            launches: vec![launch],
            parameters: Vec::new(),
            candidates,
            truncated: false,
        };
        let groups = vec![
            group(
                1,
                vec![
                    vec![4, 1],
                    vec![4, 2],
                    vec![8, 1],
                    vec![8, 2],
                    vec![16, 1],
                    vec![16, 2],
                ],
            ),
            group(
                2,
                vec![
                    vec![1, 1],
                    vec![1, 2],
                    vec![2, 1],
                    vec![2, 2],
                    vec![4, 1],
                    vec![4, 2],
                ],
            ),
            group(3, vec![vec![1], vec![2]]),
        ];
        let order = coverage_order(&groups, &[2, 3, 0], &[1, 2]);
        assert_eq!(&order[..2], &[(0, 0), (1, 1)]);
        assert!(
            order
                .iter()
                .any(|&(group, candidate)| group == 1
                    && groups[group].candidates[candidate] == [2, 1])
        );
        assert!(
            order.iter().all(|&(group, _)| group != 2),
            "inactive launches must keep their defaults"
        );
        let measured = order[..3].to_vec();
        let resumed = coverage_order(&groups, &[2, 3, 0], &[1, 2])
            .into_iter()
            .filter(|candidate| !measured.contains(candidate))
            .collect::<Vec<_>>();
        assert_eq!(
            resumed,
            order[3..],
            "continuation preserves uninterrupted search order"
        );
    }

    #[test]
    fn factored_coverage_respects_coupled_admissibility_and_deduplicates() {
        // Only diagonal assignments are admissible: neither coordinate can
        // change alone. A coupled start covers both coordinates once.
        let candidates = vec![vec![1, 1], vec![2, 2], vec![4, 4]];
        assert_eq!(coordinate_starts(&candidates, 1), vec![0, 2]);
        assert!(coordinate_starts(&[vec![1, 2]], 0).is_empty());
    }

    #[test]
    fn unobserved_group_keeps_seed_on_equal_cost() {
        let mut ranking = vec![(0, 1.0), (2, 1.0), (1, 1.0)];
        rank_group(&mut ranking, 2);
        assert_eq!(ranking[0].0, 2);
        ranking.push((3, 0.9));
        rank_group(&mut ranking, 2);
        assert_eq!(ranking[0].0, 3, "a measured improvement still wins");
    }

    #[test]
    fn cutoff_replays_cached_candidates_beyond_missing_form_starts() {
        // Candidate1 is an unmeasured form start. Candidate2 belongs to the
        // later ordinary pass and was measured by coordinate coverage.
        // No device work is admitted after cutoff; both groups still rank
        // their already measured improvements, exactly once.
        for costs in [[Some(1.0), None, Some(0.7)], [Some(1.0), None, Some(0.8)]] {
            let mut ranking = replay_group(0, 1.0, costs);
            assert_eq!(ranking.len(), 2);
            rank_group(&mut ranking, 0);
            assert_eq!(ranking[0], (2, costs[2].unwrap()));
        }
    }
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::registry::BackendName;

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

    /// A confirmation at the points `(label, median, samples)` in microseconds,
    /// every point keyed by `form`.
    fn confirmation(form: u64, points: &[(&str, f64, &[f64])]) -> Vec<PointMeasurement> {
        points
            .iter()
            .map(|(label, median, samples)| {
                let mut sorted = samples.to_vec();
                sorted.sort_by(f64::total_cmp);
                let mut deviations = samples
                    .iter()
                    .map(|sample| (sample - median).abs())
                    .collect::<Vec<_>>();
                deviations.sort_by(f64::total_cmp);
                PointMeasurement {
                    point: (*label).into(),
                    key: PointKey {
                        launches: vec![0],
                        values: [("DIRECT".to_owned(), form)].into(),
                    },
                    median_seconds: median * 1e-6,
                    deviation_seconds: deviations[deviations.len() / 2] * 1e-6,
                    samples: samples.iter().map(|sample| sample * 1e-6).collect(),
                    repetitions: 1,
                    rotation_bytes: 0,
                }
            })
            .collect()
    }

    /// Whether confirmed finalist `rival` replaces defaults confirmed as
    /// `defaults`, by the search's ranking rule and margin.
    fn replaces(rival: Vec<PointMeasurement>, defaults: Vec<PointMeasurement>) -> bool {
        let weighing = Weighing {
            weights: vec![0.5; defaults.len()],
            classes: (0..defaults.len()).collect(),
        };
        let reference = confirmed_measurement(true, defaults).unwrap();
        let rival = confirmed_measurement(false, rival).unwrap();
        weighing
            .cost(&rival, &reference)
            .improves_on(&weighing.cost(&reference, &reference), 0.02)
    }

    /// Defaults whose own re-measurement spreads at one small point (the
    /// staged K8/V4 prefill at 64 rows over 256 history on an M6, bimodal in
    /// 35 us steps) do not thereby beat a tightly confirmed finalist that is
    /// 40% ahead where the time goes; they do keep their place against a
    /// finalist that is ahead of their median only, not of their fastest
    /// samples.
    #[test]
    fn unstable_defaults_are_ranked_at_their_fastest_samples() {
        let defaults = || {
            confirmation(
                0,
                &[
                    ("m64-c256", 213.0, &[147.0, 262.0, 149.0, 213.0, 213.0]),
                    ("m512-c4096", 7822.0, &[7822.0]),
                ],
            )
        };
        assert!(unstable(&defaults()).is_some());
        let fastest = confirmed_measurement(true, defaults()).unwrap();
        assert_eq!(
            medians(&fastest),
            [147.0 * 1e-6, 7822.0 * 1e-6],
            "the defaults at their fastest samples"
        );
        // Unstable rivals still leave the ranking.
        assert!(confirmed_measurement(false, defaults()).is_err());

        let direct = confirmation(
            1,
            &[
                ("m64-c256", 96.0, &[97.0, 96.0, 98.0, 96.0, 96.0]),
                ("m512-c4096", 4563.0, &[4563.0]),
            ],
        );
        assert!(replaces(direct, defaults()));
        // Ahead of the defaults' median at the unstable point, level with
        // their fastest sample, and level elsewhere: within noise of them.
        let level = confirmation(
            1,
            &[
                ("m64-c256", 148.0, &[148.0, 147.0, 149.0, 148.0, 150.0]),
                ("m512-c4096", 7800.0, &[7800.0]),
            ],
        );
        assert!(!replaces(level, defaults()));
    }

    #[test]
    fn a_finalist_is_trusted_only_when_its_confirmed_samples_are_tight() {
        assert_eq!(unstable(&[measured(88e-6, 2e-6)]), None);
        // Samples that spread widely at any point.
        assert!(unstable(&[measured(88e-6, 2e-6), measured(85e-6, 20e-6)]).is_some());
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
    fn factored_keys_ignore_parameters_of_inactive_launches() {
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
            factored_key(implementation, &[], vec![0], &default),
            factored_key(implementation, &[], vec![0], &split)
        );
        assert_ne!(
            factored_key(implementation, &[], vec![1], &default),
            factored_key(implementation, &[], vec![1], &split)
        );
    }

    #[test]
    fn factored_keys_ignore_boundary_values_when_the_active_launch_is_unchanged() {
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
            factored_key(implementation, &boundary, vec![1], &default),
            factored_key(implementation, &boundary, vec![1], &changed)
        );
    }
}
