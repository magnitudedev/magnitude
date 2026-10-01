//! Native tuning validates every candidate against the selected reference before
//! repeated timing. The first initialized timed invocation supplies its outputs
//! and writable state; host comparison remains outside the device interval.
//! Ordinary, factored, survey and census paths use the same evidence. Timing
//! normalization and reuse do not confer numerical eligibility.

use super::plan::{self, PointShape};
use super::search::{
    self, Cost, Evaluator, ParameterValues, PointKey, SearchParameter, SearchSettings, SearchSpace,
    SearchSpaceError, SearchStop,
};
use super::timing::{self, OutputPool, PointTiming};
use super::validation::{self, PreparedPoint};
pub use super::validation::{NumericalEvidence, NumericalMetrics};
use super::{MeasureOptions, Measurement, NativePrepared};
use crate::api::device::DeviceInner;
use crate::api::kernel::{EncodedArgs, PrepareError};
use crate::api::{CallError, TensorError};
use seismic_compiler::prepared::{validate_invocation, InvocationContract};
use seismic_lang::checked::{CheckedModule, NativeImplementation, NativeSpecialization};
use seismic_lang::entry::{ElementBindings, LogicalEntry, ParameterKind, TensorAccess};
use seismic_lang::expr::SymbolValue;
use seismic_lang::ids::EntryId;
use seismic_lang::precision::PrecisionPolicy;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::cpu::CpuNativeKernels;

/// Restores a point's `&mut` tensors to their initial contents.
pub type TuningInitializer<'a> = Box<dyn FnMut() -> Result<(), TensorError> + 'a>;

/// One workload the tuned implementation serves.
pub struct TuningPoint<'a> {
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
    /// The point's cost relative to the unit's other points, estimated by
    /// the consumer. A census admits points in the order given and predicts
    /// a point's time from the previous one's by the ratio of their costs.
    pub cost: f64,
    /// A census always admits the point, whatever the time: it exercises a
    /// code path a candidate may run in serving.
    pub required: bool,
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
    /// The first passing configuration, measured at the points its ceiling
    /// admitted.
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
    #[serde(skip)]
    pub reused: bool,
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
        }
    }
}

impl std::error::Error for TuneError {}

/// A time-bounded search (production).
#[derive(Clone, Debug)]
pub struct SearchPlan {
    /// The search's time, confirmation included: exploration ends when what
    /// remains only covers confirming the finalists.
    pub allowance: Duration,
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

#[derive(Clone, Debug)]
pub enum Strategy {
    Search(SearchPlan),
    Survey(SurveyPlan),
    /// Measure the first passing configuration point by point, in the order
    /// given, admitting a point while the time so far plus its predicted time
    /// stays within `ceiling` (required points always). The result records
    /// every point, a point not admitted with weight zero and its weight
    /// folded into an admitted point.
    Census {
        min_sample_seconds: f64,
        ceiling: Duration,
    },
}

/// A completed matching search may be reused wholesale; a census seed only
/// supplies matching numerical evidence and timing for continued exploration.
#[derive(Clone, Copy)]
pub enum TuningReuse<'a> {
    Completed(&'a TuningResult),
    Seed(&'a TuningResult),
}
impl<'a> TuningReuse<'a> {
    pub fn result(self) -> &'a TuningResult {
        match self {
            Self::Completed(result) | Self::Seed(result) => result,
        }
    }
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

pub struct TuneRequest<'a> {
    pub device: &'a Arc<DeviceInner>,
    pub module: &'a CheckedModule,
    pub entry: EntryId,
    pub bindings: ElementBindings,
    /// Values of every static dimension.
    pub statics: NativeSpecialization,
    pub cpu: Option<&'static CpuNativeKernels>,
    pub points: Vec<TuningPoint<'a>>,
    pub validation: PrecisionPolicy,
    pub strategy: Strategy,
    pub reuse: Option<TuningReuse<'a>>,
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

/// Forms and measures configurations on the device for [`search::search`].
struct Live<'s, 'a> {
    formation: &'s Formation<'s>,
    space: &'s SearchSpace,
    statics: &'s NativeSpecialization,
    points: &'s [PreparedPoint<'a>],
    measurer: Measurer,
    search: MeasureOptions,
    confirmation: MeasureOptions,
    /// When exploration ends: the allowance less what confirmation needs.
    deadline: Instant,
    evaluated: HashMap<usize, Evaluated>,
    anchor: Option<usize>,
    seed: Option<&'s TuningResult>,
    time: TuningTime,
}

impl Live<'_, '_> {
    /// Measure configuration `index` at every point, reusing the census
    /// seed's measurement of the same configuration.
    fn cost(
        &mut self,
        index: usize,
        kernel: Result<Arc<NativePrepared>, Exclusion>,
        weighing: &Weighing,
    ) -> Result<Cost, Exclusion> {
        let configuration =
            Configuration::of(&specialization(self.statics, &self.space.values(index)));
        if let Some(seed) = self.seed.filter(|seed| {
            self.points.iter().all(|point| {
                seed.numerical_evidence
                    .iter()
                    .any(|evidence| evidence.identity == point.identity())
            })
        }) {
            if let Some(error) = seed.configurations.iter().find_map(|record| {
                if record.configuration != configuration {
                    return None;
                }
                match &record.outcome {
                    Outcome::Excluded(error) => Some(error.clone()),
                    _ => None,
                }
            }) {
                return Err(error);
            }
        }
        let kernel = kernel?;
        let reused = self.seed.and_then(|seed| {
            if seed.overall != configuration
                || !self
                    .points
                    .iter()
                    .all(|point| point.evidence(&kernel.artifact().0).is_some())
            {
                return None;
            }
            let measurements = seed.configurations.iter().find_map(|record| {
                if record.configuration != seed.overall {
                    return None;
                }
                match &record.outcome {
                    Outcome::Measured { points, .. } => Some(points),
                    _ => None,
                }
            })?;
            self.points
                .iter()
                .map(|point| {
                    measurements
                        .iter()
                        .find(|measured| measured.point == point.label)
                        .cloned()
                })
                .collect::<Option<Vec<_>>>()
        });
        let points = match reused {
            Some(points) => points,
            None => self.measurer.measure(
                &kernel,
                self.points,
                &self.space.values(index),
                &self.search,
            )?,
        };
        self.anchor.get_or_insert(index);
        self.evaluated.insert(
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
        self.evaluated
            .get(
                &self
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
        self.time.forming_seconds += (measuring - began).as_secs_f64();

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
        self.time.measuring_seconds += measuring.elapsed().as_secs_f64();
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
                    &self.evaluated[index].kernel,
                    self.points,
                    &self.measurer.influence,
                    &self.space.values(*index),
                    &mut self.measurer.outputs,
                    self.confirmation.min_sample_seconds,
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
        while let Err(failure) = timing::sample(&mut timings, &self.confirmation) {
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
        let confirmed = keys
            .into_iter()
            .map(|keys| {
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
                match unstable(&confirmed) {
                    Some(exclusion) => Err(exclusion),
                    None => Ok(confirmed),
                }
            })
            .collect::<Vec<_>>();
        self.time.measuring_seconds += began.elapsed().as_secs_f64();
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
                self.evaluated
                    .get_mut(index)
                    .expect("finalists were evaluated")
                    .confirmed = measured;
                Ok(cost)
            })
            .collect()
    }

    fn expired(&self) -> bool {
        Instant::now() >= self.deadline
    }
}

pub fn tune(request: TuneRequest<'_>) -> Result<TuningResult, TuneError> {
    if request.points.is_empty() || request.points.iter().any(|point| point.rotation.is_empty()) {
        return Err(TuneError::NoPoints);
    }
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
        reuse,
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
    if let Strategy::Survey(plan) = &strategy {
        widen(&mut implementation, &plan.domains)?;
    }
    // Every configuration formed below is this one entry at these bindings.
    let logical = Arc::new(
        module
            .entry(entry, &bindings)
            .map_err(|error| TuneError::Declaration(error.to_string()))?,
    );
    // Held so the reference and the search reuse its formed programs.
    let (default, _launchable) = launchable_default(
        device,
        module,
        entry,
        &logical,
        &bindings,
        cpu,
        &implementation,
        &statics,
        &points,
    )?;
    let mutable = mutable_parameters(&logical);
    if let (Some((_, parameter)), Some(point)) = (
        mutable.first(),
        points.iter().find(|point| point.initialize.is_none()),
    ) {
        return Err(TuneError::SharedMutableState {
            point: point.label.clone(),
            parameter: parameter.clone(),
        });
    }
    let implementation_identity =
        implementation_digest(device, module, entry, &bindings, &statics, cpu)?;
    let reuse = reuse.filter(|reuse| {
        let result = reuse.result();
        result.validation == validation
            && result.implementation_identity == implementation_identity
            && result.tuning_identity == device.tuning_identity()
            && result.entry == entry_name
            && result.overall.statics == *statics.statics()
            && implementation
                .validate(&result.overall.specialization())
                .is_ok()
            && match (reuse, &strategy, &result.method) {
                (TuningReuse::Seed(_), _, _) => true,
                (_, Strategy::Search(plan), TuningMethod::Search { settings, .. }) => {
                    plan.settings == *settings
                }
                (_, Strategy::Search(_), TuningMethod::Factored { .. }) => true,
                _ => false,
            }
    });
    let prepared = validation::prepare(
        device,
        module,
        entry,
        &bindings,
        &logical,
        &mutable,
        points,
        &validation,
        reuse,
        reference,
        &default,
        cpu,
    )?;
    let (points, references) = match prepared {
        validation::Preparation::Reused(result) => return Ok(result),
        validation::Preparation::Cases(points, references) => (points, references),
    };
    // A search times the points that carry weight; a point of weight zero
    // (one its census did not admit) only validates the search's choice,
    // which must pass at every point.
    let (points, unweighted): (Vec<_>, Vec<_>) = if matches!(strategy, Strategy::Search(_)) {
        points.into_iter().partition(|point| point.weight > 0.0)
    } else {
        (points, Vec::new())
    };
    let mut strategy = strategy;
    let mut min_sample_seconds = 0.0;
    if let Strategy::Search(plan) = &mut strategy {
        plan.allowance = plan.allowance.saturating_sub(validation_time(
            reuse.map(TuningReuse::result),
            &points,
            &unweighted,
        ));
        min_sample_seconds = plan.min_sample_seconds;
    }
    // A census executes each point's reference when it admits the point;
    // every other strategy measures at every point it times.
    if !matches!(strategy, Strategy::Census { .. }) {
        for point in &points {
            point.ensure_reference()?;
        }
    }
    let formation = Formation {
        device,
        module,
        entry,
        logical: &logical,
        bindings: &bindings,
        cpu,
        implementation: &implementation,
    };
    // A launch-scoped census measures the defaults as any census does; its
    // search is factored.
    if implementation.launch_scoped() && !matches!(strategy, Strategy::Census { .. }) {
        if !matches!(
            backend,
            seismic_lang::registry::BackendName::Metal | seismic_lang::registry::BackendName::Cuda
        ) {
            return Err(TuneError::Declaration(format!(
                "launch-scoped native tuning is not yet available on `{}`",
                backend.as_str(),
            )));
        }
        let Strategy::Search(plan) = strategy else {
            return Err(TuneError::Domain(
                "launch-scoped surveys require the factored search".into(),
            ));
        };
        let mut result = tune_factored(FactoredRequest {
            device,
            module,
            entry,
            logical: &logical,
            bindings: &bindings,
            statics: &statics,
            cpu,
            points: &points,
            validation,
            implementation_identity,
            implementation: &implementation,
            default: &default,
            search: plan,
        })?;
        validate_everywhere(
            &formation,
            &default,
            reuse.map(TuningReuse::result),
            &points,
            &unweighted,
            min_sample_seconds,
            &mut result,
        )?;
        result.time.reference_seconds += references.seconds();
        return Ok(result);
    }
    let mut admissible = implementation
        .admissible(&statics)
        .map_err(|error| TuneError::Declaration(error.to_string()))?;
    // A parameter read only by launches inactive at every point keeps its
    // default: nothing measures it, and no validation runs its code.
    if matches!(strategy, Strategy::Search(_)) {
        let unserved = unserved_parameters(
            &implementation,
            &statics,
            &point_shapes(device, &logical, &points)?,
        )?;
        admissible.retain(|candidate| {
            unserved
                .iter()
                .all(|name| candidate.param(name) == default.param(name))
        });
    }
    let declared = implementation
        .params
        .iter()
        .map(|parameter| SearchParameter {
            name: parameter.name.clone(),
            values: parameter.values.clone(),
            form: parameter.form,
        })
        .collect::<Vec<_>>();
    let space = SearchSpace::new(
        &declared,
        &admissible
            .iter()
            .map(|specialization| specialization.params().clone())
            .collect::<Vec<_>>(),
    )
    .map_err(TuneError::Space)?;
    let tuned = Tuned {
        formation: &formation,
        space: &space,
        statics: &statics,
        default: &default,
        seed: reuse.map(TuningReuse::result),
    };
    let (configurations, overall, method, time) = match strategy {
        Strategy::Search(plan) => tuned.search(&points, plan)?,
        Strategy::Survey(plan) => tuned.survey(&points, plan)?,
        Strategy::Census {
            min_sample_seconds,
            ceiling,
        } => tuned.census(&points, min_sample_seconds, ceiling)?,
    };
    let mut result = TuningResult {
        tuning_identity: device.tuning_identity(),
        entry: entry_name,
        backend: backend.as_str().to_owned(),
        points: configurations.points,
        validation,
        reused: false,
        numerical_evidence: configurations.evidence,
        implementation_identity,
        parameters: implementation
            .params
            .iter()
            .map(|parameter| DeclaredParameter {
                name: parameter.name.clone(),
                launch: None,
                arithmetic: parameter.arithmetic,
                form: parameter.form,
                values: parameter.values.clone(),
            })
            .collect(),
        configurations: configurations.records,
        overall: Configuration::of(&overall),
        method,
        time,
    };
    validate_everywhere(
        &formation,
        &default,
        reuse.map(TuningReuse::result),
        &points,
        &unweighted,
        min_sample_seconds,
        &mut result,
    )?;
    result.time.reference_seconds += references.seconds();
    Ok(result)
}

/// What validating a choice at `unweighted` costs: a reference and a
/// candidate execution of each point's rotation, each predicted from the
/// seed's time at the costliest point it measured, scaled by cost.
fn validation_time(
    seed: Option<&TuningResult>,
    weighted: &[PreparedPoint<'_>],
    unweighted: &[PreparedPoint<'_>],
) -> Duration {
    let Some(measured) = seed.and_then(|seed| measured_points(seed, &seed.overall)) else {
        return Duration::ZERO;
    };
    let Some((cost, seconds)) = weighted
        .iter()
        .filter_map(|point| {
            measured
                .iter()
                .find(|measurement| measurement.point == point.label)
                .map(|measurement| (point.cost, measurement.median_seconds))
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
    else {
        return Duration::ZERO;
    };
    let per_cost = seconds / cost.max(f64::MIN_POSITIVE);
    Duration::from_secs_f64(
        unweighted
            .iter()
            .map(|point| 2.0 * per_cost * point.cost * point.rotation.len() as f64)
            .sum(),
    )
}

/// Validate a search's choice at the points it was not timed at, so every
/// choice passes at every point. A choice that fails there is excluded and
/// gives way to the census seed (the first passing configuration, the
/// defaults when they pass; the defaults without a seed), which must pass
/// there too. The defaults compared with themselves (a `NativeDefault`
/// reference) pass by construction. Every point is recorded.
fn validate_everywhere(
    formation: &Formation<'_>,
    default: &NativeSpecialization,
    seed: Option<&TuningResult>,
    weighted: &[PreparedPoint<'_>],
    unweighted: &[PreparedPoint<'_>],
    min_sample_seconds: f64,
    result: &mut TuningResult,
) -> Result<(), TuneError> {
    result
        .points
        .extend(labels(unweighted).into_iter().map(|point| PointRecord {
            weight: 0.0,
            ..point
        }));
    let Some(first) = unweighted.first() else {
        return Ok(());
    };
    let trusted = (first.reference_kind() == TuningReference::NativeDefault)
        .then(|| Configuration::of(default));
    let fallback = seed.map_or_else(|| Configuration::of(default), |seed| seed.overall.clone());
    let mut candidates = vec![result.overall.clone()];
    if fallback != result.overall {
        candidates.push(fallback);
    }
    let began = Instant::now();
    let mut failures = Vec::new();
    let mut chosen = None;
    for candidate in candidates {
        if trusted.as_ref() == Some(&candidate) {
            chosen = Some((candidate, Vec::new()));
            break;
        }
        // The seed's record stands for its measurement at the timed points
        // when the search never measured it.
        if !result
            .configurations
            .iter()
            .any(|record| record.configuration == candidate)
        {
            if let Some(record) = seed.and_then(|seed| {
                seed.configurations
                    .iter()
                    .find(|record| record.configuration == candidate)
            }) {
                result.configurations.push(record.clone());
            }
        }
        match validate_at(formation, &candidate, unweighted, min_sample_seconds) {
            Ok(evidence) => {
                chosen = Some((candidate, evidence));
                break;
            }
            Err(exclusion) => {
                if let Some(record) = result
                    .configurations
                    .iter_mut()
                    .find(|record| record.configuration == candidate)
                {
                    record.outcome = Outcome::Excluded(exclusion.clone());
                }
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

struct FactoredRequest<'a, 'p> {
    device: &'a Arc<DeviceInner>,
    module: &'a CheckedModule,
    entry: EntryId,
    logical: &'a Arc<LogicalEntry>,
    bindings: &'a ElementBindings,
    statics: &'a NativeSpecialization,
    cpu: Option<&'static CpuNativeKernels>,
    points: &'a [PreparedPoint<'p>],
    validation: PrecisionPolicy,
    implementation_identity: String,
    implementation: &'a NativeImplementation,
    default: &'a NativeSpecialization,
    search: SearchPlan,
}

fn tune_factored(request: FactoredRequest<'_, '_>) -> Result<TuningResult, TuneError> {
    let FactoredRequest {
        device,
        module,
        entry,
        logical,
        bindings,
        statics,
        cpu,
        points,
        validation,
        implementation_identity,
        implementation,
        default,
        search,
    } = request;
    let started = Instant::now();
    // Until the seed is measured, the whole allowance; then the allowance
    // less what confirming the finalists against the seed needs.
    let mut deadline = started + search.allowance;
    let mut time = TuningTime::default();
    let shapes = point_shapes(device, logical, points)?;
    let partition = plan::partition(implementation, statics, &shapes)
        .map_err(|error| TuneError::Declaration(format!("factored native plan: {error:?}")))?;
    let began = Instant::now();
    // Each launch formed once with all its code variants and held, so that
    // every specialization assembled below shares them.
    let _held = NativePrepared::hold_launch_variants(
        device,
        module,
        entry,
        bindings,
        default,
        implementation,
        &partition.sources,
    )
    .map_err(|error| TuneError::DefaultUnusable(Exclusion::Formation(prepare_message(error))))?;
    time.forming_seconds += began.elapsed().as_secs_f64();
    let form = |specialization: &NativeSpecialization| {
        NativePrepared::prepare_implementation(
            device,
            module,
            entry,
            logical,
            bindings.clone(),
            specialization.clone(),
            cpu,
            implementation.clone(),
        )
        .map_err(|error| Exclusion::Formation(prepare_message(error)))
    };

    let measuring = MeasureOptions {
        samples: search.settings.samples,
        min_sample_seconds: search.min_sample_seconds,
    };
    let confirmation = MeasureOptions {
        samples: search.settings.confirmation_samples,
        min_sample_seconds: search.min_sample_seconds,
    };
    // Every configuration placed in this run writes its results here.
    let mut outputs = OutputPool::default();
    let mut records = Vec::new();
    let measure_seed = |specialization: &NativeSpecialization, outputs: &mut OutputPool| {
        let kernel = form(specialization)?;
        let measured = measure_factored(
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
        )?;
        Ok::<_, Exclusion>((kernel, measured))
    };
    let began = Instant::now();
    let mut seed = match measure_seed(default, &mut outputs) {
        Ok((kernel, measured)) => Some((default.clone(), kernel, measured)),
        Err(exclusion) => {
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
                        match measure_seed(&candidate, &mut outputs) {
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
    let (seed, default_kernel, reference) =
        seed.ok_or_else(|| TuneError::NoValidatedCandidate(excluded(&records)))?;
    let default = &seed;
    time.measuring_seconds += began.elapsed().as_secs_f64();
    deadline = started
        + search.allowance.saturating_sub(confirmation_time(
            &reference,
            search.settings.confirmed + 1,
            search.settings.confirmation_samples,
        ));
    // A point with the same active launches and parameters runs the same work
    // under every boundary assignment. Reuse its sweep sample across those
    // assignments; confirmation below still takes fresh samples.
    let mut sweep_cache = reference
        .iter()
        .enumerate()
        .map(|(point, measurement)| ((point, measurement.key.clone()), measurement.clone()))
        .collect::<BTreeMap<_, _>>();
    let weighing = Weighing::of(points);
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
            artifact: default_kernel.artifact().0.clone(),
            points: reference.clone(),
            confirmed: Vec::new(),
            validated: true,
        },
    });
    let mut boundaries = Vec::new();
    let mut complete = true;
    'boundaries: for boundary in boundary_assignments(&partition, implementation) {
        if Instant::now() >= deadline {
            complete = false;
            break;
        }
        let base = partition
            .assemble(implementation, statics, &default_choices, &boundary)
            .map_err(|error| {
                TuneError::Declaration(format!("factored native boundary: {error:?}"))
            })?;
        let baseline = if base == *default {
            reference.clone()
        } else {
            let began = Instant::now();
            let kernel = match form(&base) {
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
            match measure_factored(
                implementation,
                &partition.boundary,
                &kernel,
                &base,
                points,
                &measuring,
                None,
                Some(&reference),
                Some(&mut sweep_cache),
                &mut outputs,
            ) {
                Ok(measured) => {
                    time.measuring_seconds += began.elapsed().as_secs_f64();
                    records.push(ConfigurationRecord {
                        configuration: Configuration::of(&base),
                        outcome: Outcome::Measured {
                            artifact: kernel.artifact().0.clone(),
                            points: measured.clone(),
                            confirmed: Vec::new(),
                            validated: true,
                        },
                    });
                    measured
                }
                Err(exclusion) => {
                    records.push(ConfigurationRecord {
                        configuration: Configuration::of(&base),
                        outcome: Outcome::Excluded(exclusion),
                    });
                    continue;
                }
            }
        };
        let base_cost = weighing.cost(&baseline, &reference).total();
        let mut choices = default_choices.clone();
        let mut group_rankings = Vec::with_capacity(partition.groups.len());
        let mut score = base_cost;
        let mut interrupted = false;
        for (group_index, group) in partition.groups.iter().enumerate() {
            let mut ranking = vec![(default_choices[group_index], base_cost)];
            let mut ready = Vec::new();
            for candidate in 0..group.candidates.len() {
                if Instant::now() >= deadline {
                    complete = false;
                    interrupted = true;
                    break;
                }
                if candidate == default_choices[group_index] {
                    continue;
                }
                let mut selected = default_choices.clone();
                selected[group_index] = candidate;
                let specialization = partition
                    .assemble(implementation, statics, &selected, &boundary)
                    .map_err(|error| {
                        TuneError::Declaration(format!("factored native candidate: {error:?}"))
                    })?;
                let began = Instant::now();
                let kernel = match form(&specialization) {
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
            let measured = measure_factored_group(
                implementation,
                &partition.boundary,
                &ready,
                points,
                &measuring,
                &group.launches,
                &baseline,
                Some(&mut sweep_cache),
                &mut outputs,
            );
            time.measuring_seconds += began.elapsed().as_secs_f64();
            for (candidate, outcome) in ready.into_iter().zip(measured) {
                match outcome {
                    Ok(measured) => {
                        let cost = weighing.cost(&measured, &reference).total();
                        records.push(ConfigurationRecord {
                            configuration: Configuration::of(&candidate.specialization),
                            outcome: Outcome::Measured {
                                artifact: candidate.kernel.artifact().0.clone(),
                                points: measured,
                                confirmed: Vec::new(),
                                validated: true,
                            },
                        });
                        ranking.push((candidate.index, cost));
                    }
                    Err(exclusion) => {
                        records.push(ConfigurationRecord {
                            configuration: Configuration::of(&candidate.specialization),
                            outcome: Outcome::Excluded(exclusion),
                        });
                    }
                }
            }
            ranking.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            choices[group_index] = ranking[0].0;
            score += ranking[0].1 - base_cost;
            group_rankings.push(ranking);
            if Instant::now() >= deadline {
                complete = false;
                interrupted = true;
            }
            if interrupted {
                break;
            }
        }
        boundaries.push((score, boundary, choices, group_rankings));
        if interrupted {
            break 'boundaries;
        }
    }
    boundaries.sort_by(|left, right| left.0.total_cmp(&right.0));
    // A boundary moves work between launches. Prefer the default band split
    // when its apparent loss is within one percent of the leading score.
    if let Some(default_index) = boundaries
        .iter()
        .position(|(_, boundary, _, _)| *boundary == default_boundary)
    {
        if boundaries[default_index].0 <= boundaries[0].0 * 1.01 {
            let preferred = boundaries.remove(default_index);
            boundaries.insert(0, preferred);
        }
    }
    // The safety stop can interrupt a group's candidate formation. Its
    // already formed candidates still get their sweep, while later groups
    // retain their defaults. Confirm only that partial boundary and never
    // store its result as a complete search.
    if !complete {
        boundaries.truncate(1);
    }
    let mut overall = default.clone();
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
            let kernel = match form(&base) {
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
                    let kernel = match form(&specialization) {
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
            let kernel = form(&selected);
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
            break;
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
        tuning_identity: device.tuning_identity(),
        entry: super::entry_name(module, entry),
        backend: "metal".into(),
        points: points
            .iter()
            .map(|point| PointRecord {
                label: point.label.clone(),
                weight: point.weight,
                class: point.class.clone(),
            })
            .collect(),
        validation,
        reused: false,
        numerical_evidence: winner_evidence(points, &records, &Configuration::of(&overall))?,
        implementation_identity,
        parameters,
        configurations: records,
        overall: Configuration::of(&overall),
        method: TuningMethod::Factored {
            allowance_seconds: search.allowance.as_secs_f64(),
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
        crate::backends::OpenedKind::Metal(_) => Some(super::abi::Dialect::Metal),
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
/// declaration's values and stable source identity, never that owner.
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
                &implementation.declared_in,
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
    /// The declaration's defaults, with every launch's own parameters.
    default: &'s NativeSpecialization,
    seed: Option<&'s TuningResult>,
}

type Tuning = (Records, NativeSpecialization, TuningMethod, TuningTime);

impl Tuned<'_> {
    fn census(
        &self,
        points: &[PreparedPoint<'_>],
        min_sample_seconds: f64,
        ceiling: Duration,
    ) -> Result<Tuning, TuneError> {
        let mut measurer = Measurer::new(self.formation.implementation);
        let options = MeasureOptions {
            samples: 1,
            min_sample_seconds,
        };
        let mut records = Vec::new();
        let mut time = TuningTime::default();
        // The defaults first, then, when they fail, the first configuration
        // that passes. A launch-scoped configuration also values its
        // launches' own parameters, which only its factored search
        // assembles, so its census measures the defaults alone.
        let default = self.space.default_index();
        let alternatives = if self.formation.implementation.launch_scoped() {
            0
        } else {
            self.space.len()
        };
        for index in
            std::iter::once(default).chain((0..alternatives).filter(|index| *index != default))
        {
            let candidate = if index == default {
                self.default.clone()
            } else {
                specialization(self.statics, &self.space.values(index))
            };
            let began = Instant::now();
            let formed = self.formation.form_all(&[candidate.clone()]).remove(0);
            time.forming_seconds += began.elapsed().as_secs_f64();
            let kernel = match formed {
                Ok(kernel) => kernel,
                Err(exclusion) => {
                    records.push(ConfigurationRecord {
                        configuration: Configuration::of(&candidate),
                        outcome: Outcome::Excluded(exclusion),
                    });
                    continue;
                }
            };
            // Each point's time is its execution: reference, validated
            // invocation and samples.
            let mut admitted = Vec::new();
            let mut measured = Vec::new();
            let mut spent = 0.0;
            let mut previous: Option<(f64, f64)> = None;
            let mut closed = false;
            let mut failure = None;
            for (position, point) in points.iter().enumerate() {
                if let Some((cost, seconds)) = previous {
                    if !point.required {
                        if closed {
                            continue;
                        }
                        let predicted = seconds * point.cost / cost.max(f64::MIN_POSITIVE);
                        if spent + predicted > ceiling.as_secs_f64() {
                            closed = true;
                            continue;
                        }
                    }
                }
                let began = Instant::now();
                point.ensure_reference()?;
                match measurer.measure(
                    &kernel,
                    std::slice::from_ref(point),
                    candidate.params(),
                    &options,
                ) {
                    Ok(mut point_measured) => measured.push(point_measured.remove(0)),
                    Err(exclusion) => {
                        failure = Some(exclusion);
                        break;
                    }
                }
                let seconds = began.elapsed().as_secs_f64();
                spent += seconds;
                previous = Some((point.cost, seconds));
                admitted.push(position);
            }
            time.measuring_seconds += spent;
            if let Some(exclusion) = failure {
                records.push(ConfigurationRecord {
                    configuration: Configuration::of(&candidate),
                    outcome: Outcome::Excluded(exclusion),
                });
                continue;
            }
            records.push(ConfigurationRecord {
                configuration: Configuration::of(&candidate),
                outcome: Outcome::Measured {
                    artifact: kernel.artifact().0.clone(),
                    points: measured,
                    confirmed: Vec::new(),
                    validated: true,
                },
            });
            time.validating_seconds = points.iter().map(|p| p.validation_seconds()).sum::<f64>();
            time.measuring_seconds = (time.measuring_seconds - time.validating_seconds).max(0.);
            return Ok((
                Records {
                    points: folded(&points, &admitted),
                    evidence: winner_evidence(
                        admitted.iter().map(|position| &points[*position]),
                        &records,
                        &Configuration::of(&candidate),
                    )?,
                    records,
                },
                candidate,
                TuningMethod::Census,
                time,
            ));
        }
        Err(TuneError::NoValidatedCandidate(excluded(&records)))
    }

    fn search(&self, points: &[PreparedPoint<'_>], plan: SearchPlan) -> Result<Tuning, TuneError> {
        let began = Instant::now();
        // Confirmation re-measures the defaults and the finalists at every
        // point; the census seed tells what that costs.
        let reserve = self
            .seed
            .and_then(|seed| measured_points(seed, &seed.overall))
            .map_or(Duration::ZERO, |measured| {
                confirmation_time(
                    measured,
                    plan.settings.confirmed + 1,
                    plan.settings.confirmation_samples,
                )
            });
        let start = plan
            .start
            .iter()
            .chain(self.seed.map(|seed| &seed.overall.params))
            .filter_map(|values| self.space.index_of(values))
            .collect::<Vec<_>>();
        let mut live = Live {
            formation: self.formation,
            space: self.space,
            statics: self.statics,
            points: &points,
            measurer: Measurer::new(self.formation.implementation),
            search: MeasureOptions {
                samples: plan.settings.samples,
                min_sample_seconds: plan.min_sample_seconds,
            },
            confirmation: MeasureOptions {
                samples: plan.settings.confirmation_samples,
                min_sample_seconds: plan.min_sample_seconds,
            },
            deadline: began + plan.allowance.saturating_sub(reserve),
            evaluated: HashMap::new(),
            anchor: None,
            seed: self.seed,
            time: TuningTime::default(),
        };
        let trace = search::search(self.space, &start, &plan.settings, &mut live);
        let Live {
            evaluated,
            mut time,
            ..
        } = live;
        let default = self.space.default_index();
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
                let configuration =
                    Configuration::of(&specialization(self.statics, &self.space.values(*index)));
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
            TuningMethod::Search {
                allowance_seconds: plan.allowance.as_secs_f64(),
                settings: plan.settings,
                stop: trace.stop,
            },
            time,
        ))
    }

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

/// Every point's record, the weight of each point a census did not admit
/// folded into the largest admitted point of its class, or into the largest
/// admitted point when its class has none. Points come in ascending cost, so
/// the largest is the last.
fn folded(points: &[PreparedPoint<'_>], admitted: &[usize]) -> Vec<PointRecord> {
    let mut weights = points.iter().map(|point| point.weight).collect::<Vec<_>>();
    let largest = *admitted.last().expect("a census admits its first point");
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

/// The measurement of configuration `configuration` in `result`, when
/// measured.
fn measured_points<'r>(
    result: &'r TuningResult,
    configuration: &Configuration,
) -> Option<&'r [PointMeasurement]> {
    result
        .configurations
        .iter()
        .find(|record| record.configuration == *configuration)
        .and_then(|record| match &record.outcome {
            Outcome::Measured { points, .. } => Some(points.as_slice()),
            Outcome::Excluded(_) => None,
        })
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

/// The configuration tuning starts from and validates against, formed: the
/// declared defaults when every launch at every point fits its formed program
/// on this device, else the admissible configuration nearest them that does.
/// A program's thread limit is known only once formed (Apple M1/M2 pipelines
/// under register pressure admit fewer than the device's 1024), so declaration
/// order alone cannot guarantee the defaults launch.
#[allow(clippy::too_many_arguments)]
fn launchable_default(
    device: &Arc<DeviceInner>,
    module: &CheckedModule,
    entry: EntryId,
    logical: &Arc<LogicalEntry>,
    bindings: &ElementBindings,
    cpu: Option<&'static CpuNativeKernels>,
    implementation: &NativeImplementation,
    statics: &NativeSpecialization,
    points: &[TuningPoint<'_>],
) -> Result<(NativeSpecialization, Arc<NativePrepared>), TuneError> {
    let launchable = |specialization: &NativeSpecialization| {
        let kernel = NativePrepared::prepare_implementation(
            device,
            module,
            entry,
            logical,
            bindings.clone(),
            specialization.clone(),
            cpu,
            implementation.clone(),
        )
        .map_err(|error| Exclusion::Formation(prepare_message(error)))?;
        for args in points.iter().flat_map(|point| &point.rotation) {
            kernel
                .shape(&args.values())
                .map_err(|error| Exclusion::Execution(error.to_string()))?;
        }
        Ok::<_, Exclusion>(kernel)
    };
    let declared = implementation
        .default_specialization(statics)
        .map_err(|error| TuneError::Declaration(error.to_string()))?;
    let refusal = match launchable(&declared) {
        Ok(kernel) => return Ok((declared, kernel)),
        Err(exclusion) => exclusion,
    };
    let mut candidates = implementation
        .admissible(statics)
        .map_err(|error| TuneError::Declaration(error.to_string()))?;
    // Stable: equally distant candidates keep declaration order.
    candidates.sort_by_key(|candidate| {
        let params = candidate
            .params()
            .iter()
            .filter(|(name, value)| declared.param(name) != Some(**value))
            .count();
        let launch_params = candidate
            .launch_params()
            .iter()
            .filter(|((launch, name), value)| declared.launch_param(*launch, name) != Some(**value))
            .count();
        params + launch_params
    });
    candidates
        .into_iter()
        .filter(|candidate| *candidate != declared)
        .find_map(|candidate| {
            launchable(&candidate)
                .ok()
                .map(|kernel| (candidate, kernel))
        })
        .ok_or(TuneError::DefaultUnusable(refusal))
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
        let check = |text: String| {
            check_source(SourceSet::new(vec![SourceFile {
                path: "scale.seismic".into(),
                text,
            }]))
            .unwrap()
        };
        let first = check(source("[1, 2]"));
        let second = check(source("[1, 2]"));
        let changed = check(source("[1, 3]"));
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
