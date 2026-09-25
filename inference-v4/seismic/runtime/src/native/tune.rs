//! Tuning of a native implementation over its declared domain, fast enough
//! to run when programs are prepared.
//!
//! A [`Strategy::Search`] runs the budgeted local search of
//! [`super::search`] over the admissible configurations: the live
//! evaluator forms each batch of configurations concurrently, then measures
//! each at every point, and re-measures the finalists alternating between
//! them. Validation then walks the search's ranking and chooses the first
//! configuration that agrees with the all-defaults configuration. A
//! [`Strategy::Survey`] (development) instead forms, measures and validates
//! every admissible configuration, recording every sample, so searches can
//! be replayed against it.
//!
//! The objective is [`Cost`]: per point, its weight (share of step time)
//! times the configuration's time there relative to the defaults'. A point
//! is measured once per [`PointKey`]: the launches doing work there and the
//! parameters they read, as the declaration names them ([`Influence`]).
//! Configurations that differ only in parameters no launch at a point reads
//! share that point's measurement, so its noise cannot choose those
//! parameters.
//!
//! Validation compares every tensor an entry writes, its results and its
//! `&mut` parameters, against the all-defaults configuration: bit-exact when
//! the arithmetic parameters agree with the defaults (a difference is a
//! misclassified mapping parameter), else within the caller's tolerance.
//! The defaults are the reference and always pass. The tuner never saves or
//! restores state. A point whose entry has `&mut` parameters binds the same
//! tensors for every configuration, so it must bring an initializer that
//! restores them before each validation run; one without is rejected.

use super::search::{
    self, Cost, Evaluator, ParameterValues, PointKey, SearchSettings, SearchSpace,
    SearchSpaceError, SearchStop,
};
use super::timing::{self, PointTiming};
use super::{MeasureOptions, Measurement, NativePrepared};
use crate::api::device::DeviceInner;
use crate::api::kernel::{DecodedValue, EncodedArgs, PrepareError};
use crate::api::{CallError, TensorError};
use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::checked::{CheckedModule, NativeImplementation, NativeSpecialization};
use seismic_lang::entry::{ElementBindings, ParameterKind, TensorAccess};
use seismic_lang::ids::{EntryId, RepresentationId};
use seismic_lang::registry::{self, RepresentationKind};
use seismic_lang::types::DType;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

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
    /// Argument sets cycled through by measurement. The first is also the
    /// validation input.
    pub rotation: Vec<EncodedArgs>,
    /// Required when the entry has `&mut` parameters: called before every
    /// configuration's validation run, it restores the tensors those
    /// parameters bind in `rotation[0]`.
    pub initialize: Option<TuningInitializer<'a>>,
}

/// How configurations must agree with the all-defaults configuration.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Validation {
    BitExact,
    /// Each dense floating-point result may differ from the reference by at
    /// most `error` of the reference's norm: `‖actual − reference‖₂ ≤ error ·
    /// ‖reference‖₂` over the whole tensor. Reduced-precision operands and
    /// reassociated sums perturb every element by a share of the output's
    /// scale, not of the element's own value, so near-zero elements carry
    /// errors far above their own magnitude; the norm bound admits that and
    /// still rejects a wrong result. Every other result is bit-exact.
    Relative { error: f64 },
}

/// A configuration as recorded: its static values and parameter values.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Configuration {
    pub statics: BTreeMap<String, u64>,
    pub params: BTreeMap<String, u64>,
}

impl Configuration {
    fn of(specialization: &NativeSpecialization) -> Self {
        Self {
            statics: specialization.statics().clone(),
            params: specialization.params().clone(),
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
    /// A mapping parameter changed result bits relative to the defaults,
    /// which share its arithmetic parameters.
    MisclassifiedParameter {
        point: String,
        reference: Configuration,
    },
    /// Results disagree with the all-defaults configuration.
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
        points: Vec<PointMeasurement>,
        /// The finalists' re-measurement (empty for every other
        /// configuration).
        confirmed: Vec<PointMeasurement>,
        /// Whether its outputs were compared with the defaults' (and
        /// agreed). A search validates down its ranking only until one
        /// configuration passes; a survey validates every configuration.
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
    pub arithmetic: bool,
    pub values: Vec<u64>,
}

/// How the recorded configurations were reached.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TuningMethod {
    Search {
        budget: usize,
        settings: SearchSettings,
        stop: SearchStop,
    },
    /// Every admissible configuration, `samples` per point.
    Survey { samples: usize },
}

/// Where tuning time went.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TuningTime {
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
    pub validation: Validation,
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
    /// Configurations excluded for authoring defects rather than device
    /// limits.
    pub fn defects(&self) -> impl Iterator<Item = &ConfigurationRecord> {
        self.configurations.iter().filter(|record| {
            matches!(
                record.outcome,
                Outcome::Excluded(
                    Exclusion::Formation(_)
                        | Exclusion::MisclassifiedParameter { .. }
                        | Exclusion::Validation { .. }
                )
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
    SharedMutableState { point: String, parameter: String },
    /// A point's initializer failed.
    Initialization { point: String, detail: String },
}

impl std::fmt::Display for TuneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Declaration(message) | Self::Domain(message) => write!(f, "{message}"),
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

/// A budgeted search (production).
#[derive(Clone, Debug)]
pub struct SearchPlan {
    /// Configurations the search may evaluate, the defaults included.
    pub budget: usize,
    pub settings: SearchSettings,
    /// Minimum device time of one sample; sets repetitions per sample.
    pub min_sample_seconds: f64,
    /// Configurations evaluated with the defaults at the start (winners of
    /// the same declaration elsewhere); inadmissible ones are skipped.
    pub start: Vec<ParameterValues>,
    /// The safety stop: past it, the search ends with the best found.
    pub deadline: Option<Instant>,
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
    pub validation: Validation,
    pub strategy: Strategy,
}

/// What one run wrote, read back at once: the in-place parameters are
/// overwritten by the next configuration.
#[derive(Clone)]
enum Observed {
    Tensor {
        representation: RepresentationId,
        bytes: Vec<u8>,
    },
    Scalar(ArgumentValue),
}

/// Every result, then every `&mut` parameter, of one run.
type Outputs = Vec<Observed>;

/// Forms configurations of one entry's implementation.
struct Formation<'s> {
    device: &'s Arc<DeviceInner>,
    module: &'s CheckedModule,
    entry: EntryId,
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

fn measurement_failure(points: &[TuningPoint<'_>], point: usize, error: CallError) -> Exclusion {
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
struct Placed {
    timings: Vec<PointTiming>,
    keys: Vec<PointKey>,
}

/// Place `kernel`'s calls at every point and key them.
fn place(
    kernel: &Arc<NativePrepared>,
    points: &[TuningPoint<'_>],
    influence: &Influence,
    values: &ParameterValues,
) -> Result<Placed, Exclusion> {
    let timings = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            PointTiming::new(kernel, point.rotation.clone())
                .map_err(|error| measurement_failure(points, index, error))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let keys = timings
        .iter()
        .map(|timing| influence.key(timing.working_launches(), values))
        .collect();
    Ok(Placed { timings, keys })
}

/// Measures configurations at the tuning points, once per point and key.
struct Measurer {
    influence: Influence,
    /// Every point measured so far, by point and key.
    measured: HashMap<(usize, PointKey), PointMeasurement>,
}

impl Measurer {
    fn new(implementation: &NativeImplementation) -> Self {
        Self {
            influence: Influence::of(implementation),
            measured: HashMap::new(),
        }
    }

    /// The measurement of `kernel` at every point: points whose key was
    /// measured before reuse that measurement, the others are sampled.
    fn measure(
        &mut self,
        kernel: &Arc<NativePrepared>,
        points: &[TuningPoint<'_>],
        values: &ParameterValues,
        options: &MeasureOptions,
    ) -> Result<Vec<PointMeasurement>, Exclusion> {
        let Placed { timings, keys } = place(kernel, points, &self.influence, values)?;
        let (fresh, mut timings): (Vec<usize>, Vec<PointTiming>) = timings
            .into_iter()
            .enumerate()
            .filter(|(point, _)| !self.measured.contains_key(&(*point, keys[*point].clone())))
            .unzip();
        timing::sample(&mut timings, options)
            .map_err(|failure| measurement_failure(points, fresh[failure.point], failure.error))?;
        for (point, timing) in fresh.into_iter().zip(&timings) {
            let key = keys[point].clone();
            let measurement = point_measurement(&points[point].label, key.clone(), timing.measurement());
            self.measured.insert((point, key), measurement);
        }
        Ok(keys
            .into_iter()
            .enumerate()
            .map(|(point, key)| self.measured[&(point, key)].clone())
            .collect())
    }
}

/// Bring the device to its sustained clock after forming a batch left it
/// idle ([`timing::warm`]), with the first formed configuration's first
/// point. A configuration that cannot run here fails the same way when it
/// is measured and is excluded there.
fn warm(formed: &[Result<Arc<NativePrepared>, Exclusion>], points: &[TuningPoint<'_>]) {
    let Some(kernel) = formed.iter().find_map(|kernel| kernel.as_ref().ok()) else {
        return;
    };
    if let Ok(point) = PointTiming::new(kernel, points[0].rotation.clone()) {
        let _ = timing::warm(&point);
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
            detail: format!("confirmed samples deviate {:.0}% from their median", deviation * 100.0),
        })
    })
}

/// The points' weights and classes: how the objective weighs them.
struct Weighing {
    weights: Vec<f64>,
    classes: Vec<usize>,
}

impl Weighing {
    fn of(points: &[TuningPoint<'_>]) -> Self {
        Self {
            weights: points.iter().map(|point| point.weight).collect(),
            classes: search::classes(points.iter().map(|point| point.class.as_deref())),
        }
    }

    /// The cost of `measured` relative to the defaults' measurement at the
    /// same points.
    fn cost(&self, measured: &[PointMeasurement], reference: &[PointMeasurement]) -> Cost {
        Cost::relative(
            measured.iter().map(|measurement| measurement.key.clone()).collect(),
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

fn specialization(statics: &NativeSpecialization, values: &ParameterValues) -> NativeSpecialization {
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
    points: &'s [TuningPoint<'a>],
    measurer: Measurer,
    search: MeasureOptions,
    confirmation: MeasureOptions,
    deadline: Option<Instant>,
    evaluated: HashMap<usize, Evaluated>,
    time: TuningTime,
}

impl Live<'_, '_> {
    /// The defaults' search measurement: the reference of every cost.
    fn reference(&self) -> Result<&[PointMeasurement], Exclusion> {
        self.evaluated
            .get(&self.space.default_index())
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
        warm(&formed, self.points);
        let weighing = Weighing::of(self.points);
        let costs = batch
            .iter()
            .zip(formed)
            .map(|(index, kernel)| {
                let kernel = kernel?;
                let points =
                    self.measurer
                        .measure(&kernel, self.points, &self.space.values(*index), &self.search)?;
                self.evaluated.insert(
                    *index,
                    Evaluated {
                        kernel,
                        points: points.clone(),
                        confirmed: Vec::new(),
                    },
                );
                Ok(weighing.cost(&points, self.reference()?))
            })
            .collect();
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
            failed.insert((point, key), measurement_failure(points, point, failure.error));
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
        // The finalists begin with the defaults, the reference.
        let reference = confirmed[0].clone();
        finalists
            .iter()
            .zip(confirmed)
            .map(|(index, measured)| {
                let measured = measured?;
                let reference = reference.as_ref().map_err(Clone::clone)?;
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
        self.deadline.is_some_and(|deadline| Instant::now() >= deadline)
    }
}

/// Validates configurations against the all-defaults configuration.
struct Validator<'r, 'a> {
    implementation: &'r NativeImplementation,
    default: &'r NativeSpecialization,
    reference: Vec<Outputs>,
    mutable: &'r [usize],
    validation: Validation,
    points: &'r mut [TuningPoint<'a>],
}

impl Validator<'_, '_> {
    /// Run `kernel` at every point and compare with the reference: bit-exact
    /// when its arithmetic parameters equal the defaults', else within the
    /// tolerance.
    fn check(
        &mut self,
        kernel: &Arc<NativePrepared>,
        candidate: &NativeSpecialization,
    ) -> Result<Result<(), Exclusion>, TuneError> {
        let exact = arithmetic_values(self.implementation, candidate)
            == arithmetic_values(self.implementation, self.default);
        let outputs = match run_points(kernel, self.points, self.mutable)? {
            Err(exclusion) => return Ok(Err(exclusion)),
            Ok(outputs) => outputs,
        };
        let mode = if exact {
            Validation::BitExact
        } else {
            self.validation
        };
        Ok(self
            .points
            .iter()
            .zip(self.reference.iter().zip(&outputs))
            .find_map(|(point, (expected, actual))| {
                compare(expected, actual, mode).err().map(|detail| {
                    if exact {
                        Exclusion::MisclassifiedParameter {
                            point: point.label.clone(),
                            reference: Configuration::of(self.default),
                        }
                    } else {
                        Exclusion::Validation {
                            point: point.label.clone(),
                            detail,
                        }
                    }
                })
            })
            .map_or(Ok(()), Err))
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
    } = request;
    let backend = super::backend_name(&device.kind);
    let entry_name = super::entry_name(module, entry);
    let mut implementation = super::on_device(
        device,
        module.native_implementation(entry, backend).cloned().ok_or_else(|| {
            TuneError::Declaration(format!(
                "`{entry_name}` has no native implementation for `{}`",
                backend.as_str()
            ))
        })?,
    );
    if let Strategy::Survey(plan) = &strategy {
        widen(&mut implementation, &plan.domains)?;
    }
    let admissible = implementation
        .admissible(&statics)
        .map_err(|error| TuneError::Declaration(error.to_string()))?;
    let default = implementation
        .default_specialization(&statics)
        .map_err(|error| TuneError::Declaration(error.to_string()))?;
    let mutable = mutable_parameters(module, entry, &bindings)?;
    if let (Some((_, parameter)), Some(point)) = (
        mutable.first(),
        points.iter().find(|point| point.initialize.is_none()),
    ) {
        return Err(TuneError::SharedMutableState {
            point: point.label.clone(),
            parameter: parameter.clone(),
        });
    }
    let mutable = mutable.into_iter().map(|(ordinal, _)| ordinal).collect::<Vec<_>>();
    let declared = implementation
        .params
        .iter()
        .map(|parameter| (parameter.name.clone(), parameter.values.clone()))
        .collect::<Vec<_>>();
    let space = SearchSpace::new(
        &declared,
        &admissible
            .iter()
            .map(|specialization| specialization.params().clone())
            .collect::<Vec<_>>(),
    )
    .map_err(TuneError::Space)?;
    let formation = Formation {
        device,
        module,
        entry,
        bindings: &bindings,
        cpu,
        implementation: &implementation,
    };
    let tuned = Tuned {
        formation: &formation,
        space: &space,
        statics: &statics,
        default: &default,
        mutable: &mutable,
        validation,
    };
    let (configurations, overall, method, time) = match strategy {
        Strategy::Search(plan) => tuned.search(points, plan)?,
        Strategy::Survey(plan) => tuned.survey(points, plan)?,
    };
    Ok(TuningResult {
        tuning_identity: device.tuning_identity(),
        entry: entry_name,
        backend: backend.as_str().to_owned(),
        points: configurations.points,
        validation,
        parameters: implementation
            .params
            .iter()
            .map(|parameter| DeclaredParameter {
                name: parameter.name.clone(),
                arithmetic: parameter.arithmetic,
                values: parameter.values.clone(),
            })
            .collect(),
        configurations: configurations.records,
        overall: Configuration::of(&overall),
        method,
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
    use sha2::{Digest, Sha256};
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
    digest.update(format!("{implementation:?}").as_bytes());
    // Backends whose implementations are rendered source; CPU implementations
    // are compiled into the binary and identified by their compiled digest.
    let dialect = match &device.kind {
        crate::backends::OpenedKind::Cpu(_) => {
            let kernels = cpu.ok_or_else(|| {
                TuneError::Declaration(format!("`{entry_name}` has no compiled CPU native functions in this build"))
            })?;
            digest.update(kernels.digest.as_bytes());
            None
        }
        #[cfg(target_os = "macos")]
        crate::backends::OpenedKind::Metal(_) => Some(super::abi::Dialect::Metal),
        crate::backends::OpenedKind::Cuda(_) => Some(super::abi::Dialect::Cuda),
        #[cfg(not(target_os = "macos"))]
        crate::backends::OpenedKind::Vulkan(opened) => Some(super::abi::Dialect::Vulkan(opened.features())),
    };
    if let Some(dialect) = dialect {
        let logical = module
            .entry(entry, bindings)
            .map_err(|error| TuneError::Declaration(error.to_string()))?;
        let asset = module.native_asset(entry, backend).ok_or_else(|| {
            TuneError::Declaration(format!("native asset of `{entry_name}` is absent from the module"))
        })?;
        digest.update(
            super::abi::render_source(dialect, &logical, bindings, implementation, &default, asset)
                .as_bytes(),
        );
    }
    Ok(crate::telemetry::hex(&digest.finalize()))
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
}

/// One entry's tuning, shared by both strategies.
struct Tuned<'s> {
    formation: &'s Formation<'s>,
    space: &'s SearchSpace,
    statics: &'s NativeSpecialization,
    default: &'s NativeSpecialization,
    mutable: &'s [usize],
    validation: Validation,
}

type Tuning = (Records, NativeSpecialization, TuningMethod, TuningTime);

impl Tuned<'_> {
    fn search(&self, mut points: Vec<TuningPoint<'_>>, plan: SearchPlan) -> Result<Tuning, TuneError> {
        let default_index = self.space.default_index();
        let start = plan
            .start
            .iter()
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
            deadline: plan.deadline,
            evaluated: HashMap::new(),
            time: TuningTime::default(),
        };
        let trace = search::search(self.space, &start, plan.budget, &plan.settings, &mut live);
        let Live {
            evaluated,
            mut time,
            ..
        } = live;
        let default_kernel = match trace
            .evaluated
            .iter()
            .find(|(index, _)| *index == default_index)
            .map(|(_, result)| result)
        {
            Some(Ok(_)) => evaluated[&default_index].kernel.clone(),
            Some(Err(exclusion)) => return Err(TuneError::DefaultUnusable(exclusion.clone())),
            None => unreachable!("the search evaluates the defaults"),
        };

        // Validate down the ranking; the defaults are the reference and
        // always pass.
        let began = Instant::now();
        let reference = run_points(&default_kernel, &mut points, self.mutable)?
            .map_err(TuneError::DefaultUnusable)?;
        let mut validator = Validator {
            implementation: self.formation.implementation,
            default: self.default,
            reference,
            mutable: self.mutable,
            validation: self.validation,
            points: &mut points,
        };
        let mut verdicts: HashMap<usize, Result<(), Exclusion>> = HashMap::new();
        let mut chosen = default_index;
        for &index in &trace.ranking {
            let verdict = if index == default_index {
                Ok(())
            } else {
                let candidate = specialization(self.statics, &self.space.values(index));
                validator.check(&evaluated[&index].kernel, &candidate)?
            };
            let passed = verdict.is_ok();
            verdicts.insert(index, verdict);
            if passed {
                chosen = index;
                break;
            }
        }
        time.validating_seconds += began.elapsed().as_secs_f64();

        let records = trace
            .evaluated
            .iter()
            .map(|(index, result)| {
                let configuration =
                    Configuration::of(&specialization(self.statics, &self.space.values(*index)));
                // A finalist that could not be confirmed is excluded; the
                // defaults stay the fallback choice whatever their
                // confirmation showed.
                let unconfirmed = trace
                    .confirmed
                    .iter()
                    .find(|(finalist, _)| finalist == index && *index != default_index)
                    .and_then(|(_, confirmed)| confirmed.as_ref().err());
                let outcome = match (result, unconfirmed, verdicts.get(index)) {
                    (Err(exclusion), _, _)
                    | (Ok(_), Some(exclusion), _)
                    | (Ok(_), None, Some(Err(exclusion))) => Outcome::Excluded(exclusion.clone()),
                    (Ok(_), None, verdict) => {
                        let measured = &evaluated[index];
                        Outcome::Measured {
                            artifact: measured.kernel.artifact().0.clone(),
                            points: measured.points.clone(),
                            confirmed: measured.confirmed.clone(),
                            validated: verdict.is_some(),
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
                records,
            },
            specialization(self.statics, &self.space.values(chosen)),
            TuningMethod::Search {
                budget: plan.budget,
                settings: plan.settings,
                stop: trace.stop,
            },
            time,
        ))
    }

    fn survey(&self, mut points: Vec<TuningPoint<'_>>, plan: SurveyPlan) -> Result<Tuning, TuneError> {
        let options = MeasureOptions {
            samples: plan.samples,
            min_sample_seconds: plan.min_sample_seconds,
        };
        let mut time = TuningTime::default();
        let began = Instant::now();
        let default_kernel = self
            .formation
            .form_all(std::slice::from_ref(self.default))
            .pop()
            .expect("one configuration was formed")
            .map_err(TuneError::DefaultUnusable)?;
        time.forming_seconds += began.elapsed().as_secs_f64();
        let began = Instant::now();
        let reference = run_points(&default_kernel, &mut points, self.mutable)?
            .map_err(TuneError::DefaultUnusable)?;
        time.validating_seconds += began.elapsed().as_secs_f64();
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
            warm(&formed, &points);
            for ((index, candidate), kernel) in batch.iter().zip(&specializations).zip(formed) {
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
                    Ok((kernel, measured)) => {
                        let mut validator = Validator {
                            implementation: self.formation.implementation,
                            default: self.default,
                            reference: reference.clone(),
                            mutable: self.mutable,
                            validation: self.validation,
                            points: &mut points,
                        };
                        match validator.check(&kernel, candidate)? {
                            Err(exclusion) => Outcome::Excluded(exclusion),
                            Ok(()) => Outcome::Measured {
                                artifact: kernel.artifact().0.clone(),
                                points: measured,
                                confirmed: Vec::new(),
                                validated: true,
                            },
                        }
                    }
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
        let reference = match &outcomes[self.space.default_index()] {
            Outcome::Measured { points, .. } => points.clone(),
            Outcome::Excluded(exclusion) => return Err(TuneError::DefaultUnusable(exclusion.clone())),
        };
        let weighing = Weighing::of(&points);
        let chosen = (0..outcomes.len())
            .filter_map(|index| {
                measured(&outcomes[index]).map(|points| (index, weighing.cost(&points, &reference).total()))
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .expect("the defaults were measured");
        let records = outcomes
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

fn labels(points: &[TuningPoint<'_>]) -> Vec<PointRecord> {
    points
        .iter()
        .map(|point| PointRecord {
            label: point.label.clone(),
            weight: point.weight,
            class: point.class.clone(),
        })
        .collect()
}

fn arithmetic_values(
    implementation: &NativeImplementation,
    specialization: &NativeSpecialization,
) -> Vec<u64> {
    implementation
        .params
        .iter()
        .filter(|parameter| parameter.arithmetic)
        .map(|parameter| {
            specialization
                .param(&parameter.name)
                .expect("admissible configurations value every parameter")
        })
        .collect()
}

/// Ordinal and name of every `&mut` tensor parameter.
fn mutable_parameters(
    module: &CheckedModule,
    entry: EntryId,
    bindings: &ElementBindings,
) -> Result<Vec<(usize, String)>, TuneError> {
    let logical = module
        .entry(entry, bindings)
        .map_err(|error| TuneError::Declaration(error.to_string()))?;
    Ok(logical
        .schema()
        .parameters()
        .iter()
        .enumerate()
        .filter(|(_, parameter)| {
            matches!(
                parameter.kind,
                ParameterKind::Tensor {
                    access: TensorAccess::Mutable,
                    ..
                }
            )
        })
        .map(|(ordinal, parameter)| (ordinal, parameter.name.clone()))
        .collect())
}

fn prepare_message(error: PrepareError) -> String {
    match error {
        PrepareError::Source(error) => error.to_string(),
        PrepareError::Preparation(error) => error.to_string(),
    }
}

/// Run every point's validation input once: initialize its in-place
/// state, call, and read back the results and the `&mut` parameters. An
/// initializer failure is the case's error; a call failure excludes the
/// configuration.
fn run_points(
    kernel: &Arc<NativePrepared>,
    points: &mut [TuningPoint<'_>],
    mutable: &[usize],
) -> Result<Result<Vec<Outputs>, Exclusion>, TuneError> {
    let mut outputs = Vec::with_capacity(points.len());
    for point in points {
        if let Some(initialize) = &mut point.initialize {
            initialize().map_err(|error| TuneError::Initialization {
                point: point.label.clone(),
                detail: error.to_string(),
            })?;
        }
        let args = point.rotation[0].clone();
        let observed = kernel
            .call(args.clone())
            .map_err(|error| Exclusion::Execution(error.to_string()))
            .and_then(|results| {
                let written = mutable.iter().map(|ordinal| {
                    DecodedValue::Tensor(
                        args.tensor(*ordinal)
                            .expect("a mutable parameter is a tensor argument")
                            .clone(),
                    )
                });
                results
                    .into_values()
                    .into_iter()
                    .chain(written)
                    .map(observe)
                    .collect::<Result<Outputs, _>>()
            });
        match observed {
            Ok(observed) => outputs.push(observed),
            Err(exclusion) => return Ok(Err(exclusion)),
        }
    }
    Ok(Ok(outputs))
}

fn observe(value: DecodedValue) -> Result<Observed, Exclusion> {
    match value {
        DecodedValue::Tensor(tensor) => Ok(Observed::Tensor {
            representation: tensor.representation(),
            bytes: tensor
                .read_to_host()
                .map_err(|error| Exclusion::Execution(error.to_string()))?,
        }),
        DecodedValue::Scalar(value) => Ok(Observed::Scalar(value)),
    }
}

fn compare(expected: &Outputs, actual: &Outputs, validation: Validation) -> Result<(), String> {
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        match (expected, actual) {
            (
                Observed::Tensor {
                    representation,
                    bytes: expected_bytes,
                },
                Observed::Tensor {
                    bytes: actual_bytes,
                    ..
                },
            ) => {
                if expected_bytes == actual_bytes {
                    continue;
                }
                let Validation::Relative { error } = validation else {
                    return Err(format!("output {index} differs"));
                };
                let dtype = match &registry::representation_info(*representation).kind {
                    RepresentationKind::Dense(dtype) => *dtype,
                    _ => return Err(format!("packed output {index} differs")),
                };
                let decode = |bytes: &[u8], element: usize| -> f64 {
                    match dtype {
                        DType::F32 => f64::from(f32::from_le_bytes(
                            bytes[element * 4..element * 4 + 4]
                                .try_into()
                                .expect("f32 element"),
                        )),
                        DType::F16 => f64::from(registry::f16_to_f32(u16::from_le_bytes(
                            bytes[element * 2..element * 2 + 2]
                                .try_into()
                                .expect("f16 element"),
                        ))),
                        DType::BF16 => f64::from(f32::from_bits(
                            u32::from(u16::from_le_bytes(
                                bytes[element * 2..element * 2 + 2]
                                    .try_into()
                                    .expect("bf16 element"),
                            )) << 16,
                        )),
                        _ => f64::NAN,
                    }
                };
                if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
                    return Err(format!("integer output {index} differs"));
                }
                let elements = expected_bytes.len() / dtype.bytes() as usize;
                let (mut difference, mut norm) = (0.0f64, 0.0f64);
                for element in 0..elements {
                    let reference = decode(expected_bytes, element);
                    let value = decode(actual_bytes, element);
                    // Equal non-finite values (masked scores) agree; any
                    // other non-finite value is a wrong result.
                    if value == reference || (value.is_nan() && reference.is_nan()) {
                        norm += if reference.is_finite() { reference * reference } else { 0.0 };
                        continue;
                    }
                    if !(value.is_finite() && reference.is_finite()) {
                        return Err(format!(
                            "output {index} element {element}: {value} vs {reference}"
                        ));
                    }
                    difference += (value - reference) * (value - reference);
                    norm += reference * reference;
                }
                let (difference, norm) = (difference.sqrt(), norm.sqrt());
                if difference > error * norm {
                    return Err(format!(
                        "output {index}: error norm {difference:.4e} exceeds {error} of the reference norm {norm:.4e} ({:.4})",
                        difference / norm
                    ));
                }
            }
            (Observed::Scalar(expected), Observed::Scalar(actual)) => {
                if !scalar_equal(expected, actual) {
                    return Err(format!("scalar output {index} differs"));
                }
            }
            _ => return Err(format!("output {index} changed kind")),
        }
    }
    Ok(())
}

fn scalar_equal(expected: &ArgumentValue, actual: &ArgumentValue) -> bool {
    match (expected, actual) {
        (ArgumentValue::F32(left), ArgumentValue::F32(right)) => left.to_bits() == right.to_bits(),
        (left, right) => left == right,
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
}
