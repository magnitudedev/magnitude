//! Tuning of a native implementation over its declared domain, fast enough
//! to run when programs are prepared.
//!
//! A [`Strategy::Search`] runs the budgeted local search of
//! [`super::search`] over the admissible configurations: the live
//! evaluator forms each batch of configurations concurrently, then measures
//! each at every point (weighted cost `Σ weight × median`), and re-measures
//! the finalists alternating between them. Validation then walks the
//! search's ranking and chooses the first configuration that agrees with
//! the all-defaults configuration. A [`Strategy::Survey`] (development)
//! instead forms, measures and validates every admissible configuration,
//! recording every sample, so searches can be replayed against it.
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
    self, Evaluator, ParameterValues, SearchSettings, SearchSpace, SearchSpaceError, SearchStop,
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
    /// Share of the objective.
    pub weight: f64,
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

/// The complete, serializable outcome of tuning one implementation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuningResult {
    pub tuning_identity: String,
    pub entry: String,
    pub backend: String,
    /// Points in the order given, with their weights.
    pub points: Vec<(String, f64)>,
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

/// Place `kernel`'s calls at every point.
fn timings(
    kernel: &Arc<NativePrepared>,
    points: &[TuningPoint<'_>],
) -> Result<Vec<PointTiming>, Exclusion> {
    points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            PointTiming::new(kernel, point.rotation.clone())
                .map_err(|error| measurement_failure(points, index, error))
        })
        .collect()
}

/// The measurements of one configuration's points and its weighted cost.
fn costed(points: &[TuningPoint<'_>], timings: &[PointTiming]) -> (Vec<PointMeasurement>, f64) {
    let measured = points
        .iter()
        .zip(timings)
        .map(|(point, timing)| point_measurement(&point.label, timing.measurement()))
        .collect::<Vec<_>>();
    let cost = points
        .iter()
        .zip(&measured)
        .map(|(point, measurement)| point.weight * measurement.median_seconds)
        .sum();
    (measured, cost)
}

fn measure(
    kernel: &Arc<NativePrepared>,
    points: &[TuningPoint<'_>],
    options: &MeasureOptions,
) -> Result<(Vec<PointMeasurement>, f64), Exclusion> {
    let mut timings = timings(kernel, points)?;
    timing::sample(&mut timings, options)
        .map_err(|failure| measurement_failure(points, failure.point, failure.error))?;
    Ok(costed(points, &timings))
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
    search: MeasureOptions,
    confirmation: MeasureOptions,
    deadline: Option<Instant>,
    evaluated: HashMap<usize, Evaluated>,
    time: TuningTime,
}

impl Evaluator for Live<'_, '_> {
    fn evaluate(&mut self, batch: &[usize]) -> Vec<Result<f64, Exclusion>> {
        let specializations = batch
            .iter()
            .map(|index| specialization(self.statics, &self.space.values(*index)))
            .collect::<Vec<_>>();
        let began = Instant::now();
        let formed = self.formation.form_all(&specializations);
        let measuring = Instant::now();
        self.time.forming_seconds += (measuring - began).as_secs_f64();
        let costs = batch
            .iter()
            .zip(formed)
            .map(|(index, kernel)| {
                let kernel = kernel?;
                let (points, cost) = measure(&kernel, self.points, &self.search)?;
                self.evaluated.insert(
                    *index,
                    Evaluated {
                        kernel,
                        points,
                        confirmed: Vec::new(),
                    },
                );
                Ok(cost)
            })
            .collect();
        self.time.measuring_seconds += measuring.elapsed().as_secs_f64();
        costs
    }

    fn confirm(&mut self, finalists: &[usize]) -> Vec<Result<f64, Exclusion>> {
        let began = Instant::now();
        let count = self.points.len();
        let mut results: Vec<Option<Result<f64, Exclusion>>> = vec![None; finalists.len()];
        // Finalists still being confirmed, by position in `finalists`.
        let mut active = (0..finalists.len()).collect::<Vec<_>>();
        while !active.is_empty() {
            // Every point of every active finalist, finalist by finalist:
            // sampling round by round alternates between the finalists.
            let placed = active
                .iter()
                .map(|&finalist| timings(&self.evaluated[&finalists[finalist]].kernel, self.points))
                .collect::<Vec<_>>();
            if let Some(position) = placed.iter().position(Result::is_err) {
                let failed = active.remove(position);
                results[failed] = placed.into_iter().nth(position).map(|placed| {
                    Err(placed.err().expect("the failed placement is an error"))
                });
                continue;
            }
            let mut flat = placed
                .into_iter()
                .flat_map(|placed| placed.expect("placements succeeded"))
                .collect::<Vec<_>>();
            match timing::sample(&mut flat, &self.confirmation) {
                Ok(()) => {
                    for (position, &finalist) in active.iter().enumerate() {
                        let timings = &flat[position * count..(position + 1) * count];
                        let (points, cost) = costed(self.points, timings);
                        self.evaluated
                            .get_mut(&finalists[finalist])
                            .expect("finalists were evaluated")
                            .confirmed = points;
                        results[finalist] = Some(Ok(cost));
                    }
                    active.clear();
                }
                Err(failure) => {
                    let failed = active.remove(failure.point / count);
                    results[failed] = Some(Err(measurement_failure(
                        self.points,
                        failure.point % count,
                        failure.error,
                    )));
                }
            }
        }
        self.time.measuring_seconds += began.elapsed().as_secs_f64();
        results
            .into_iter()
            .map(|result| result.expect("every finalist was confirmed or failed"))
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
    let mut implementation = module
        .native_implementation(entry, backend)
        .cloned()
        .ok_or_else(|| {
            TuneError::Declaration(format!(
                "`{entry_name}` has no native implementation for `{}`",
                backend.as_str()
            ))
        })?;
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
/// (parameters and their domains, `where`, launches, scratch) and, for Metal
/// and CUDA, the source rendered for the defaults at these bindings and
/// static values (the asset with its inlined includes, and the ABI prefix).
/// Embedders key stored tuning results with it.
pub fn implementation_digest(
    device: &Arc<DeviceInner>,
    module: &CheckedModule,
    entry: EntryId,
    bindings: &ElementBindings,
    statics: &NativeSpecialization,
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
    // are compiled into the binary and identified by their declaration.
    let dialect = match &device.kind {
        crate::backends::OpenedKind::Cpu(_) => None,
        #[cfg(target_os = "macos")]
        crate::backends::OpenedKind::Metal(_) => Some(super::abi::Dialect::Metal),
        crate::backends::OpenedKind::Cuda(_) => Some(super::abi::Dialect::Cuda),
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
    points: Vec<(String, f64)>,
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
                let outcome = match (result, verdicts.get(index)) {
                    (Err(exclusion), _) | (Ok(_), Some(Err(exclusion))) => {
                        Outcome::Excluded(exclusion.clone())
                    }
                    (Ok(_), verdict) => {
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
        let mut outcomes: Vec<Option<(Outcome, f64)>> = vec![None; self.space.len()];
        let order = (0..self.space.len()).collect::<Vec<_>>();
        for batch in order.chunks(workers) {
            let specializations = batch
                .iter()
                .map(|index| specialization(self.statics, &self.space.values(*index)))
                .collect::<Vec<_>>();
            let began = Instant::now();
            let formed = self.formation.form_all(&specializations);
            time.forming_seconds += began.elapsed().as_secs_f64();
            for ((index, candidate), kernel) in batch.iter().zip(&specializations).zip(formed) {
                let began = Instant::now();
                let measured = kernel.and_then(|kernel| {
                    measure(&kernel, &points, &options).map(|(measured, cost)| (kernel, measured, cost))
                });
                time.measuring_seconds += began.elapsed().as_secs_f64();
                let began = Instant::now();
                let outcome = match measured {
                    Err(exclusion) => (Outcome::Excluded(exclusion), f64::INFINITY),
                    Ok((kernel, measured, cost)) => {
                        let mut validator = Validator {
                            implementation: self.formation.implementation,
                            default: self.default,
                            reference: reference.clone(),
                            mutable: self.mutable,
                            validation: self.validation,
                            points: &mut points,
                        };
                        match validator.check(&kernel, candidate)? {
                            Err(exclusion) => (Outcome::Excluded(exclusion), f64::INFINITY),
                            Ok(()) => (
                                Outcome::Measured {
                                    artifact: kernel.artifact().0.clone(),
                                    points: measured,
                                    confirmed: Vec::new(),
                                    validated: true,
                                },
                                cost,
                            ),
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
        let chosen = (0..outcomes.len())
            .min_by(|left, right| outcomes[*left].1.total_cmp(&outcomes[*right].1))
            .filter(|index| outcomes[*index].1.is_finite())
            .unwrap_or(self.space.default_index());
        let records = outcomes
            .into_iter()
            .enumerate()
            .map(|(index, (outcome, _))| ConfigurationRecord {
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

fn labels(points: &[TuningPoint<'_>]) -> Vec<(String, f64)> {
    points
        .iter()
        .map(|point| (point.label.clone(), point.weight))
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

fn point_measurement(label: &str, measurement: Measurement) -> PointMeasurement {
    PointMeasurement {
        point: label.to_owned(),
        median_seconds: measurement.median,
        deviation_seconds: measurement.deviation,
        samples: measurement.samples,
        repetitions: measurement.repetitions,
        rotation_bytes: measurement.rotation_bytes,
    }
}
