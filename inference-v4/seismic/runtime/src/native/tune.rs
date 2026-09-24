//! Exhaustive tuning of a native implementation over its declared domain.
//!
//! Every admissible configuration is formed, validated and measured at every
//! tuning point; the choice is exact over the measured table. Arithmetic
//! parameters (those that change a row's arithmetic order) take one value
//! across all points; mapping parameters are chosen per point.

use super::{MeasureOptions, Measurement, NativePrepared};
use crate::api::device::DeviceInner;
use crate::api::kernel::{DecodedValue, EncodedArgs, PrepareError};
use crate::api::CallError;
use seismic_compiler::prepared::ArgumentValue;
use seismic_lang::checked::{CheckedModule, NativeImplementation, NativeSpecialization};
use seismic_lang::entry::ElementBindings;
use seismic_lang::ids::EntryId;
use seismic_lang::registry::{self, RepresentationKind};
use seismic_lang::types::DType;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use super::cpu::CpuNativeKernels;

/// One workload the tuned implementation serves.
pub struct TuningPoint {
    pub label: String,
    /// Share of the objective.
    pub weight: f64,
    /// Argument sets cycled through by measurement. The first is also the
    /// validation input.
    pub rotation: Vec<EncodedArgs>,
}

/// How configurations must agree with the all-defaults configuration.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Validation {
    BitExact,
    /// Dense floating-point results may differ by at most
    /// `absolute + relative * |reference|`; every other result is bit-exact.
    Tolerance {
        absolute: f64,
        relative: f64,
    },
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

/// Why a configuration was not measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Exclusion {
    /// Formation failed: an authoring defect (compile error) or toolchain
    /// failure.
    Formation(String),
    /// The device rejected a call, for example a launch beyond its limits.
    Execution(String),
    /// A mapping parameter changed result bits relative to another
    /// configuration with the same arithmetic parameters.
    MisclassifiedParameter {
        point: String,
        reference: Configuration,
    },
    /// Results disagree with the all-defaults configuration.
    Validation { point: String, detail: String },
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
    },
    Excluded(Exclusion),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfigurationRecord {
    pub configuration: Configuration,
    pub outcome: Outcome,
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
    /// Every admissible configuration in domain order.
    pub configurations: Vec<ConfigurationRecord>,
    /// The chosen configuration of each point, in point order. Arithmetic
    /// parameters are equal across points.
    pub chosen: Vec<Configuration>,
    /// The single configuration with the least weighted cost over all
    /// points, for a consumer that prepares one configuration.
    pub overall: Configuration,
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
    /// The all-defaults configuration could not be formed or run; it is the
    /// validation reference.
    DefaultUnusable(Exclusion),
    /// Every configuration was excluded.
    NothingMeasured(TuningResult),
    Measurement(CallError),
}

impl std::fmt::Display for TuneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Declaration(message) => write!(f, "{message}"),
            Self::NoPoints => f.write_str("tuning needs at least one point"),
            Self::DefaultUnusable(exclusion) => {
                write!(
                    f,
                    "the all-defaults configuration is unusable: {exclusion:?}"
                )
            }
            Self::NothingMeasured(_) => f.write_str("every configuration was excluded"),
            Self::Measurement(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for TuneError {}

pub struct TuneRequest<'a> {
    pub device: &'a Arc<DeviceInner>,
    pub module: &'a CheckedModule,
    pub entry: EntryId,
    pub bindings: ElementBindings,
    /// Values of every static dimension.
    pub statics: NativeSpecialization,
    pub cpu: Option<&'static CpuNativeKernels>,
    pub points: Vec<TuningPoint>,
    pub validation: Validation,
    pub measure: MeasureOptions,
}

type Outputs = Vec<DecodedValue>;

pub fn tune(request: TuneRequest<'_>) -> Result<TuningResult, TuneError> {
    if request.points.is_empty() || request.points.iter().any(|point| point.rotation.is_empty()) {
        return Err(TuneError::NoPoints);
    }
    let backend = super::backend_name(&request.device.kind);
    let entry_name = request
        .module
        .entries()
        .iter()
        .find(|candidate| candidate.id == request.entry)
        .expect("entry belongs to its module")
        .name
        .clone();
    let implementation = request
        .module
        .native_implementation(request.entry, backend)
        .cloned()
        .ok_or_else(|| {
            TuneError::Declaration(format!(
                "`{entry_name}` has no native implementation for `{}`",
                backend.as_str()
            ))
        })?;
    let configurations = implementation
        .admissible(&request.statics)
        .map_err(|error| TuneError::Declaration(error.to_string()))?;
    let default = implementation
        .default_specialization(&request.statics)
        .map_err(|error| TuneError::Declaration(error.to_string()))?;

    // Form every configuration concurrently.
    let formed = form_all(&request, &configurations);

    // Validation reference: the all-defaults configuration.
    let default_index = configurations
        .iter()
        .position(|configuration| *configuration == default)
        .expect("the default configuration is admissible");
    let reference = match &formed[default_index] {
        Ok(kernel) => run_points(kernel, &request.points).map_err(TuneError::DefaultUnusable)?,
        Err(exclusion) => return Err(TuneError::DefaultUnusable(exclusion.clone())),
    };

    // Validate against the reference, and within each arithmetic group
    // against its first surviving member.
    let arithmetic =
        |specialization: &NativeSpecialization| arithmetic_values(&implementation, specialization);
    let mut group_reference: BTreeMap<Vec<u64>, (Configuration, Vec<Outputs>)> = BTreeMap::new();
    let mut records = Vec::with_capacity(configurations.len());
    let mut survivors: Vec<(usize, Arc<NativePrepared>)> = Vec::new();
    for (index, (specialization, kernel)) in configurations.iter().zip(formed).enumerate() {
        let configuration = Configuration::of(specialization);
        let kernel = match kernel {
            Ok(kernel) => kernel,
            Err(exclusion) => {
                records.push(ConfigurationRecord {
                    configuration,
                    outcome: Outcome::Excluded(exclusion),
                });
                continue;
            }
        };
        let outputs = if index == default_index {
            Ok(reference.clone())
        } else {
            run_points(&kernel, &request.points)
        };
        let outputs = match outputs {
            Ok(outputs) => outputs,
            Err(exclusion) => {
                records.push(ConfigurationRecord {
                    configuration,
                    outcome: Outcome::Excluded(exclusion),
                });
                continue;
            }
        };
        let group = arithmetic(specialization);
        let exclusion =
            if let Some((group_configuration, group_outputs)) = group_reference.get(&group) {
                request
                    .points
                    .iter()
                    .zip(group_outputs.iter().zip(&outputs))
                    .find(|(_, (expected, actual))| {
                        compare(expected, actual, Validation::BitExact).is_err()
                    })
                    .map(|(point, _)| Exclusion::MisclassifiedParameter {
                        point: point.label.clone(),
                        reference: group_configuration.clone(),
                    })
            } else {
                None
            };
        let exclusion = exclusion.or_else(|| {
            request
                .points
                .iter()
                .zip(reference.iter().zip(&outputs))
                .find_map(|(point, (expected, actual))| {
                    compare(expected, actual, request.validation)
                        .err()
                        .map(|detail| Exclusion::Validation {
                            point: point.label.clone(),
                            detail,
                        })
                })
        });
        if let Some(exclusion) = exclusion {
            records.push(ConfigurationRecord {
                configuration,
                outcome: Outcome::Excluded(exclusion),
            });
            continue;
        }
        group_reference
            .entry(group)
            .or_insert_with(|| (configuration.clone(), outputs));
        survivors.push((records.len(), kernel.clone()));
        records.push(ConfigurationRecord {
            configuration,
            outcome: Outcome::Measured {
                artifact: kernel.artifact().0.clone(),
                points: Vec::new(),
            },
        });
    }

    // Measure every survivor at every point.
    let mut medians: Vec<(usize, Vec<f64>)> = Vec::with_capacity(survivors.len());
    for (record, kernel) in &survivors {
        let mut points = Vec::with_capacity(request.points.len());
        for point in &request.points {
            let measurement = kernel
                .measure(point.rotation.clone(), &request.measure)
                .map_err(TuneError::Measurement)?;
            points.push(point_measurement(&point.label, measurement));
        }
        medians.push((
            *record,
            points.iter().map(|point| point.median_seconds).collect(),
        ));
        let Outcome::Measured {
            points: recorded, ..
        } = &mut records[*record].outcome
        else {
            unreachable!("survivors are measured records");
        };
        *recorded = points;
    }

    let mut result = TuningResult {
        tuning_identity: request.device.tuning_identity(),
        entry: entry_name,
        backend: backend.as_str().to_owned(),
        points: request
            .points
            .iter()
            .map(|point| (point.label.clone(), point.weight))
            .collect(),
        validation: request.validation,
        configurations: records,
        chosen: Vec::new(),
        overall: Configuration::of(&default),
    };
    if medians.is_empty() {
        return Err(TuneError::NothingMeasured(result));
    }
    result.chosen = choose(&implementation, &result, &medians, &request.points);
    let weighted = |times: &Vec<f64>| {
        request
            .points
            .iter()
            .zip(times)
            .map(|(point, time)| point.weight * time)
            .sum::<f64>()
    };
    let (overall, _) = medians
        .iter()
        .min_by(|(_, left), (_, right)| weighted(left).total_cmp(&weighted(right)))
        .expect("at least one configuration was measured");
    result.overall = result.configurations[*overall].configuration.clone();
    Ok(result)
}

/// Exact choice over the measured table.
fn choose(
    implementation: &NativeImplementation,
    result: &TuningResult,
    medians: &[(usize, Vec<f64>)],
    points: &[TuningPoint],
) -> Vec<Configuration> {
    let group = |record: usize| {
        arithmetic_values(
            implementation,
            &result.configurations[record].configuration.specialization(),
        )
    };
    // Candidate arithmetic assignments in domain order.
    let mut assignments: Vec<Vec<u64>> = Vec::new();
    for (record, _) in medians {
        let assignment = group(*record);
        if !assignments.contains(&assignment) {
            assignments.push(assignment);
        }
    }
    let best_in = |assignment: &Vec<u64>, point: usize| {
        medians
            .iter()
            .filter(|(record, _)| group(*record) == *assignment)
            .min_by(|(_, left), (_, right)| left[point].total_cmp(&right[point]))
            .map(|(record, times)| (*record, times[point]))
            .expect("every assignment has a measured member")
    };
    let cost = |assignment: &Vec<u64>| {
        points
            .iter()
            .enumerate()
            .map(|(point, tuning)| tuning.weight * best_in(assignment, point).1)
            .sum::<f64>()
    };
    let chosen = assignments
        .iter()
        .min_by(|left, right| cost(left).total_cmp(&cost(right)))
        .expect("at least one configuration was measured");
    (0..points.len())
        .map(|point| {
            result.configurations[best_in(chosen, point).0]
                .configuration
                .clone()
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

fn form_all(
    request: &TuneRequest<'_>,
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
                            NativePrepared::prepare(
                                request.device,
                                request.module,
                                request.entry,
                                request.bindings.clone(),
                                specialization.clone(),
                                request.cpu,
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

fn prepare_message(error: PrepareError) -> String {
    match error {
        PrepareError::Source(error) => error.to_string(),
        PrepareError::Preparation(error) => error.to_string(),
    }
}

fn run_points(
    kernel: &Arc<NativePrepared>,
    points: &[TuningPoint],
) -> Result<Vec<Outputs>, Exclusion> {
    points
        .iter()
        .map(|point| {
            kernel
                .call(point.rotation[0].clone())
                .map(|results| results.into_values())
                .map_err(|error| Exclusion::Execution(error.to_string()))
        })
        .collect()
}

fn compare(expected: &Outputs, actual: &Outputs, validation: Validation) -> Result<(), String> {
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        match (expected, actual) {
            (DecodedValue::Tensor(expected), DecodedValue::Tensor(actual)) => {
                let expected_bytes = expected.read_to_host().map_err(|error| error.to_string())?;
                let actual_bytes = actual.read_to_host().map_err(|error| error.to_string())?;
                if expected_bytes == actual_bytes {
                    continue;
                }
                let Validation::Tolerance { absolute, relative } = validation else {
                    return Err(format!("result {index} differs"));
                };
                let dtype = match &registry::representation_info(expected.representation()).kind {
                    RepresentationKind::Dense(dtype) => *dtype,
                    _ => return Err(format!("packed result {index} differs")),
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
                    return Err(format!("integer result {index} differs"));
                }
                let elements = expected_bytes.len() / dtype.bytes() as usize;
                for element in 0..elements {
                    let reference = decode(&expected_bytes, element);
                    let value = decode(&actual_bytes, element);
                    let bound = absolute + relative * reference.abs();
                    if !((value - reference).abs() <= bound
                        || (value.is_nan() && reference.is_nan()))
                    {
                        return Err(format!(
                            "result {index} element {element}: {value} vs {reference} exceeds {bound}"
                        ));
                    }
                }
            }
            (DecodedValue::Scalar(expected), DecodedValue::Scalar(actual)) => {
                if !scalar_equal(expected, actual) {
                    return Err(format!("scalar result {index} differs"));
                }
            }
            _ => return Err(format!("result {index} changed kind")),
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
