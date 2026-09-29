//! Empirical validation of direct-native entries against an explicit numerical reference.
//! This does not establish compiler applicability for a native implementation.
use super::timing::PointTiming;
use super::tune::{
    Exclusion, Outcome, TuneError, TuningInitializer, TuningPoint, TuningReference, TuningResult,
    TuningReuse,
};
use crate::api::{
    device::DeviceInner,
    kernel::{self, DecodedValue, EncodedArgs},
};
use seismic_compiler::prepared::ArgumentValue;
use seismic_compiler::{
    feedback::{FeedbackOptions, PreparationOptions},
    numerics::{compare_element_bits, input_subject, result_subject, PolicyIdentity},
};
use seismic_lang::{
    checked::CheckedModule,
    entry::{ElementBindings, LogicalEntry},
    ids::EntryId,
    precision::PrecisionPolicy,
    registry::{self, RepresentationKind},
    types::DType,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    ops::Deref,
    rc::Rc,
    sync::Arc,
    time::Instant,
};

pub(super) type Initializer<'a> = Rc<RefCell<TuningInitializer<'a>>>;

#[derive(Clone, Debug)]
enum Observation {
    Tensor {
        representation: seismic_lang::ids::RepresentationId,
        shape: Vec<u64>,
        bytes: Vec<Arc<[u8]>>,
    },
    Scalar(ArgumentValue),
}

// State views overlap across row classes and layer rotations. Retain identical
// immutable pages once, while still observing and comparing every byte of every case.
const OBSERVATION_PAGE: usize = 64 * 1024;
#[derive(Default)]
struct ReferencePages {
    pages: HashMap<[u8; 32], Vec<Arc<[u8]>>>,
    retained: usize,
}
impl ReferencePages {
    fn capture(&mut self, bytes: &[u8]) -> Result<Vec<Arc<[u8]>>, Exclusion> {
        bytes
            .chunks(OBSERVATION_PAGE)
            .map(|page| {
                let key: [u8; 32] = Sha256::digest(page).into();
                let matches = self.pages.entry(key).or_default();
                if let Some(existing) = matches.iter().find(|existing| existing.as_ref() == page) {
                    return Ok(existing.clone());
                }
                let retained = self
                    .retained
                    .checked_add(page.len())
                    .filter(|&n| n <= 1024 * 1024 * 1024)
                    .ok_or_else(|| {
                        Exclusion::Execution(
                            "unique reference observations exceed the 1 GiB per-unit limit".into(),
                        )
                    })?;
                let page: Arc<[u8]> = page.into();
                self.retained = retained;
                matches.push(page.clone());
                Ok(page)
            })
            .collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NumericalMetrics {
    pub maximum_absolute_error: f64,
    pub maximum_envelope_usage: f64,
    pub worst_subject: String,
    pub worst_element: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NumericalEvidence {
    pub reference: TuningReference,
    pub identity: String,
    pub case: String,
    pub candidate: String,
    pub metrics: BTreeMap<String, NumericalMetrics>,
}

pub(super) struct PreparedPoint<'a> {
    point: TuningPoint<'a>,
    case: Rc<ReferenceCase<'a>>,
}
impl<'a> Deref for PreparedPoint<'a> {
    type Target = TuningPoint<'a>;
    fn deref(&self) -> &Self::Target {
        &self.point
    }
}
impl<'a> PreparedPoint<'a> {
    pub(super) fn reweighted(&self, weight: f64) -> Self {
        Self {
            point: TuningPoint {
                label: self.label.clone(),
                weight,
                class: self.class.clone(),
                rotation: self.rotation.clone(),
                initialize: None,
            },
            case: self.case.clone(),
        }
    }
    pub(super) fn validate(
        &self,
        timing: &mut PointTiming<'a>,
        minimum_seconds: f64,
    ) -> Result<(), Exclusion> {
        timing.initialize = self.case.initialize.clone();
        let candidate = timing.artifact().to_owned();
        if let Some(verdict) = self.case.verdicts.borrow().get(&candidate) {
            return verdict.as_ref().map(|_| ()).map_err(Clone::clone);
        }
        let mut metrics = BTreeMap::new();
        let verdict = timing
            .observe_first(minimum_seconds, |rotation, values, args| {
                let started = Instant::now();
                let result = (|| {
                    let actual = observe(values, args, &self.case.mutable, None)?;
                    compare(
                        &self.case.reference[rotation],
                        &actual,
                        &self.case.subjects,
                        &self.case.policy,
                        &mut metrics,
                    )
                    .map_err(|detail| Exclusion::Validation {
                        point: self.label.clone(),
                        detail,
                    })
                })();
                *self.case.validation_seconds.borrow_mut() += started.elapsed().as_secs_f64();
                result
            })
            .map(|()| NumericalEvidence {
                reference: self.case.reference_kind,
                identity: self.case.identity.clone(),
                case: self.label.clone(),
                candidate: candidate.clone(),
                metrics,
            });
        self.case
            .verdicts
            .borrow_mut()
            .insert(candidate, verdict.clone());
        verdict.map(|_| ())
    }
    pub(super) fn evidence(&self, candidate: &str) -> Option<NumericalEvidence> {
        self.case
            .verdicts
            .borrow()
            .get(candidate)
            .and_then(|v| v.as_ref().ok())
            .cloned()
    }
    pub(super) fn identity(&self) -> &str {
        &self.case.identity
    }
    pub(super) fn validation_seconds(&self) -> f64 {
        *self.case.validation_seconds.borrow()
    }
}
struct ReferenceCase<'a> {
    reference_kind: TuningReference,
    initialize: Option<Initializer<'a>>,
    policy: Arc<PrecisionPolicy>,
    subjects: Vec<String>,
    mutable: Vec<usize>,
    reference: Vec<Vec<Observation>>,
    identity: String,
    verdicts: RefCell<BTreeMap<String, Result<NumericalEvidence, Exclusion>>>,
    validation_seconds: RefCell<f64>,
}

pub(super) enum Preparation<'a> {
    Cases(Vec<PreparedPoint<'a>>),
    Reused(TuningResult),
}

pub(super) fn prepare<'a>(
    device: &Arc<DeviceInner>,
    module: &CheckedModule,
    entry: EntryId,
    bindings: &ElementBindings,
    logical: &LogicalEntry,
    mutable: &[usize],
    points: Vec<TuningPoint<'a>>,
    policy: &PrecisionPolicy,
    reuse: Option<TuningReuse<'_>>,
    reference_kind: TuningReference,
    default: &seismic_lang::checked::NativeSpecialization,
    cpu: Option<&'static super::cpu::CpuNativeKernels>,
) -> Result<Preparation<'a>, TuneError> {
    if matches!(policy, PrecisionPolicy::Unconstrained) {
        return Err(TuneError::Declaration(
            "native tuning requires a bounded or exact precision policy".into(),
        ));
    }
    let subjects: Vec<_> = logical
        .schema()
        .results()
        .iter()
        .map(|result| result_subject(&result.path))
        .chain(mutable.iter().map(|&i| input_subject(i)))
        .collect();
    if let PrecisionPolicy::Bounded {
        outputs, inputs, ..
    } = policy
    {
        for subject in outputs.keys() {
            if !subjects.contains(subject) {
                return Err(TuneError::Declaration(format!(
                    "unknown numerical subject {subject}; available: {subjects:?}"
                )));
            }
        }
        if !inputs.is_empty() {
            return Err(TuneError::Declaration(
                "native tuning cases do not support input-range assumptions".into(),
            ));
        }
    }
    // Vulkan currently exposes only the explicit native route. Its portable
    // semantics execute on the host CPU, with canonical copies of the inputs.
    let reference_device = if reference_kind == TuningReference::Portable
        && super::backend_name(&device.kind) == registry::BackendName::Vulkan
    {
        crate::devices::Catalog::discover()
            .map_err(|e| TuneError::Reference(e.to_string()))?
            .open_backend(registry::BackendName::Cpu)
            .map_err(|e| TuneError::Reference(e.to_string()))?
    } else {
        device.clone()
    };
    let native_reference = if reference_kind == TuningReference::NativeDefault {
        Some(Arc::new(
            super::NativePrepared::prepare(
                device,
                module,
                entry,
                bindings.clone(),
                default.clone(),
                cpu,
            )
            .map_err(|e| TuneError::Reference(super::tune::prepare_message(e)))?,
        ))
    } else {
        None
    };
    let policy = Arc::new(policy.clone());
    let mut identified = Vec::new();
    for mut point in points {
        let initialize = point.initialize.take().map(|f| Rc::new(RefCell::new(f)));
        let mut digest = Sha256::new();
        digest.update(b"native-validation-v4");
        digest.update(format!("{reference_kind:?}"));
        if let Some(reference) = &native_reference {
            digest.update(&reference.artifact.0);
        }
        digest.update(seismic_lang::reference_math::VERSION);
        digest.update(logical.identity().digest());
        digest.update(logical.module_hash().digest());
        digest.update(device.tuning_identity());
        digest.update(reference_device.tuning_identity());
        digest.update(PolicyIdentity::of(&policy).0);
        digest.update(point.label.as_bytes());
        digest.update((point.rotation.len() as u64).to_le_bytes());
        for args in &point.rotation {
            if let Some(reset) = &initialize {
                reset.borrow_mut()().map_err(|e| TuneError::Reference(e.to_string()))?;
            }
            for (value, tensor) in args.values().iter().zip(args.tensors()) {
                if let Some(tensor) = tensor {
                    let bytes = tensor
                        .read_to_host()
                        .map_err(|e| TuneError::Reference(e.to_string()))?;
                    let descriptor = tensor.descriptor();
                    digest.update(
                        format!(
                            "{:?}",
                            (
                                registry::representation_info(descriptor.representation).name,
                                descriptor.extents,
                                descriptor.strides
                            )
                        )
                        .as_bytes(),
                    );
                    digest.update((bytes.len() as u64).to_le_bytes());
                    digest.update(&bytes);
                } else {
                    digest.update(format!("{value:?}").as_bytes());
                }
            }
        }
        identified.push((point, initialize, crate::telemetry::hex(&digest.finalize())));
    }
    let previous = reuse.map(|reuse| reuse.result().clone());
    let previous_artifact = previous.as_ref().and_then(|result| {
        result.configurations.iter().find_map(|record| {
            if record.configuration != result.overall {
                return None;
            }
            match &record.outcome {
                Outcome::Measured {
                    artifact,
                    validated: true,
                    ..
                } => Some(artifact.clone()),
                _ => None,
            }
        })
    });
    let evidence = |identity: &str| {
        previous
            .as_ref()
            .and_then(|result| {
                result.numerical_evidence.iter().find(|evidence| {
                    evidence.identity == identity
                        && Some(&evidence.candidate) == previous_artifact.as_ref()
                })
            })
            .cloned()
    };
    if matches!(reuse, Some(TuningReuse::Completed(_)))
        && identified.iter().all(|(point, _, identity)| {
            evidence(identity).is_some()
                && previous.as_ref().unwrap().points.iter().any(|stored| {
                    stored.label == point.label
                        && stored.weight == point.weight
                        && stored.class == point.class
                })
        })
        && previous
            .as_ref()
            .is_some_and(|result| result.points.len() == identified.len())
    {
        let mut previous = previous.unwrap();
        previous.reused = true;
        return Ok(Preparation::Reused(previous));
    }
    // Zero search constructs the required source implementation without timing-profile
    // acquisition or feedback observations. Exact applicability is compiler-derived.
    let reference = if reference_kind == TuningReference::Portable {
        Some(Arc::new(
            kernel::prepare(
                module,
                entry,
                bindings.clone(),
                &reference_device,
                PreparationOptions::feedback(
                    PrecisionPolicy::Exact,
                    FeedbackOptions {
                        search_time: std::time::Duration::ZERO,
                        ..Default::default()
                    },
                ),
            )
            .map_err(|e| TuneError::Reference(super::tune::prepare_message(e)))?,
        ))
    } else {
        None
    };
    let mut pages = ReferencePages::default();
    let mut prepared = Vec::new();
    for (point, initialize, identity) in identified {
        let mut expected = Vec::new();
        for args in &point.rotation {
            if let Some(reset) = &initialize {
                reset.borrow_mut()().map_err(|e| TuneError::Reference(e.to_string()))?;
            }
            // Slab addressing is a native binding detail. Portable state is canonical
            // and separately owned, so candidate execution cannot corrupt its reference.
            let reference_args = if native_reference.is_some() {
                args.clone()
            } else {
                args.map_tensors(|ordinal, tensor| {
                    if mutable.contains(&ordinal)
                        || tensor.is_slabbed()
                        || !Arc::ptr_eq(device, &reference_device)
                    {
                        let bytes = tensor
                            .read_to_host()
                            .map_err(|e| TuneError::Reference(e.to_string()))?;
                        crate::api::tensor::TensorInner::from_host(
                            &reference_device,
                            tensor.representation(),
                            tensor.extents(),
                            &bytes,
                        )
                        .map(Arc::new)
                        .map_err(|e| TuneError::Reference(e.to_string()))
                    } else {
                        Ok(tensor.clone())
                    }
                })?
            };
            let results = if let Some(reference) = &native_reference {
                reference.call(reference_args.clone())
            } else {
                kernel::call(
                    reference.as_ref().expect("portable reference"),
                    reference_args.clone(),
                )
            }
            .map_err(|e| TuneError::Reference(e.to_string()))?;
            let values = results.into_values();
            let observed = observe(values, &reference_args, mutable, Some(&mut pages))
                .map_err(|e| TuneError::Reference(format!("{e:?}")))?;
            expected.push(observed);
        }
        let mut verdicts = BTreeMap::new();
        if let Some(evidence) = evidence(&identity) {
            verdicts.insert(evidence.candidate.clone(), Ok(evidence));
        }
        let case = Rc::new(ReferenceCase {
            reference_kind,
            initialize,
            policy: policy.clone(),
            subjects: subjects.clone(),
            mutable: mutable.to_vec(),
            reference: expected,
            identity,
            verdicts: RefCell::new(verdicts),
            validation_seconds: RefCell::new(0.),
        });
        prepared.push(PreparedPoint { point, case });
    }
    Ok(Preparation::Cases(prepared))
}

fn observe(
    values: Vec<DecodedValue>,
    args: &EncodedArgs,
    mutable: &[usize],
    mut pages: Option<&mut ReferencePages>,
) -> Result<Vec<Observation>, Exclusion> {
    values
        .into_iter()
        .chain(mutable.iter().map(|&i| {
            DecodedValue::Tensor(args.tensor(i).expect("checked writable tensor").clone())
        }))
        .map(|value| match value {
            DecodedValue::Tensor(tensor) => Ok(Observation::Tensor {
                representation: tensor.representation(),
                shape: tensor.descriptor().extents,
                bytes: {
                    let bytes = tensor
                        .read_to_host()
                        .map_err(|e| Exclusion::Execution(e.to_string()))?;
                    match pages.as_deref_mut() {
                        Some(pages) => pages.capture(&bytes)?,
                        None => bytes.chunks(OBSERVATION_PAGE).map(Arc::from).collect(),
                    }
                },
            }),
            DecodedValue::Scalar(value) => Ok(Observation::Scalar(value)),
        })
        .collect()
}
fn scalar_bits(value: &ArgumentValue) -> Option<(DType, u32)> {
    Some(match value {
        ArgumentValue::F32(v) => (DType::F32, v.to_bits()),
        ArgumentValue::F16(v) => (DType::F16, u32::from(*v)),
        ArgumentValue::BF16(v) => (DType::BF16, u32::from(*v)),
        ArgumentValue::I32(v) => (DType::I32, *v as u32),
        ArgumentValue::U32(v) => (DType::U32, *v),
        ArgumentValue::Bool(v) => (DType::Bool, u32::from(*v)),
        _ => return None,
    })
}
fn element(
    policy: &PrecisionPolicy,
    subject: &str,
    dtype: DType,
    reference: u32,
    actual: u32,
    index: usize,
    metrics: &mut NumericalMetrics,
) -> Result<(), String> {
    let measured = compare_element_bits(policy, subject, dtype, reference, actual);
    if !measured.accepted {
        return Err(format!("subject {subject} element {index}: reference bits {reference:#x}, actual bits {actual:#x}, absolute error {}, relative error {}, ULPs {}, tolerance {:?}",measured.absolute_error,measured.relative_error,measured.ulps,policy.tolerance(subject)));
    }
    let reference_value = if dtype.is_float() {
        seismic_lang::reference_math::conversion::exact_f64(dtype, reference)
    } else {
        0.
    };
    let envelope = policy
        .tolerance(subject)
        .map_or(0., |v| v.envelope(reference_value));
    let usage = if envelope > 0. && measured.absolute_error.is_finite() {
        measured.absolute_error / envelope
    } else {
        0.
    };
    if measured.absolute_error.is_finite() {
        metrics.maximum_absolute_error =
            metrics.maximum_absolute_error.max(measured.absolute_error);
    }
    if usage > metrics.maximum_envelope_usage {
        metrics.maximum_envelope_usage = usage;
        metrics.worst_subject = subject.to_owned();
        metrics.worst_element = index;
    }
    Ok(())
}
fn compare(
    expected: &[Observation],
    actual: &[Observation],
    subjects: &[String],
    policy: &PrecisionPolicy,
    metrics: &mut BTreeMap<String, NumericalMetrics>,
) -> Result<(), String> {
    if expected.len() != actual.len() || expected.len() != subjects.len() {
        return Err("result/state subject count differs".into());
    }
    for ((expected, actual), subject) in expected.iter().zip(actual).zip(subjects) {
        let metrics = metrics.entry(subject.clone()).or_default();
        match (expected, actual) {
            (
                Observation::Tensor {
                    representation: er,
                    shape: es,
                    bytes: eb,
                },
                Observation::Tensor {
                    representation: ar,
                    shape: as_,
                    bytes: ab,
                },
            ) => {
                if er != ar
                    || es != as_
                    || eb.len() != ab.len()
                    || eb.iter().zip(ab).any(|(e, a)| e.len() != a.len())
                {
                    return Err(format!(
                        "subject {subject}: representation, shape or byte length differs"
                    ));
                }
                if eb == ab {
                    continue;
                }
                let RepresentationKind::Dense(dtype) = registry::representation_info(*er).kind
                else {
                    return Err(format!("subject {subject}: packed storage differs"));
                };
                let width = dtype.bytes() as usize;
                for (page_index, (expected, actual)) in eb.iter().zip(ab).enumerate() {
                    if expected == actual {
                        continue;
                    }
                    for (index, (e, a)) in expected
                        .chunks_exact(width)
                        .zip(actual.chunks_exact(width))
                        .enumerate()
                    {
                        if e == a {
                            continue;
                        }
                        let bits = |v: &[u8]| {
                            let mut b = [0; 4];
                            b[..v.len()].copy_from_slice(v);
                            u32::from_le_bytes(b)
                        };
                        element(
                            policy,
                            subject,
                            dtype,
                            bits(e),
                            bits(a),
                            (page_index * OBSERVATION_PAGE) / width + index,
                            metrics,
                        )?;
                    }
                }
            }
            (Observation::Scalar(e), Observation::Scalar(a)) => {
                match (scalar_bits(e), scalar_bits(a)) {
                    (Some((ed, eb)), Some((ad, ab))) if ed == ad => {
                        element(policy, subject, ed, eb, ab, 0, metrics)?
                    }
                    _ if e == a => (),
                    _ => return Err(format!("subject {subject}: discrete scalar differs")),
                }
            }
            _ => return Err(format!("subject {subject}: value kind differs")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::precision::{Limit, Tolerance};
    fn policy() -> PrecisionPolicy {
        PrecisionPolicy::bounded(Tolerance {
            absolute: Limit::new(0.01).unwrap(),
            relative: Limit::ZERO,
            relative_floor: Limit::ZERO,
            ulps: None,
        })
    }
    fn tensor(dtype: DType, values: &[u32]) -> Observation {
        Observation::Tensor {
            representation: registry::dense(dtype),
            shape: vec![values.len() as u64],
            bytes: vec![values
                .iter()
                .flat_map(|bits| bits.to_le_bytes())
                .collect::<Vec<_>>()
                .into()],
        }
    }
    #[test]
    fn reference_pages_share_identical_prefixes_but_keep_distinct_tails() {
        let mut pages = ReferencePages::default();
        let mut bytes = vec![3; OBSERVATION_PAGE * 2];
        let first = pages.capture(&bytes).unwrap();
        assert!(Arc::ptr_eq(&first[0], &first[1]));
        bytes[OBSERVATION_PAGE + 5] = 4;
        let second = pages.capture(&bytes).unwrap();
        assert!(Arc::ptr_eq(&first[0], &second[0]));
        assert!(!Arc::ptr_eq(&first[1], &second[1]));
        assert_eq!(pages.retained, 2 * OBSERVATION_PAGE);
        assert_eq!(first[1][5], 3, "captured bytes remain immutable");
        assert_eq!(second[1][5], 4);
    }

    #[test]
    fn bounded_float_passes_but_a_localized_defect_and_discrete_change_fail() {
        let reference = vec![tensor(DType::F32, &[1f32.to_bits(); 1000])];
        let mut values = vec![1f32.to_bits(); 1000];
        values[731] = 1.005f32.to_bits();
        let subjects = vec!["value".into()];
        assert!(compare(
            &reference,
            &[tensor(DType::F32, &values)],
            &subjects,
            &policy(),
            &mut BTreeMap::new()
        )
        .is_ok());
        values[731] = 1.1f32.to_bits();
        assert!(compare(
            &reference,
            &[tensor(DType::F32, &values)],
            &subjects,
            &policy(),
            &mut BTreeMap::new()
        )
        .unwrap_err()
        .contains("element 731"));
        assert!(compare(
            &[tensor(DType::U32, &[1])],
            &[tensor(DType::U32, &[2])],
            &subjects,
            &policy(),
            &mut BTreeMap::new()
        )
        .is_err());
    }
    #[test]
    fn shapes_and_special_behavior_are_checked() {
        let expected = tensor(DType::F32, &[0]);
        let mut wrong_shape = expected.clone();
        if let Observation::Tensor { shape, .. } = &mut wrong_shape {
            *shape = vec![1, 1];
        }
        let subjects = vec!["i0".into()];
        assert!(compare(
            &[expected.clone()],
            &[wrong_shape],
            &subjects,
            &policy(),
            &mut BTreeMap::new()
        )
        .is_err());
        for bits in [
            (-0f32).to_bits(),
            f32::INFINITY.to_bits(),
            f32::NAN.to_bits(),
        ] {
            assert!(compare(
                &[expected.clone()],
                &[tensor(DType::F32, &[bits])],
                &subjects,
                &policy(),
                &mut BTreeMap::new()
            )
            .is_err());
        }
    }
}
