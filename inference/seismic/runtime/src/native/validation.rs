//! Empirical validation of direct-native entries against an explicit numerical reference.
//! This does not establish compiler applicability for a native implementation.
use super::timing::PointTiming;
use super::tune::{Exclusion, TuneError, TuningInitializer, TuningPoint, TuningReference};
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
    ops::{Deref, Range},
    rc::Rc,
    sync::Arc,
    time::Instant,
};

pub(super) type Initializer<'a> = Rc<RefCell<TuningInitializer<'a>>>;

/// A `&mut` parameter's ordinal and the leading rows observed of it (`None`:
/// the whole tensor).
type Observed = (usize, Option<Range<u64>>);

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

/// What every point's reference execution shares: the reference
/// implementation, where it runs, and the observations retained so far.
pub(super) struct References {
    device: Arc<DeviceInner>,
    reference_device: Arc<DeviceInner>,
    native: Option<Arc<Arc<super::NativePrepared>>>,
    portable: Option<Arc<kernel::PreparedAny>>,
    mutable: Vec<(usize, String)>,
    pages: RefCell<ReferencePages>,
    seconds: RefCell<f64>,
}

impl References {
    /// Time spent executing references so far.
    pub(super) fn seconds(&self) -> f64 {
        *self.seconds.borrow()
    }
}
impl<'a> Deref for PreparedPoint<'a> {
    type Target = TuningPoint<'a>;
    fn deref(&self) -> &Self::Target {
        &self.point
    }
}
impl<'a> PreparedPoint<'a> {
    /// Execute this point's reference, once, before its first validation.
    pub(super) fn ensure_reference(&self) -> Result<(), TuneError> {
        if self.case.reference.borrow().is_some() {
            return Ok(());
        }
        let started = Instant::now();
        let references = &self.case.references;
        let mut expected = Vec::new();
        for args in &self.point.rotation {
            if let Some(reset) = &self.case.initialize {
                reset.borrow_mut()().map_err(|e| TuneError::Reference(e.to_string()))?;
            }
            // Slab addressing is a native binding detail. Portable state is canonical
            // and separately owned, so candidate execution cannot corrupt its reference.
            let reference_args = if references.native.is_some() {
                args.clone()
            } else {
                args.map_tensors(|ordinal, tensor| {
                    if references.mutable.iter().any(|(i, _)| *i == ordinal)
                        || tensor.is_slabbed()
                        || !Arc::ptr_eq(&references.device, &references.reference_device)
                    {
                        let bytes = tensor
                            .read_to_host()
                            .map_err(|e| TuneError::Reference(e.to_string()))?;
                        crate::api::tensor::TensorInner::from_host(
                            &references.reference_device,
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
            let results = match (&references.native, &references.portable) {
                (Some(native), _) => native.call(reference_args.clone()),
                (None, Some(portable)) => kernel::call(portable, reference_args.clone()),
                (None, None) => unreachable!("a reference implementation is prepared"),
            }
            .map_err(|e| TuneError::Reference(e.to_string()))?;
            expected.push(
                observe(
                    results.into_values(),
                    &reference_args,
                    &self.case.mutable,
                    Some(&mut *references.pages.borrow_mut()),
                )
                .map_err(|e| TuneError::Reference(format!("{e:?}")))?,
            );
        }
        *self.case.reference.borrow_mut() = Some(expected);
        *references.seconds.borrow_mut() += started.elapsed().as_secs_f64();
        Ok(())
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
                    let reference = self.case.reference.borrow();
                    compare(
                        &reference
                            .as_ref()
                            .expect("a point's reference executes before its validation")[rotation],
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
    pub(super) fn validation_seconds(&self) -> f64 {
        *self.case.validation_seconds.borrow()
    }
    /// Weigh this point by `weight` in the search's objective.
    pub(super) fn reweigh(&mut self, weight: f64) {
        self.point.weight = weight;
    }
    /// What candidates are compared with.
    pub(super) fn reference_kind(&self) -> TuningReference {
        self.case.reference_kind
    }
}
struct ReferenceCase<'a> {
    reference_kind: TuningReference,
    initialize: Option<Initializer<'a>>,
    policy: Arc<PrecisionPolicy>,
    subjects: Vec<String>,
    mutable: Vec<Observed>,
    references: Rc<References>,
    /// The reference's observations per rotation entry, once executed.
    reference: RefCell<Option<Vec<Vec<Observation>>>>,
    identity: String,
    verdicts: RefCell<BTreeMap<String, Result<NumericalEvidence, Exclusion>>>,
    validation_seconds: RefCell<f64>,
}

/// Prepares points for validation as their inputs arrive: the policy, the
/// reference implementation and what every point's case identity covers.
pub(super) struct Validator {
    reference_kind: TuningReference,
    policy: Arc<PrecisionPolicy>,
    subjects: Vec<String>,
    mutable: Vec<(usize, String)>,
    references: Rc<References>,
    /// What every point's identity starts from.
    identity: Sha256,
}

impl Validator {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        device: &Arc<DeviceInner>,
        module: &CheckedModule,
        entry: EntryId,
        bindings: &ElementBindings,
        logical: &LogicalEntry,
        mutable: &[(usize, String)],
        policy: &PrecisionPolicy,
        reference_kind: TuningReference,
        default: &seismic_lang::checked::NativeSpecialization,
        cpu: Option<&'static super::cpu::CpuNativeKernels>,
    ) -> Result<Self, TuneError> {
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
            .chain(mutable.iter().map(|(i, _)| input_subject(*i)))
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
        let native = if reference_kind == TuningReference::NativeDefault {
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
        // Zero search constructs the required source implementation without timing-profile
        // acquisition or feedback observations. Exact applicability is compiler-derived.
        let portable = if reference_kind == TuningReference::Portable {
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
        let policy = Arc::new(policy.clone());
        // A case's identity covers its argument structure, not tensor
        // contents: tuning inputs are generated test data, and the caller's
        // tuning key names what they are generated from.
        let mut identity = Sha256::new();
        identity.update(b"native-validation-v5");
        identity.update(format!("{reference_kind:?}"));
        if let Some(reference) = &native {
            identity.update(&reference.artifact.0);
        }
        identity.update(seismic_lang::reference_math::VERSION);
        identity.update(logical.identity().digest());
        identity.update(logical.module_hash().digest());
        identity.update(device.tuning_identity());
        identity.update(reference_device.tuning_identity());
        identity.update(PolicyIdentity::of(&policy).0);
        Ok(Self {
            reference_kind,
            policy,
            subjects,
            mutable: mutable.to_vec(),
            references: Rc::new(References {
                device: device.clone(),
                reference_device,
                native,
                portable,
                mutable: mutable.to_vec(),
                pages: RefCell::new(ReferencePages::default()),
                seconds: RefCell::new(0.),
            }),
            identity,
        })
    }

    /// Time spent executing references so far.
    pub(super) fn reference_seconds(&self) -> f64 {
        self.references.seconds()
    }

    /// `point` ready for validation; its reference executes on first use.
    pub(super) fn point<'a>(
        &self,
        mut point: TuningPoint<'a>,
    ) -> Result<PreparedPoint<'a>, TuneError> {
        if let Some((_, parameter)) = self.mutable.first().filter(|_| point.initialize.is_none()) {
            return Err(TuneError::SharedMutableState {
                point: point.label.clone(),
                parameter: parameter.clone(),
            });
        }
        if let Some(name) = point
            .written
            .keys()
            .find(|name| !self.mutable.iter().any(|(_, parameter)| parameter == *name))
        {
            return Err(TuneError::Declaration(format!(
                "point `{}` declares written rows of `{name}`, which is not a `&mut` parameter",
                point.label
            )));
        }
        let observed: Vec<Observed> = self
            .mutable
            .iter()
            .map(|(ordinal, name)| (*ordinal, point.written.get(name).cloned()))
            .collect();
        let initialize = point.initialize.take().map(|f| Rc::new(RefCell::new(f)));
        let mut digest = self.identity.clone();
        digest.update(point.label.as_bytes());
        digest.update(format!("{observed:?}").as_bytes());
        digest.update((point.rotation.len() as u64).to_le_bytes());
        for args in &point.rotation {
            for (value, tensor) in args.values().iter().zip(args.tensors()) {
                if let Some(tensor) = tensor {
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
                } else {
                    digest.update(format!("{value:?}").as_bytes());
                }
            }
        }
        let case = Rc::new(ReferenceCase {
            reference_kind: self.reference_kind,
            initialize,
            policy: self.policy.clone(),
            subjects: self.subjects.clone(),
            mutable: observed,
            references: self.references.clone(),
            reference: RefCell::new(None),
            identity: crate::telemetry::hex(&digest.finalize()),
            verdicts: RefCell::new(BTreeMap::new()),
            validation_seconds: RefCell::new(0.),
        });
        Ok(PreparedPoint { point, case })
    }
}

fn observe(
    values: Vec<DecodedValue>,
    args: &EncodedArgs,
    mutable: &[Observed],
    mut pages: Option<&mut ReferencePages>,
) -> Result<Vec<Observation>, Exclusion> {
    let states = mutable
        .iter()
        .map(|(i, rows)| {
            let tensor = args.tensor(*i).expect("checked writable tensor");
            match rows {
                Some(rows) => tensor
                    .slice_leading(rows.start, rows.end)
                    .map(|slice| DecodedValue::Tensor(Arc::new(slice)))
                    .map_err(|e| Exclusion::Execution(e.to_string())),
                None => Ok(DecodedValue::Tensor(tensor.clone())),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    values
        .into_iter()
        .chain(states)
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
