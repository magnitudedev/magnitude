//! Numerical policy, evidence and comparison shared by the compiler, runtime and tests.
//!
//! Policy describes an observable entry result. It is deliberately independent from
//! implementation syntax: source code supplies computations, while the caller supplies the
//! acceptable deviation from the reference computation.

use crate::types::DType;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

/// A finite, nonnegative floating-point number with stable equality and hashing.
#[derive(Clone, Copy, Debug, Default)]
pub struct Limit(f64);

impl Limit {
    pub const ZERO: Self = Self(0.0);

    pub fn new(value: f64) -> Result<Self, String> {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "precision limit must be finite and nonnegative, got {value}"
            ));
        }
        Ok(Self(if value == 0.0 { 0.0 } else { value }))
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

impl PartialEq for Limit {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for Limit {}
impl PartialOrd for Limit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Limit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl Hash for Limit {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// A finite floating-point value with stable equality and hashing.
#[derive(Clone, Copy, Debug, Default)]
pub struct Finite(f64);

impl Finite {
    pub fn new(value: f64) -> Result<Self, String> {
        value
            .is_finite()
            .then_some(Self(if value == 0.0 { 0.0 } else { value }))
            .ok_or_else(|| format!("range endpoint must be finite, got {value}"))
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

impl PartialEq for Finite {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for Finite {}
impl Hash for Finite {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Maximum permitted deviation of one observable floating-point element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tolerance {
    pub absolute: Limit,
    pub relative: Limit,
    /// Stabilizes relative error around zero. This changes the metric, so it is identity.
    pub relative_floor: Limit,
    /// An additional ULP requirement. `None` means the envelope alone is authoritative.
    pub ulps: Option<u64>,
}

impl Tolerance {
    pub const EXACT: Self = Self {
        absolute: Limit::ZERO,
        relative: Limit::ZERO,
        relative_floor: Limit::ZERO,
        ulps: Some(0),
    };

    pub fn envelope(self, reference: f64) -> f64 {
        self.absolute.get() + self.relative.get() * reference.abs().max(self.relative_floor.get())
    }
}

impl Default for Tolerance {
    fn default() -> Self {
        Self::EXACT
    }
}

/// The weakest evidence a bounded production policy is willing to trust.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceRequirement {
    /// Only compiler proofs and exact equivalence.
    #[default]
    Proven,
    /// A matching whole-witness qualification record may also establish the bound.
    Qualified,
}

/// Required behavior for exceptional floating-point values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpecialPolicy {
    pub nan: bool,
    pub infinity: bool,
    pub signed_zero: bool,
    pub subnormal: bool,
}

impl SpecialPolicy {
    pub const PRESERVE: Self = Self {
        nan: true,
        infinity: true,
        signed_zero: true,
        subnormal: true,
    };
}

impl Default for SpecialPolicy {
    fn default() -> Self {
        Self::PRESERVE
    }
}

/// A caller-supplied closed interval used by static numerical analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InputRange {
    pub minimum: Finite,
    pub maximum: Finite,
}

impl InputRange {
    pub fn new(minimum: f64, maximum: f64) -> Result<Self, String> {
        let (minimum, maximum) = (Finite::new(minimum)?, Finite::new(maximum)?);
        if minimum.get() > maximum.get() {
            return Err(format!(
                "input range minimum {} exceeds maximum {}",
                minimum.get(),
                maximum.get()
            ));
        }
        Ok(Self { minimum, maximum })
    }

    pub fn magnitude(maximum: f64) -> Result<Self, String> {
        let maximum = Limit::new(maximum)?.get();
        Self::new(-maximum, maximum)
    }
}

/// Observable numerical contract of one compilation request.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrecisionPolicy {
    /// Identical finite values and preserved exceptional-value behavior.
    Exact,
    /// Explicit output bounds. Overrides are keyed by entry parameter name.
    Bounded {
        default: Tolerance,
        outputs: BTreeMap<String, Tolerance>,
        evidence: EvidenceRequirement,
        specials: SpecialPolicy,
        inputs: BTreeMap<String, InputRange>,
    },
    /// Exploration only. Unknown numerical behavior is selectable and the result is labelled.
    Unconstrained,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self::Exact
    }
}

impl PrecisionPolicy {
    pub fn bounded(default: Tolerance) -> Self {
        Self::Bounded {
            default,
            outputs: BTreeMap::new(),
            evidence: EvidenceRequirement::Proven,
            specials: SpecialPolicy::PRESERVE,
            inputs: BTreeMap::new(),
        }
    }

    pub fn tolerance(&self, output: &str) -> Option<Tolerance> {
        match self {
            Self::Exact => Some(Tolerance::EXACT),
            Self::Bounded {
                default, outputs, ..
            } => Some(outputs.get(output).copied().unwrap_or(*default)),
            Self::Unconstrained => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceClass {
    Exact,
    Proven,
    Qualified,
    Unknown,
}

/// Numerical freedoms present in one authored implementation. These are attribution, not
/// permission and not evidence: a policy still needs an exact/proven/qualified assessment.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NumericalEffect {
    AlternativeImplementation,
    ReassociatedReduction,
    ApproximateTranscendental(String),
    BackendIntrinsic { target: String, operation: String },
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ErrorMetrics {
    pub maximum_absolute: f64,
    pub maximum_relative: f64,
    pub maximum_ulps: u64,
    pub differing: u64,
    pub compared: u64,
    pub nan_mismatches: u64,
    pub infinity_mismatches: u64,
    pub signed_zero_mismatches: u64,
    pub subnormal_mismatches: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ComparisonReport {
    pub metrics: ErrorMetrics,
    pub worst_element: Option<usize>,
    pub beyond_tolerance: u64,
}

impl ErrorMetrics {
    pub fn is_exact(self) -> bool {
        self.differing == 0
            && self.nan_mismatches == 0
            && self.infinity_mismatches == 0
            && self.signed_zero_mismatches == 0
            && self.subnormal_mismatches == 0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputAssessment {
    pub output: String,
    pub dtype: DType,
    pub evidence: EvidenceClass,
    pub metrics: ErrorMetrics,
    pub worst_element: Option<usize>,
    pub attribution: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericalAssessment {
    pub evidence: EvidenceClass,
    /// Policy against which every retained qualification output was checked elementwise.
    /// Aggregate maxima cannot later reconstruct a combined absolute/relative envelope, so
    /// qualified evidence is never detached from the policy that was actually validated.
    pub validated_policy: Option<PrecisionPolicy>,
    pub outputs: Vec<OutputAssessment>,
    pub reasons: Vec<String>,
}

impl NumericalAssessment {
    pub fn exact() -> Self {
        Self {
            evidence: EvidenceClass::Exact,
            validated_policy: None,
            outputs: Vec::new(),
            reasons: Vec::new(),
        }
    }

    pub fn unknown(reason: impl Into<String>) -> Self {
        Self {
            evidence: EvidenceClass::Unknown,
            validated_policy: None,
            outputs: Vec::new(),
            reasons: vec![reason.into()],
        }
    }

    pub fn qualified(
        validated_policy: PrecisionPolicy,
        outputs: Vec<OutputAssessment>,
    ) -> Result<Self, String> {
        if !matches!(validated_policy, PrecisionPolicy::Bounded { .. }) {
            return Err("qualification requires a bounded validation policy".into());
        }
        if outputs.is_empty() {
            return Err("a qualification must assess at least one observable output".into());
        }
        if outputs
            .iter()
            .any(|output| output.evidence != EvidenceClass::Qualified)
        {
            return Err("every output of a qualified assessment needs qualified evidence".into());
        }
        Ok(Self {
            evidence: EvidenceClass::Qualified,
            validated_policy: Some(validated_policy),
            outputs,
            reasons: Vec::new(),
        })
    }

    pub fn satisfies(&self, policy: &PrecisionPolicy) -> bool {
        match policy {
            PrecisionPolicy::Unconstrained => true,
            PrecisionPolicy::Exact => {
                self.evidence == EvidenceClass::Exact
                    && self.outputs.iter().all(|output| output.metrics.is_exact())
            }
            PrecisionPolicy::Bounded {
                evidence, specials, ..
            } => {
                let evidence_ok = match (self.evidence, evidence) {
                    (EvidenceClass::Exact | EvidenceClass::Proven, _) => true,
                    (EvidenceClass::Qualified, EvidenceRequirement::Qualified) => true,
                    _ => false,
                };
                let validated = match self.evidence {
                    EvidenceClass::Qualified => {
                        self.validated_policy.as_ref().is_some_and(|validated| {
                            policy_is_at_least_as_permissive(policy, validated)
                        })
                    }
                    _ => true,
                };
                evidence_ok
                    && validated
                    && (self.evidence == EvidenceClass::Exact || !self.outputs.is_empty())
                    && self.outputs.iter().all(|output| {
                        let Some(tolerance) = policy.tolerance(&output.output) else {
                            return false;
                        };
                        let m = output.metrics;
                        // Proven assessments carry conservative independent maxima. Qualified
                        // assessments were checked against `validated_policy` elementwise.
                        (self.evidence == EvidenceClass::Qualified
                            || (m.maximum_absolute <= tolerance.absolute.get()
                                && m.maximum_relative <= tolerance.relative.get()
                                && tolerance.ulps.is_none_or(|limit| m.maximum_ulps <= limit)))
                            && (!specials.nan || m.nan_mismatches == 0)
                            && (!specials.infinity || m.infinity_mismatches == 0)
                            && (!specials.signed_zero || m.signed_zero_mismatches == 0)
                            && (!specials.subnormal || m.subnormal_mismatches == 0)
                    })
            }
        }
    }
}

/// Whether every behavior admitted by `validated` is also admitted by `requested`.
/// Output names and input-domain facts are identity: evidence is not widened to another domain.
fn policy_is_at_least_as_permissive(
    requested: &PrecisionPolicy,
    validated: &PrecisionPolicy,
) -> bool {
    let (
        PrecisionPolicy::Bounded {
            default: requested_default,
            outputs: requested_outputs,
            specials: requested_specials,
            inputs: requested_inputs,
            ..
        },
        PrecisionPolicy::Bounded {
            default: validated_default,
            outputs: validated_outputs,
            specials: validated_specials,
            inputs: validated_inputs,
            ..
        },
    ) = (requested, validated)
    else {
        return false;
    };
    if requested_inputs != validated_inputs
        || (requested_specials.nan && !validated_specials.nan)
        || (requested_specials.infinity && !validated_specials.infinity)
        || (requested_specials.signed_zero && !validated_specials.signed_zero)
        || (requested_specials.subnormal && !validated_specials.subnormal)
    {
        return false;
    }
    let looser = |requested: Tolerance, validated: Tolerance| {
        requested.absolute >= validated.absolute
            && requested.relative >= validated.relative
            && requested.relative_floor >= validated.relative_floor
            && match (requested.ulps, validated.ulps) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(requested), Some(validated)) => requested >= validated,
            }
    };
    if !looser(*requested_default, *validated_default) {
        return false;
    }
    let names = requested_outputs.keys().chain(validated_outputs.keys());
    names.into_iter().all(|name| {
        looser(
            requested_outputs
                .get(name)
                .copied()
                .unwrap_or(*requested_default),
            validated_outputs
                .get(name)
                .copied()
                .unwrap_or(*validated_default),
        )
    })
}

fn ordered_f32(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

fn ordered_bits(bits: u64, width: u32) -> u64 {
    let sign = 1_u64 << (width - 1);
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1_u64 << width) - 1
    };
    if bits & sign == 0 {
        bits | sign
    } else {
        (!bits) & mask
    }
}

fn float_bits(dtype: DType, value: f32) -> Option<(u64, u32)> {
    match dtype {
        DType::F32 => Some((u64::from(value.to_bits()), 32)),
        DType::BF16 => Some((
            u64::from(crate::numeric::bf16_round(value).to_bits() >> 16),
            16,
        )),
        DType::F16 => Some((u64::from(crate::numeric::f16_bits(value)), 16)),
        _ => None,
    }
}

fn is_subnormal(dtype: DType, value: f32) -> bool {
    match dtype {
        DType::F32 => value.is_subnormal(),
        DType::BF16 => {
            let bits = crate::numeric::bf16_round(value).to_bits() >> 16;
            bits & 0x7f80 == 0 && bits & 0x007f != 0
        }
        DType::F16 => {
            let bits = crate::numeric::f16_bits(value);
            bits & 0x7c00 == 0 && bits & 0x03ff != 0
        }
        _ => false,
    }
}

pub fn ulp_distance_f32(left: f32, right: f32) -> u64 {
    u64::from(ordered_f32(left).abs_diff(ordered_f32(right)))
}

pub fn ulp_distance(dtype: DType, left: f32, right: f32) -> Option<u64> {
    let ((left, width), (right, other_width)) =
        (float_bits(dtype, left)?, float_bits(dtype, right)?);
    (width == other_width).then(|| ordered_bits(left, width).abs_diff(ordered_bits(right, width)))
}

/// Compare dense values using their published dtype. When `tolerance` is present the report also
/// counts elementwise violations of the combined absolute/relative envelope, ULP requirement and
/// special-value policy.
pub fn compare_dense(
    reference: &[f64],
    candidate: &[f64],
    dtype: DType,
    tolerance: Option<Tolerance>,
    specials: SpecialPolicy,
) -> Result<ComparisonReport, String> {
    if reference.len() != candidate.len() {
        return Err(format!(
            "comparison length differs: {} reference values, {} candidate values",
            reference.len(),
            candidate.len()
        ));
    }
    let mut report = ComparisonReport {
        metrics: ErrorMetrics {
            compared: reference.len() as u64,
            ..ErrorMetrics::default()
        },
        ..ComparisonReport::default()
    };
    for (index, (&reference, &candidate)) in reference.iter().zip(candidate).enumerate() {
        if !dtype.is_float() {
            if reference != candidate {
                report.metrics.differing += 1;
                report.beyond_tolerance += 1;
                report.worst_element.get_or_insert(index);
            }
            continue;
        }
        let (reference, candidate) = (reference as f32, candidate as f32);
        let same_nan = reference.is_nan() && candidate.is_nan();
        if reference.is_nan() || candidate.is_nan() {
            if !same_nan {
                report.metrics.nan_mismatches += 1;
                report.metrics.differing += 1;
                if specials.nan {
                    report.beyond_tolerance += 1;
                }
                report.worst_element.get_or_insert(index);
            }
            continue;
        }
        if reference.is_infinite() || candidate.is_infinite() {
            if reference.to_bits() != candidate.to_bits() {
                report.metrics.infinity_mismatches += 1;
                report.metrics.differing += 1;
                if specials.infinity {
                    report.beyond_tolerance += 1;
                }
                report.worst_element.get_or_insert(index);
            }
            continue;
        }
        let signed_zero_mismatch =
            reference == 0.0 && candidate == 0.0 && reference.to_bits() != candidate.to_bits();
        if signed_zero_mismatch {
            report.metrics.signed_zero_mismatches += 1;
        }
        let subnormal_mismatch = is_subnormal(dtype, reference) != is_subnormal(dtype, candidate)
            && (is_subnormal(dtype, reference) || is_subnormal(dtype, candidate));
        if subnormal_mismatch {
            report.metrics.subnormal_mismatches += 1;
        }
        let same =
            float_bits(dtype, reference).map(|v| v.0) == float_bits(dtype, candidate).map(|v| v.0);
        if same {
            if (signed_zero_mismatch && specials.signed_zero)
                || (subnormal_mismatch && specials.subnormal)
            {
                report.beyond_tolerance += 1;
            }
            continue;
        }
        report.metrics.differing += 1;
        report.worst_element.get_or_insert(index);
        let absolute = f64::from((candidate - reference).abs());
        let floor = tolerance.map_or(0.0, |value| value.relative_floor.get());
        let relative = absolute / f64::from(reference.abs()).max(floor);
        let ulps = ulp_distance(dtype, reference, candidate).unwrap_or(u64::MAX);
        if absolute > report.metrics.maximum_absolute {
            report.metrics.maximum_absolute = absolute;
            report.worst_element = Some(index);
        }
        report.metrics.maximum_relative = report.metrics.maximum_relative.max(relative);
        report.metrics.maximum_ulps = report.metrics.maximum_ulps.max(ulps);
        if let Some(tolerance) = tolerance {
            let envelope = tolerance.envelope(f64::from(reference));
            let ulps_ok = tolerance.ulps.is_none_or(|limit| ulps <= limit);
            if absolute > envelope
                || !ulps_ok
                || (signed_zero_mismatch && specials.signed_zero)
                || (subnormal_mismatch && specials.subnormal)
            {
                report.beyond_tolerance += 1;
            }
        }
    }
    Ok(report)
}

/// Compare one dense f32 output. This is the common implementation used by qualification and
/// backend/reference validation; policy checking remains separate from measurement.
pub fn compare_f32(
    reference: &[f32],
    candidate: &[f32],
    relative_floor: f64,
) -> Result<(ErrorMetrics, Option<usize>), String> {
    let floor = Limit::new(relative_floor)?;
    let reference: Vec<f64> = reference.iter().map(|value| f64::from(*value)).collect();
    let candidate: Vec<f64> = candidate.iter().map(|value| f64::from(*value)).collect();
    let report = compare_dense(
        &reference,
        &candidate,
        DType::F32,
        Some(Tolerance {
            relative_floor: floor,
            absolute: Limit::new(f64::MAX)?,
            relative: Limit::ZERO,
            ulps: None,
        }),
        SpecialPolicy::PRESERVE,
    )?;
    Ok((report.metrics, report.worst_element))
}

/// Construct qualified evidence for one observable f32 output using the shared comparator.
pub fn qualify_f32(
    output: impl Into<String>,
    reference: &[f32],
    candidate: &[f32],
    policy: PrecisionPolicy,
    attribution: Vec<String>,
) -> Result<NumericalAssessment, String> {
    let reference: Vec<f64> = reference.iter().map(|value| f64::from(*value)).collect();
    let candidate: Vec<f64> = candidate.iter().map(|value| f64::from(*value)).collect();
    qualify_dense_output(
        output,
        &reference,
        &candidate,
        DType::F32,
        policy,
        attribution,
    )
}

/// Construct policy-bound qualification evidence for one observable dense output. A failed
/// comparison is a qualification failure, never a weaker evidence record.
pub fn qualify_dense_output(
    output: impl Into<String>,
    reference: &[f64],
    candidate: &[f64],
    dtype: DType,
    policy: PrecisionPolicy,
    attribution: Vec<String>,
) -> Result<NumericalAssessment, String> {
    let output = output.into();
    let tolerance = policy
        .tolerance(&output)
        .ok_or("unconstrained exploration cannot produce qualification evidence")?;
    if !matches!(policy, PrecisionPolicy::Bounded { .. }) {
        return Err("qualification requires a bounded validation policy".into());
    }
    let specials = match &policy {
        PrecisionPolicy::Bounded { specials, .. } => *specials,
        _ => unreachable!(),
    };
    let report = compare_dense(reference, candidate, dtype, Some(tolerance), specials)?;
    if report.beyond_tolerance != 0 {
        return Err(format!(
            "output `{output}` has {} element(s) beyond the precision policy; worst element {:?}",
            report.beyond_tolerance, report.worst_element
        ));
    }
    NumericalAssessment::qualified(
        policy,
        vec![OutputAssessment {
            output,
            dtype,
            evidence: EvidenceClass::Qualified,
            metrics: report.metrics,
            worst_element: report.worst_element,
            attribution,
        }],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_are_stable_identity_values() {
        assert!(Limit::new(f64::NAN).is_err());
        assert!(Limit::new(-1.0).is_err());
        assert_eq!(Limit::new(-0.0).unwrap(), Limit::ZERO);
        assert_eq!(InputRange::new(-2.0, 3.0).unwrap().minimum.get(), -2.0);
        assert!(InputRange::new(3.0, -2.0).is_err());
    }

    #[test]
    fn comparison_distinguishes_special_values_and_ulps() {
        let reference = [0.0, f32::INFINITY, f32::NAN, 1.0];
        let candidate = [
            -0.0,
            f32::NEG_INFINITY,
            1.0,
            f32::from_bits(1.0f32.to_bits() + 1),
        ];
        let (metrics, worst) = compare_f32(&reference, &candidate, 1e-30).unwrap();
        assert_eq!(metrics.signed_zero_mismatches, 1);
        assert_eq!(metrics.infinity_mismatches, 1);
        assert_eq!(metrics.nan_mismatches, 1);
        assert_eq!(metrics.maximum_ulps, 1);
        assert_eq!(worst, Some(3));
    }

    #[test]
    fn qualification_is_bound_to_the_checked_combined_envelope() {
        let checked = PrecisionPolicy::Bounded {
            default: Tolerance {
                absolute: Limit::new(0.01).unwrap(),
                relative: Limit::new(0.1).unwrap(),
                relative_floor: Limit::new(0.001).unwrap(),
                ulps: None,
            },
            outputs: BTreeMap::new(),
            evidence: EvidenceRequirement::Qualified,
            specials: SpecialPolicy::PRESERVE,
            inputs: BTreeMap::new(),
        };
        // 0.105 is allowed around 1.0 by the combined envelope (0.11), despite exceeding
        // both independently retained component limits.
        let assessment = qualify_f32("out", &[1.0], &[1.105], checked.clone(), vec![]).unwrap();
        assert!(assessment.satisfies(&checked));

        let mut tighter = checked.clone();
        let PrecisionPolicy::Bounded { default, .. } = &mut tighter else {
            unreachable!()
        };
        default.relative = Limit::new(0.05).unwrap();
        assert!(!assessment.satisfies(&tighter));

        let mut looser = checked.clone();
        let PrecisionPolicy::Bounded { default, .. } = &mut looser else {
            unreachable!()
        };
        default.absolute = Limit::new(0.02).unwrap();
        assert!(assessment.satisfies(&looser));
    }
}
