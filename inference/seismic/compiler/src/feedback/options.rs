use seismic_lang::expr::SymbolValue;
use seismic_lang::ids::StableEntryId;
use std::time::Duration;

/// Entry-scoped invocation constraints. Generated typed range builders supply
/// ordinals from that entry's schema; no expression or compiler symbol ids cross
/// the public preparation boundary.
#[derive(Clone, Debug)]
pub struct InvocationScope {
    pub(super) entry: StableEntryId,
    pub(super) constraints: Vec<Constraint>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[doc(hidden)]
pub enum InvocationParameter {
    Dimension(u32),
    Scalar(u32),
    RangeStart(u32),
    RangeEnd(u32),
}
#[derive(Clone, Debug)]
pub(super) struct Constraint {
    pub parameter: InvocationParameter,
    pub lower: SymbolValue,
    pub upper: SymbolValue,
}
impl InvocationScope {
    #[doc(hidden)]
    pub fn for_entry(entry: StableEntryId) -> Self {
        Self {
            entry,
            constraints: Vec::new(),
        }
    }
    #[doc(hidden)]
    pub fn constrain(
        &mut self,
        parameter: InvocationParameter,
        lower: SymbolValue,
        upper: SymbolValue,
    ) {
        self.constraints.push(Constraint {
            parameter,
            lower,
            upper,
        });
    }
}

#[derive(Clone, Debug)]
pub struct FeedbackOptions {
    pub search_time: Duration,
    pub optimize_for: Option<InvocationScope>,
    pub seed: u64,
    pub experiment_memory_bytes: u64,
    pub reference_work_limit: u64,
    pub archive_points: usize,
    pub archive_candidates: usize,
}
impl Default for FeedbackOptions {
    fn default() -> Self {
        Self {
            search_time: Duration::from_secs(60),
            optimize_for: None,
            seed: 0,
            experiment_memory_bytes: 256 * 1024 * 1024,
            reference_work_limit: 1_000_000,
            archive_points: 1024,
            archive_candidates: 4096,
        }
    }
}

#[derive(Clone, Debug)]
pub enum EvaluationMethod {
    Analytical,
    Feedback(FeedbackOptions),
}
#[derive(Clone, Debug)]
pub struct PreparationOptions {
    pub precision: seismic_lang::precision::PrecisionPolicy,
    pub evaluation: EvaluationMethod,
}
impl Default for PreparationOptions {
    fn default() -> Self {
        Self {
            precision: Default::default(),
            evaluation: EvaluationMethod::Analytical,
        }
    }
}

impl PreparationOptions {
    pub fn analytical(precision: seismic_lang::precision::PrecisionPolicy) -> Self {
        Self {
            precision,
            evaluation: EvaluationMethod::Analytical,
        }
    }
    pub fn feedback(
        precision: seismic_lang::precision::PrecisionPolicy,
        options: FeedbackOptions,
    ) -> Self {
        Self {
            precision,
            evaluation: EvaluationMethod::Feedback(options),
        }
    }
}

impl EvaluationMethod {
    /// Requested preparation identity, separate from the final result's evidence.
    pub fn fingerprint(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        digest.update(b"seismic-preparation-method-v3");
        match self {
            Self::Analytical => digest.update([0]),
            Self::Feedback(options) => {
                digest.update([1]);
                digest.update(options.search_time.as_nanos().to_le_bytes());
                digest.update(options.seed.to_le_bytes());
                digest.update(options.experiment_memory_bytes.to_le_bytes());
                digest.update(options.reference_work_limit.to_le_bytes());
                digest.update((options.archive_points as u64).to_le_bytes());
                digest.update((options.archive_candidates as u64).to_le_bytes());
                if let Some(scope) = &options.optimize_for {
                    digest.update([1]);
                    digest.update(scope.entry.digest());
                    digest.update((scope.constraints.len() as u64).to_le_bytes());
                    for constraint in &scope.constraints {
                        let (tag, ordinal) = match constraint.parameter {
                            InvocationParameter::Dimension(i) => (0, i),
                            InvocationParameter::Scalar(i) => (1, i),
                            InvocationParameter::RangeStart(i) => (2, i),
                            InvocationParameter::RangeEnd(i) => (3, i),
                        };
                        digest.update([tag]);
                        digest.update(ordinal.to_le_bytes());
                        for value in [&constraint.lower, &constraint.upper] {
                            let (tag, bits) = match value {
                                SymbolValue::Nat(v) => (0, v.to_bytes_le()),
                                SymbolValue::Int(v) => (1, v.to_signed_bytes_le()),
                                SymbolValue::F32(v) => (2, v.to_bits().to_le_bytes().to_vec()),
                                SymbolValue::F16(v) => (3, v.to_le_bytes().to_vec()),
                                SymbolValue::BF16(v) => (4, v.to_le_bytes().to_vec()),
                                SymbolValue::I32(v) => (5, v.to_le_bytes().to_vec()),
                                SymbolValue::U32(v) => (6, v.to_le_bytes().to_vec()),
                                SymbolValue::Bool(v) => (7, [u8::from(*v)].to_vec()),
                            };
                            digest.update([tag]);
                            digest.update((bits.len() as u64).to_le_bytes());
                            digest.update(bits);
                        }
                    }
                } else {
                    digest.update([0]);
                }
            }
        }
        digest.finalize().into()
    }
}

/// One fixed invocation value or inclusive range, used by generated entry builders.
#[derive(Clone, Debug)]
pub struct InvocationRange<T> {
    pub lower: T,
    pub upper: T,
}
impl<T: Clone> From<T> for InvocationRange<T> {
    fn from(value: T) -> Self {
        Self {
            lower: value.clone(),
            upper: value,
        }
    }
}
impl<T> From<std::ops::RangeInclusive<T>> for InvocationRange<T> {
    fn from(range: std::ops::RangeInclusive<T>) -> Self {
        let (lower, upper) = range.into_inner();
        Self { lower, upper }
    }
}
