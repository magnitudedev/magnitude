//! Numerical applicability follows the selected source construction.
//! A required body executes the checked operations; a replacement needs an
//! established whole-outcome relation before it can be admitted.
use crate::portable::outcome_relation::{self, DerivedOutcome};
use seismic_lang::expr::{BoolExpr, ExprArena};
pub use seismic_lang::precision;
use seismic_lang::precision::{PrecisionPolicy, Tolerance};

/// Resolution of the actual selected body and all of its selected callees.
/// This is produced only by source construction, never by observations.
#[derive(Clone, Debug)]
pub struct NumericalApplicability {
    derived: DerivedOutcome,
}
impl NumericalApplicability {
    pub(crate) fn required_source() -> Self {
        Self {
            derived: DerivedOutcome::Exact,
        }
    }

    pub(crate) fn selected_child(&mut self, child: &Self) {
        if matches!(self.derived, DerivedOutcome::Exact) {
            self.derived = match child.derived {
                DerivedOutcome::Exact => DerivedOutcome::Exact,
                DerivedOutcome::DiscreteProvenFloatingUnbounded => DerivedOutcome::Pending(
                    "nonexact helper replacement has no composed whole-entry analysis",
                ),
                DerivedOutcome::Pending(reason) => DerivedOutcome::Pending(reason),
            };
        }
    }

    pub(crate) fn selected_replacement<B: seismic_native_target::TargetFamily>(
        arena: &ExprArena,
        program: &seismic_lang::entry::SemanticProgram,
        function: &seismic_lang::entry::SemanticFunction,
        bindings: &crate::portable::BindingArena,
        executable: &seismic_ir::execution::ClosedExecutableIr<B>,
        children: &Self,
    ) -> Self {
        if !matches!(children.derived, DerivedOutcome::Exact) {
            return children.clone();
        }
        Self {
            derived: outcome_relation::derive(arena, program, function, bindings, executable),
        }
    }
    pub fn is_exact(&self) -> bool {
        matches!(self.derived, DerivedOutcome::Exact)
    }
    pub fn pending_reason(&self) -> Option<&'static str> {
        match self.derived {
            DerivedOutcome::Pending(reason) => Some(reason),
            DerivedOutcome::DiscreteProvenFloatingUnbounded => {
                Some("floating numerical bound is not established")
            }
            DerivedOutcome::Exact => None,
        }
    }
    pub(crate) fn basis(&self, policy: &PrecisionPolicy) -> Option<NumericalBasis> {
        match (&self.derived, policy) {
            (DerivedOutcome::Exact, _) => Some(NumericalBasis::Exact),
            (DerivedOutcome::DiscreteProvenFloatingUnbounded, PrecisionPolicy::Unconstrained) => {
                Some(NumericalBasis::Unknown)
            }
            _ => None,
        }
    }
    #[cfg(test)]
    pub(crate) fn unresolved_fixture() -> Self {
        Self {
            derived: DerivedOutcome::Pending("native-only fixture has no checked outcome"),
        }
    }
}

/// Content identity of a precision policy (§15.1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PolicyIdentity(pub [u8; 32]);

impl PolicyIdentity {
    pub fn of(policy: &PrecisionPolicy) -> Self {
        internals::policy_identity(policy)
    }
}

/// Explanation of the accepted derived numerical regions of a candidate.
/// This is descriptive metadata, not an independently attachable admission.
#[derive(Clone, Debug)]
pub struct NumericalAssessment {
    pub regions: Vec<NumericalRegion>,
}

/// The explanation applies only where this actual compiled predicate holds.
#[derive(Clone, Debug)]
pub struct NumericalRegion {
    pub scope: seismic_lang::expr::compiled::CompiledPredicate,
    pub basis: NumericalBasis,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumericalBasis {
    Exact,
    /// Accepted under the explicitly unconstrained policy.
    Unknown,
}

/// Policy projection of construction-owned applicability. An unresolved
/// alternative remains structural domain membership, never an executable grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StructuralNumericalObligation {
    analytic: BoolExpr,
}
impl StructuralNumericalObligation {
    pub fn analytic_predicate(self) -> BoolExpr {
        self.analytic
    }
}
pub fn structural_obligation(
    arena: &mut ExprArena,
    applicability: &NumericalApplicability,
    policy: &PrecisionPolicy,
) -> StructuralNumericalObligation {
    StructuralNumericalObligation {
        analytic: arena.bool(applicability.basis(policy).is_some()),
    }
}

mod internals {
    use super::*;
    use sha2::{Digest, Sha256};

    pub(super) fn policy_identity(policy: &PrecisionPolicy) -> PolicyIdentity {
        let mut digest = Sha256::new();
        digest.update(b"seismic-precision-policy-v2");
        match policy {
            PrecisionPolicy::Exact => digest.update([0]),
            PrecisionPolicy::Unconstrained => digest.update([1]),
            PrecisionPolicy::Bounded {
                default,
                outputs,
                specials,
                inputs,
            } => {
                digest.update([2]);
                tolerance(&mut digest, *default);
                digest.update([
                    specials.nan as u8,
                    specials.infinity as u8,
                    specials.signed_zero as u8,
                    specials.subnormal as u8,
                ]);
                digest.update((outputs.len() as u64).to_le_bytes());
                for (name, value) in outputs {
                    bytes(&mut digest, name.as_bytes());
                    tolerance(&mut digest, *value);
                }
                digest.update((inputs.len() as u64).to_le_bytes());
                for (name, range) in inputs {
                    bytes(&mut digest, name.as_bytes());
                    digest.update(range.minimum.get().to_bits().to_le_bytes());
                    digest.update(range.maximum.get().to_bits().to_le_bytes());
                }
            }
        }
        PolicyIdentity(digest.finalize().into())
    }

    fn bytes(digest: &mut Sha256, value: &[u8]) {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value);
    }

    fn tolerance(digest: &mut Sha256, value: Tolerance) {
        digest.update(value.absolute.get().to_bits().to_le_bytes());
        digest.update(value.relative.get().to_bits().to_le_bytes());
        digest.update(value.relative_floor.get().to_bits().to_le_bytes());
        match value.ulps {
            Some(ulps) => {
                digest.update([1]);
                digest.update(ulps.to_le_bytes());
            }
            None => digest.update([0]),
        }
    }

    pub(super) fn output_key(path: &[u32]) -> String {
        if path.is_empty() {
            "value".to_owned()
        } else {
            format!(
                "r{}",
                path.iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join("_")
            )
        }
    }
}
mod comparison;
pub use comparison::{compare_element, ElementComparison};
mod outcome_comparison;
pub use outcome_comparison::{
    compare_outcome, Comparison, ComparisonError, ComparisonSubject, Difference, ObservedInput,
    ObservedInvocation, ObservedResult, ObservedTensor, ObservedValue,
};

mod validation;
pub use validation::{ValidationCase, ValidationObservation};

mod demand;
#[cfg(test)]
mod demand_tests;

/// Canonical public numerical subjects. Writable input state currently shares
/// the owner's `value` policy; individual writable leaves are not separate names.
pub fn subject_names(entry: &seismic_lang::checked::EntryInfo) -> Vec<String> {
    let mut names: std::collections::BTreeSet<_> = entry
        .results
        .iter()
        .map(|r| internals::output_key(&r.path))
        .collect();
    if entry.parameters.iter().any(|p| {
        matches!(
            p.kind,
            seismic_lang::checked::ParameterSummaryKind::Tensor {
                access: seismic_lang::checked::TensorAccess::Mutable
                    | seismic_lang::checked::TensorAccess::Owned,
                ..
            }
        )
    }) {
        names.insert("value".into());
    }
    names.into_iter().collect()
}
pub fn validate_policy_subjects(
    entry: &seismic_lang::checked::EntryInfo,
    policy: &PrecisionPolicy,
) -> Result<(), String> {
    if let PrecisionPolicy::Bounded { outputs, .. } = policy {
        let names = subject_names(entry);
        for name in outputs.keys() {
            if !names.contains(name) {
                return Err(format!(
                    "unknown numerical subject {name}; available subjects: {names:?}"
                ));
            }
        }
    }
    Ok(())
}
