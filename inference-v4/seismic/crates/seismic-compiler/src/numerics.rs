//! Numerical precision as a planning dimension (spec §9).
//!
//! Every implementation's transfer is derived from its actual operations,
//! order, data types, reductions, approximations, and intrinsic semantics.
//! Admissibility under the caller's policy is a predicate over decisions
//! and invocation symbols, encoded into the solver problem; it is never
//! checked after selection. Evidence is keyed exactly (§9.3) and its
//! availability is part of the same predicate.
//!
//! W6 owns derivation and encoding internals.

use crate::implementation::ImplementationIdentity;
use crate::target::NumericalEnvironmentIdentity;
use seismic_lang::expr::{BoolExpr, CmpOp, DecisionId, ExprArena, NatExpr};
use seismic_lang::precision::{
    EvidenceRequirement, InputRange, PrecisionPolicy, SpecialPolicy, Tolerance,
};
use seismic_lang::types::DType;
use std::collections::BTreeMap;

pub use seismic_lang::precision;

/// The derived numerical transfer of one implementation: per output, a
/// conservative outward-rounded bound expression over invocation symbols
/// and decisions, plus the discrete effects that make the implementation
/// differ from the reference.
#[derive(Clone, Debug)]
pub struct NumericalTransfer {
    outputs: Vec<OutputTransfer>,
    effects: Vec<NumericalEffect>,
    operations: Vec<NumericalOperation>,
    /// Input ranges the analytical transfer was derived over. A bounded
    /// policy must declare an equal or narrower range for each named input.
    input_assumptions: BTreeMap<String, InputRange>,
    /// Transfers of inlined callees, guarded by the exact finite choice arm
    /// that selects them. Keeping the guard with the transfer prevents call
    /// composition from turning an unselected approximate body into a global
    /// precision failure.
    children: Vec<ConditionalNumericalTransfer>,
}

#[derive(Clone, Debug)]
pub(crate) struct ConditionalNumericalTransfer {
    pub(crate) selection: Option<(DecisionId, i64)>,
    pub(crate) role: seismic_lang::entry::NumericalRole,
    pub(crate) transfer: Box<NumericalTransfer>,
}

impl NumericalTransfer {
    pub(crate) fn new(
        outputs: Vec<OutputTransfer>,
        effects: Vec<NumericalEffect>,
        operations: Vec<NumericalOperation>,
        input_assumptions: BTreeMap<String, InputRange>,
        children: Vec<ConditionalNumericalTransfer>,
    ) -> Self {
        Self {
            outputs,
            effects,
            operations,
            input_assumptions,
            children,
        }
    }
    pub fn outputs(&self) -> &[OutputTransfer] {
        &self.outputs
    }
    pub fn effects(&self) -> &[NumericalEffect] {
        &self.effects
    }
    pub fn operations(&self) -> &[NumericalOperation] {
        &self.operations
    }
    pub fn input_assumptions(&self) -> &BTreeMap<String, InputRange> {
        &self.input_assumptions
    }
    pub(crate) fn children(&self) -> &[ConditionalNumericalTransfer] {
        &self.children
    }
    /// Heap storage retained by this transfer, including recursively inlined
    /// callees. Budget accounting deliberately follows owned capacities so a
    /// transfer cannot hide an unbounded allocation behind its shallow size.
    pub(crate) fn retained_bytes(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>()
            .saturating_add(
                self.outputs
                    .capacity()
                    .saturating_mul(std::mem::size_of::<OutputTransfer>()),
            )
            .saturating_add(
                self.effects
                    .capacity()
                    .saturating_mul(std::mem::size_of::<NumericalEffect>()),
            )
            .saturating_add(
                self.operations
                    .capacity()
                    .saturating_mul(std::mem::size_of::<NumericalOperation>()),
            )
            .saturating_add(
                self.children
                    .capacity()
                    .saturating_mul(std::mem::size_of::<ConditionalNumericalTransfer>()),
            );
        for output in &self.outputs {
            bytes = bytes.saturating_add(
                output
                    .path
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()),
            );
        }
        // BTreeMap has no capacity API. Charge every retained key/value plus
        // the key's separately allocated UTF-8 buffer; the conservative
        // pointer overhead covers the tree links and allocator bookkeeping.
        for (name, _range) in &self.input_assumptions {
            bytes = bytes
                .saturating_add(std::mem::size_of::<(String, InputRange)>())
                .saturating_add(name.capacity())
                .saturating_add(3 * std::mem::size_of::<usize>());
        }
        for child in &self.children {
            bytes = bytes.saturating_add(child.transfer.retained_bytes());
        }
        bytes
    }
    /// True when the implementation reproduces the reference exactly.
    pub fn is_exact(&self) -> bool {
        self.effects.is_empty()
            && self.operations.is_empty()
            && self
                .outputs
                .iter()
                .all(|output| matches!(output.bound, ErrorBound::Exact))
            && self.children.iter().all(|child| {
                child.role == seismic_lang::entry::NumericalRole::Reference
                    && child.transfer.is_exact()
            })
    }
}

/// One ordered operation-level numerical fact and its exact dynamic
/// multiplicity. Kernel and ordinal are stable construction ordinals.
#[derive(Clone, Debug)]
pub struct NumericalOperation {
    pub kernel: u32,
    pub ordinal: u32,
    pub effect: NumericalEffect,
    pub multiplicity: Option<NatExpr>,
}

#[derive(Clone, Debug)]
pub struct OutputTransfer {
    pub path: Vec<u32>,
    pub dtype: DType,
    pub bound: ErrorBound,
    pub specials: SpecialGuarantees,
}

/// Exceptional-value semantics proved for one published output. These are
/// explicit transfer facts; policy checking never guesses them from effect
/// names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpecialGuarantees {
    pub nan: bool,
    pub infinity: bool,
    pub signed_zero: bool,
    pub subnormal: bool,
}

/// A conservative error envelope as arena expressions: absolute and
/// relative components scaled by shape-dependent rounding counts.
#[derive(Clone, Debug)]
pub enum ErrorBound {
    Exact,
    /// `abs <= absolute * roundings`, `rel <= relative * roundings` in ulps
    /// of the published dtype.
    Analytic {
        roundings: NatExpr,
        /// Worst-case absolute error contributed per rounding.
        absolute: f64,
        /// Worst-case relative error contributed per rounding.
        relative: f64,
        /// Worst-case ULP error contributed per rounding.
        ulps: u32,
    },
    /// No analytic bound is derivable; admissible only with evidence or an
    /// unconstrained policy.
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NumericalEffect {
    ReassociatedReduction,
    Contraction,
    ApproximateTranscendental(seismic_lang::intrinsics::MathOp),
    NarrowAccumulator(DType),
    FlushToZero,
    BackendIntrinsic(seismic_lang::ids::IntrinsicId),
    AlternativeBody,
}

/// Exact key of one piece of empirical evidence (§9.3).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EvidenceKey {
    pub implementation: ImplementationIdentity,
    /// Stable decision ordinal and value. Arena-local `DecisionId` values are
    /// deliberately absent from persisted evidence identities.
    pub decisions: Vec<(u32, i64)>,
    pub target: NumericalEnvironmentIdentity,
    /// Digest of every selected native kernel's numerical-mode identity.
    pub native_numerics: [u8; 32],
    /// Digest of the semantic domain predicate the evidence covers.
    pub domain: [u8; 32],
    pub policy: PolicyIdentity,
    pub corpus: String,
    pub qualification_version: u32,
}

/// Content identity of a precision policy (§15.1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PolicyIdentity(pub [u8; 32]);

impl PolicyIdentity {
    pub fn of(policy: &PrecisionPolicy) -> Self {
        internals::policy_identity(policy)
    }
}

/// One qualified record.
#[derive(Clone, Debug)]
pub struct EvidenceRecord {
    pub key: EvidenceKey,
}

/// The immutable evidence catalog a preparation receives.
#[derive(Clone, Debug, Default)]
pub struct EvidenceCatalog {
    records: Vec<EvidenceRecord>,
}

impl EvidenceCatalog {
    pub fn new(records: Vec<EvidenceRecord>) -> Self {
        Self { records }
    }
    pub fn records(&self) -> &[EvidenceRecord] {
        &self.records
    }
}

/// The assessment carried by a frozen plan and its executable variant
/// (§11.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumericalAssessment {
    Exact,
    Proven {
        policy: PolicyIdentity,
    },
    Qualified {
        policy: PolicyIdentity,
        evidence: Vec<EvidenceKey>,
    },
    /// Selectable only under an unconstrained policy.
    Unknown,
}

/// Builds the admissibility predicate of one implementation under a policy:
/// a `BoolExpr` over decisions and invocation symbols that is true exactly
/// when the transfer satisfies the policy (analytically or by matching
/// evidence). Encoded into the solver by `plan_space` (§9.3).
pub fn admissibility(
    arena: &mut ExprArena,
    transfer: &NumericalTransfer,
    policy: &PrecisionPolicy,
    evidence: &EvidenceCatalog,
    implementation: &ImplementationIdentity,
    decisions: &[(DecisionId, &'static str)],
    target: &NumericalEnvironmentIdentity,
    native_numerics: [u8; 32],
    domain: [u8; 32],
) -> BoolExpr {
    internals::admissibility(
        arena,
        transfer,
        policy,
        evidence,
        implementation,
        decisions,
        target,
        native_numerics,
        domain,
    )
}

/// The assessment of one frozen selection under its guard.
pub fn assess(
    arena: &ExprArena,
    transfer: &NumericalTransfer,
    policy: &PrecisionPolicy,
    evidence: &EvidenceCatalog,
    implementation: &ImplementationIdentity,
    target: &NumericalEnvironmentIdentity,
    native_numerics: [u8; 32],
    domain: [u8; 32],
    decisions: &[(DecisionId, i64)],
) -> NumericalAssessment {
    internals::assess(
        arena,
        transfer,
        policy,
        evidence,
        implementation,
        target,
        native_numerics,
        domain,
        decisions,
    )
}

mod internals {
    use super::*;
    use sha2::{Digest, Sha256};

    pub(super) fn policy_identity(policy: &PrecisionPolicy) -> PolicyIdentity {
        let mut digest = Sha256::new();
        digest.update(b"seismic-precision-policy-v1");
        match policy {
            PrecisionPolicy::Exact => digest.update([0]),
            PrecisionPolicy::Unconstrained => digest.update([1]),
            PrecisionPolicy::Bounded {
                default,
                outputs,
                evidence,
                specials,
                inputs,
            } => {
                digest.update([2]);
                tolerance(&mut digest, *default);
                digest.update([match evidence {
                    EvidenceRequirement::Proven => 0,
                    EvidenceRequirement::Qualified => 1,
                }]);
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

    pub(super) fn admissibility(
        arena: &mut ExprArena,
        transfer: &NumericalTransfer,
        policy: &PrecisionPolicy,
        evidence: &EvidenceCatalog,
        implementation: &ImplementationIdentity,
        decisions: &[(DecisionId, &'static str)],
        target: &NumericalEnvironmentIdentity,
        native_numerics: [u8; 32],
        domain: [u8; 32],
    ) -> BoolExpr {
        if matches!(policy, PrecisionPolicy::Unconstrained) {
            return arena.bool(true);
        }
        if matches!(policy, PrecisionPolicy::Exact) {
            return exact_predicate(arena, transfer);
        }

        let analytic = analytic_predicate(arena, transfer, policy);
        let allow_qualified = matches!(
            policy,
            PrecisionPolicy::Bounded {
                evidence: EvidenceRequirement::Qualified,
                ..
            }
        );
        if !allow_qualified {
            return analytic;
        }
        let policy = PolicyIdentity::of(policy);
        let qualified =
            evidence
                .records()
                .iter()
                .filter(|record| {
                    record.key.implementation == *implementation
                        && record.key.target == *target
                        && record.key.native_numerics == native_numerics
                        && record.key.domain == domain
                        && record.key.policy == policy
                        && record.key.decisions.len() == decisions.len()
                        && record.key.decisions.iter().enumerate().all(
                            |(ordinal, (record_ordinal, _))| *record_ordinal as usize == ordinal,
                        )
                })
                .filter_map(|record| {
                    let terms = record
                        .key
                        .decisions
                        .iter()
                        .map(|(ordinal, value)| {
                            decisions
                                .get(*ordinal as usize)
                                .map(|decision| arena.decision_is(decision.0, *value))
                        })
                        .collect::<Option<Vec<_>>>()?;
                    Some(arena.all(&terms))
                })
                .collect::<Vec<_>>();
        let evidence = arena.any(&qualified);
        arena.or(analytic, evidence)
    }

    fn analytic_predicate(
        arena: &mut ExprArena,
        transfer: &NumericalTransfer,
        policy: &PrecisionPolicy,
    ) -> BoolExpr {
        let PrecisionPolicy::Bounded {
            specials, inputs, ..
        } = policy
        else {
            return arena.bool(false);
        };
        if transfer.input_assumptions().iter().any(|(name, required)| {
            inputs.get(name).is_none_or(|provided| {
                provided.minimum.get() < required.minimum.get()
                    || provided.maximum.get() > required.maximum.get()
            })
        }) {
            return arena.bool(false);
        }
        let mut terms = Vec::with_capacity(
            transfer
                .outputs()
                .len()
                .saturating_add(transfer.children().len()),
        );
        for output in transfer.outputs() {
            let key = output_key(&output.path);
            let tolerance = policy
                .tolerance(&key)
                .expect("bounded policy has a tolerance");
            let special = special_ok(output.specials, *specials);
            let error = match output.bound {
                ErrorBound::Exact => arena.bool(true),
                ErrorBound::Unknown => arena.bool(false),
                ErrorBound::Analytic {
                    roundings,
                    absolute,
                    relative,
                    ulps,
                } => {
                    // The hybrid tolerance allows `absolute + relative *
                    // relative_floor` around zero, while the independent
                    // relative proof controls larger reference values.
                    let absolute_limit = tolerance.absolute.get()
                        + tolerance.relative.get() * tolerance.relative_floor.get();
                    let absolute_ok = coefficient_bound(arena, roundings, absolute, absolute_limit);
                    let relative_ok =
                        coefficient_bound(arena, roundings, relative, tolerance.relative.get());
                    let ulp_ok = match (ulps, tolerance.ulps) {
                        (0, _) => arena.bool(true),
                        (_, Some(limit)) => {
                            let maximum = limit / u64::from(ulps);
                            let maximum = arena.nat(maximum);
                            arena.nat_cmp(CmpOp::Le, roundings, maximum)
                        }
                        (_, None) => arena.bool(false),
                    };
                    let magnitude_ok = arena.and(absolute_ok, relative_ok);
                    arena.and(magnitude_ok, ulp_ok)
                }
            };
            let special = arena.bool(special);
            terms.push(arena.and(error, special));
        }
        // Until the operation-level analyser has composed a callee's bound
        // through the caller's downstream arithmetic, only an exact selected
        // child can satisfy an analytical parent proof. This is deliberately
        // conservative and, unlike independently checking both tolerances,
        // cannot admit two local errors whose sum exceeds the caller policy.
        for child in transfer.children() {
            let child_exact = if child.role == seismic_lang::entry::NumericalRole::Reference {
                exact_predicate(arena, &child.transfer)
            } else {
                arena.bool(false)
            };
            let condition = match child.selection {
                Some((decision, value)) => arena.decision_is(decision, value),
                None => arena.bool(true),
            };
            terms.push(arena.implies(condition, child_exact));
        }
        arena.all(&terms)
    }

    fn exact_predicate(arena: &mut ExprArena, transfer: &NumericalTransfer) -> BoolExpr {
        let local = transfer.effects().is_empty()
            && transfer.operations().is_empty()
            && transfer
                .outputs()
                .iter()
                .all(|output| matches!(output.bound, ErrorBound::Exact));
        let mut terms = vec![arena.bool(local)];
        for child in transfer.children() {
            let exact = if child.role == seismic_lang::entry::NumericalRole::Reference {
                exact_predicate(arena, &child.transfer)
            } else {
                arena.bool(false)
            };
            let condition = match child.selection {
                Some((decision, value)) => arena.decision_is(decision, value),
                None => arena.bool(true),
            };
            terms.push(arena.implies(condition, exact));
        }
        arena.all(&terms)
    }

    fn coefficient_bound(
        arena: &mut ExprArena,
        count: NatExpr,
        coefficient: f64,
        limit: f64,
    ) -> BoolExpr {
        if coefficient == 0.0 {
            return arena.bool(true);
        }
        if !coefficient.is_finite() || coefficient < 0.0 {
            return arena.bool(false);
        }
        let ratio = (limit / coefficient).floor();
        if ratio >= u64::MAX as f64 {
            return arena.bool(true);
        }
        let maximum = if ratio <= 0.0 { 0 } else { ratio as u64 };
        let maximum = arena.nat(maximum);
        arena.nat_cmp(CmpOp::Le, count, maximum)
    }

    fn special_ok(guarantee: SpecialGuarantees, requested: SpecialPolicy) -> bool {
        (!requested.nan || guarantee.nan)
            && (!requested.infinity || guarantee.infinity)
            && (!requested.signed_zero || guarantee.signed_zero)
            && (!requested.subnormal || guarantee.subnormal)
    }

    fn output_key(path: &[u32]) -> String {
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

    pub(super) fn assess(
        _arena: &ExprArena,
        transfer: &NumericalTransfer,
        policy: &PrecisionPolicy,
        evidence: &EvidenceCatalog,
        implementation: &ImplementationIdentity,
        target: &NumericalEnvironmentIdentity,
        native_numerics: [u8; 32],
        domain: [u8; 32],
        decisions: &[(DecisionId, i64)],
    ) -> NumericalAssessment {
        if selected_exact(transfer, decisions) {
            return NumericalAssessment::Exact;
        }
        if matches!(policy, PrecisionPolicy::Unconstrained) {
            return NumericalAssessment::Unknown;
        }

        // The analytical predicate is part of the frozen guard. If it is
        // not identically false after fixing decisions, every invocation
        // admitted by that guard satisfies the proof.
        let mut probe =
            evidence
                .records()
                .iter()
                .filter(|record| {
                    record.key.implementation == *implementation
                        && record.key.target == *target
                        && record.key.native_numerics == native_numerics
                        && record.key.domain == domain
                        && record.key.policy == PolicyIdentity::of(policy)
                        && record.key.decisions.len() == decisions.len()
                        && record.key.decisions.iter().enumerate().all(
                            |(ordinal, (record_ordinal, _))| *record_ordinal as usize == ordinal,
                        )
                        && record.key.decisions.iter().all(|(ordinal, expected)| {
                            decisions
                                .get(*ordinal as usize)
                                .is_some_and(|(_, actual)| actual == expected)
                        })
                })
                .map(|record| record.key.clone())
                .collect::<Vec<_>>();
        if !probe.is_empty() {
            probe.sort_by(|a, b| {
                a.corpus
                    .cmp(&b.corpus)
                    .then(a.qualification_version.cmp(&b.qualification_version))
            });
            return NumericalAssessment::Qualified {
                policy: PolicyIdentity::of(policy),
                evidence: probe,
            };
        }
        NumericalAssessment::Proven {
            policy: PolicyIdentity::of(policy),
        }
    }

    fn selected_exact(transfer: &NumericalTransfer, decisions: &[(DecisionId, i64)]) -> bool {
        let local = transfer.effects().is_empty()
            && transfer.operations().is_empty()
            && transfer
                .outputs()
                .iter()
                .all(|output| matches!(output.bound, ErrorBound::Exact));
        local
            && transfer.children().iter().all(|child| {
                let selected = child.selection.is_none_or(|(decision, expected)| {
                    decisions
                        .iter()
                        .find(|(candidate, _)| *candidate == decision)
                        .is_some_and(|(_, actual)| *actual == expected)
                });
                !selected
                    || (child.role == seismic_lang::entry::NumericalRole::Reference
                        && selected_exact(&child.transfer, decisions))
            })
    }
}
