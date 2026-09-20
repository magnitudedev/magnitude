//! Numerical transfer taxonomy and composition, hosted here so every crate
//! below the compiler shares the one definition. The registry-level form
//! (`seismic_lang::intrinsics::NumericalTransfer`) describes authored
//! capability effects and is converted here.

use seismic_lang::{
    intrinsics::{ErrorBound, IntrinsicId, PrimitiveId, ReduceOp},
    logical::{EffectiveTargetIdentity, LogicalIdentity},
    precision::{EvidenceRequirement, NumericalAssessment, PrecisionPolicy, Tolerance},
    types::{DType, ExtentExpr, RuntimeExtentId, ValueType},
};

// ---------------------------------------------------------------------------
// Reduction topology (identical shape to the compiler crate's
// `terminal::reduction::ReductionTopology`; this is the single hosted
// definition the compiler crate re-exports)
// ---------------------------------------------------------------------------

/// Exact reduction topology: what the strategy's data flow looks like. The
/// `inner` of a wrapping topology is the per-participant fold.
#[derive(Clone, Debug, PartialEq)]
pub enum ReductionTopology {
    /// A serial fold over one axis, ascending coordinates.
    SerialAxis { axis: usize, length: ExtentExpr },
    /// Independent outer coordinates are parallel; each folds `inner`.
    ParallelOuter {
        outer_axes: Vec<ExtentExpr>,
        inner: Box<ReductionTopology>,
    },
    /// Fixed fan-in combining tree of `depth` levels over `inner` folds.
    Tree {
        fan_in: u32,
        depth: u32,
        inner: Box<ReductionTopology>,
    },
    /// One hardware subgroup of `width` participants combines `inner` folds.
    Subgroup {
        width: u32,
        inner: Box<ReductionTopology>,
    },
    /// One workgroup of `participants` combines `inner` folds (workgroup
    /// staging allocation is an exact hard resource of the alternative).
    Workgroup {
        participants: u32,
        inner: Box<ReductionTopology>,
    },
    /// Matrix-fragment accumulation across `fragments` tiles.
    Matrix {
        fragments: u32,
        inner: Box<ReductionTopology>,
    },
    /// The reduced axis is cut into the given slice lengths; partials combine
    /// through `inner`.
    Split {
        cuts: Vec<ExtentExpr>,
        inner: Box<ReductionTopology>,
    },
    /// Multiple launches each reduce a pass; the results combine through
    /// `inner` in launch order.
    MultiLaunch {
        passes: u32,
        inner: Box<ReductionTopology>,
    },
}

impl ReductionTopology {
    /// The length of the directly-serial fold, when this topology's innermost
    /// step is a serial axis.
    pub fn serial_length(&self) -> Option<&ExtentExpr> {
        match self {
            ReductionTopology::SerialAxis { length, .. } => Some(length),
            _ => None,
        }
    }

    /// Whether the reduced axis is folded in the reference ascending order:
    /// a serial axis, or parallel outer coordinates each folding serially.
    /// Reassociating topologies (tree/subgroup/workgroup/matrix/split/
    /// multi-launch) do not preserve the reference order.
    pub fn is_reference_order(&self) -> bool {
        match self {
            ReductionTopology::SerialAxis { .. } => true,
            ReductionTopology::ParallelOuter { inner, .. } => inner.is_reference_order(),
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Symbolic rounding counts
// ---------------------------------------------------------------------------

/// Number of rounding steps a `Round` transfer introduces, kept symbolic so
/// loops, reductions, and traversals compose without losing the analytical
/// bound. `Unbounded` marks a count no closed form bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CountExpr {
    Const(u64),
    /// Steps along one serial path add.
    Add(Box<CountExpr>, Box<CountExpr>),
    /// A step count repeated by a traversal length multiplies.
    Mul(Box<CountExpr>, Box<CountExpr>),
    /// The number of elements visited along one extent (a runtime value).
    Elements(ExtentExpr),
    Unbounded,
}

impl CountExpr {
    pub fn one() -> CountExpr {
        CountExpr::Const(1)
    }

    /// Sum of two step counts with constant folding.
    pub fn add(a: CountExpr, b: CountExpr) -> CountExpr {
        match (a, b) {
            (CountExpr::Const(a), CountExpr::Const(b)) => match a.checked_add(b) {
                Some(sum) => CountExpr::Const(sum),
                None => CountExpr::Unbounded,
            },
            (a, b) => CountExpr::Add(Box::new(a), Box::new(b)),
        }
    }

    /// A step count repeated by a traversal length.
    pub fn mul(a: CountExpr, b: CountExpr) -> CountExpr {
        match (a, b) {
            (CountExpr::Const(a), CountExpr::Const(b)) => match a.checked_mul(b) {
                Some(product) => CountExpr::Const(product),
                None => CountExpr::Unbounded,
            },
            (a, b) => CountExpr::Mul(Box::new(a), Box::new(b)),
        }
    }

    /// Checked evaluation against concrete runtime extents. `None` means the
    /// count is not analytically bounded here (runtime extent unknown or
    /// overflow); the transfer then requires evidence.
    pub fn eval(&self, runtime: &dyn Fn(RuntimeExtentId) -> Option<u64>) -> Option<u64> {
        match self {
            CountExpr::Const(n) => Some(*n),
            CountExpr::Add(a, b) => a.eval(runtime)?.checked_add(b.eval(runtime)?),
            CountExpr::Mul(a, b) => a.eval(runtime)?.checked_mul(b.eval(runtime)?),
            CountExpr::Elements(extent) => extent.as_static().or_else(|| match extent {
                ExtentExpr::Runtime(id) => runtime(*id),
                _ => None,
            }),
            CountExpr::Unbounded => None,
        }
    }
}

/// Unit roundoff of one float dtype (half-ulp of 1.0). Integer dtypes never
/// round; a `Round` transfer never carries them.
pub fn unit_roundoff(dtype: DType) -> f64 {
    match dtype {
        DType::F32 => 2f64.powi(-24),
        DType::F16 => 2f64.powi(-11),
        DType::BF16 => 2f64.powi(-8),
        DType::I32 | DType::U32 | DType::Bool => 0.0,
    }
}

/// Identity of one exact capability signature: the intrinsic plus its concrete
/// argument types. Evidence and transfer attributions key on this.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CapabilitySignatureId {
    pub intrinsic: IntrinsicId,
    pub arguments: Vec<ValueType>,
}

impl CapabilitySignatureId {
    pub fn new(intrinsic: IntrinsicId, arguments: Vec<ValueType>) -> Self {
        Self {
            intrinsic,
            arguments,
        }
    }
}

// ---------------------------------------------------------------------------
// The transfer taxonomy
// ---------------------------------------------------------------------------

/// Deviation of one physical mapping from the registry reference numerics.
#[derive(Clone, Debug, PartialEq)]
pub enum NumericalTransfer {
    /// Bit-identical to the reference computation.
    Exact,
    /// Additional or differently placed roundings: bounded by
    /// `count` steps at `dtype`'s unit roundoff.
    Round { dtype: DType, count: CountExpr },
    /// The reduction visits elements in a different order than the reference
    /// ascending fold; the deviation is data-dependent.
    Reassociate {
        op: ReduceOp,
        topology: ReductionTopology,
    },
    /// A bounded deviation from the reference `operation`.
    Approximate {
        operation: PrimitiveId,
        bound: ErrorBound,
    },
    /// A capability-supplied value; `bound: None` is unqualified.
    Capability {
        signature: CapabilitySignatureId,
        bound: Option<ErrorBound>,
    },
    /// No analytical bound; only evidence keyed to the complete identity can
    /// qualify an alternative carrying this transfer.
    Unknown { reason: String },
}

impl NumericalTransfer {
    /// Convert a registry-level (authored capability) transfer into the
    /// strategy-level taxonomy. Registry `Approximate` effects are keyed by
    /// operation name only, so their bounds cannot attach to a `PrimitiveId`
    /// here; they convert to `Unknown` pending evidence.
    pub fn from_registry(
        effect: &seismic_lang::intrinsics::NumericalTransfer,
    ) -> NumericalTransfer {
        use seismic_lang::intrinsics::NumericalTransfer as Registry;
        match effect {
            Registry::Exact => NumericalTransfer::Exact,
            Registry::Round { dtype } => NumericalTransfer::Round {
                dtype: *dtype,
                count: CountExpr::one(),
            },
            Registry::Approximate { operation, bound } => NumericalTransfer::Unknown {
                reason: format!(
                    "authored approximate operation `{operation}` carries no primitive identity"
                ),
            }
            .with_bound_note(bound.as_ref()),
            Registry::Capability { signature, bound } => NumericalTransfer::Capability {
                signature: CapabilitySignatureId::new(signature.clone(), Vec::new()),
                bound: *bound,
            },
            Registry::Unknown { reason } => NumericalTransfer::Unknown {
                reason: reason.clone(),
            },
        }
    }

    fn with_bound_note(self, bound: Option<&ErrorBound>) -> Self {
        match (self, bound) {
            (NumericalTransfer::Unknown { reason }, Some(ErrorBound { relative, absolute })) => {
                NumericalTransfer::Unknown {
                    reason: format!(
                        "{reason} (retained bound: rel {relative:e}, abs {absolute:e})"
                    ),
                }
            }
            (other, _) => other,
        }
    }
}

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------

/// Compose two transfers along one data path: `outer` is applied to the result
/// of `inner`. The rules are conservative and analytical where a bound exists:
///
/// - `Exact` is the identity on both sides.
/// - `Unknown` absorbs everything, merging reasons.
/// - `Round` after `Round` adds step counts at the wider-error dtype.
/// - `Reassociate` subsumes countable rounding and same-operator re-folding;
///   different operators reassociated together are `Unknown`.
/// - `Approximate`/bounded `Capability` absorbs a `Round` by adding
///   `count * unit_roundoff(dtype)` to its bound when the count is closed-form;
///   otherwise the composition is `Unknown`.
/// - `Unbounded` counts and unqualified capabilities compose to `Unknown`.
pub fn compose(outer: &NumericalTransfer, inner: &NumericalTransfer) -> NumericalTransfer {
    use NumericalTransfer::*;
    match (outer, inner) {
        (Exact, inner) => inner.clone(),
        (outer, Exact) => outer.clone(),
        (Unknown { reason: r1 }, Unknown { reason: r2 }) => Unknown {
            reason: format!("{r1}; {r2}"),
        },
        (Unknown { reason }, _) => Unknown {
            reason: reason.clone(),
        },
        (_, Unknown { reason }) => Unknown {
            reason: reason.clone(),
        },
        // Reassociation subsumes countable rounding and same-op refolding.
        (Reassociate { op, topology }, Round { .. }) => Reassociate {
            op: *op,
            topology: topology.clone(),
        },
        (Reassociate { op, topology }, Reassociate { op: inner_op, .. }) => {
            if op == inner_op {
                Reassociate {
                    op: *op,
                    topology: topology.clone(),
                }
            } else {
                Unknown {
                    reason: "reassociated reductions of different operators composed".into(),
                }
            }
        }
        // A rounding count before a reassociated reduction is subsumed by it.
        (Round { .. }, Reassociate { op, topology }) => Reassociate {
            op: *op,
            topology: topology.clone(),
        },
        // Reassociation composed with approximation or a capability has no
        // analytical bound.
        (Reassociate { op, .. }, Approximate { operation, .. }) => Unknown {
            reason: format!(
                "reassociated `{}` composed with `{}`",
                op.name(),
                operation.name()
            ),
        },
        (Reassociate { op, .. }, Capability { signature, .. }) => Unknown {
            reason: format!(
                "reassociated `{}` composed with capability `{}`",
                op.name(),
                signature.intrinsic.path()
            ),
        },
        // A rounding count on top of a bounded approximation adds to its bound.
        (Round { dtype, count }, Approximate { operation, bound }) => {
            absorb_rounding_into_bound(operation.clone(), *bound, *dtype, count, "approximation")
        }
        (
            Round { dtype, count },
            Capability {
                signature,
                bound: Some(bound),
            },
        ) => match count.eval(&|_| None) {
            Some(steps) => Capability {
                signature: signature.clone(),
                bound: Some(ErrorBound {
                    relative: bound.relative + unit_roundoff(*dtype) * steps as f64,
                    absolute: bound.absolute,
                }),
            },
            None => Unknown {
                reason: "capability bound composed with a runtime-dependent rounding count".into(),
            },
        },
        (
            Round { .. },
            Capability {
                signature,
                bound: None,
            },
        ) => Unknown {
            reason: format!(
                "unqualified capability `{}` admits no analytical bound",
                signature.intrinsic.path()
            ),
        },
        (
            Round {
                dtype: d1,
                count: c1,
            },
            Round {
                dtype: d2,
                count: c2,
            },
        ) => {
            // The wider-error dtype governs the composed deviation.
            let dtype = if unit_roundoff(*d1) >= unit_roundoff(*d2) {
                *d1
            } else {
                *d2
            };
            Round {
                dtype,
                count: CountExpr::add(c1.clone(), c2.clone()),
            }
        }
        (Approximate { operation, bound }, Round { dtype, count }) => {
            absorb_rounding_into_bound(operation.clone(), *bound, *dtype, count, "approximation")
        }
        (
            Approximate {
                operation: o1,
                bound: b1,
            },
            Approximate {
                operation: o2,
                bound: b2,
            },
        ) => {
            if o1 == o2 {
                Approximate {
                    operation: o1.clone(),
                    bound: ErrorBound {
                        relative: b1.relative + b2.relative,
                        absolute: b1.absolute + b2.absolute,
                    },
                }
            } else {
                Unknown {
                    reason: "approximations of different operations composed".into(),
                }
            }
        }
        (Approximate { .. }, Reassociate { .. } | Capability { .. }) => Unknown {
            reason: "approximation composed with reassociation or an unqualified capability".into(),
        },
        (
            Capability {
                signature,
                bound: Some(bound),
            },
            Round { dtype, count },
        ) => {
            if let Some(steps) = count.eval(&|_| None) {
                Capability {
                    signature: signature.clone(),
                    bound: Some(ErrorBound {
                        relative: bound.relative + unit_roundoff(*dtype) * steps as f64,
                        absolute: bound.absolute,
                    }),
                }
            } else {
                Unknown {
                    reason: "capability bound composed with a runtime-dependent rounding count"
                        .into(),
                }
            }
        }
        (
            Capability {
                signature,
                bound: Some(_),
            },
            Approximate { .. },
        ) => Unknown {
            reason: format!(
                "capability `{}` composed with an approximation",
                signature.intrinsic.path()
            ),
        },
        (
            Capability {
                signature,
                bound: None,
            },
            Round { .. } | Reassociate { .. } | Approximate { .. } | Capability { .. },
        ) => Unknown {
            reason: format!(
                "unqualified capability `{}` admits no analytical bound",
                signature.intrinsic.path()
            ),
        },
        (
            Capability {
                signature,
                bound: Some(_),
            },
            Reassociate { .. },
        ) => Unknown {
            reason: format!(
                "capability `{}` composed with reassociation",
                signature.intrinsic.path()
            ),
        },
        (
            Capability { signature, .. },
            Capability {
                signature: inner_signature,
                ..
            },
        ) => Unknown {
            reason: format!(
                "capabilities `{}` and `{}` composed",
                signature.intrinsic.path(),
                inner_signature.intrinsic.path()
            ),
        },
    }
}

/// Absorb `count` roundings at `dtype` into a numeric bound; a count with no
/// closed form degrades the composition to `Unknown`.
fn absorb_rounding_into_bound(
    operation: PrimitiveId,
    bound: ErrorBound,
    dtype: DType,
    count: &CountExpr,
    what: &str,
) -> NumericalTransfer {
    match count.eval(&|_| None) {
        Some(steps) => NumericalTransfer::Approximate {
            operation,
            bound: ErrorBound {
                relative: bound.relative + unit_roundoff(dtype) * steps as f64,
                absolute: bound.absolute,
            },
        },
        None => NumericalTransfer::Unknown {
            reason: format!("{what} bound composed with a runtime-dependent rounding count"),
        },
    }
}

/// Compose a whole candidate: authored effects and every mapping transfer, in
/// application order.
pub fn compose_all(transfers: &[NumericalTransfer]) -> NumericalTransfer {
    transfers
        .iter()
        .fold(NumericalTransfer::Exact, |acc, next| compose(next, &acc))
}

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// Result of checking one transfer against a caller policy.
#[derive(Clone, Debug, PartialEq)]
pub enum PolicyDecision {
    Satisfied,
    /// The analytical bound is absent or looser than the policy; only evidence
    /// keyed to the complete identity can qualify the alternative.
    RequiresEvidence {
        reason: String,
    },
    /// No evidence can admit this transfer under the policy.
    Violated {
        reason: String,
    },
}

/// Check one transfer against a caller `PrecisionPolicy`. `runtime` resolves
/// runtime extents so symbolic rounding counts can be bounded analytically.
///
/// Strict (`Exact`) policies are satisfied only by `Exact` (or a proved
/// zero-bound equivalence); the ordered universal strategies — whose transfers
/// are `Exact` — therefore always remain available under strict policies.
pub fn satisfies_policy(
    transfer: &NumericalTransfer,
    policy: &PrecisionPolicy,
    runtime: &dyn Fn(RuntimeExtentId) -> Option<u64>,
) -> PolicyDecision {
    if matches!(policy, PrecisionPolicy::Unconstrained) {
        return PolicyDecision::Satisfied;
    }
    match transfer {
        NumericalTransfer::Exact => PolicyDecision::Satisfied,
        NumericalTransfer::Round { dtype, count } => match policy {
            PrecisionPolicy::Exact => PolicyDecision::Violated {
                reason: format!("strict policy rejects extra rounding at {}", dtype.name()),
            },
            PrecisionPolicy::Bounded { default, .. } => {
                let Some(steps) = count.eval(runtime) else {
                    return PolicyDecision::RequiresEvidence {
                        reason: "rounding count is not analytically bounded".into(),
                    };
                };
                let bound = unit_roundoff(*dtype) * steps as f64;
                if default.relative.get() >= bound {
                    PolicyDecision::Satisfied
                } else {
                    PolicyDecision::RequiresEvidence {
                        reason: format!(
                            "analytical rounding bound {bound:e} exceeds the default tolerance"
                        ),
                    }
                }
            }
            PrecisionPolicy::Unconstrained => unreachable!("handled above"),
        },
        NumericalTransfer::Reassociate { op, .. } => match policy {
            PrecisionPolicy::Exact => PolicyDecision::Violated {
                reason: format!("strict policy rejects reassociated `{}`", op.name()),
            },
            PrecisionPolicy::Bounded { .. } => PolicyDecision::RequiresEvidence {
                reason: format!(
                    "reassociated `{}` has a data-dependent deviation; evidence required",
                    op.name()
                ),
            },
            PrecisionPolicy::Unconstrained => unreachable!("handled above"),
        },
        NumericalTransfer::Approximate { operation, bound } => {
            let zero = bound.relative == 0.0 && bound.absolute == 0.0;
            match policy {
                PrecisionPolicy::Exact => {
                    if zero {
                        PolicyDecision::Satisfied
                    } else {
                        PolicyDecision::Violated {
                            reason: format!("strict policy rejects approximate `{operation}`"),
                        }
                    }
                }
                PrecisionPolicy::Bounded { default, .. } => {
                    if default.relative.get() >= bound.relative
                        && default.absolute.get() >= bound.absolute
                    {
                        PolicyDecision::Satisfied
                    } else {
                        PolicyDecision::RequiresEvidence {
                            reason: format!(
                                "approximate bound for `{operation}` exceeds the default tolerance"
                            ),
                        }
                    }
                }
                PrecisionPolicy::Unconstrained => unreachable!("handled above"),
            }
        }
        NumericalTransfer::Capability { signature, bound } => match bound {
            None => PolicyDecision::RequiresEvidence {
                reason: format!("capability `{}` is unqualified", signature.intrinsic.path()),
            },
            Some(bound) => {
                let zero = bound.relative == 0.0 && bound.absolute == 0.0;
                match policy {
                    PrecisionPolicy::Exact if !zero => PolicyDecision::Violated {
                        reason: format!(
                            "strict policy rejects capability `{}`",
                            signature.intrinsic.path()
                        ),
                    },
                    PrecisionPolicy::Bounded { default, .. }
                        if default.relative.get() < bound.relative
                            || default.absolute.get() < bound.absolute =>
                    {
                        PolicyDecision::RequiresEvidence {
                            reason: format!(
                                "capability bound for `{}` exceeds the default tolerance",
                                signature.intrinsic.path()
                            ),
                        }
                    }
                    _ => PolicyDecision::Satisfied,
                }
            }
        },
        NumericalTransfer::Unknown { reason } => PolicyDecision::RequiresEvidence {
            reason: reason.clone(),
        },
    }
}

// ---------------------------------------------------------------------------
// Evidence
// ---------------------------------------------------------------------------

/// Toolchain identity participating in evidence keys.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ToolchainId(pub String);

/// Fingerprint of the workload (entry, shapes, elements) the evidence covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkloadFingerprint(pub [u8; 32]);

/// Fingerprint of the complete plan assignment the evidence covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AssignmentFingerprint(pub [u8; 32]);

/// The complete identity an evidence record is keyed to: logical program,
/// workload, target, toolchain, assignment, and the precision policy it was
/// validated against. Evidence is never widened to another identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EvidenceKey {
    pub logical: LogicalIdentity,
    pub workload: WorkloadFingerprint,
    pub target: EffectiveTargetIdentity,
    pub toolchain: ToolchainId,
    pub assignment: AssignmentFingerprint,
    pub precision: PrecisionPolicy,
}

impl EvidenceKey {
    pub fn new(
        logical: LogicalIdentity,
        workload: WorkloadFingerprint,
        target: EffectiveTargetIdentity,
        toolchain: ToolchainId,
        assignment: AssignmentFingerprint,
        precision: PrecisionPolicy,
    ) -> Self {
        Self {
            logical,
            workload,
            target,
            toolchain,
            assignment,
            precision,
        }
    }
}

/// One accepted numerical evidence record: a qualification measured under
/// exactly its key.
#[derive(Clone, Debug, PartialEq)]
pub struct NumericalEvidence {
    pub key: EvidenceKey,
    pub assessment: NumericalAssessment,
}

impl NumericalEvidence {
    /// Whether this record qualifies a candidate with exactly this identity
    /// under a policy at least as strict as the one it was validated against.
    pub fn qualifies(&self, key: &EvidenceKey) -> bool {
        &self.key == key && self.assessment.satisfies(&key.precision)
    }
}

/// Whether a bounded policy is willing to trust qualification records at all.
pub fn accepts_evidence(policy: &PrecisionPolicy) -> bool {
    match policy {
        PrecisionPolicy::Bounded { evidence, .. } => *evidence == EvidenceRequirement::Qualified,
        _ => false,
    }
}

/// Default tolerance of a bounded policy (the whole-candidate envelope when no
/// per-output override applies).
pub fn default_tolerance(policy: &PrecisionPolicy) -> Option<Tolerance> {
    match policy {
        PrecisionPolicy::Bounded { default, .. } => Some(*default),
        _ => None,
    }
}
