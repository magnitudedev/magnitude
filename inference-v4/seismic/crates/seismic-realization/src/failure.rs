//! The closed failure taxonomy shared by the compiler and the runtime
//! (package A1 owns; C0 freezes the classes).
//!
//! Every compiler or runtime error site belongs to exactly one class. There
//! is no miscellaneous runtime compiler error, and no string channel through
//! which compiler-owned structure can be reclassified as an invocation or
//! system error: no failure type in this module implements `From<String>`
//! or `From<&str>`, and every variant that carries text also carries the
//! typed identity (slot, field, relation, value) it describes.
//!
//! | class                | owner phase                      | production form                   |
//! |----------------------|----------------------------------|-----------------------------------|
//! | SourceDiagnostic     | checking / logical construction  | `CompileFailure::InvalidSemanticProgram` |
//! | Applicability        | logical construction / W1 domains| `CompileFailure::NoApplicableImplementation` |
//! | PlanningInfeasible   | solver                           | `CompileFailure::PlanningInfeasible` |
//! | CompilerDefect       | any construction phase           | `CompileFailure::CompilerBug`     |
//! | Toolchain            | native assembly                  | `CompileFailure::ToolchainFailure`|
//! | ExternalSystem       | assembly / execution             | `CompileFailure::SystemFailure` / `ExecutionFailure::External` |
//! | InvalidInvocation    | `CompiledPlan::prepare`          | `ExecutionFailure::Invocation`    |
//! | SafetyViolation      | retained guard at execution      | `ExecutionFailure::Safety`        |
//!
//! Every failure type of this module answers `class()`; `CompileFailure`
//! (owner P1, `seismic-compiler`) classifies its own six production variants.

use crate::ids::{
    BufferSlot, GuardIx, InvocationValueId, ObligationRef, ScalarSlot, StatusFieldIx,
};
use crate::invocation::InvocationPredicate;
use seismic_lang::logical::specialization::{ShapeDomain, ShapeFieldId};
use std::fmt;

/// The eight production failure classes. Used by the C0 ledger to classify
/// every error site; a site that fits none is a defect in the ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FailureClass {
    SourceDiagnostic,
    Applicability,
    PlanningInfeasible,
    CompilerDefect,
    InvalidInvocation,
    SafetyViolation,
    Toolchain,
    ExternalSystem,
}

/// The work package whose construction invariant a defect contradicts. A
/// defect is assigned to its owner, never patched downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Package {
    L1,
    W1,
    A1,
    O1,
    S1,
    D1,
    K1,
    M1,
    P1,
    B1Cpu,
    B1Metal,
    B1Cuda,
    N1,
    R1,
    E1,
}

/// A contradicted compiler invariant. Constructed only inside compiler
/// crates; reported as `CompileFailure::CompilerBug`; never retried and never
/// caught downstream to continue with another candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompilerDefect {
    pub package: Package,
    pub invariant: String,
}

impl CompilerDefect {
    pub fn new(package: Package, invariant: impl Into<String>) -> CompilerDefect {
        CompilerDefect {
            package,
            invariant: invariant.into(),
        }
    }

    pub fn class(&self) -> FailureClass {
        FailureClass::CompilerDefect
    }
}

impl fmt::Display for CompilerDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "compiler defect ({:?}): {}", self.package, self.invariant)
    }
}

impl std::error::Error for CompilerDefect {}

/// An invocation rejected before submission for facts supplied by the
/// invocation. Structural compiler facts never appear here.
///
/// The arithmetic variants name the contract element that failed:
/// `ScalarOutsideDomain` is the retained static interval of one ABI scalar,
/// `IndexOutOfBounds` is one `ScalarDomain::Index` bound, `Predicate` is one
/// relational predicate of the invocation contract, and `ArithmeticDomain`
/// is one derived value whose `DomainProof::Predicate` relation is false.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InvalidInvocation {
    MissingBuffer { slot: BufferSlot, path: String, plane: String },
    MissingScalar { slot: ScalarSlot, name: String },
    MissingShape { field: ShapeFieldId, name: String },
    BufferDevice { slot: BufferSlot },
    BufferBytes { slot: BufferSlot, required: u64, actual: u64 },
    BufferAlignment { slot: BufferSlot, required: u64, actual: u64 },
    /// The supplied scalar has no value in its ABI representation (not an
    /// integer, negative where unsigned, not finite, ...).
    ScalarRepresentation { slot: ScalarSlot, reason: String },
    /// The supplied scalar lies outside the retained static interval
    /// `[min, max]` of the compiled envelope.
    ScalarOutsideDomain { slot: ScalarSlot, value: u64, min: u64, max: u64 },
    /// The supplied index scalar is not below its invocation-known bound.
    IndexOutOfBounds { slot: ScalarSlot, value: u64, bound: u64 },
    ShapeOutsideDomain { field: ShapeFieldId, value: u64, domain: ShapeDomain },
    Alias { left: BufferSlot, right: BufferSlot },
    /// The relational predicate of the invocation contract is false; the
    /// retained predicate names its operands.
    Predicate { predicate: InvocationPredicate, description: String },
    /// The partial arithmetic operation producing derived `value` has no
    /// defined result for these inputs (division by zero, subtraction below
    /// zero); its `DomainProof::Predicate` relation is false.
    ArithmeticDomain { value: InvocationValueId, description: String },
}

impl InvalidInvocation {
    pub fn class(&self) -> FailureClass {
        FailureClass::InvalidInvocation
    }
}

impl fmt::Display for InvalidInvocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBuffer { path, plane, .. } => {
                write!(f, "unbound tensor `{path}` plane `{plane}`")
            }
            Self::MissingScalar { name, .. } => write!(f, "unbound scalar `{name}`"),
            Self::MissingShape { name, .. } => write!(f, "unbound shape field `{name}`"),
            Self::BufferDevice { slot } => {
                write!(f, "buffer slot {} belongs to another device", slot.index())
            }
            Self::BufferBytes {
                slot,
                required,
                actual,
            } => write!(
                f,
                "buffer slot {} requires {required} bytes, {actual} supplied",
                slot.index()
            ),
            Self::BufferAlignment {
                slot,
                required,
                actual,
            } => write!(
                f,
                "buffer slot {} requires alignment {required}, offset alignment {actual}",
                slot.index()
            ),
            Self::ScalarRepresentation { slot, reason } => {
                write!(f, "scalar slot {} is not representable: {reason}", slot.index())
            }
            Self::ScalarOutsideDomain {
                slot,
                value,
                min,
                max,
            } => write!(
                f,
                "scalar slot {} value {value} is outside the compiled interval [{min}, {max}]",
                slot.index()
            ),
            Self::IndexOutOfBounds { slot, value, bound } => write!(
                f,
                "index scalar slot {} value {value} is not below its bound {bound}",
                slot.index()
            ),
            Self::ShapeOutsideDomain {
                field,
                value,
                domain,
            } => write!(
                f,
                "shape field {} value {value} is outside the compiled domain {domain:?}",
                field.0
            ),
            Self::Alias { left, right } => write!(
                f,
                "buffers {} and {} overlap but must be disjoint",
                left.index(),
                right.index()
            ),
            Self::Predicate { description, .. } => {
                write!(f, "invocation predicate failed: {description}")
            }
            Self::ArithmeticDomain { value, description } => write!(
                f,
                "invocation value {} is undefined: {description}",
                value.0
            ),
        }
    }
}

impl std::error::Error for InvalidInvocation {}

/// The kind of retained data-dependent safety obligation that failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SafetyKind {
    ExtentPositive,
    IndexInBounds,
    RangeInBounds,
    DivisorNonZero,
    SignedDivisionNoOverflow,
    ShiftInRange,
    ShapeProductFits,
    /// The guard dominating one result-field read failed: the value-dependent
    /// use of that result block field is not established.
    ResultFieldUsable,
}

/// The site at which a retained data-dependent safety obligation failed:
/// a schedule-level `Guard` step (`GuardIx`) or a kernel-`Check` status
/// field (`StatusFieldIx`, per-block, guard order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SafetyViolationSource {
    Guard(GuardIx),
    KernelCheck(StatusFieldIx),
}

/// A retained data-dependent safety obligation failed during execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetyViolation {
    pub source: SafetyViolationSource,
    pub obligation: ObligationRef,
    pub kind: SafetyKind,
}

impl SafetyViolation {
    pub fn class(&self) -> FailureClass {
        FailureClass::SafetyViolation
    }
}

impl fmt::Display for SafetyViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let site = match self.source {
            SafetyViolationSource::Guard(guard) => format!("guard {}", guard.index()),
            SafetyViolationSource::KernelCheck(field) => {
                format!("kernel check status field {}", field.index())
            }
        };
        write!(
            f,
            "retained safety obligation {:?} failed at {site}",
            self.kind
        )
    }
}

impl std::error::Error for SafetyViolation {}

/// Which external system failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalStage {
    Allocation,
    Driver,
    Submission,
    DeviceLost,
}

/// The device, driver, allocation, or submission system failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalFailure {
    pub stage: ExternalStage,
    pub detail: String,
}

impl ExternalFailure {
    pub fn class(&self) -> FailureClass {
        FailureClass::ExternalSystem
    }
}

impl fmt::Display for ExternalFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "external {:?} failure: {}", self.stage, self.detail)
    }
}

impl std::error::Error for ExternalFailure {}

/// Every execution `Result` is one of these three. Missing compiler structure
/// is not representable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionFailure {
    Invocation(InvalidInvocation),
    Safety(SafetyViolation),
    External(ExternalFailure),
}

impl ExecutionFailure {
    pub fn class(&self) -> FailureClass {
        match self {
            Self::Invocation(failure) => failure.class(),
            Self::Safety(failure) => failure.class(),
            Self::External(failure) => failure.class(),
        }
    }
}

impl fmt::Display for ExecutionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invocation(failure) => write!(f, "invalid invocation: {failure}"),
            Self::Safety(failure) => write!(f, "safety violation: {failure}"),
            Self::External(failure) => write!(f, "{failure}"),
        }
    }
}

impl std::error::Error for ExecutionFailure {}

impl From<InvalidInvocation> for ExecutionFailure {
    fn from(failure: InvalidInvocation) -> Self {
        Self::Invocation(failure)
    }
}

impl From<SafetyViolation> for ExecutionFailure {
    fn from(failure: SafetyViolation) -> Self {
        Self::Safety(failure)
    }
}

impl From<ExternalFailure> for ExecutionFailure {
    fn from(failure: ExternalFailure) -> Self {
        Self::External(failure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::DenseIndex;

    #[test]
    fn every_execution_failure_reports_its_class() {
        let invocation = ExecutionFailure::Invocation(InvalidInvocation::IndexOutOfBounds {
            slot: ScalarSlot::from_index(0),
            value: 4,
            bound: 4,
        });
        assert_eq!(invocation.class(), FailureClass::InvalidInvocation);

        let external = ExecutionFailure::External(ExternalFailure {
            stage: ExternalStage::Allocation,
            detail: "arena exhausted".to_string(),
        });
        assert_eq!(external.class(), FailureClass::ExternalSystem);

        assert_eq!(
            CompilerDefect::new(Package::A1, "interval proof absent").class(),
            FailureClass::CompilerDefect
        );
    }
}
