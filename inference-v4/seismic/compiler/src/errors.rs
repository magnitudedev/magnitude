//! The complete real-world error taxonomy (spec §13.2).
//!
//! Every variant names a condition expected from source, invocation, devices,
//! toolchains, capacity, infeasibility, or data-dependent safety. No variant
//! means "two compiler phases disagreed"; that is a panic under §13.3.
//! Adding a variant requires demonstrating a real external condition.

use seismic_lang::checked::SourceError;
use std::fmt;

pub use seismic_lang::bundle::CheckedBundleError;
use seismic_target::NativeCompilationError;

/// Target discovery/opening failures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetError {
    UnsupportedDevice(String),
    UnsupportedDriver(String),
    UnsupportedToolchain(String),
    DeviceUnavailable(String),
    /// Device/profile evidence was acquired, but could not certify the
    /// analytical contract. The immutable device description remains valid.
    InvalidExecutionProfile(String),
}

/// Report of solver resource exhaustion during mandatory feasibility or
/// coverage work (§5.4, §10.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolverBudgetReport {
    pub phase: SolverPhase,
    pub work_units: u64,
    pub elapsed_ms: u64,
    pub memory_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverPhase {
    Feasibility,
    Coverage,
}

/// Preparation failures (`for_device`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreparationError {
    /// No registered implementation applies to this entry on this target.
    NoApplicableImplementation(NoApplicableReport),
    /// Implementations exist, but none satisfies the precision policy over
    /// some part of the target domain.
    NumericalPolicyInfeasible(NumericalInfeasibleReport),
    /// The entry domain cannot be represented on this target (address or
    /// integer range).
    TargetDomainUnrepresentable(String),
    SolverResourceExhausted(SolverBudgetReport),
    NativeCompilation(NativeCompilationError),
    /// A direct native entry could not reserve its fixed invocation storage.
    NativeWorkspaceAllocation(String),
    /// A direct native specialization does not match its declaration, or the
    /// entry has no native implementation for the device's backend.
    NativeSpecialization(String),
    /// A direct native implementation binds more buffers than the backend's
    /// argument table holds (Metal: 31).
    NativeBufferSlots {
        entry: String,
        slots: usize,
        limit: usize,
    },
    /// The device's backend is native-only (Vulkan): it has no planned
    /// route, so planned preparation and feedback are refused.
    PlannedRouteUnavailable {
        backend: seismic_lang::registry::BackendName,
    },
    /// The selected evaluator could not consume this sealed domain. No
    /// partial evaluated domain exists.
    Evaluation(crate::evaluation::EvaluationError),
    Feedback(crate::feedback::FeedbackError),
    /// A supposedly sealed candidate set violated a structural identity
    /// invariant and therefore was not published.
    InvalidCandidateDomain(String),
    Planning(crate::planning::PlanningError),
    /// The checked reference implementation could not establish total
    /// post-native legality and duration coverage over the target domain.
    UniversalClosure(String),
    /// Mandatory source construction awaits an implemented semantic/storage requirement.
    UnfinishedConstruction(crate::candidate_domain::ConstructionPending),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoApplicableReport {
    pub entry: String,
    /// Selected implementation and its established incompatibility.
    pub declined: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NumericalInfeasibleReport {
    pub entry: String,
    pub reasons: Vec<String>,
}

/// Failures a valid frozen plan can still meet in native compilation
/// (§11.4). "Unsupported feature", "too much local memory", "invalid
/// geometry", "missing binding" and "wrong storage kind" are not permitted
/// here: their possibility means target profiling or closed construction is
/// incomplete.
/// Invocation rejections at the generated call boundary (§12.2). All occur
/// before allocation or launch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InvocationError {
    WrongDevice {
        parameter: String,
    },
    WrongRepresentation {
        parameter: String,
    },
    ShapeMismatch {
        parameter: String,
        axis: u32,
    },
    InvalidTensorDescriptor {
        parameter: String,
    },
    ScalarOutOfDomain {
        parameter: String,
    },
    IllegalAliasing {
        first: String,
        second: String,
    },
    /// The invocation is outside the prepared target domain.
    OutsideTargetDomain,
    AllocationCapacity {
        required: seismic_lang::expr::BigUint,
        available: u64,
    },
    /// A dimension fixed when a native implementation was prepared has a
    /// different value in this invocation.
    StaticDimension {
        dimension: String,
        expected: u64,
        actual: seismic_lang::expr::BigUint,
    },
    /// A tensor parameter whose extents are all static renders constant
    /// canonical strides into its native source, so it must be bound with
    /// exactly those strides.
    NoncanonicalStaticTensor {
        parameter: String,
    },
}

/// Execution failures after a valid invocation was accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionError {
    DataCheckFailed(CheckFailure),
    AllocationFailed(String),
    /// Resource refusal after the invocation's source prefix may have executed.
    AllocationCapacity { required: seismic_lang::expr::BigUint, available: u64 },
    /// A closed executable violated a compiler-owned construction premise.
    ConstructionContradiction(String),
    SubmissionFailed(String),
    SynchronizationFailed(String),
    DeviceLost(String),
    /// Issuing the schedule failed after earlier work was accepted, and
    /// reaching that work's terminal boundary independently failed as well.
    /// Both failures are retained because either one may determine recovery.
    IssueAndCompletionFailed {
        issue: Box<ExecutionError>,
        completion: Box<ExecutionError>,
    },
}

/// A checked source operation that stopped execution, with display location.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckFailure {
    pub failure: seismic_lang::failure::SourceFailure,
    /// Source location of the check.
    pub path: String,
    pub line: u32,
}

/// The one public compile-time error of a consumer build.
#[derive(Debug)]
pub enum BuildError {
    Source(SourceError),
    Io(std::io::Error),
}

impl fmt::Display for TargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDevice(s) => write!(f, "unsupported device: {s}"),
            Self::UnsupportedDriver(s) => write!(f, "unsupported driver: {s}"),
            Self::UnsupportedToolchain(s) => write!(f, "unsupported toolchain: {s}"),
            Self::DeviceUnavailable(s) => write!(f, "device unavailable: {s}"),
            Self::InvalidExecutionProfile(s) => write!(f, "invalid execution profile: {s}"),
        }
    }
}
impl std::error::Error for TargetError {}

impl fmt::Display for PreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoApplicableImplementation(r) => {
                write!(f, "no applicable implementation for `{}`", r.entry)?;
                for (implementation, reason) in &r.declined {
                    write!(f, "; {implementation}: {reason}")?;
                }
                Ok(())
            }
            Self::NumericalPolicyInfeasible(r) => {
                write!(
                    f,
                    "numerical policy infeasible for `{}`: {}",
                    r.entry,
                    r.reasons.join("; ")
                )
            }
            Self::TargetDomainUnrepresentable(s) => write!(f, "target domain unrepresentable: {s}"),
            Self::SolverResourceExhausted(r) => write!(
                f,
                "solver resources exhausted during {:?} after {} work units",
                r.phase, r.work_units
            ),
            Self::NativeCompilation(e) => write!(f, "native compilation failed: {e}"),
            Self::NativeWorkspaceAllocation(s) => write!(f, "native invocation workspace: {s}"),
            Self::NativeBufferSlots { entry, slots, limit } => write!(
                f,
                "native implementation of `{entry}` binds {slots} buffers; the backend admits {limit}"
            ),
            Self::NativeSpecialization(s) => write!(f, "native specialization: {s}"),
            Self::PlannedRouteUnavailable { backend } => write!(
                f,
                "`{}` is a native-only backend: planned preparation is unavailable",
                backend.as_str()
            ),
            Self::Evaluation(e) => write!(f, "candidate evaluation failed: {e:?}"),
            Self::Feedback(e) => write!(f, "feedback evaluation failed: {e:?}"),
            Self::InvalidCandidateDomain(s) => write!(f, "invalid candidate domain: {s}"),
            Self::Planning(e) => write!(f, "planning failed: {e}"),
            Self::UnfinishedConstruction(reason) => write!(f, "source reservation construction is unfinished: {reason:?}"),
            Self::UniversalClosure(s) => write!(f, "universal implementation is not total: {s}"),
        }
    }
}
impl std::error::Error for PreparationError {}

impl fmt::Display for InvocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongDevice { parameter } => {
                write!(f, "parameter `{parameter}` is on the wrong device")
            }
            Self::WrongRepresentation { parameter } => {
                write!(
                    f,
                    "parameter `{parameter}` has the wrong element representation"
                )
            }
            Self::ShapeMismatch { parameter, axis } => {
                write!(
                    f,
                    "parameter `{parameter}` axis {axis} does not satisfy the call schema"
                )
            }
            Self::InvalidTensorDescriptor { parameter } => {
                write!(f, "parameter `{parameter}` has an invalid tensor descriptor")
            }
            Self::ScalarOutOfDomain { parameter } => {
                write!(f, "scalar `{parameter}` is outside its domain")
            }
            Self::IllegalAliasing { first, second } => {
                write!(f, "parameters `{first}` and `{second}` alias illegally")
            }
            Self::OutsideTargetDomain => {
                f.write_str("invocation is outside the prepared target domain")
            }
            Self::AllocationCapacity {
                required,
                available,
            } => {
                write!(
                    f,
                    "allocation needs {required} bytes but {available} are available"
                )
            }
            Self::StaticDimension {
                dimension,
                expected,
                actual,
            } => write!(
                f,
                "dimension `{dimension}` is {actual}, but the native implementation was prepared for {expected}"
            ),
            Self::NoncanonicalStaticTensor { parameter } => write!(
                f,
                "`{parameter}` has static extents, so the native implementation requires its canonical strides"
            ),
        }
    }
}
impl std::error::Error for InvocationError {}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataCheckFailed(c) => {
                write!(f, "check failed at {}:{}: {}", c.path, c.line, c.failure)
            }
            Self::AllocationFailed(s) => write!(f, "allocation failed: {s}"),
            Self::AllocationCapacity { required, available } => write!(f, "allocation requires {required} bytes; {available} bytes available"),
            Self::ConstructionContradiction(detail) => write!(f, "closed executable construction contradiction: {detail}"),
            Self::SubmissionFailed(s) => write!(f, "submission failed: {s}"),
            Self::SynchronizationFailed(s) => write!(f, "synchronization failed: {s}"),
            Self::DeviceLost(s) => write!(f, "device lost: {s}"),
            Self::IssueAndCompletionFailed { issue, completion } => write!(
                f,
                "execution issue failed ({issue}); terminal completion also failed ({completion})"
            ),
        }
    }
}
impl std::error::Error for ExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IssueAndCompletionFailed { issue, .. } => Some(issue.as_ref()),
            _ => None,
        }
    }
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}
impl std::error::Error for BuildError {}
