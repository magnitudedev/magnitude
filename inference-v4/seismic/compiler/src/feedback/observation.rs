//! Runtime implements this contract; the compiler never imports runtime.
use crate::executable::ExecutableVariant;
use crate::prepared::InvocationContract;
use seismic_lang::expr::{compiled::InvocationValues, SymbolValue};
use seismic_lang::ids::RepresentationId;
use seismic_lang::precision::PrecisionPolicy;
use seismic_target::TargetFamily;
use std::time::Duration;

#[derive(Clone, Debug)]
pub enum CaseArgument {
    Tensor {
        representation: RepresentationId,
        extents: Vec<u64>,
    },
    Scalar(SymbolValue),
    Index(seismic_lang::expr::BigUint),
    Range {
        start: seismic_lang::expr::BigUint,
        end: seismic_lang::expr::BigUint,
    },
}
#[derive(Clone, Debug)]
pub struct ObservationCase {
    pub values: InvocationValues,
    pub arguments: Vec<CaseArgument>,
    pub seed: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct ObservationProtocol {
    pub warmup: usize,
    pub trials: usize,
}
impl ObservationProtocol {
    pub const SCREEN: Self = Self {
        warmup: 1,
        trials: 3,
    };
}

pub struct ObservationRequest<'a, T: TargetFamily, H> {
    /// Exact preparation candidate whose retained native handles are observed.
    pub candidate: crate::evaluation::PreparedCandidateId,
    pub reference: seismic_lang::entry::LogicalEntryView<'a>,
    pub executable: &'a ExecutableVariant<T, H>,
    pub precision: &'a PrecisionPolicy,
    pub case: &'a ObservationCase,
    pub protocol: ObservationProtocol,
    pub memory_limit: u64,
    pub reference_work_limit: u64,
    pub deadline: Option<std::time::Instant>,
}

impl<T: TargetFamily, H> ObservationRequest<'_, T, H> {
    pub fn invocation(&self) -> &InvocationContract {
        self.executable.invocation_contract()
    }
}

#[derive(Clone, Debug)]
pub struct Observation {
    pub support: super::ObservationSupport,
    pub samples: Vec<Duration>,
    pub setup_time: Duration,
    pub checking_time: Duration,
    /// Versioned observer environment identity, distinct from code compatibility.
    pub environment: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObservationError {
    Unsupported(String),
    Capacity {
        required: seismic_lang::expr::BigUint,
        limit: u64,
    },
    InvalidCase(String),
    ReferenceLimit {
        work_limit: u64,
    },
    DeadlineExpired,
    /// A completed execution contradicted its checked observation contract.
    Contract(String),
    IncorrectResult(String),
    Execution(crate::errors::ExecutionError),
}

pub trait ControlledObserver<T: TargetFamily, H> {
    /// Revalidated at each explicit preparation call, independently of code compatibility.
    fn environment(&mut self) -> Result<[u8; 32], ObservationError>;
    /// Diagnostic execution of an already applicable executable. The result has
    /// no path back into candidate applicability or selection.
    fn validate(
        &mut self,
        _request: ObservationRequest<'_, T, H>,
        _case: crate::numerics::ValidationCase,
    ) -> Result<crate::numerics::ValidationObservation, ObservationError> {
        Err(ObservationError::Unsupported(
            "diagnostic observation unavailable".into(),
        ))
    }
    fn observe(
        &mut self,
        request: ObservationRequest<'_, T, H>,
    ) -> Result<Observation, ObservationError>;
}
