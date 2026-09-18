//! Invocation storage bindings and finite derivation budgets shared by backend models.
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Allocation {
    pub id: u64,
    pub bytes: u64,
    pub alignment: u64,
    /// Bytes constrained by this workload. Other external bytes are unknown, not
    /// zero. Only values needed to determine control or addressing must be known.
    pub known_bytes: BTreeMap<u64, u8>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferBinding {
    pub allocation: u64,
    pub offset: u64,
    pub bytes: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarWorkload {
    pub identity: String,
    pub allocations: Vec<Allocation>,
    /// Ordered exactly as ScalarProgram::buffers. Equal allocation IDs denote
    /// actual shared storage even when parameter names differ.
    pub buffers: Vec<BufferBinding>,
    /// The shared eight-byte-slot scalar ABI, checked against its typed schema.
    pub scalars: Vec<u8>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DerivationLimits {
    pub instructions: u64,
    pub operations: usize,
}

/// A construction limit does not constrain the execution's legal domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerivationLimit {
    Instructions(u64),
    Operations(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DerivationError {
    Exhausted(DerivationLimit),
    Analysis(String),
}
impl std::fmt::Display for DerivationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted(DerivationLimit::Instructions(limit)) => {
                write!(f, "model derivation instruction budget exhausted ({limit})")
            }
            Self::Exhausted(DerivationLimit::Operations(limit)) => {
                write!(f, "model derivation operation budget exhausted ({limit})")
            }
            Self::Analysis(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for DerivationError {}
impl From<String> for DerivationError {
    fn from(message: String) -> Self {
        Self::Analysis(message)
    }
}
impl From<&str> for DerivationError {
    fn from(message: &str) -> Self {
        Self::Analysis(message.into())
    }
}
impl From<DerivationError> for String {
    fn from(error: DerivationError) -> Self {
        error.to_string()
    }
}
