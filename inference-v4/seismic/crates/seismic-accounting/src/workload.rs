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
