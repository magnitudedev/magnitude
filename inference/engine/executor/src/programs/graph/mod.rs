//! Graph construction shared by every operator: the draft abstraction over
//! prepared and checked entries, the readout, and the draft taps. Each
//! operator's own block fragment lives with the operator
//! (`operators::<op>::graph`).

pub(crate) mod draft;
pub(crate) mod readout;
pub(crate) mod tap;

use crate::operators::{attention, gated_delta, routed};
use seismic::{KernelDomainViolation, NativeGraphMetadataError};
use std::fmt;

/// Why a graph could not be built, certified or sealed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    /// A kernel call at statics outside its kernel's domain on this backend:
    /// the model's shapes are beyond what the engine's kernels execute.
    KernelDomain(KernelDomainViolation),
    /// Any other failure: an engine defect.
    Invalid(String),
}

impl GraphError {
    /// The error within `context`. A domain violation names its own call.
    pub(crate) fn context(self, context: impl fmt::Display) -> Self {
        match self {
            Self::KernelDomain(violation) => Self::KernelDomain(violation),
            Self::Invalid(message) => Self::Invalid(format!("{context}: {message}")),
        }
    }
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelDomain(violation) => violation.fmt(formatter),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for GraphError {}

impl From<NativeGraphMetadataError> for GraphError {
    fn from(error: NativeGraphMetadataError) -> Self {
        match error {
            NativeGraphMetadataError::Inadmissible(violation) => Self::KernelDomain(violation),
            error => Self::Invalid(error.to_string()),
        }
    }
}

impl From<String> for GraphError {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

impl From<&str> for GraphError {
    fn from(message: &str) -> Self {
        Self::Invalid(message.to_owned())
    }
}

/// The structural form a row class selects in every row-dependent branch of
/// graph construction. The topology code branches on these same predicates,
/// so classes of equal form share node order, edges and exports, and one
/// certified layout can serve all of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct RowForm {
    attention_decode: bool,
    recurrent_chunked: bool,
    routed_decode: bool,
}

impl RowForm {
    pub(crate) fn of(rows: u64) -> Self {
        Self {
            attention_decode: attention::graph::decodes(rows),
            recurrent_chunked: gated_delta::graph::chunked(rows),
            routed_decode: routed::fused_graph::decodes(rows),
        }
    }
}
