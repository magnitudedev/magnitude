use std::fmt;

/// A malformed immutable target description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetDescriptionError {
    BackendIdentityMismatch,
    EmptyLimit(&'static str),
    InvalidAlignment(&'static str),
    DuplicateResourceName(&'static str),
}

impl fmt::Display for TargetDescriptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BackendIdentityMismatch => {
                f.write_str("target identity names a different backend family")
            }
            Self::EmptyLimit(name) => write!(f, "target limit `{name}` is zero"),
            Self::InvalidAlignment(name) => {
                write!(f, "target alignment `{name}` is not a nonzero power of two")
            }
            Self::DuplicateResourceName(name) => {
                write!(f, "target addressable resource `{name}` is declared twice")
            }
        }
    }
}

impl std::error::Error for TargetDescriptionError {}

/// Failures a fully described kernel may meet in the native toolchain.
///
/// Unsupported semantics and target-limit violations are intentionally absent:
/// those are description, construction, or reconciliation defects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeCompilationError {
    ToolchainFailure(String),
    MalformedToolchainOutput(String),
    DeviceLost(String),
    CacheFailure(String),
    ToolchainResourceExhausted(String),
}

impl fmt::Display for NativeCompilationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ToolchainFailure(message) => write!(f, "toolchain failure: {message}"),
            Self::MalformedToolchainOutput(message) => {
                write!(f, "malformed toolchain output: {message}")
            }
            Self::DeviceLost(message) => write!(f, "device lost: {message}"),
            Self::CacheFailure(message) => write!(f, "native cache failure: {message}"),
            Self::ToolchainResourceExhausted(message) => {
                write!(f, "toolchain resources exhausted: {message}")
            }
        }
    }
}

impl std::error::Error for NativeCompilationError {}
