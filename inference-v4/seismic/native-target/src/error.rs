use std::fmt;
use std::path::PathBuf;

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
    /// The installation's native toolchain cannot be used on this host.
    ToolchainUnavailable(ToolchainUnavailable),
    /// The device's architecture is not one the installed toolchain targets.
    UnsupportedArchitecture {
        architecture: String,
        supported: Vec<String>,
    },
    ToolchainFailure(String),
    MalformedToolchainOutput(String),
    DeviceLost(String),
    CacheFailure(String),
    ToolchainResourceExhausted(String),
}

/// Why an owned native toolchain library cannot be used. `library` is the
/// file name the installation layout defines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolchainUnavailable {
    /// The toolchain's directory could not be determined.
    Unlocated { reason: String },
    /// The library is absent from the resolved directory.
    Missing { library: String, directory: PathBuf },
    /// The library is present but cannot be loaded or lacks an entry point.
    Unusable {
        library: String,
        path: PathBuf,
        reason: String,
    },
    /// The library is present with a version this build cannot use.
    IncompatibleVersion {
        library: String,
        path: PathBuf,
        found: String,
        required: String,
    },
}

impl fmt::Display for ToolchainUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unlocated { reason } => {
                write!(f, "the toolchain directory cannot be determined: {reason}")
            }
            Self::Missing { library, directory } => {
                write!(f, "{library} is not present in {}", directory.display())
            }
            Self::Unusable {
                library,
                path,
                reason,
            } => write!(f, "{library} at {} is unusable: {reason}", path.display()),
            Self::IncompatibleVersion {
                library,
                path,
                found,
                required,
            } => write!(
                f,
                "{library} at {} is version {found}, but this build requires {required}",
                path.display()
            ),
        }
    }
}

impl fmt::Display for NativeCompilationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ToolchainUnavailable(reason) => write!(f, "toolchain unavailable: {reason}"),
            Self::UnsupportedArchitecture {
                architecture,
                supported,
            } => write!(
                f,
                "architecture {architecture} is not supported by the installed toolchain (supported: {})",
                supported.join(", ")
            ),
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
