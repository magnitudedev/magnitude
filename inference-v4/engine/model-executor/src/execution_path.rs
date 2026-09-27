use std::fmt;

/// One execution implementation selected for the lifetime of an engine.
/// Entries may never choose or retry a different path independently.
///
/// `Native` is backend-neutral: its backend is the backend of the device the
/// engine opens (Metal, CUDA or CPU), and every native entry is prepared from
/// that backend's declarations. Errors name the path and the backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExecutionPath {
    Native,
    Planned,
}

impl fmt::Display for ExecutionPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Native => "native",
            Self::Planned => "planned",
        })
    }
}
