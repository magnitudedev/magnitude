use std::fmt;

/// One execution implementation selected for the lifetime of an engine.
/// Entries may never choose or retry a different path independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExecutionPath {
    NativeMetal,
    Planned,
}

impl fmt::Display for ExecutionPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NativeMetal => "native-metal",
            Self::Planned => "planned",
        })
    }
}
