use std::{fmt, io, path::PathBuf};

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Invalid(String),
    Truncated { offset: u64 },
    AmbiguousProjectors(Vec<PathBuf>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "artifact source: {error}"),
            Self::Invalid(message) => f.write_str(message),
            Self::Truncated { offset } => write!(f, "truncated GGUF header at byte {offset}"),
            Self::AmbiguousProjectors(paths) => write!(
                f,
                "multiple sibling projector artifacts found: {}",
                paths
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

impl Error {
    /// Whether the material itself is truncated or not a valid container,
    /// as opposed to a source that could not be read.
    pub fn unreadable(&self) -> bool {
        match self {
            Self::Invalid(_) | Self::Truncated { .. } => true,
            Self::Io(error) => error.kind() == io::ErrorKind::UnexpectedEof,
            Self::AmbiguousProjectors(_) => false,
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Invalid(_) | Self::Truncated { .. } | Self::AmbiguousProjectors(_) => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}
