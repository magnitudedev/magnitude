use std::{fmt, io, path::PathBuf};

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Invalid(String),
    AmbiguousProjectors(Vec<PathBuf>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "artifact source: {error}"),
            Self::Invalid(message) => f.write_str(message),
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

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Invalid(_) | Self::AmbiguousProjectors(_) => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}
