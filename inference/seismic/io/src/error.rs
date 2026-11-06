use std::{fmt, io};

/// Why a read, or a reader's opening, failed.
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    /// A read the file or its destination cannot hold: outside the file, or
    /// past its destination's region.
    Invalid(String),
    /// A read cancelled before it started (its [`crate::Handle`] dropped): its
    /// range was not read.
    Cancelled,
    /// A reader thread the read needs could not start.
    ThreadStart(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "file read: {error}"),
            Self::Invalid(message) => f.write_str(message),
            Self::Cancelled => f.write_str("cancelled before it was read"),
            Self::ThreadStart(error) => write!(f, "a reader thread could not start: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) | Self::ThreadStart(error) => Some(error),
            Self::Invalid(_) | Self::Cancelled => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<crate::OpenError> for Error {
    fn from(error: crate::OpenError) -> Self {
        match error {
            crate::OpenError::Io(error) => Self::Io(error),
            crate::OpenError::NotAFile => Self::Invalid("not a regular file".into()),
        }
    }
}
