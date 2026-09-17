//! Device-independent validation of stored model artifacts and weight roles.
pub mod descriptor;
pub mod gguf;
pub mod mlx;
pub mod residency;
pub mod safetensors;
pub mod source;
use std::{fmt, io};
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Invalid(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "weight source: {e}"),
            Self::Invalid(e) => f.write_str(e),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
