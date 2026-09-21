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
    Seismic(crate::Error),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "weight source: {e}"),
            Self::Invalid(e) => f.write_str(e),
            Self::Seismic(e) => write!(f, "weight execution: {e}"),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Seismic(e) => Some(e),
            Self::Invalid(_) => None,
        }
    }
}
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<crate::Error> for Error {
    fn from(error: crate::Error) -> Self {
        Self::Seismic(error)
    }
}

macro_rules! seismic_error {
    ($source:ty) => {
        impl From<$source> for Error {
            fn from(error: $source) -> Self {
                Self::Seismic(error.into())
            }
        }
    };
}

seismic_error!(seismic::TensorError);
seismic_error!(seismic::LoadError);
seismic_error!(seismic::CallError);
