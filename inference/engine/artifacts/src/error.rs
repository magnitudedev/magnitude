use std::{fmt, io, path::PathBuf};

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Invalid(String),
    Truncated { offset: u64 },
    AmbiguousProjectors(Vec<PathBuf>),
    /// A well-formed header whose tensors use type ids this parser doesn't know (a quantization
    /// newer than it): each id with how many tensors use it, and the directory of every other
    /// tensor, whose metadata is intact.
    UnknownTensorTypes {
        types: Vec<(u32, u32)>,
        directory: Box<crate::gguf::Directory>,
    },
    /// A metadata array's payload that ends before the elements its count names.
    ShortPayload,
    /// A metadata array's payload that holds bytes past the elements its count names.
    TrailingPayload,
    /// A digest asked of a metadata array that isn't text ([`crate::gguf::Array::digest`]).
    DigestOfNonText,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "artifact source: {error}"),
            Self::Invalid(message) => f.write_str(message),
            Self::Truncated { offset } => write!(f, "truncated GGUF header at byte {offset}"),
            Self::ShortPayload => {
                f.write_str("a GGUF metadata array's payload ends before its elements")
            }
            Self::TrailingPayload => {
                f.write_str("a GGUF metadata array's payload holds bytes past its elements")
            }
            Self::DigestOfNonText => f.write_str("only a GGUF text array has a digest"),
            Self::UnknownTensorTypes { types, .. } => write!(
                f,
                "unknown GGUF tensor types {}",
                types
                    .iter()
                    .map(|(id, count)| format!("{id} ({count} tensors)"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
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
            Self::Invalid(_)
            | Self::Truncated { .. }
            | Self::ShortPayload
            | Self::TrailingPayload => true,
            Self::Io(error) => error.kind() == io::ErrorKind::UnexpectedEof,
            Self::AmbiguousProjectors(_)
            | Self::UnknownTensorTypes { .. }
            | Self::DigestOfNonText => false,
        }
    }

    /// The weight types a header names that this parser doesn't know, as the engine reports
    /// unsupported weight types: `type <id>`, since an unknown id has no name.
    pub fn unknown_weight_types(&self) -> Option<Vec<crate::support::WeightTypeCount>> {
        match self {
            Self::UnknownTensorTypes { types, .. } => Some(
                types
                    .iter()
                    .map(|(id, count)| crate::support::WeightTypeCount {
                        ggml_type: format!("type {id}"),
                        tensor_count: *count,
                    })
                    .collect(),
            ),
            _ => None,
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Invalid(_)
            | Self::Truncated { .. }
            | Self::AmbiguousProjectors(_)
            | Self::UnknownTensorTypes { .. }
            | Self::ShortPayload
            | Self::TrailingPayload
            | Self::DigestOfNonText => None,
        }
    }
}

impl From<seismic_io::OpenError> for Error {
    fn from(error: seismic_io::OpenError) -> Self {
        match error {
            seismic_io::OpenError::Io(error) => Self::Io(error),
            seismic_io::OpenError::NotAFile => Self::Invalid("artifact source is not a regular file".into()),
        }
    }
}

impl From<seismic_io::Error> for Error {
    fn from(error: seismic_io::Error) -> Self {
        match error {
            seismic_io::Error::Io(error) => Self::Io(error),
            error => Self::Io(io::Error::other(error)),
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}
