use std::{fmt, io, path::PathBuf};

/// Why a split GGUF's shard list or split metadata could not be resolved or merged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SplitError {
    /// An empty shard list was supplied.
    Empty,
    /// A single path names shard `index + 1` of `count` rather than the first shard.
    NotFirstShard {
        path: PathBuf,
        index: u64,
        count: u64,
    },
    /// A split first shard's filename is not valid UTF-8 or does not match
    /// `<prefix>-00001-of-<count:05>.gguf`.
    MisnamedFirstShard {
        path: PathBuf,
        count: u64,
    },
    /// Malformed or inconsistent `split.no` / `split.count` / `split.tensors.count` metadata.
    InconsistentMetadata(String),
    /// Shard at 1-based position `expected` does not declare itself shard `expected` of `count`.
    ShardPositionMismatch {
        expected: u64,
        count: u64,
    },
    /// Split shards disagree on GGUF version, byte order, or alignment.
    HeaderMismatch,
    /// A tensor name appears in more than one split shard.
    DuplicateTensor {
        name: String,
    },
    /// Declared `split.tensors.count` does not match the merged shard tensor count.
    TensorCountMismatch {
        declared: u64,
        actual: usize,
    },
    /// Shard count, index, or merged data offset overflowed `u64`.
    OffsetOverflow,
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("split GGUF has no shards"),
            Self::NotFirstShard { path, index, count } => write!(
                f,
                "{} is shard {} of {count} in a split GGUF; open its first shard",
                path.display(),
                index + 1
            ),
            Self::MisnamedFirstShard { path, count } => write!(
                f,
                "first shard {} of a {count}-file split GGUF is not named <prefix>-00001-of-{count:05}.gguf",
                path.display()
            ),
            Self::InconsistentMetadata(detail) => f.write_str(detail),
            Self::ShardPositionMismatch { expected, count } => write!(
                f,
                "shard {expected} does not declare itself shard {expected} of {count}"
            ),
            Self::HeaderMismatch => {
                f.write_str("split GGUF shards differ in version, byte order or alignment")
            }
            Self::DuplicateTensor { name } => {
                write!(f, "tensor {name:?} appears in more than one split shard")
            }
            Self::TensorCountMismatch { declared, actual } => write!(
                f,
                "split GGUF declares {declared} tensors but its shards hold {actual}"
            ),
            Self::OffsetOverflow => f.write_str("split GGUF shard index or data offset overflows"),
        }
    }
}

impl std::error::Error for SplitError {}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Invalid(String),
    Split(SplitError),
    Truncated { offset: u64 },
    AmbiguousProjectors(Vec<PathBuf>),
    /// A well-formed header whose tensors use type ids this parser doesn't know (a quantization
    /// newer than it): each id with how many tensors use it, and the directory of every other
    /// tensor, whose metadata is intact.
    UnknownTensorTypes {
        types: Vec<(u32, u32)>,
        directory: Box<crate::gguf::Directory>,
    },
    /// A metadata array read from a header skeleton whose payload, apart from it, can't be read.
    PayloadRead {
        key: String,
        error: io::Error,
    },
    /// A metadata array read from a header skeleton whose payload, apart from it, is not the
    /// bytes its length and digest name.
    DamagedPayload {
        key: String,
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
            Self::Split(error) => fmt::Display::fmt(error, f),
            Self::Truncated { offset } => write!(f, "truncated GGUF header at byte {offset}"),
            Self::PayloadRead { key, error } => {
                write!(f, "the payload of GGUF metadata {key:?}: {error}")
            }
            Self::DamagedPayload { key } => {
                write!(f, "the payload of GGUF metadata {key:?} is damaged")
            }
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
            | Self::Split(_)
            | Self::Truncated { .. }
            | Self::ShortPayload
            | Self::TrailingPayload => true,
            Self::Io(error) => error.kind() == io::ErrorKind::UnexpectedEof,
            Self::AmbiguousProjectors(_)
            | Self::UnknownTensorTypes { .. }
            | Self::PayloadRead { .. }
            | Self::DamagedPayload { .. }
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
            Self::Io(error) | Self::PayloadRead { error, .. } => Some(error),
            Self::Invalid(_)
            | Self::Split(_)
            | Self::Truncated { .. }
            | Self::AmbiguousProjectors(_)
            | Self::UnknownTensorTypes { .. }
            | Self::DamagedPayload { .. }
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

impl From<SplitError> for Error {
    fn from(error: SplitError) -> Self {
        Self::Split(error)
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}
