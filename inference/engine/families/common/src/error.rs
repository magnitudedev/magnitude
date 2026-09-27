//! Why a header could not be read. Families convert it into their own error
//! type; the wording here is the default a family may keep.

use magnitude_artifacts::gguf::Encoding;
use magnitude_family_contracts::DefinitionError;
use std::{error, fmt};

#[derive(Clone, Debug, PartialEq)]
pub enum HeaderError {
    /// An architecture key the family does not interpret.
    UnknownMetadata(String),
    MissingMetadata(String),
    /// A key present with a value outside what it admits.
    MetadataType {
        key: String,
        expected: &'static str,
    },
    MissingWeight(String),
    WeightShape {
        name: String,
        expected: Vec<u64>,
        received: Vec<u64>,
    },
    /// A weight read as a matrix is not one.
    NotMatrix { name: String, shape: Vec<u64> },
    /// A stored second-level scale that is not one F32 value per matrix.
    CompanionScale {
        name: String,
        matrices: u64,
        encoding: Encoding,
        shape: Vec<u64>,
    },
    /// A stored tensor no role bound.
    UnboundWeight(String),
    /// A rotary table over an odd, empty or oversized dimension range, or a
    /// base that does not decay.
    Rotary { rotated: u64, width: u64, base: f64 },
    /// YaRN parameters outside their admitted ranges.
    Yarn,
    Definition(DefinitionError),
}

impl fmt::Display for HeaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownMetadata(key) => write!(formatter, "unknown metadata {key:?}"),
            Self::MissingMetadata(key) => write!(formatter, "missing metadata {key}"),
            Self::MetadataType { key, expected } => write!(formatter, "{key} must be {expected}"),
            Self::MissingWeight(name) => write!(formatter, "missing weight {name:?}"),
            Self::WeightShape {
                name,
                expected,
                received,
            } => write!(
                formatter,
                "weight {name:?}: expected {expected:?}, received {received:?}"
            ),
            Self::NotMatrix { name, shape } => write!(
                formatter,
                "weight {name:?} must be a matrix, received {shape:?}"
            ),
            Self::CompanionScale {
                name,
                matrices,
                encoding,
                shape,
            } => write!(
                formatter,
                "scale {name:?}: expected F32 [{matrices}], received {encoding:?} {shape:?}"
            ),
            Self::UnboundWeight(name) => {
                write!(formatter, "artifact contains unbound weight role {name:?}")
            }
            Self::Rotary {
                rotated,
                width,
                base,
            } => write!(
                formatter,
                "invalid rotary: {rotated} of {width} dimensions at base {base}"
            ),
            Self::Yarn => formatter.write_str("invalid YaRN scaling"),
            Self::Definition(error) => error.fmt(formatter),
        }
    }
}

impl error::Error for HeaderError {}

impl From<DefinitionError> for HeaderError {
    fn from(error: DefinitionError) -> Self {
        Self::Definition(error)
    }
}
