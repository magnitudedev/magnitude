//! Why a header could not be read. Families convert it into their own error
//! type; the wording here is the default a family may keep.

use magnitude_artifacts::{gguf::Encoding, UnsupportedReason};
use magnitude_family_contracts::{DefinitionError, FamilyError};
use std::{error, fmt};

#[derive(Clone, Debug, PartialEq)]
pub enum HeaderError {
    /// Every architecture key the family does not interpret.
    UnknownMetadata(Vec<String>),
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
    /// Every stored tensor no role bound.
    UnboundWeights(Vec<String>),
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
            Self::UnknownMetadata(keys) => write!(formatter, "unknown metadata {keys:?}"),
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
            Self::UnboundWeights(names) => {
                write!(formatter, "artifact contains unbound weight roles {names:?}")
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

impl HeaderError {
    /// The unsupported features this error names: unknown keys, keys or
    /// tensors stored in a form the family doesn't read (such as a weight
    /// with an extra unit dimension), unbound tensors, rope scaling outside
    /// the admitted ranges, and an output head absent because it is tied to
    /// the embedding. A header missing anything else the family needs, or
    /// whose values contradict each other, is `Unreadable`.
    pub fn reasons(&self) -> Vec<UnsupportedReason> {
        match self {
            Self::UnknownMetadata(keys) => keys
                .iter()
                .map(|key| UnsupportedReason::UnrecognizedMetadata { key: key.clone() })
                .collect(),
            Self::MetadataType { key, .. } => {
                vec![UnsupportedReason::UnrecognizedMetadata { key: key.clone() }]
            }
            Self::UnboundWeights(names) => names
                .iter()
                .map(|name| UnsupportedReason::UnrecognizedTensor { name: name.clone() })
                .collect(),
            Self::WeightShape { name, .. }
            | Self::NotMatrix { name, .. }
            | Self::CompanionScale { name, .. } => {
                vec![UnsupportedReason::UnrecognizedTensor { name: name.clone() }]
            }
            Self::Yarn => vec![UnsupportedReason::RopeScaling {
                kind: "yarn".into(),
            }],
            Self::MissingWeight(name) if name == OUTPUT_HEAD => {
                vec![UnsupportedReason::TiedOutputHead]
            }
            Self::MissingMetadata(_)
            | Self::MissingWeight(_)
            | Self::Rotary { .. }
            | Self::Definition(_) => vec![UnsupportedReason::Unreadable],
        }
    }
}

/// The output projection a GGUF stores apart from the embedding unless the
/// two are tied.
const OUTPUT_HEAD: &str = "output.weight";

impl From<HeaderError> for FamilyError {
    fn from(error: HeaderError) -> Self {
        FamilyError::unsupported(error.reasons(), error.to_string())
    }
}

impl From<DefinitionError> for HeaderError {
    fn from(error: DefinitionError) -> Self {
        Self::Definition(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_weight_in_a_form_the_family_does_not_read_names_that_tensor() {
        let error = HeaderError::WeightShape {
            name: "blk.0.ffn_gate_inp_shexp.weight".into(),
            expected: vec![2048],
            received: vec![1, 2048],
        };
        assert_eq!(
            error.reasons(),
            [UnsupportedReason::UnrecognizedTensor {
                name: "blk.0.ffn_gate_inp_shexp.weight".into()
            }]
        );
    }

    #[test]
    fn a_header_missing_what_the_family_needs_is_unreadable_unless_its_head_is_tied() {
        assert_eq!(
            HeaderError::MissingWeight("output.weight".into()).reasons(),
            [UnsupportedReason::TiedOutputHead]
        );
        for error in [
            HeaderError::MissingWeight("blk.0.attn_q.weight".into()),
            HeaderError::MissingMetadata("lfm2.block_count".into()),
            HeaderError::Rotary {
                rotated: 3,
                width: 64,
                base: 10_000.0,
            },
        ] {
            assert_eq!(error.reasons(), [UnsupportedReason::Unreadable], "{error}");
        }
    }
}
