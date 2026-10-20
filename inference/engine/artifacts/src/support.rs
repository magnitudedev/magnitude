//! Why model material cannot run, typed where the engine's layers raise it.
//!
//! Families, the tokenizer adapter, weight planning and the execution
//! backends each name what they refuse in this one vocabulary, so a refusal
//! keeps its cause from the layer that raised it to the host.

use serde::{Deserialize, Serialize};
use std::fmt;

/// One cause a model or drafter cannot run. `TargetMismatch` is a drafter
/// or projector that does not fit its target; `MissingMetadata` is a
/// well-formed file that omits a declaration the engine requires and never
/// infers (a block drafter's attention causality); `Unreadable` is a
/// truncated or invalid GGUF, or a header missing what its family needs;
/// `DraftArchitecture` is a separate drafter whose `general.architecture`
/// no drafter family binds (a model architecture's prediction head the
/// engine drafts with only when it is built in, or not at all).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnsupportedReason {
    Architecture {
        architecture: String,
    },
    WeightTypes {
        types: Vec<WeightTypeCount>,
    },
    RopeScaling {
        kind: String,
    },
    UnrecognizedTensor {
        name: String,
    },
    UnrecognizedMetadata {
        key: String,
    },
    MissingMetadata {
        key: String,
    },
    Tokenizer {
        model: String,
        pre: Option<String>,
    },
    ChatTemplate,
    KernelDomain {
        backend: String,
        entry: String,
        statics: Vec<(String, u64)>,
    },
    Backend {
        backend: String,
    },
    Unreadable,
    TargetMismatch {
        detail: String,
    },
    TiedOutputHead,
    DraftArchitecture {
        architecture: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WeightTypeCount {
    pub ggml_type: String,
    pub tensor_count: u32,
}

impl fmt::Display for UnsupportedReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Architecture { architecture } => {
                write!(formatter, "architecture {architecture:?}")
            }
            Self::WeightTypes { types } => {
                formatter.write_str("weight types ")?;
                for (index, count) in types.iter().enumerate() {
                    if index > 0 {
                        formatter.write_str(", ")?;
                    }
                    write!(
                        formatter,
                        "{} ({} tensors)",
                        count.ggml_type, count.tensor_count
                    )?;
                }
                Ok(())
            }
            Self::RopeScaling { kind } => write!(formatter, "rope scaling {kind:?}"),
            Self::UnrecognizedTensor { name } => write!(formatter, "tensor {name:?}"),
            Self::UnrecognizedMetadata { key } => write!(formatter, "metadata {key:?}"),
            Self::MissingMetadata { key } => write!(formatter, "missing metadata {key:?}"),
            Self::Tokenizer { model, pre } => {
                write!(
                    formatter,
                    "tokenizer model {model:?}, pre-tokenizer {pre:?}"
                )
            }
            Self::ChatTemplate => formatter.write_str("chat template"),
            Self::KernelDomain {
                backend,
                entry,
                statics,
            } => {
                write!(formatter, "`{entry}` on {backend} at ")?;
                for (index, (name, value)) in statics.iter().enumerate() {
                    if index > 0 {
                        formatter.write_str(", ")?;
                    }
                    write!(formatter, "{name}={value}")?;
                }
                Ok(())
            }
            Self::Backend { backend } => write!(formatter, "the {backend} backend"),
            Self::Unreadable => formatter.write_str("unreadable GGUF"),
            Self::TargetMismatch { detail } => write!(formatter, "target mismatch: {detail}"),
            Self::TiedOutputHead => formatter.write_str("tied output head"),
            Self::DraftArchitecture { architecture } => {
                write!(formatter, "separate drafter of architecture {architecture:?}")
            }
        }
    }
}
