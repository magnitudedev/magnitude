use serde::{Deserialize, Serialize};

/// Stable vocabulary identity shared by host preparation, generation, and
/// numerical execution. Artifact ownership is the lowest common layer that
/// can define it without coupling model-family adapters to an executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokenId(pub u32);
