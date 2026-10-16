//! The one extension point for model architectures (integration spec §4.3).
//!
//! A family recognizes artifact headers, builds the family-neutral
//! [`ModelDefinition`], and supplies its input adapter and media placeholder
//! policy. Everything downstream consumes the definition and the adapter
//! trait only, so adding a family changes nothing above this contract.

use crate::{ModelDefinition, ModelInputAdapter, SublayerIndex, TapPoint};
use magnitude_artifacts::{
    gguf::{Directory, Value},
    PackageIdentity, TokenId, UnsupportedReason,
};
use std::{error, fmt};

/// A registered family rejected material it recognized: every unsupported
/// feature it can name, and the developer text that describes the refusal.
/// `reasons` is empty when the material is malformed rather than outside a
/// supported feature set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyError {
    pub reasons: Vec<UnsupportedReason>,
    pub detail: String,
}

impl FamilyError {
    /// A refusal with no typed cause.
    pub fn invalid(detail: impl Into<String>) -> Self {
        Self {
            reasons: Vec::new(),
            detail: detail.into(),
        }
    }

    pub fn unsupported(reasons: Vec<UnsupportedReason>, detail: impl Into<String>) -> Self {
        Self {
            reasons,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for FamilyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl error::Error for FamilyError {}

/// The package tokenizer's view a family needs to configure its adapter:
/// the identity of a special marker that must encode to exactly one token.
pub trait MarkerTokens {
    fn marker(&self, text: &str) -> Result<TokenId, FamilyError>;
}

/// The input adapter a family configures for one package.
pub type FamilyInputAdapter = Box<dyn ModelInputAdapter + Send + Sync>;

pub trait ModelFamily: Send + Sync {
    /// Stable family name, for diagnostics.
    fn name(&self) -> &'static str;

    /// The `general.architecture` values this family claims. Claiming an
    /// architecture is recognition only: whether the family admits the
    /// file's feature set is decided by [`Self::inspect`], so a refusal names
    /// what it does not admit instead of disowning the file.
    fn architectures(&self) -> &'static [&'static str];

    /// Whether this family claims the target's architecture.
    fn recognizes(&self, target: &Directory) -> bool {
        target
            .value("general.architecture")
            .and_then(Value::string)
            .is_some_and(|architecture| self.architectures().contains(&architecture))
    }

    /// Build the numerical definition of a recognized package from headers.
    fn inspect(
        &self,
        target: &Directory,
        projector: Option<&Directory>,
        identity: PackageIdentity,
    ) -> Result<ModelDefinition, FamilyError>;

    /// The adapter that turns tokens and prepared media into the closed
    /// numerical input contract, configured with the package's markers.
    fn input_adapter(
        &self,
        definition: &ModelDefinition,
        markers: &dyn MarkerTokens,
    ) -> Result<FamilyInputAdapter, FamilyError>;

    /// The text one image renders as in the family's chat template, when the
    /// definition has a vision component.
    fn media_placeholder(&self, definition: &ModelDefinition) -> Option<&'static str>;

    /// The target residual a draft component taps at the artifact's layer
    /// `layer` (a draft names target layers by their GGUF index): the input
    /// of that layer. By default a GGUF layer is one decoder block, and
    /// `layer == blocks` is the final pre-norm residual; a family whose
    /// layers are not blocks overrides this.
    fn layer_entry(
        &self,
        definition: &ModelDefinition,
        layer: u64,
    ) -> Result<TapPoint, FamilyError> {
        let blocks = definition.decoder.blocks.len() as u64;
        match layer.cmp(&blocks) {
            std::cmp::Ordering::Less => Ok(TapPoint::Sublayer(SublayerIndex {
                block: u32::try_from(layer)
                    .map_err(|_| FamilyError::invalid(format!("target layer {layer} exceeds u32")))?,
                sublayer: 0,
            })),
            std::cmp::Ordering::Equal => Ok(TapPoint::Exit),
            std::cmp::Ordering::Greater => Err(FamilyError::invalid(format!(
                "target layer {layer} is beyond the target's {blocks} layers"
            ))),
        }
    }
}
