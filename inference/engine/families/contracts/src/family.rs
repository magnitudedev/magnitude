//! The one extension point for model architectures (integration spec §4.3).
//!
//! A family recognizes artifact headers, builds the family-neutral
//! [`ModelDefinition`], and supplies its input adapter and media placeholder
//! policy. Everything downstream consumes the definition and the adapter
//! trait only, so adding a family changes nothing above this contract.

use crate::{ModelDefinition, ModelInputAdapter, SublayerIndex, TapPoint};
use magnitude_artifacts::{gguf::Directory, PackageIdentity, TokenId};
use std::{error, fmt};

/// A registered family rejected material it recognized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyError(pub String);

impl fmt::Display for FamilyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
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

    /// Whether this family claims the target from its header metadata alone.
    /// Recognition interprets no geometry.
    fn recognizes(&self, target: &Directory) -> bool;

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
                    .map_err(|_| FamilyError(format!("target layer {layer} exceeds u32")))?,
                sublayer: 0,
            })),
            std::cmp::Ordering::Equal => Ok(TapPoint::Exit),
            std::cmp::Ordering::Greater => Err(FamilyError(format!(
                "target layer {layer} is beyond the target's {blocks} layers"
            ))),
        }
    }
}
