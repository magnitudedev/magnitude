//! Nemotron-H as a registered model family.

use crate::{inspect_components, recognize};
use magnitude_artifacts::{gguf::Directory, PackageIdentity};
use magnitude_family_common::TextInput;
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, MarkerTokens, ModelDefinition, ModelFamily, TapPoint,
};

pub struct NemotronHFamily;

impl ModelFamily for NemotronHFamily {
    fn name(&self) -> &'static str {
        "nemotron_h"
    }

    fn recognizes(&self, target: &Directory) -> bool {
        recognize(target).is_ok()
    }

    fn inspect(
        &self,
        target: &Directory,
        projector: Option<&Directory>,
        identity: PackageIdentity,
    ) -> Result<ModelDefinition, FamilyError> {
        inspect_components(target, projector, identity)
            .map_err(|error| FamilyError(error.to_string()))
    }

    /// The model is text only and has no rotary embedding, so a row's single
    /// coordinate is its absolute position.
    fn input_adapter(
        &self,
        definition: &ModelDefinition,
        _markers: &dyn MarkerTokens,
    ) -> Result<FamilyInputAdapter, FamilyError> {
        Ok(Box::new(TextInput::new(definition)))
    }

    fn media_placeholder(&self, _definition: &ModelDefinition) -> Option<&'static str> {
        None
    }

    /// Each GGUF layer is one sublayer, in execution order: layer `l` is the
    /// residual entering the `l`-th sublayer, and the sublayer count is the
    /// final pre-norm residual.
    fn layer_entry(
        &self,
        definition: &ModelDefinition,
        layer: u64,
    ) -> Result<TapPoint, FamilyError> {
        let sublayers = definition
            .decoder
            .sublayers()
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        match usize::try_from(layer) {
            Ok(index) if index < sublayers.len() => Ok(TapPoint::Sublayer(sublayers[index])),
            Ok(index) if index == sublayers.len() => Ok(TapPoint::Exit),
            _ => Err(FamilyError(format!(
                "target layer {layer} is beyond the model's {} layers",
                sublayers.len()
            ))),
        }
    }
}
