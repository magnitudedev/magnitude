//! Nemotron-H as a registered model family.

use crate::inputs::{input_adapter, IMAGE_MARKERS, IMAGE_PLACEHOLDER};
use crate::{inspect_components, ARCHITECTURE};
use magnitude_artifacts::{gguf::Directory, PackageIdentity};
use magnitude_family_common::TextInput;
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, ImageMarkers, MarkerTokens, ModelDefinition, ModelFamily,
    TapPoint,
};

pub struct NemotronHFamily;

impl ModelFamily for NemotronHFamily {
    fn name(&self) -> &'static str {
        "nemotron_h"
    }

    fn architectures(&self) -> &'static [&'static str] {
        &[ARCHITECTURE]
    }

    fn inspect(
        &self,
        target: &Directory,
        projector: Option<&Directory>,
        identity: PackageIdentity,
    ) -> Result<ModelDefinition, FamilyError> {
        inspect_components(target, projector, identity)
            .map_err(FamilyError::from)
    }

    /// The model has no rotary embedding, so a row's single coordinate is its
    /// absolute position; images expand between their markers.
    fn input_adapter(
        &self,
        definition: &ModelDefinition,
        markers: &dyn MarkerTokens,
    ) -> Result<FamilyInputAdapter, FamilyError> {
        if definition.vision.is_none() {
            return Ok(Box::new(TextInput::new(definition)));
        }
        let [start, row, end] = IMAGE_MARKERS;
        let markers = ImageMarkers::new(
            markers.marker(start)?,
            markers.marker(row)?,
            markers.marker(end)?,
        )
        .map_err(|error| FamilyError::invalid(error.to_string()))?;
        Ok(Box::new(input_adapter(markers)))
    }

    fn media_placeholder(&self, definition: &ModelDefinition) -> Option<&'static str> {
        definition.vision.as_ref().map(|_| IMAGE_PLACEHOLDER)
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
            _ => Err(FamilyError::mismatch(format!(
                "target layer {layer} is beyond the model's {} layers",
                sublayers.len()
            ))),
        }
    }
}
