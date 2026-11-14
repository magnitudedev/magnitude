//! GLM-5.3-Flash as a registered model family.

use crate::{inspect_components, FAMILY_ID};
use magnitude_artifacts::{gguf::Directory, BoundaryRule, PackageIdentity};
use magnitude_family_common::TextInput;
use magnitude_family_contracts::{
    cell_raster_controls, FamilyError, FamilyInputAdapter, ImageMarkers, MarkerTokens,
    ModelDefinition, ModelFamily, SequentialImageInput,
};

/// An image's start, row and end markers: the text one image renders as,
/// which input preparation expands to one row per merged cell.
const IMAGE_MARKERS: [&str; 3] = ["<|begin_of_image|>", "<|image|>", "<|end_of_image|>"];
const IMAGE_PLACEHOLDER: &str = "<|begin_of_image|><|image|><|end_of_image|>";

pub struct Glm5NextFamily;

impl ModelFamily for Glm5NextFamily {
    fn name(&self) -> &'static str {
        FAMILY_ID
    }

    fn architectures(&self) -> &'static [&'static str] {
        &[FAMILY_ID]
    }

    fn inspect(
        &self,
        target: &Directory,
        projector: Option<&Directory>,
        identity: PackageIdentity,
    ) -> Result<ModelDefinition, FamilyError> {
        inspect_components(target, projector, identity)
    }

    /// Every token sits at its absolute position (the model has no
    /// positions beyond the order of its rows); an image expands between its
    /// markers to one row per merged cell, and without a projector media is
    /// refused.
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
        // The decoder reads media rows causally, so a span may be split.
        Ok(Box::new(SequentialImageInput::new(
            FAMILY_ID,
            Some(markers),
            BoundaryRule::Causal,
            cell_raster_controls,
        )))
    }

    fn media_placeholder(&self, definition: &ModelDefinition) -> Option<&'static str> {
        definition.vision.as_ref().map(|_| IMAGE_PLACEHOLDER)
    }
}
