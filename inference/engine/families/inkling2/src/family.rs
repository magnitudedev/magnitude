//! Inkling 2 as a registered model family.

use crate::{ARCHITECTURE, ARCHITECTURE_LOOP, inspect_components};
use magnitude_artifacts::{BoundaryRule, PackageIdentity, gguf::Directory};
use magnitude_family_common::TextInput;
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, ImageMarkers, MarkerTokens, ModelDefinition, ModelFamily,
    SequentialImageInput, cell_raster_controls,
};

/// Image boundary markers in Inkling 2's chat template.
const IMAGE_MARKERS: [&str; 3] = ["<|image_start|>", "<|image_pad|>", "<|image_end|>"];
const IMAGE_PLACEHOLDER: &str = "<|image_start|><|image_pad|><|image_end|>";

pub struct Inkling2Family;

impl ModelFamily for Inkling2Family {
    fn name(&self) -> &'static str {
        ARCHITECTURE
    }

    fn architectures(&self) -> &'static [&'static str] {
        &[ARCHITECTURE, ARCHITECTURE_LOOP]
    }

    fn inspect(
        &self,
        target: &Directory,
        projector: Option<&Directory>,
        identity: PackageIdentity,
    ) -> Result<ModelDefinition, FamilyError> {
        inspect_components(target, projector, identity)
    }

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
        Ok(Box::new(SequentialImageInput::new(
            ARCHITECTURE,
            Some(markers),
            BoundaryRule::Causal,
            cell_raster_controls,
        )))
    }

    fn media_placeholder(&self, definition: &ModelDefinition) -> Option<&'static str> {
        definition.vision.as_ref().map(|_| IMAGE_PLACEHOLDER)
    }
}
