//! LFM2 (dense and routed) and LFM3 as a registered model family.

use crate::{
    inputs::{input_adapter, IMAGE_MARKERS, IMAGE_PLACEHOLDER},
    inspect_components,
};
use magnitude_artifacts::{gguf::Directory, PackageIdentity};
use magnitude_family_common::TextInput;
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, ImageMarkers, MarkerTokens, ModelDefinition, ModelFamily,
};

pub struct Lfm2Family;

impl ModelFamily for Lfm2Family {
    fn name(&self) -> &'static str {
        "lfm2"
    }

    fn architectures(&self) -> &'static [&'static str] {
        &["lfm2", "lfm2moe", "lfm3"]
    }

    fn inspect(
        &self,
        target: &Directory,
        projector: Option<&Directory>,
        identity: PackageIdentity,
    ) -> Result<ModelDefinition, FamilyError> {
        inspect_components(target, projector, identity)
    }

    /// Text rows carry their absolute positions; an LFM3-VL image expands
    /// between its markers to one row per merged cell.
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
}
