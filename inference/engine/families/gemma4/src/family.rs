//! Gemma 4 (dense, routed and per-layer-input variants) as a registered model
//! family.

use crate::inputs::{input_adapter, IMAGE_PLACEHOLDER};
use crate::{inspect_components, ARCHITECTURE};
use magnitude_artifacts::{gguf::Directory, PackageIdentity};
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, ImageMarkers, MarkerTokens, ModelDefinition, ModelFamily,
};

pub struct Gemma4Family;

impl ModelFamily for Gemma4Family {
    fn name(&self) -> &'static str {
        ARCHITECTURE
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
    }

    fn input_adapter(
        &self,
        definition: &ModelDefinition,
        markers: &dyn MarkerTokens,
    ) -> Result<FamilyInputAdapter, FamilyError> {
        let markers = definition
            .vision
            .as_ref()
            .map(|_| {
                ImageMarkers::new(
                    markers.marker("<|image>")?,
                    markers.marker("<|image|>")?,
                    markers.marker("<image|>")?,
                )
                .map_err(|error| FamilyError::invalid(error.to_string()))
            })
            .transpose()?;
        Ok(Box::new(input_adapter(markers)))
    }

    fn media_placeholder(&self, definition: &ModelDefinition) -> Option<&'static str> {
        definition.vision.as_ref().map(|_| IMAGE_PLACEHOLDER)
    }
}
