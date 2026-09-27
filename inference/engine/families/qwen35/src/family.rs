//! Qwen 3.5 (dense and routed) as a registered model family.

use crate::inputs::{QwenImageTokens, QwenInputAdapter};
use crate::{inspect_components, recognize};
use magnitude_artifacts::{gguf::Directory, PackageIdentity};
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, MarkerTokens, ModelDefinition, ModelFamily,
};

/// The text one image renders as in the Qwen chat template: the image pad
/// between the vision delimiters, which input preparation expands to the
/// image's merged patch rows.
const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

pub struct Qwen35Family;

impl ModelFamily for Qwen35Family {
    fn name(&self) -> &'static str {
        "qwen35"
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
        inspect_components(target, projector, identity).map_err(|error| FamilyError(error.to_string()))
    }

    fn input_adapter(
        &self,
        definition: &ModelDefinition,
        markers: &dyn MarkerTokens,
    ) -> Result<FamilyInputAdapter, FamilyError> {
        let image_tokens = definition
            .vision
            .as_ref()
            .map(|_| {
                QwenImageTokens::new(
                    markers.marker("<|image_pad|>")?,
                    markers.marker("<|vision_start|>")?,
                    markers.marker("<|vision_end|>")?,
                )
                .map_err(|error| FamilyError(error.to_string()))
            })
            .transpose()?;
        Ok(Box::new(QwenInputAdapter::new(image_tokens)))
    }

    fn media_placeholder(&self, definition: &ModelDefinition) -> Option<&'static str> {
        definition.vision.as_ref().map(|_| IMAGE_PLACEHOLDER)
    }
}
