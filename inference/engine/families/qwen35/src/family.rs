//! Qwen 3.5 (dense and routed) as a registered model family.

use crate::{FAMILY_IDS, inspect_components};
use magnitude_artifacts::{PackageIdentity, gguf::Directory};
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, MarkerTokens, ModelDefinition, ModelFamily,
};
use magnitude_family_qwen::inputs::{QwenImageTokens, QwenInputAdapter};

/// The text one image renders as in the Qwen chat template: the image pad
/// between the vision delimiters, which input preparation expands to the
/// image's merged patch rows.
const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

pub struct Qwen35Family;

impl ModelFamily for Qwen35Family {
    fn name(&self) -> &'static str {
        "qwen35"
    }

    fn architectures(&self) -> &'static [&'static str] {
        FAMILY_IDS
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
        if definition.vision.as_ref().is_some_and(|v| {
            matches!(
                v.preprocessing.resize,
                magnitude_family_contracts::VisionResize::Slices { .. }
            )
        }) {
            return crate::minicpm::inputs::adapter(definition, markers);
        }
        let image_tokens = definition
            .vision
            .as_ref()
            .map(|_| {
                QwenImageTokens::new(
                    markers.marker("<|image_pad|>")?,
                    markers.marker("<|vision_start|>")?,
                    markers.marker("<|vision_end|>")?,
                )
                .map_err(|error| FamilyError::invalid(error.to_string()))
            })
            .transpose()?;
        Ok(Box::new(QwenInputAdapter::new(FAMILY_IDS, image_tokens)))
    }

    fn media_placeholder(&self, definition: &ModelDefinition) -> Option<&'static str> {
        definition.vision.as_ref().map(|v| {
            if matches!(
                v.preprocessing.resize,
                magnitude_family_contracts::VisionResize::Slices { .. }
            ) {
                "<|image_pad|>"
            } else {
                IMAGE_PLACEHOLDER
            }
        })
    }
}
