//! The strict `llama` feature set as a registered model family.

use crate::{inspect_components, recognize};
use magnitude_artifacts::{gguf::Directory, PackageIdentity};
use magnitude_family_common::TextInput;
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, MarkerTokens, ModelDefinition, ModelFamily,
};

pub struct LlamaFamily;

impl ModelFamily for LlamaFamily {
    fn name(&self) -> &'static str {
        "llama"
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

    /// Text-only input: every token sits at its absolute position.
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
}
