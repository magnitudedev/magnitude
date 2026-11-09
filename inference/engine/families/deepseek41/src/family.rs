//! DeepSeek V4.1 Flash as a registered model family.

use crate::{inspect_components, FAMILY_ID};
use magnitude_artifacts::{gguf::Directory, PackageIdentity};
use magnitude_family_common::TextInput;
use magnitude_family_contracts::{
    FamilyError, FamilyInputAdapter, MarkerTokens, ModelDefinition, ModelFamily,
};

pub struct DeepSeek41Family;

impl ModelFamily for DeepSeek41Family {
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

    /// Text-only input until the vision projector is interpreted: every
    /// token sits at its absolute position, and media is refused.
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
